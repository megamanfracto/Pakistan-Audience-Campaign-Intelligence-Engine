"""Project 3 - Pakistan Social Audience & Campaign Intelligence Engine (Colab-friendly).
pip install pandas numpy scikit-learn  |  python engine.py -> posts.csv, comments.csv, results.json
DATA: SIMULATED demo data (source='synthetic'), same schema as Table A / Table B in the brief.
To use real comments: replace `com` below with a DataFrame having comment_text, language_type, topic, sentiment, post_id."""
import json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
rng = np.random.default_rng(42)

# ---------- 1. Data: Table A (posts) + Table B (comments) ----------
CT = {'Reel': 1.6, 'Product photo': 1.0, 'Influencer post': 1.8, 'Styling video': 1.3, 'Sale announcement': 1.1, 'Eid campaign': 2.0}
PL = {'Instagram': 1.0, 'TikTok': 1.7, 'Facebook': 0.7}
CAMP = ['11.11 Sale', '12.12 Sale', 'Eid Collection', 'Summer Lawn']
T = {  # topic: (sentiment, [(text, language_type)])
 'price': ('neutral', [('Price please?', 'English'), ('Rate kya hai is suit ka?', 'Roman Urdu'), ('قیمت کیا ہے؟', 'Urdu'), ('Price bohat zyada hai.', 'Roman Urdu')]),
 'discount': ('neutral', [('Any discount on this suit?', 'English'), ('Sale mein kab lagega?', 'Roman Urdu'), ('کوئی رعایت ملے گی؟', 'Urdu'), ('Discount code do na', 'Roman Urdu')]),
 'availability': ('neutral', [('Is this available in Karachi?', 'English'), ('Stock mein hai ya khatam?', 'Roman Urdu'), ('کیا یہ دستیاب ہے؟', 'Urdu'), ('Lahore store pe milega?', 'Roman Urdu')]),
 'sizing': ('neutral', [('Is medium size in stock?', 'English'), ('Size chart bhej dein', 'Roman Urdu'), ('سائز کتنا ہے؟', 'Urdu'), ('Length kitni hai shirt ki?', 'Roman Urdu')]),
 'fabric': ('neutral', [('Is the fabric pure cotton?', 'English'), ('Kapra kaisa hai, quality achi hai?', 'Roman Urdu'), ('کپڑے کی کوالٹی کیسی ہے؟', 'Urdu'), ('Fabric thin lag raha hai', 'Roman Urdu')]),
 'delivery': ('negative', [('Ordered last week, still waiting.', 'English'), ('Delivery kitne din mein hogi?', 'Roman Urdu'), ('آرڈر ابھی تک نہیں پہنچا', 'Urdu'), ('Parcel abhi tak nahi aaya', 'Roman Urdu')]),
 'purchase_intent': ('positive', [('Inbox me details send kar dein', 'Roman Urdu'), ('I want to order this, DM me', 'English'), ('آرڈر کرنا ہے، تفصیل بھیجیں', 'Urdu'), ('Mujhe ye lena hai, WhatsApp number?', 'Roman Urdu')]),
 'praise': ('positive', [('Very beautiful dress!', 'English'), ('Bohat khoobsurat hai', 'Roman Urdu'), ('بہت خوبصورت', 'Urdu'), ('Love the colours, stunning', 'English')]),
}
TP = dict(price=.2, discount=.12, availability=.12, sizing=.08, fabric=.1, delivery=.1, purchase_intent=.1, praise=.18)

def comments_for(pid, camp, kind):
    p = dict(TP)
    if 'Sale' in camp: p['discount'] *= 2.5; p['price'] *= 1.4
    if kind == 'delivery_storm': p['delivery'] *= 8
    k = list(p); w = np.array([p[x] for x in k]); w /= w.sum()
    out = []
    for t in rng.choice(k, size=rng.integers(3, 9), p=w):
        s, txts = T[t]; tx, lang = txts[rng.integers(4)]
        out.append((pid, tx + rng.choice(['', ' plz', ' 😍', '??', ' pls reply']), lang, t, s))
    return out

P, C = [], []
for i in range(2400):
    ct, pl, cp = rng.choice(list(CT)), rng.choice(list(PL)), rng.choice(CAMP)
    likes = int(rng.lognormal(6.3, .5) * CT[ct] * PL[pl])
    com_n, sh = int(likes * rng.uniform(.05, .2)), int(likes * rng.uniform(.02, .08))
    r, kind = rng.random(), 'normal'
    if r < .02: likes, com_n, sh, kind = likes * 6, com_n * 6, sh * 6, 'viral'
    elif r < .035: com_n, kind = com_n * 4, 'delivery_storm'
    elif r < .05: likes, com_n, kind = int(likes * .15), int(com_n * .3), 'flop'
    views = int(likes * rng.uniform(12, 30)) if (ct in ('Reel', 'Styling video') or pl == 'TikTok') else None
    pid = f'P{i+1:04d}'
    P.append(dict(post_id=pid, brand=rng.choice(['Brand A', 'Brand B', 'Brand C']), platform=pl, campaign=cp, content_type=ct,
                  product_category='Lawn suit', likes=likes, comments_count=com_n, shares=sh, views=views,
                  days_since_post=int(rng.integers(1, 30)), source='synthetic', sim_pattern=kind))
    C += comments_for(pid, cp, kind)
posts = pd.DataFrame(P)
com = pd.DataFrame(C, columns=['post_id', 'comment_text', 'language_type', 'topic', 'sentiment'])
posts['eng'] = posts.likes + 2 * posts.comments_count + 3 * posts.shares

# ---------- Q1: which content attracts attention ----------
g = posts.groupby('content_type').agg(posts=('post_id', 'count'), median_eng=('eng', 'median'), avg_comments=('comments_count', 'mean')).round(0).sort_values('median_eng', ascending=False)
pc = posts.pivot_table(index='content_type', columns='platform', values='eng', aggfunc='median').round(0).reindex(g.index)
camp = posts.groupby('campaign').eng.median().round(0).sort_values(ascending=False)

# ---------- Q2 + Q4: comment clustering (char n-grams handle Roman Urdu spelling variants) ----------
K = 24
X = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=3).fit_transform(com.comment_text)
com['cluster'] = KMeans(K, n_init=10, random_state=0).fit_predict(X)
stop = ['hai', 'ka', 'ki', 'ko', 'se', 'mein', 'kya', 'is', 'the', 'this', 'plz', 'pls', 'reply', 'me', 'ya', 'in', 'on', 'to', 'of']
vw = TfidfVectorizer(min_df=3, stop_words=stop); W = vw.fit_transform(com.comment_text); words = np.array(vw.get_feature_names_out())
segs = []
for c in range(K):
    m = (com.cluster == c).values; sub = com[m]; top = sub.topic.mode()[0]
    segs.append(dict(id=c, share=round(m.mean() * 100, 1), terms=list(words[np.asarray(W[m].mean(0)).ravel().argsort()[::-1][:3]]),
                     topic=top, purity=round((sub.topic == top).mean() * 100), sentiment=sub.sentiment.value_counts(normalize=True).mul(100).round().astype(int).to_dict(),
                     samples=list(sub.comment_text.str.strip().drop_duplicates().head(3))))
segs.sort(key=lambda s: -s['share'])
purity = round(sum(s['purity'] * s['share'] for s in segs) / sum(s['share'] for s in segs), 1)
tshare = (com.topic.value_counts(normalize=True) * 100).round(1)

# ---------- Q3: unusual posts (robust vs comparable posts + comment mix) ----------
grp = posts.groupby(['platform', 'content_type'])
posts['peer'] = grp.eng.transform('median'); posts['ratio'] = posts.eng / posts.peer
posts['z'] = (np.log1p(posts.eng) - grp.eng.transform(lambda s: np.log1p(s).median())) / (grp.eng.transform(lambda s: np.log1p(s).std()) + 1e-9)
posts['deliv'] = posts.post_id.map(com.assign(d=com.topic.eq('delivery')).groupby('post_id').d.mean()).fillna(0)
posts['cr'] = posts.comments_count / (posts.likes + 1)
posts['anom'] = IsolationForest(contamination=.05, random_state=0).fit_predict(posts[['z', 'deliv', 'cr']]) == -1
def why(r):
    if r.ratio > 2: return f'{r.ratio:.1f}x the engagement of comparable posts'
    if r.ratio < .5: return f'only {r.ratio:.1f}x the engagement of comparable posts'
    if r.deliv >= .4: return f'{r.deliv*100:.0f}% of comments are delivery complaints'
    return 'unusual comments-to-likes ratio'
an = posts[posts.anom].copy(); an['why'] = an.apply(why, axis=1); an['score'] = an.z.abs()
truth = posts.sim_pattern != 'normal'
recall = round((posts.anom & truth).sum() / truth.sum() * 100)
precision = round((posts.anom & truth).sum() / posts.anom.sum() * 100)

# ---------- Q5: evidence-based recommendations ----------
t = tshare
rec = [
 f"Content: {g.index[0]} earns the most engagement (median {int(g.median_eng.iloc[0]):,}) and {g.index[-1]} the least ({int(g.median_eng.iloc[-1]):,}). Shift next-campaign budget toward {g.index[0]}; run a 2-week A/B test against {g.index[-1]}.",
 f"Promotions: price ({t['price']}%) and discount ({t['discount']}%) questions make up {t['price']+t['discount']:.0f}% of comments. State the price and any bundle/discount in the caption so the team stops answering it comment by comment.",
 f"Customer service: delivery comments are {t['delivery']}% of all comments and {int((posts.sim_pattern == 'delivery_storm').sum())} posts show delivery-complaint spikes. Pin a delivery-time note and send proactive tracking updates during 11.11 / 12.12.",
 f"Product pages: availability ({t['availability']}%) and sizing ({t['sizing']}%) questions suggest adding a city/stock line and a size chart to every lawn-suit post.",
 f"Sales: {t['purchase_intent']}% of comments are purchase-intent ('inbox me details'). Set an auto-reply with a WhatsApp/order link and measure how many convert.",
 f"Platform test: {camp.index[0]} posts perform best on median engagement ({int(camp.iloc[0]):,}); repeat that campaign format on TikTok, where engagement multiplier is highest.",
]
res = dict(n_posts=len(posts), n_comments=len(com), langs=com.language_type.value_counts().to_dict(), content=g.reset_index().to_dict('records'),
           matrix=pc.reset_index().fillna(0).to_dict('records'), campaigns=camp.to_dict(), topics=tshare.to_dict(), segments=segs, purity=purity,
           anomalies=an.sort_values('score', ascending=False).head(10)[['post_id', 'platform', 'content_type', 'campaign', 'eng', 'why']].to_dict('records'),
           n_anom=int(posts.anom.sum()), recall=recall, precision=precision, recs=rec)
json.dump(res, open('results.json', 'w'), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
posts.drop(columns=['eng', 'peer', 'ratio', 'z', 'deliv', 'cr', 'anom']).to_csv('posts.csv', index=False)
com.to_csv('comments.csv', index=False, encoding='utf-8-sig')
print(len(posts), len(com), 'purity', purity, 'anomaly recall/precision', recall, precision)
