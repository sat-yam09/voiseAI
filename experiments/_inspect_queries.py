import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

gt_queries = {}
with open('data/chunks/chunks_256_32.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        qid = str(c.get('query_id', ''))
        if qid not in gt_queries:
            gt_queries[qid] = {'query': c.get('query', ''), 'selected': 0, 'total': 0, 'target_lang': c.get('target_lang', '')}
        gt_queries[qid]['total'] += 1
        if c.get('is_selected'):
            gt_queries[qid]['selected'] += 1

gt = {k: v for k, v in gt_queries.items() if v['selected'] > 0}
print('Total queries: %d, with ground truth: %d' % (len(gt_queries), len(gt)))
print()

for i, (qid, info) in enumerate(list(gt.items())[:30]):
    q = info['query']
    s = info['selected']
    t = info['total']
    lang = info['target_lang']
    print('  QID=%s: selected=%d/%d lang=%s  query=%s' % (qid, s, t, lang, q[:100]))
