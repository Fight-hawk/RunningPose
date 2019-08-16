import json

finished = False
j = 0
with open('./examples/res/jiulin.json', 'w', encoding='utf-8') as w:
    data = []
    with open('./examples/res/alphapose-results_jiulin.json', 'r', encoding='utf-8') as r:
        result = json.load(r)
        for i in range(0, 36800):
            if finished:
                obj = {'score': None, 'keypoints': None, 'image_id': None, 'category_id': None}
                data.append(obj)
                break
            elif int(result[j]['image_id'].replace('.jpg', '')) > i:
                obj = {'score': None, 'keypoints': None, 'image_id': None, 'category_id': None}
                data.append(obj)
            elif int(result[j]['image_id'].replace('.jpg', '')) == i:
                data.append(result[j])
                j += 1
                if j >= len(result):
                    finished = True
            else:
                j += 1
                if j >= len(result):
                    finished = True

    json.dump(data, w)