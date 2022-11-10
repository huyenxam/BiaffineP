def get_pred_entity(cate_pred, span_scores,label_set, is_flat_ner= True):
    top_span = []
    for i in range(len(cate_pred)):
        for j in range(i,len(cate_pred)):
            if cate_pred[i][j]>0:
                tmp = (label_set[cate_pred[i][j].item()], i, j,span_scores[i][j].item())
                top_span.append(tmp)
    top_span = sorted(top_span, reverse=True, key=lambda x: x[3])
    
    
    if not top_span:
        top_span = [('ANSWER', 0, 0)]

    return top_span[0]

def get_entities(input_tensor, label, label_set):

    input_tensor, cate_pred = input_tensor.max(dim=-1)
    predict_entity = get_pred_entity(cate_pred, input_tensor, label_set, True)
    return predict_entity

def batch_computeF1(labels, preds, seq_lengths, label_set):
    predict_entitys = []
    for i in range(len(labels)):
        label = labels[i]
        pred = preds[i]
        true_len = seq_lengths[i].item()
        pred = pred[:true_len, :true_len]
        label = label[:true_len, :true_len]
        predict_entity = get_entities(pred, label, label_set)
        predict_entitys.append(predict_entity)
    return predict_entitys