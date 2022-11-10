from metrics.normalize_answer import normalize_answer, chuan_hoa_dau_cau_tieng_viet

def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    # prediction_tokens = chuan_hoa_dau_cau_tieng_viet(normalize_answer(prediction))
    # ground_truth_tokens = chuan_hoa_dau_cau_tieng_viet(normalize_answer(ground_truth))

    prediction_tokens = normalize_answer(prediction.replace("_", " "))
    ground_truth_tokens = normalize_answer(ground_truth.replace("_", " "))

    return (prediction_tokens == ground_truth_tokens)
