# formulas, metrics and coefficients
def precision_metric(tp, fp):
    return tp / float(tp + fp)


def recall_metric(tp, fn):
    return tp / float(tp + fn)


def f1score_metric(precision, recall):
    return 2 * precision * recall / float(precision + recall)


# rect(x, y, w, h)
def iou_coeff(rect1, rect2):
    return intersection_area(rect1, rect2) / float(union_area(rect1, rect2))


def union_area(rect1, rect2):
    return (rect1[2] * rect1[3]) + (rect2[2] * rect2[3]) - (intersection_area(rect1, rect2))


def intersection_area(rect1, rect2):
    w = min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - max(rect1[0], rect2[0])
    h = min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - max(rect1[1], rect2[1])
    if w < 0 or h < 0:
        return 0
    else:
        return w * h
