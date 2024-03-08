scaling_dict = {"pants-fire": 0, "false": 20, "barely-true": 40, "half-true": 60, "mostly-true": 80, "true": 100}

def scale_and_combine(truthfullness, automl):
    average_score = 0
    for label, score in automl:
        average_score += scaling_dict[label] * score
    
    bounded_score = max(0.25, min(average_score, 0.75))
    normalized_score = (bounded_score - 0.25) / (0.75 - 0.25)
    scaled_variable = normalized_score * 100

    if abs(truthfullness - scaled_variable) > 50:
        return truthfullness
    elif abs(truthfullness - scaled_variable) > 25:
        return truthfullness * .80 + scaled_variable * .20
    else:
        return truthfullness * .60 + scaled_variable * .40