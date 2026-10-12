import re


def normalize_number(text):
    if not isinstance(text, str):
        return str(text)

    text = re.sub(r'\s', '', text)

    sci_notation = re.search(r'([-+]?\d*\.?\d+)[×xX]10\^?([-+]?\d+)', text)
    if sci_notation:
        base = float(sci_notation.group(1))
        exp = int(sci_notation.group(2))
        return str(base * (10 ** exp))

    number_match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)
    if number_match:
        return number_match.group(1)

    return text


def verify_float(reference, prediction, tolerance=0.05):
    try:
        ref_num = float(normalize_number(reference))
        pred_num = float(normalize_number(prediction))

        if ref_num == 0:
            return abs(pred_num) < tolerance

        relative_error = abs((pred_num - ref_num) / ref_num)
        return relative_error <= tolerance
    except (ValueError, TypeError):
        return normalize_number(reference) == normalize_number(prediction)


def verify_string(reference, prediction):
    ref_clean = re.sub(r'\s', '', reference.lower())
    pred_clean = re.sub(r'\s', '', prediction.lower())

    return ref_clean == pred_clean


def verify_answer(reference, prediction, method='float', tolerance=0.05):
    if method == 'float':
        return verify_float(reference, prediction, tolerance)
    elif method == 'string':
        return verify_string(reference, prediction)
    else:
        try:
            return verify_float(reference, prediction, tolerance)
        except:
            return verify_string(reference, prediction)
