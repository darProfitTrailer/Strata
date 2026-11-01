"""Validate reported benchmark measurements without coercing or estimating them."""
import math


def finite_nonnegative(value):
    """Whether a measurement can be represented by finite report arithmetic."""
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def measured_median(values):
    """Median without overflowing the sum of the two central values; null if unknown."""
    values = [value for value in values if value is not None]
    if not values or not all(finite_nonnegative(value) for value in values):
        return None
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    low, high = values[middle - 1:middle + 1]
    result = low + (high - low) / 2
    return result if finite_nonnegative(result) else None


def measured_ratio(numerator, denominator):
    """A finite ratio, or null when the operands/result cannot be represented."""
    if not finite_nonnegative(numerator) or not finite_nonnegative(denominator) or denominator == 0:
        return None
    try:
        result = numerator / denominator
    except OverflowError:
        return None
    return result if finite_nonnegative(result) else None


def validated_metrics(usage, timings):
    errors = []

    def obj(value, name):
        if value is None:
            return {}
        if not isinstance(value, dict):
            errors.append(f'{name} must be an object or null')
            return {}
        return value.copy()

    def number(values, key, path, integer=False):
        value = values.get(key)
        if value is None:
            return
        valid = finite_nonnegative(value) and (not integer or type(value) is int)
        if not valid:
            errors.append(f'{path}.{key} must be a finite nonnegative ' + ('integer' if integer else 'number'))
            values[key] = None

    u, t = obj(usage, 'usage'), obj(timings, 'timings')
    details = obj(u.get('prompt_tokens_details'), 'usage.prompt_tokens_details')
    for key in ('completion_tokens', 'prompt_tokens', 'total_tokens'):
        number(u, key, 'usage', integer=True)
    number(details, 'cached_tokens', 'usage.prompt_tokens_details', integer=True)
    if 'prompt_tokens_details' in u:
        u['prompt_tokens_details'] = details
    for key in ('prompt_n', 'predicted_n', 'cache_n', 'draft_n', 'draft_n_accepted'):
        number(t, key, 'timings', integer=True)
    for key in ('prompt_per_second', 'predicted_per_second', 'prompt_ms', 'predicted_ms'):
        number(t, key, 'timings')
    offered, accepted = t.get('draft_n'), t.get('draft_n_accepted')
    if offered is not None and accepted is not None and accepted > offered:
        errors.append('accepted draft tokens exceed offered draft tokens')
        t['draft_n_accepted'] = None
    cached = details.get('cached_tokens')
    if cached is None:
        cached = t.get('cache_n')
    if cached is not None and u.get('prompt_tokens') is not None and cached > u['prompt_tokens']:
        errors.append('cached tokens exceed prompt tokens')
        details['cached_tokens'] = None
        t['cache_n'] = None
    return u, t, errors
