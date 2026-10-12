from typing import List


DEFAULT_INSPECT_SYSTEM = (
    'You are a reasoning assistant. Before solving, you must inspect the premises. '
    'You will output a machine executable inspection trace and a cleaned premise list.'
)

DEFAULT_SOLVE_SYSTEM = (
    'You are a careful problem solver. Use only the inspected premises. '
    'Solve step by step and output the final answer in a single line.'
)


def format_premises(premises: List[str]) -> str:
    return '\n'.join([f'{i+1}. {p}' for i, p in enumerate(premises)])


def build_inspection_user_prompt(question: str, premises: List[str]) -> str:
    prem_txt = format_premises(premises)
    return (
        'Task: Inspect the following premise list before solving.\n'
        'You must remove redundant and irrelevant premises, fix obvious errors when possible, and reorder premises into a consistent logical order.\n\n'
        'Output format must be exactly:\n'
        '<trace>JSON_LIST</trace>\n'
        '<inspect>INSPECTED_PREMISE_LIST</inspect>\n\n'
        'Trace JSON_LIST must be a list of objects with keys id, op, and optional arg.\n'
        'Supported ops: keep, delete, rewrite, reorder.\n'
        'For rewrite, arg is the rewritten premise string.\n'
        'For reorder, you may either provide a full list of premise ids as arg or use move ops with arg as the new position.\n\n'
        f'Question: {question}\n\n'
        f'Premises:\n{prem_txt}\n'
    )


def build_solver_user_prompt(question: str, inspected_premises: List[str]) -> str:
    prem_txt = format_premises(inspected_premises)
    return (
        f'Question: {question}\n\n'
        f'Inspected premises:\n{prem_txt}\n\n'
        'Solve using only the inspected premises. Provide the final answer only.'
    )
