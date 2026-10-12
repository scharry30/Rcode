import argparse

from utils.io_utils import read_jsonl
from utils.logging_utils import get_logger


def main(args):
    logger = get_logger('rts_eval', log_file=args.log_file)
    rows = read_jsonl(args.pred_jsonl)
    if not rows:
        raise ValueError('Empty predictions')

    total = len(rows)
    correct = sum(1 for r in rows if bool(r.get('correct')))
    elim_ok = sum(1 for r in rows if bool(r.get('elim_ok')))
    inspect_ok = sum(1 for r in rows if bool(r.get('inspect_ok')))

    logger.info(f'total {total}')
    logger.info(f'correct {correct} acc {correct/total:.4f}')
    logger.info(f'inspect_ok {inspect_ok} rate {inspect_ok/total:.4f}')
    logger.info(f'elimination_ok {elim_ok} rate {elim_ok/total:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_jsonl', type=str, required=True)
    parser.add_argument('--log_file', type=str, default=None)
    args = parser.parse_args()
    main(args)
