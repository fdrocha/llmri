#!/usr/bin/env python3
import argparse
import math
import random
import sys
import util


def extract_random_lines(fn, nlines, min_length):
    lines = [line.strip() for line in open(fn, 'r')]
    if len(lines) < nlines * 4: return 0
    for _ in range(nlines):
        i = random.randint(0, len(lines) - 1)
        outstr = ""
        while len(outstr) < min_length and i < len(lines):
            outstr += util.encode_line(lines[i])
            i += 1
        print(outstr)
    return nlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('NLINES', type=int, default=500)
    parser.add_argument('LENGTH', type=int, default=200)
    args = parser.parse_args()

    lines_left = args.NLINES
    fns = [line.strip() for line in sys.stdin]
    random.shuffle(fns)
    lines_per_file = int(math.ceil(len(fns) / args.NLINES))

    while lines_left > 0:
        fn = fns.pop()
        lines_to_get = min(lines_left, lines_per_file)
        lines_left -= extract_random_lines(fn, lines_to_get, args.LENGTH)


if __name__ == '__main__':
    main()
