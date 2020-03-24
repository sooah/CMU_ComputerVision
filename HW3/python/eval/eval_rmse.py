#!/usr/bin/python3
import sys
import math


def main():
    """main"""
    # the first argument is our golden file
    golden_in = open(sys.argv[1], 'r')

    # test is the second arg
    test_in = open(sys.argv[2], 'r')

    line = 0

    error = False

    E = 0.0
    num_ratings = 0

    while True:
        g_line = golden_in.readline()
        t_line = test_in.readline()

        line = line + 1

        if len(g_line) == 0 and len(t_line) == 0:
            # normally got to the end of each file at the same time
            break
        elif len(g_line) == 0 and len(t_line) > 0:
            print('Error: prematurely reached the end of the test file')
            error = True
            break
        elif len(g_line) > 0 and len(t_line) == 0:
            print('Error: answer incomplete!')
            error = True
            break

        # this is a rating line
        rating = -1
        rating_t = -1
        try:
            rating = float(g_line.strip())
        except Exception:
            print('Error: couldn\'t parse rating line in golden file: %s' %
                  g_line.strip())
            error = True
            break

        try:
            rating_t = float(t_line.strip())
        except Exception:
            print('Error: couldn\'t parse rating line in test file: %s' %
                  t_line.strip())
            error = True
            break

        delta = rating_t - rating
        E = E + (delta * delta)
        num_ratings = num_ratings + 1

    if not error:
        # print out the RMSE
        print(str(math.sqrt(E / num_ratings)))


if __name__ == '__main__':
    main()
