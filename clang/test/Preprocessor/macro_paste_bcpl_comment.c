// RUN: clang-cc %s -Eonly -fms-extensions=0 2>&1 | grep error

#define COMM1 / ## /
COMM1

