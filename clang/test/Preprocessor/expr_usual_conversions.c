// RUN: clang %s -E  2>&1 | grep warning | wc -l | grep 2

#define INTMAX_MIN (-9223372036854775807LL -1)

#if (-42 + 0U) / -2
foo
#endif

