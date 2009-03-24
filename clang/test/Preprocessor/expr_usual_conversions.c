// RUN: clang-cc %s -E  2>&1 | grep warning | wc -l | grep 2

#define INTMAX_MIN (-9223372036854775807LL -1)

#if (-42 + 0U) / -2
foo
#endif

// Shifts don't want the usual conversions: PR2279
#if (2 << 1U) - 30 >= 0
#error
#endif

