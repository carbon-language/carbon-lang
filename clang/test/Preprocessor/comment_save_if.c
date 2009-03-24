// RUN: clang-cc %s -E -CC -pedantic 2>&1 | grep -v '^/' | not grep warning

#if 1 /*bar */

#endif /*foo*/

