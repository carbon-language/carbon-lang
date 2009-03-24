// RUN: clang-cc %s -E -Wundef -Werror 2>&1 | grep error | count 1 &&
// RUN: clang-cc %s -E -Werror 2>&1 | not grep error 

#if foo   // Should generate an warning
#endif

#ifdef foo
#endif

#if defined(foo)
#endif

