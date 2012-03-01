// RUN: %clang_cc1 %s
// RUN: %clang_cc1 %s -E
#ifndef __has_attribute
#error Should have __has_attribute
#endif

#if __has_attribute(something_we_dont_have)
#error Bad
#endif

#if !__has_attribute(__always_inline__) || \
    !__has_attribute(always_inline)
#error Clang should have this
#endif
