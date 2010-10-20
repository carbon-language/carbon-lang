// RUN: %clang_cc1 -E %s -o - | FileCheck %s

// CHECK: always_inline
#if __has_attribute(always_inline)
int always_inline();
#endif

// CHECK: no_dummy_attribute
#if !__has_attribute(dummy_attribute)
int no_dummy_attribute();
#endif

