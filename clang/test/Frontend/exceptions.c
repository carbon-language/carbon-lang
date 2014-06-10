// RUN: %clang_cc1 -fms-compatibility -fexceptions -fcxx-exceptions -verify %s
// expected-no-diagnostics

#if defined(__EXCEPTIONS)
#error __EXCEPTIONS should not be defined.
#endif
