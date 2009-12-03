// RUN: clang -E -fexceptions %s -o - | FileCheck --check-prefix=CHECK-EXCEPTIONS %s
// RUN: clang -E -fno-exceptions %s -o - | FileCheck --check-prefix=CHECK-NO-EXCEPTIONS %s

#if __has_feature(cxx_exceptions)
int foo();
#else
int bar();
#endif

// CHECK-EXCEPTIONS: foo
// CHECK-NO-EXCEPTIONS: bar
