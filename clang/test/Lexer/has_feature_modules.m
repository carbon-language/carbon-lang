// RUN: %clang_cc1 -E -fmodules %s -o - | FileCheck --check-prefix=CHECK-HAS-MODULES %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-MODULES %s

#if __has_feature(objc_modules)
int has_modules();
#else
int no_modules();
#endif

// CHECK-HAS-MODULES: has_modules
// CHECK-NO-MODULES: no_modules
