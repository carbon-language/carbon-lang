// XFAIL: hexagon
// RUN: rm -rf %t
// RUN: %clang -Wunused-local-typedef -c -x objective-c++ -fcxx-modules -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=CHECK_1
// RUN: %clang -Wunused-local-typedef -c -x objective-c++ -fcxx-modules -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=CHECK_2 -allow-empty

// For modules, the warning should only fire the first time, when the module is
// built.
// CHECK_1: warning: unused typedef
// CHECK_2-NOT: warning: unused typedef
@import warn_unused_local_typedef;
