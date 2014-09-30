// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// RUN: %clang -fcxx-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// RUN: %clang -fmodules -fno-cxx-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// CHECK-NO-MODULES-NOT: -fmodules

// RUN: %clang -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// RUN: %clang -fmodules -fno-cxx-modules -fcxx-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// CHECK-HAS-MODULES: -fmodules
