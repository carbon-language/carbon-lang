// RUN: %clang -fmodules -fno-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// CHECK-NO-MODULES-NOT: -fmodules

// RUN: %clang -fmodules -fno-modules -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// CHECK-HAS-MODULES: -fmodules

// RUN: %clang -target x86_64-apple-darwin10 -fmodules -fno-modules -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-AUTOLINK %s
// CHECK-HAS-AUTOLINK: -fmodules-autolink

// RUN: %clang -fmodules -fno-modules -fno-modules-autolink -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-AUTOLINK %s
// CHECK-NO-AUTOLINK-NOT: -fmodules-autolink

