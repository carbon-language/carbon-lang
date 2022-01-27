// Gentoo disables -faddrsig by default
// XFAIL: gentoo

// RUN: %clang -### -target x86_64-unknown-linux -c %s 2>&1 | FileCheck -check-prefix=ADDRSIG %s
// RUN: %clang -### -target x86_64-pc-win32 -c %s 2>&1 | FileCheck -check-prefix=ADDRSIG %s
// RUN: %clang -### -target x86_64-unknown-linux -fno-integrated-as -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s
// RUN: %clang -### -target x86_64-unknown-linux -fno-integrated-as -faddrsig -c %s 2>&1 | FileCheck -check-prefix=ADDRSIG %s
// RUN: %clang -### -target x86_64-unknown-linux -fno-addrsig -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s
// RUN: %clang -### -target x86_64-apple-darwin -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s
// RUN: %clang -### -target x86_64-scei-ps4 -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s
// RUN: %clang -### -target x86_64-linux-android21 -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s

// ADDRSIG: -faddrsig
// NO-ADDRSIG-NOT: -faddrsig
