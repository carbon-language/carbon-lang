// RUN: %clang -triple i386 -### -no-integrated-as %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOIAS

// NOIAS: -no-integrated-as

// RUN: %clang -triple i386 -### -integrated-as %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix IAS

// IAS-NOT: -no-integrated-as

// RUN: %clang -triple i386 -### %s 2>&1 | FileCheck %s -check-prefix DEFAULT

// DEFAULT-NOT: -no-integrated-as

// RUN: %clang -triple msp430 -### %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-IAS-DEFAULT

// NO-IAS-DEFAULT-NOT: -no-integrated-as

