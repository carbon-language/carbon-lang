// RUN: %clang -### %s 2>&1 | FileCheck %s -check-prefix=NO-REALIGN
// RUN: %clang -### -mno-stackrealign -mstackrealign %s 2>&1 | FileCheck %s -check-prefix=REALIGN
// RUN: %clang -### -mstackrealign -mno-stackrealign %s 2>&1 | FileCheck %s -check-prefix=NO-REALIGN
// REQUIRES: clang-driver

// REALIGN: -mstackrealign
// NO-REALIGN-NOT: -mstackrealign
