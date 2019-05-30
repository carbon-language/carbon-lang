// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+dsp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-DSP < %t %s
// CHECK-DSP: "-target-feature" "+dsp"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FP < %t %s
// CHECK-FP: "-target-feature" "+fp-armv8"
// CHECK-FP-NOT: "-target-feature" "+fp64"
// CHECK-FP-NOT: "-target-feature" "+d32"
// CHECK-FP: "-target-feature" "+fullfp16"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+fp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FPDP < %t %s
// CHECK-FPDP: "-target-feature" "+fp-armv8"
// CHECK-FPDP: "-target-feature" "+fullfp16"
// CHECK-FPDP: "-target-feature" "+fp64"
// CHECK-FPDP-NOT: "-target-feature" "+d32"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVE < %t %s
// CHECK-MVE: "-target-feature" "+mve"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVEFP < %t %s
// CHECK-MVEFP: "-target-feature" "+mve.fp"
// CHECK-MVEFP-NOT: "-target-feature" "+fp64"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp+fp.dp  -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MVEFP_DP < %t %s
// CHECK-MVEFP_DP: "-target-feature" "+mve.fp"
// CHECK-MVEFP_DP: "-target-feature" "+fp64"

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1m.main+fp -S %s
double foo (double a) { return a; }
