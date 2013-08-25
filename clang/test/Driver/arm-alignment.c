// RUN: %clang -target arm-none-gnueeabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED < %t %s

// CHECK-UNALIGNED: "-backend-option" "-arm-no-strict-align"

// RUN: %clang -target arm-none-gnueeabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED < %t %s

// CHECK-ALIGNED: "-backend-option" "-arm-strict-align"
