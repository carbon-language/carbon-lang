// RUN: %clang -target arm-none-gnueeabi -ffixed-r9 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R9 < %t %s

// CHECK-FIXED-R9: "-backend-option" "-arm-reserve-r9"
