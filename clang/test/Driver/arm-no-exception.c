// RUN: %clang -target arm-none-gnueeabi -fno-exceptions -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOEH < %t %s

// CHECK-NOEH: "-backend-option" "-arm-disable-ehabi"
