// RUN: %clang -target arm-none-gnueabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED < %t %s

// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED < %t %s

// CHECK-UNALIGNED: "-backend-option" "-arm-no-strict-align"


// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED < %t %s

// CHECK-ALIGNED: "-backend-option" "-arm-strict-align"
