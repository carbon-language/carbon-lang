// RUN: %clang -target arm-none-gnueabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target arm64-none-gnueabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM64 < %t %s

// RUN: %clang -target arm64-none-gnueabi -mstrict-align -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM64 < %t %s

// RUN: %clang -target arm64-none-gnueabi -mno-unaligned-access -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM64 < %t %s

// CHECK-UNALIGNED-ARM: "-backend-option" "-arm-no-strict-align"
// CHECK-UNALIGNED-ARM64: "-backend-option" "-arm64-no-strict-align"


// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm64-none-gnueabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM64 < %t %s

// RUN: %clang -target arm64-none-gnueabi -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM64 < %t %s

// RUN: %clang -target arm64-none-gnueabi -munaligned-access -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM64 < %t %s

// RUN: %clang -target arm64-none-gnueabi -munaligned-access -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM64 < %t %s

// CHECK-ALIGNED-ARM: "-backend-option" "-arm-strict-align"
// CHECK-ALIGNED-ARM64: "-backend-option" "-arm64-strict-align"
