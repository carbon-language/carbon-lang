// RUN: %clang -target arm-none-gnueabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target aarch64-none-gnueabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mstrict-align -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mno-unaligned-access -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-AARCH64 < %t %s

// CHECK-UNALIGNED-ARM: "-backend-option" "-arm-no-strict-align"
// CHECK-UNALIGNED-AARCH64: "-backend-option" "-aarch64-no-strict-align"


// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -munaligned-access -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -munaligned-access -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mkernel -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// CHECK-ALIGNED-ARM: "-backend-option" "-arm-strict-align"
// CHECK-ALIGNED-AARCH64: "-backend-option" "-aarch64-strict-align"

// Make sure that v6M cores always trigger the unsupported aligned accesses error
// for all supported architecture triples.
// RUN: not %clang -c -target thumbv6m-none-gnueabi -mcpu=cortex-m0 -munaligned-access %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-UNALIGN-NOT-SUPPORTED %s
// RUN: not %clang -c -target thumb-none-gnueabi -mcpu=cortex-m0 -munaligned-access %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-UNALIGN-NOT-SUPPORTED %s

// CHECK-UNALIGN-NOT-SUPPORTED: error: the v6m sub-architecture does not support unaligned accesses
