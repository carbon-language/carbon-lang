// RUN: %clang -### %s 2>&1 | FileCheck %s -check-prefix=TLSDIRECT
// RUN: %clang -### -mno-tls-direct-seg-refs -mtls-direct-seg-refs %s 2>&1 | FileCheck %s -check-prefix=TLSDIRECT
// RUN: %clang -### -mtls-direct-seg-refs -mno-tls-direct-seg-refs %s 2>&1 | FileCheck %s -check-prefix=NO-TLSDIRECT
// REQUIRES: clang-driver

// NO-TLSDIRECT: -mno-tls-direct-seg-refs
// TLSDIRECT-NOT: -mno-tls-direct-seg-refs
