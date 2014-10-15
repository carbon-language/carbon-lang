// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -faltivec -ffreestanding -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -faltivec -ffreestanding -fno-lax-vector-conversions -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -faltivec -ffreestanding -x c++ -S -o - %s | FileCheck %s

#include <altivec.h>

// Verify that simply including <altivec.h> does not generate any code
// (i.e. all inline routines in the header are marked "static")

// CHECK: .text
// CHECK-NEXT: .file
// CHECK-NEXT: {{^$}}
// CHECK-NEXT: .ident{{.*$}}
// CHECK-NEXT: .section ".note.GNU-stack","",@progbits
// CHECK-NOT: .
