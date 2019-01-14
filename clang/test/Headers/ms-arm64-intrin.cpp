// RUN: %clang_cc1 -triple arm64-windows \
// RUN: -fms-compatibility -fms-compatibility-version=17.00 \
// RUN: -ffreestanding -fsyntax-only -Werror \
// RUN: -isystem %S/Inputs/include %s -S -o - 2>&1 | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <intrin.h>

void f() {
// CHECK: nop
  __nop();
}
