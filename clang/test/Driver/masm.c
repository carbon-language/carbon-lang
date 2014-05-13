// REQUIRES: x86-registered-target
// RUN: %clang -target i386-unknown-linux -masm=intel %s -S -o - | FileCheck --check-prefix=CHECK-INTEL %s
// RUN: %clang -target i386-unknown-linux -masm=att %s -S -o - | FileCheck --check-prefix=CHECK-ATT %s
// RUN: not %clang -target i386-unknown-linux -masm=somerequired %s -S -o - 2>&1 | FileCheck --check-prefix=CHECK-SOMEREQUIRED %s
// REQUIRES: arm-registered-target
// RUN: %clang -target arm-unknown-eabi -masm=intel %s -S -o - 2>&1 | FileCheck --check-prefix=CHECK-ARM %s

int f() {
// CHECK-ATT: movl      $0, %eax
// CHECK-INTEL: mov     eax, 0
// CHECK-SOMEREQUIRED: error: unsupported argument 'somerequired' to option 'masm='
// CHECK-ARM: warning: argument unused during compilation: '-masm=intel'
  return 0;
}
