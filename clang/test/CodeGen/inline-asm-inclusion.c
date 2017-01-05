// RUN: %clang_cc1 -triple i686--- -I %p/include -S -o - %s | FileCheck %s
// REQUIRES: x86-registered-target

__asm__(".include \"module.x\"");
void function(void) {
  __asm__(".include \"function.x\"");
}

// CHECK: MODULE = 1
// CHECK: FUNCTION = 1

