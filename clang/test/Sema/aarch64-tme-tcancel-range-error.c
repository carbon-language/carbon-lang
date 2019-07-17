// RUN: %clang_cc1 -triple aarch64-eabi -target-feature +tme -verify %s
void t_cancel() {
  __builtin_arm_tcancel(0x12345u); // expected-error{{argument value 74565 is outside the valid range [0, 65535]}}
}
