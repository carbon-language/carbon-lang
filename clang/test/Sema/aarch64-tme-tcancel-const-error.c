// RUN: %clang_cc1 -triple aarch64-eabi -target-feature +tme -verify %s
void t_cancel(unsigned short u) {
  __builtin_arm_tcancel(u); // expected-error{{argument to '__builtin_arm_tcancel' must be a constant integer}}
}
