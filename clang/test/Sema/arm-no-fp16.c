// RUN: %clang_cc1 -triple thumbv7-none-eabi %s -target-feature +neon -target-feature -fp16 -fsyntax-only -verify

#include <arm_neon.h>

float16x4_t test_vcvt_f16_f32(float32x4_t a) {
  return vcvt_f16_f32(a); // expected-warning{{implicit declaration of function 'vcvt_f16_f32'}}  expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float32x4_t test_vcvt_f32_f16(float16x4_t a) {
  return vcvt_f32_f16(a); // expected-warning{{implicit declaration of function 'vcvt_f32_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float32x4_t'}}
}

float16x4_t test_vrnda_f16(float16x4_t a) {
  return vrnda_f16(a); // expected-warning{{implicit declaration of function 'vrnda_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndaq_f16(float16x8_t a) {
  return vrndaq_f16(a); // expected-warning{{implicit declaration of function 'vrndaq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrnd_f16(float16x4_t a) {
  return vrnd_f16(a); // expected-warning{{implicit declaration of function 'vrnd_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndq_f16(float16x8_t a) {
  return vrndq_f16(a); // expected-warning{{implicit declaration of function 'vrndq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndi_f16(float16x4_t a) {
  return vrndi_f16(a); // expected-warning{{implicit declaration of function 'vrndi_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndiq_f16(float16x8_t a) {
  return vrndiq_f16(a); // expected-warning{{implicit declaration of function 'vrndiq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndm_f16(float16x4_t a) {
  return vrndm_f16(a); // expected-warning{{implicit declaration of function 'vrndm_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndmq_f16(float16x8_t a) {
  return vrndmq_f16(a); // expected-warning{{implicit declaration of function 'vrndmq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndn_f16(float16x4_t a) {
  return vrndn_f16(a); // expected-warning{{implicit declaration of function 'vrndn_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndnq_f16(float16x8_t a) {
  return vrndnq_f16(a); // expected-warning{{implicit declaration of function 'vrndnq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndp_f16(float16x4_t a) {
  return vrndp_f16(a); // expected-warning{{implicit declaration of function 'vrndp_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndpq_f16(float16x8_t a) {
  return vrndpq_f16(a); // expected-warning{{implicit declaration of function 'vrndpq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndx_f16(float16x4_t a) {
  return vrndx_f16(a); // expected-warning{{implicit declaration of function 'vrndx_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndxq_f16(float16x8_t a) {
  return vrndxq_f16(a); // expected-warning{{implicit declaration of function 'vrndxq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}
