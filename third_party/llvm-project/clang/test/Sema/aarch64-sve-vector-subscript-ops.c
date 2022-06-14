// RUN: %clang_cc1 -verify -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

void subscript(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
               svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
               svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
               svbool_t b) {
  (void)b[0];    // expected-error{{subscript of svbool_t is not allowed}}
  (void)b[0.f];  // expected-error{{subscript of svbool_t is not allowed}}
  (void)b[0.];   // expected-error{{subscript of svbool_t is not allowed}}

  (void)i8[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i8[0.];  // expected-error{{array subscript is not an integer}}

  (void)u8[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u8[0.];  // expected-error{{array subscript is not an integer}}

  (void)i16[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i16[0.];  // expected-error{{array subscript is not an integer}}

  (void)u16[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u16[0.];  // expected-error{{array subscript is not an integer}}

  (void)i32[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i32[0.];  // expected-error{{array subscript is not an integer}}

  (void)u32[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u32[0.];  // expected-error{{array subscript is not an integer}}

  (void)i64[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i64[0.];  // expected-error{{array subscript is not an integer}}

  (void)u64[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u64[0.];  // expected-error{{array subscript is not an integer}}

  (void)f16[0.f]; // expected-error{{array subscript is not an integer}}
  (void)f16[0.];  // expected-error{{array subscript is not an integer}}

  (void)f32[0.f]; // expected-error{{array subscript is not an integer}}
  (void)f32[0.];  // expected-error{{array subscript is not an integer}}

  (void)f64[0.f]; // expected-error{{array subscript is not an integer}}
  (void)f64[0.];  // expected-error{{array subscript is not an integer}}
}
