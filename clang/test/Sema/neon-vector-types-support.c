// RUN: %clang_cc1 %s -triple armv7 -fsyntax-only -verify

typedef __attribute__((neon_vector_type(2))) int int32x2_t; // expected-error{{'neon_vector_type' attribute is not supported for this target}}
typedef __attribute__((neon_polyvector_type(16))) short poly8x16_t; // expected-error{{'neon_polyvector_type' attribute is not supported for this target}}
