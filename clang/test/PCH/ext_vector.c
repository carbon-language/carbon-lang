// Test this without pch.
// RUN: clang-cc -include %S/ext_vector.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/ext_vector.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

int test(float4 f4) {
  return f4.xy; // expected-error{{float2}}
}
