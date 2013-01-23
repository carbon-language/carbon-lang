// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

#pragma OPENCL EXTENSION cl_khr_fp16 : disable

half half_disabled(half *p, // expected-error{{declaring function return value of type 'half' is not allowed}}
                   half h)  // expected-error{{declaring function argument of type 'half' is not allowed}} 
{
  half a[2]; // expected-error{{declaring variable of type 'half [2]' is not allowed}}
  half b;    // expected-error{{declaring variable of type 'half' is not allowed}}

  b = *p;    // expected-error{{dereferencing pointer of type 'half *' is not allowed}}
  *p = b;    // expected-error{{dereferencing pointer of type 'half *' is not allowed}}

  b = p[1];  // expected-error {{subscript to array of type 'half' is not allowed}}
  p[1] = b;  // expected-error {{subscript to array of type 'half' is not allowed}}

  float c = 1.0f;
  b = (half) c;  // expected-error{{casting to type 'half' is not allowed}}
  c = (float) h; // expected-error{{casting from type 'half' is not allowed}}

  return h;
}

// Exactly the same as above but with the cl_khr_fp16 extension enabled.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half half_enabled(half *p, half h)
{
  half a[2];
  half b;

  b = *p;
  *p = b;

  b = p[1];
  p[1] = b;

  float c = 1.0f;
  b = (half) c;
  c = (float) h;

  return h;
}
