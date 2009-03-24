// RUN: clang-cc %s -fsyntax-only -verify

void __attribute__((fastcall)) foo(float *a) { 
}

void __attribute__((stdcall)) bar(float *a) { 
}

void __attribute__((fastcall(1))) baz(float *a) { // expected-error {{attribute requires 0 argument(s)}}
}
