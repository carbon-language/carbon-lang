// RUN: %clang_cc1 -triple thumbv7-windows -fms-compatibility -fsyntax-only -verify %s

int __cdecl cdecl(int a, int b, int c, int d) { // expected-warning {{calling convention '__cdecl' ignored for this target}}
  return a + b + c + d;
}

float __stdcall stdcall(float a, float b, float c, float d) { // expected-warning {{calling convention '__stdcall' ignored for this target}}
  return a + b + c + d;
}

