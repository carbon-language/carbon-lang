// In module: expected-note{{previous declaration}}




// In module: expected-note{{previous definition is here}}

// Test redeclarations of functions where the original declaration is
// still hidden.

@import linkage_merge_left; // excludes "sub"

extern int f0(float); // expected-error{{conflicting types for 'f0'}}
static int f1(float); // okay: considered distinct
static int f2(float); // okay: considered distinct
extern int f3(float); // okay: considered distinct

extern float v0; // expected-error{{redefinition of 'v0' with a different type: 'float' vs 'int'}}
static float v1;
static float v2;
extern float v3;

typedef float T0;

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c -fmodule-cache-path %t -emit-module -fmodule-name=linkage_merge_left %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -w %s -verify
