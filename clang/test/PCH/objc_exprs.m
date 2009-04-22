// Test this without pch.
// RUN: clang-cc -fblocks -include %S/objc_exprs.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -x objective-c-header -emit-pch -fblocks -o %t %S/objc_exprs.h &&
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -verify %s 



int *A1 = (objc_string)0;   // expected-warning {{'struct objc_object *'}}

char A2 = (objc_encode){};  // expected-error {{initializer element is not a compile-time constant}} \
                               expected-warning {{char [2]}}

int *A3 = (objc_protocol)0; // expected-warning {{aka 'Protocol *'}}



