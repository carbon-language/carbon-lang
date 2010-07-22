// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-static_assert.h -verify -std=c++0x %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -std=c++0x -emit-pch -o %t %S/cxx-static_assert.h
// RUN: %clang_cc1 -include-pch %t -verify -std=c++0x %s 

// expected-error {{static_assert failed "N is not 2!"}}

T<1> t1; // expected-note {{in instantiation of template class 'T<1>' requested here}}
T<2> t2;
