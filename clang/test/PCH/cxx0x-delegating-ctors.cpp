// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx0x-delegating-ctors.h -std=c++0x -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -std=c++0x -emit-pch -o %t %S/cxx0x-delegating-ctors.h
// RUN: %clang_cc1 -std=c++0x -include-pch %t -fsyntax-only -verify %s 

foo::foo() : foo(1) { } // expected-error{{delegates to itself}}
