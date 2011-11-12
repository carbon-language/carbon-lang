// Test that infinite loop in rdar://10418538 was fixed.

// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -include %s -include %s %s

// With PCH
// RUN: %clang_cc1 -fsyntax-only -verify %s -chain-include %s -chain-include %s

#ifndef HEADER1
#define HEADER1
//===----------------------------------------------------------------------===//
// Primary header

@class I;

//===----------------------------------------------------------------------===//
#elif !defined(HEADER2)
#define HEADER2
#if !defined(HEADER1)
#error Header inclusion order messed up
#endif

//===----------------------------------------------------------------------===//
// Dependent header

@interface I
@end

@interface I(Cat1)
@end

@interface I(Cat2)
@end

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

void f(I* i) {
  [i meth]; // expected-warning {{not found}}
}

//===----------------------------------------------------------------------===//
#endif
