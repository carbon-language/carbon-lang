// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s 

@interface NSException @end
void opaque();

namespace test0 {
  void test() {
    try {
    } catch (NSException *e) { // expected-warning {{catching Objective C exceptions in C++ in the non-unified exception model [-Wobjc-nonunified-exceptions]}}
    }
  }
}
