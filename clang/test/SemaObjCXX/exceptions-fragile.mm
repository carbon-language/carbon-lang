// RUN: %clang_cc1 -fexceptions -fsyntax-only -verify %s 

@interface NSException @end
void opaque();

namespace test0 {
  void test() {
    try {
    } catch (NSException *e) { // expected-error {{can't catch Objective C exceptions in C++ in the non-unified exception model}}
    }
  }
}
