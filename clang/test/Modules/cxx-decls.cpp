// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

// expected-no-diagnostics

@import cxx_decls.imported;

void test_delete(int *p) {
  // We can call the normal global deallocation function even though it has only
  // ever been explicitly declared in an unimported submodule.
  delete p;
}
