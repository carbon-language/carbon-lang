// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify

// rdar://15522601
class MyClass {
 static void meth();
};
void MyClass::meth() { }
extern "C" {
  void _ZN7MyClass4methEv() { } // expected-error {{definition with same mangled name as another definition}}
}
