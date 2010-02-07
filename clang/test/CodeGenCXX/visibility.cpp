// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

#define HIDDEN __attribute__((visibility("hidden")))
#define PROTECTED __attribute__((visibility("protected")))
#define DEFAULT __attribute__((visibility("default")))

// CHECK: @_ZN5Test425VariableInHiddenNamespaceE = hidden global i32 10

namespace Test1 {
  // CHECK: define hidden void @_ZN5Test11fEv
  void HIDDEN f() { }
  
}

namespace Test2 {
  struct HIDDEN A {
    void f();
  };

  // A::f is a member function of a hidden class.
  // CHECK: define hidden void @_ZN5Test21A1fEv
  void A::f() { }
}
 
namespace Test3 {
  struct HIDDEN A {
    struct B {
      void f();
    };
  };

  // B is a nested class where its parent class is hidden.
  // CHECK: define hidden void @_ZN5Test31A1B1fEv
  void A::B::f() { }  
}

namespace Test4 HIDDEN {
  int VariableInHiddenNamespace = 10;

  // Test4::g is in a hidden namespace.
  // CHECK: define hidden void @_ZN5Test41gEv
  void g() { } 
  
  struct DEFAULT A {
    void f();
  };
  
  // A has default visibility.
  // CHECK: define void @_ZN5Test41A1fEv
  void A::f() { } 
}

namespace Test5 {

  namespace NS HIDDEN {
    // f is in NS which is hidden.
    // CHECK: define hidden void @_ZN5Test52NS1fEv()
    void f() { }
  }
  
  namespace NS {
    // g is in NS, but this NS decl is not hidden.
    // CHECK: define void @_ZN5Test52NS1gEv
    void g() { }
  }
}
