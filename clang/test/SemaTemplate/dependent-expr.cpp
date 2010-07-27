// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5908
template <typename Iterator>
void Test(Iterator it) {
  *(it += 1);
}

namespace PR6045 {
  template<unsigned int r>
  class A
  {
    static const unsigned int member = r;
    void f();
  };
  
  template<unsigned int r>
  const unsigned int A<r>::member;
  
  template<unsigned int r>
  void A<r>::f() 
  {
    unsigned k;
    (void)(k % member);
  }
}

namespace PR7198 {
  struct A
  {
    ~A() { }
  };

  template<typename T>
  struct B {
    struct C : A {};
    void f()
    {
      C c = C();
    }
  };
}

namespace PR7724 {
  template<typename OT> int myMethod()
  { return 2 && sizeof(OT); }
}
