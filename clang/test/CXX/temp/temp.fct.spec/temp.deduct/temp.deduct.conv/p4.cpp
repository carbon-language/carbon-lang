// RUN: clang-cc -fsyntax-only %s

struct AnyT {
  template<typename T>
  operator T();
};

void test_cvqual_ref(AnyT any) {
  const int &cir = any;  
}

struct AnyThreeLevelPtr {
  template<typename T>
  operator T***() const;
  // FIXME: Can't handle definitions of member templates yet
#if 0
  {
    T x = 0;
    x = 0; // will fail if T is deduced to a const type
           // (EDG and GCC get this wrong)
    return 0;
  }
#endif
};

void test_deduce_with_qual(AnyThreeLevelPtr a3) {
  int * const * const * const ip = a3;
}

struct X { };

struct AnyPtrMem {
  template<typename Class, typename T>
  operator T Class::*() const;
  // FIXME: Can't handle definitions of member templates yet
#if 0
  {
    T x = 0;
    x = 0; // will fail if T is deduced to a const type.
           // (EDG and GCC get this wrong)
    return 0;
  }
#endif
};

void test_deduce_ptrmem_with_qual(AnyPtrMem apm) {
  const float X::* pm = apm;
}
