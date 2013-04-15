// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

struct A {
  // This conversion operator allows implicit conversion to bool but not to other integer types.
  typedef A * (A::*MemberPointer);
  operator MemberPointer() const { return m_ptr ? &A::m_ptr : 0; }

  A *m_ptr;

  A *getPtr();
  typedef A * (A::*MemberFnPointer)(void);
};

void testConditionalUse() {
  A obj;

  obj.m_ptr = &obj;
  clang_analyzer_eval(obj.m_ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(&A::m_ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj); // expected-warning{{TRUE}}

  obj.m_ptr = 0;
  clang_analyzer_eval(obj.m_ptr); // expected-warning{{FALSE}}
  clang_analyzer_eval(A::MemberPointer(0)); // expected-warning{{FALSE}}
  clang_analyzer_eval(obj); // expected-warning{{FALSE}}

  clang_analyzer_eval(&A::getPtr); // expected-warning{{TRUE}}
  clang_analyzer_eval(A::MemberFnPointer(0)); // expected-warning{{FALSE}}
}


void testComparison() {
  clang_analyzer_eval(&A::getPtr == &A::getPtr); // expected-warning{{TRUE}}

  // FIXME: Should be TRUE.
  clang_analyzer_eval(&A::m_ptr == &A::m_ptr); // expected-warning{{UNKNOWN}}
}

namespace PR15742 {
  template <class _T1, class _T2> struct A {
    A (const _T1 &, const _T2 &);
  };
  
  typedef void *NPIdentifier;

  template <class T> class B {
  public:
    typedef A<NPIdentifier, bool (T::*) (const NPIdentifier *, unsigned,
                                         NPIdentifier *)> MethodMapMember;
  };

  class C : public B<C> {
  public:
    bool Find(const NPIdentifier *, unsigned, NPIdentifier *);
  };

  void InitStaticData () {
    C::MethodMapMember(0, &C::Find); // don't crash
  }
}

// ---------------
// FALSE NEGATIVES
// ---------------

bool testDereferencing() {
  A obj;
  obj.m_ptr = 0;

  A::MemberPointer member = &A::m_ptr;

  // FIXME: Should be TRUE.
  clang_analyzer_eval(obj.*member == 0); // expected-warning{{UNKNOWN}}

  member = 0;

  // FIXME: Should emit a null dereference.
  return obj.*member; // no-warning
}
