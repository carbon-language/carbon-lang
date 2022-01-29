// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection -verify\
// RUN:                    -Wno-tautological-compare\
// RUN:                    -x c %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection -verify\
// RUN:                    -Wno-tautological-compare\
// RUN:                    -x c++ -std=c++14 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection -verify\
// RUN:                    -Wno-tautological-compare\
// RUN:                    -x c++ -std=c++17 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection -verify\
// RUN:                    -Wno-tautological-compare\
// RUN:                    -DINLINE -x c %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection -verify\
// RUN:                    -Wno-tautological-compare\
// RUN:                    -DINLINE -x c++ -std=c++14 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection -verify\
// RUN:                    -Wno-tautological-compare\
// RUN:                    -DINLINE -x c++ -std=c++17 %s

void clang_analyzer_eval(int);

struct S {
  int field;

#if __cplusplus
  const struct S *getThis() const { return this; }
  const struct S *operator +() const { return this; }

  bool check() const { return this == this; }
  bool operator !() const { return this != this; }

  int operator *() const { return field; }
#endif
};

#if __cplusplus
const struct S *operator -(const struct S &s) { return &s; }
bool operator ~(const struct S &s) { return (&s) != &s; }
#endif


#ifdef INLINE
struct S getS() {
  struct S s = { 42 };
  return s;
}
#else
struct S getS();
#endif


void testAssignment() {
  struct S s = getS();

  if (s.field != 42) return;
  clang_analyzer_eval(s.field == 42); // expected-warning{{TRUE}}

  s.field = 0;
  clang_analyzer_eval(s.field == 0); // expected-warning{{TRUE}}

#if __cplusplus
  clang_analyzer_eval(s.getThis() == &s); // expected-warning{{TRUE}}
  clang_analyzer_eval(+s == &s); // expected-warning{{TRUE}}
  clang_analyzer_eval(-s == &s); // expected-warning{{TRUE}}

  clang_analyzer_eval(s.check()); // expected-warning{{TRUE}}
  clang_analyzer_eval(!s); // expected-warning{{FALSE}}
  clang_analyzer_eval(~s); // expected-warning{{FALSE}}

  clang_analyzer_eval(*s == 0); // expected-warning{{TRUE}}
#endif
}


void testImmediateUse() {
  int x = getS().field;

  if (x != 42) return;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}

#if __cplusplus
  clang_analyzer_eval((void *)getS().getThis() == (void *)&x); // expected-warning{{FALSE}}
  clang_analyzer_eval((void *)+getS() == (void *)&x); // expected-warning{{FALSE}}
  clang_analyzer_eval((void *)-getS() == (void *)&x); // expected-warning{{FALSE}}

  clang_analyzer_eval(getS().check()); // expected-warning{{TRUE}}
  clang_analyzer_eval(!getS()); // expected-warning{{FALSE}}
  clang_analyzer_eval(~getS()); // expected-warning{{FALSE}}
#endif
}

int getConstrainedField(struct S s) {
  if (s.field != 42) return 42;
  return s.field;
}

int getAssignedField(struct S s) {
  s.field = 42;
  return s.field;
}

void testArgument() {
  clang_analyzer_eval(getConstrainedField(getS()) == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(getAssignedField(getS()) == 42); // expected-warning{{TRUE}}
}

void testImmediateUseParens() {
  int x = ((getS())).field;

  if (x != 42) return;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}

  clang_analyzer_eval(getConstrainedField(((getS()))) == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(getAssignedField(((getS()))) == 42); // expected-warning{{TRUE}}

#if __cplusplus
  clang_analyzer_eval(((getS())).check()); // expected-warning{{TRUE}}
  clang_analyzer_eval(!((getS()))); // expected-warning{{FALSE}}
  clang_analyzer_eval(~((getS()))); // expected-warning{{FALSE}}
#endif
}


//--------------------
// C++-only tests
//--------------------

#if __cplusplus
void testReferenceAssignment() {
  const S &s = getS();

  if (s.field != 42) return;
  clang_analyzer_eval(s.field == 42); // expected-warning{{TRUE}}

  clang_analyzer_eval(s.getThis() == &s); // expected-warning{{TRUE}}
  clang_analyzer_eval(+s == &s); // expected-warning{{TRUE}}

  clang_analyzer_eval(s.check()); // expected-warning{{TRUE}}
  clang_analyzer_eval(!s); // expected-warning{{FALSE}}
  clang_analyzer_eval(~s); // expected-warning{{FALSE}}

  clang_analyzer_eval(*s == 42); // expected-warning{{TRUE}}
}


int getConstrainedFieldRef(const S &s) {
  if (s.field != 42) return 42;
  return s.field;
}

bool checkThis(const S &s) {
  return s.getThis() == &s;
}

bool checkThisOp(const S &s) {
  return +s == &s;
}

bool checkThisStaticOp(const S &s) {
  return -s == &s;
}

void testReferenceArgument() {
  clang_analyzer_eval(getConstrainedFieldRef(getS()) == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(checkThis(getS())); // expected-warning{{TRUE}}
  clang_analyzer_eval(checkThisOp(getS())); // expected-warning{{TRUE}}
  clang_analyzer_eval(checkThisStaticOp(getS())); // expected-warning{{TRUE}}
}


int getConstrainedFieldOp(S s) {
  if (*s != 42) return 42;
  return *s;
}

int getConstrainedFieldRefOp(const S &s) {
  if (*s != 42) return 42;
  return *s;
}

void testImmediateUseOp() {
  int x = *getS();
  if (x != 42) return;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}

  clang_analyzer_eval(getConstrainedFieldOp(getS()) == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(getConstrainedFieldRefOp(getS()) == 42); // expected-warning{{TRUE}}
}

namespace EmptyClass {
  struct Base {
    int& x;

    Base(int& x) : x(x) {}
  };

  struct Derived : public Base {
    Derived(int& x) : Base(x) {}

    void operator=(int a) { x = a; }
  };

  Derived ref(int& a) { return Derived(a); }

  // There used to be a warning here, because analyzer treated Derived as empty.
  int test() {
    int a;
    ref(a) = 42;
    return a; // no warning
  }
}

#if __cplusplus >= 201703L
namespace aggregate_inheritance_cxx17 {
struct A {
  int x;
};

struct B {
  int y;
};

struct C: B {
  int z;
};

struct D: A, C {
  int w;
};

void foo() {
  D d{1, 2, 3, 4};
  clang_analyzer_eval(d.x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(d.y == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(d.z == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(d.w == 4); // expected-warning{{TRUE}}
}
} // namespace aggregate_inheritance_cxx17
#endif

namespace flex_array_inheritance_cxx17 {
struct A {
  int flexible_array[];
};

struct B {
  long cookie;
};

struct C : B {
  A a;
};

void foo() {
  C c{}; // no-crash
}
} // namespace flex_array_inheritance_cxx17
#endif
