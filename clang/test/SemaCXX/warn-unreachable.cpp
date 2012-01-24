// RUN: %clang_cc1 %s -fcxx-exceptions -fexceptions -fsyntax-only -verify -fblocks -Wunreachable-code -Wno-unused-value

int &halt() __attribute__((noreturn));
int &live();
int dead();
int liveti() throw(int);
int (*livetip)() throw(int);

int test1() {
  try {
    live();
  } catch (int i) {
    live();
  }
  return 1;
}

void test2() {
  try {
    live();
  } catch (int i) {
    live();
  }
  try {
    liveti();
  } catch (int i) {
    live();
  }
  try {
    livetip();
  } catch (int i) {
    live();
  }
  throw 1;
  dead();       // expected-warning {{will never be executed}}
}


void test3() {
  halt()
    --;         // expected-warning {{will never be executed}}
  // FIXME: The unreachable part is just the '?', but really all of this
  // code is unreachable and shouldn't be separately reported.
  halt()        // expected-warning {{will never be executed}}
    ? 
    dead() : dead();
  live(),
    float       
      (halt()); // expected-warning {{will never be executed}}
}

void test4() {
  struct S {
    int mem;
  } s;
  S &foor();
  halt(), foor()// expected-warning {{will never be executed}}
    .mem;       
}

void test5() {
  struct S {
    int mem;
  } s;
  S &foor() __attribute__((noreturn));
  foor()
    .mem;       // expected-warning {{will never be executed}}
}

void test6() {
  struct S {
    ~S() { }
    S(int i) { }
  };
  live(),
    S
      (halt());  // expected-warning {{will never be executed}}
}

// Don't warn about unreachable code in template instantiations, as
// they may only be unreachable in that specific instantiation.
void isUnreachable();

template <typename T> void test_unreachable_templates() {
  T::foo();
  isUnreachable();  // no-warning
}

struct TestUnreachableA {
  static void foo() __attribute__((noreturn));
};
struct TestUnreachableB {
  static void foo();
};

void test_unreachable_templates_harness() {
  test_unreachable_templates<TestUnreachableA>();
  test_unreachable_templates<TestUnreachableB>(); 
}

// Do warn about non-dependent unreachable code in templates
// Warn even if the template is never instantiated

template<typename T> void test_non_dependent_unreachable_templates() {
  TestUnreachableA::foo();
  isUnreachable(); // expected-warning {{will never be executed}}
}

// Warn only once even if the template is instantiated multiple times

template<typename T> void test_non_dependent_unreachable_templates2() {
  TestUnreachableA::foo();
  isUnreachable(); // expected-warning {{will never be executed}}
}

template void test_non_dependent_unreachable_templates2<int>();
template void test_non_dependent_unreachable_templates2<long>();

// Do warn about explict template specializations, as they represent
// actual concrete functions that somebody wrote.

template <typename T> void funcToSpecialize() {}
template <> void funcToSpecialize<int>() {
  halt();
  dead(); // expected-warning {{will never be executed}}
}

// Ensure we don't regress a fix involving undefined bases to template
// destructors when computing the CFG for unreachable code analysis
template<int> struct imp;
template<int a>
struct aligned_storage : imp<a> {
 ~aligned_storage() { }
};

// is this valid?
template<typename T>
class outer {
  class inner;
  void func() {
    inner t;
  }
};
