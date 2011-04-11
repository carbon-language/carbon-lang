// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-checker=core,core.experimental -analyzer-store=region -verify -fblocks -analyzer-opt-analyze-nested-blocks %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-checker=core,core.experimental -analyzer-store=region -verify -fblocks   -analyzer-opt-analyze-nested-blocks %s

// Test basic handling of references.
char &test1_aux();
char *test1() {
  return &test1_aux();
}

// Test test1_aux() evaluates to char &.
char test1_as_rvalue() {
  return test1_aux();
}

// Test passing a value as a reference.  The 'const' in test2_aux() adds
// an ImplicitCastExpr, which is evaluated as an lvalue.
int test2_aux(const int &n);
int test2(int n) {
  return test2_aux(n);
}

int test2_b_aux(const short &n);
int test2_b(int n) {
  return test2_b_aux(n);
}

// Test getting the lvalue of a derived and converting it to a base.  This
// previously crashed.
class Test3_Base {};
class Test3_Derived : public Test3_Base {};

int test3_aux(Test3_Base &x);
int test3(Test3_Derived x) {
  return test3_aux(x);
}

//===---------------------------------------------------------------------===//
// Test CFG support for C++ condition variables.
//===---------------------------------------------------------------------===//

int test_init_in_condition_aux();
int test_init_in_condition() {
  if (int x = test_init_in_condition_aux()) { // no-warning
    return 1;
  }
  return 0;
}

int test_init_in_condition_switch() {
  switch (int x = test_init_in_condition_aux()) { // no-warning
    case 1:
      return 0;
    case 2:
      if (x == 2)
        return 0;
      else {
        // Unreachable.
        int *p = 0;
        *p = 0xDEADBEEF; // no-warning
      }
    default:
      break;
  }
  return 0;
}

int test_init_in_condition_while() {
  int z = 0;
  while (int x = ++z) { // no-warning
    if (x == 2)
      break;
  }
  
  if (z == 2)
    return 0;
  
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
  return 0;
}


int test_init_in_condition_for() {
  int z = 0;
  for (int x = 0; int y = ++z; ++x) {
    if (x == y) // no-warning
      break;
  }
  if (z == 1)
    return 0;
    
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
  return 0;
}

//===---------------------------------------------------------------------===//
// Test handling of 'this' pointer.
//===---------------------------------------------------------------------===//

class TestHandleThis {
  int x;

  TestHandleThis();  
  int foo();
  int null_deref_negative();
  int null_deref_positive();  
};

int TestHandleThis::foo() {
  // Assume that 'x' is initialized.
  return x + 1; // no-warning
}

int TestHandleThis::null_deref_negative() {
  x = 10;
  if (x == 10) {
    return 1;
  }
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
  return 0;  
}

int TestHandleThis::null_deref_positive() {
  x = 10;
  if (x == 9) {
    return 1;
  }
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{null pointer}}
  return 0;  
}

// PR 7675 - passing literals by-reference
void pr7675(const double &a);
void pr7675(const int &a);
void pr7675(const char &a);
void pr7675_i(const _Complex double &a);

void pr7675_test() {
  pr7675(10.0);
  pr7675(10);
  pr7675('c');
  pr7675_i(4.0i);
  // Add null deref to ensure we are analyzing the code up to this point.
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{null pointer}}
}

// <rdar://problem/8375510> - CFGBuilder should handle temporaries.
struct R8375510 {
  R8375510();
  ~R8375510();
  R8375510 operator++(int);
};

int r8375510(R8375510 x, R8375510 y) {
  for (; ; x++) { }
}

// PR8419 -- this used to crash.

class String8419 {
 public:
  char& get(int n);
  char& operator[](int n);
};

char& get8419();

void Test8419() {
  String8419 s;
  ++(s.get(0));
  get8419()--;  // used to crash
  --s[0];       // used to crash
  s[0] &= 1;    // used to crash
  s[0]++;       // used to crash
}

// PR8426 -- this used to crash.

void Use(void* to);

template <class T> class Foo {
  ~Foo();
  struct Bar;
  Bar* bar_;
};

template <class T> Foo<T>::~Foo() {
  Use(bar_);
  T::DoSomething();
  bar_->Work();
}

// PR8427 -- this used to crash.

class Dummy {};

bool operator==(Dummy, int);

template <typename T>
class Foo2 {
  bool Bar();
};

template <typename T>
bool Foo2<T>::Bar() {
  return 0 == T();
}

// PR8433 -- this used to crash.

template <typename T>
class Foo3 {
 public:
  void Bar();
  void Baz();
  T value_;
};

template <typename T>
void Foo3<T>::Bar() {
  Baz();
  value_();
}

//===---------------------------------------------------------------------===//
// Handle misc. C++ constructs.
//===---------------------------------------------------------------------===//

namespace fum {
  int i = 3;
};

void test_namespace() {
  // Previously triggered a crash.
  using namespace fum;
  int x = i;
}

// Test handling methods that accept references as parameters, and that
// variables are properly invalidated.
class RDar9203355 {
  bool foo(unsigned valA, long long &result) const;
  bool foo(unsigned valA, int &result) const;
};
bool RDar9203355::foo(unsigned valA, int &result) const {
  long long val;
  if (foo(valA, val) ||
      (int)val != val) // no-warning
    return true;
  result = val; // no-warning
  return false;
}

// Test handling of new[].
void rdar9212512() {
  int *x = new int[10];
  for (unsigned i = 0 ; i < 2 ; ++i) {
    // This previously triggered an uninitialized values warning.
    x[i] = 1;  // no-warning
  }
}

// Test basic support for dynamic_cast<>.
struct Rdar9212495_C { virtual void bar() const; };
class Rdar9212495_B : public Rdar9212495_C {};
class Rdar9212495_A : public Rdar9212495_B {};
const Rdar9212495_A& rdar9212495(const Rdar9212495_C* ptr) {
  const Rdar9212495_A& val = dynamic_cast<const Rdar9212495_A&>(*ptr);
  
  if (&val == 0) {
    val.bar(); // FIXME: This should eventually be a null dereference.
  }
  
  return val;
}

// Test constructors invalidating arguments.  Previously this raised
// an uninitialized value warning.
extern "C" void __attribute__((noreturn)) PR9645_exit(int i);

class PR9645_SideEffect
{
public:
  PR9645_SideEffect(int *pi); // caches pi in i_
  void Read(int *pi); // copies *pi into *i_
private:
  int *i_;
};

void PR9645() {
  int i;

  PR9645_SideEffect se(&i);
  int j = 1;
  se.Read(&j); // this has a side-effect of initializing i.

  PR9645_exit(i); // no-warning
}

PR9645_SideEffect::PR9645_SideEffect(int *pi) : i_(pi) {}
void PR9645_SideEffect::Read(int *pi) { *i_ = *pi; }

// Invalidate fields during C++ method calls.
class RDar9267815 {
  int x;
  void test();
  void test_pos();
  void test2();
  void invalidate();
};

void RDar9267815::test_pos() {
  int *p = 0;
  if (x == 42)
    return;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}
void RDar9267815::test() {
  int *p = 0;
  if (x == 42)
    return;
  if (x == 42)
    *p = 0xDEADBEEF; // no-warning
}

void RDar9267815::test2() {
  int *p = 0;
  if (x == 42)
    return;
  invalidate();
  if (x == 42)
    *p = 0xDEADBEEF; // expected-warning {{null}}
}

