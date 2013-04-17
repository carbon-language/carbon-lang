// RUN: %clang_cc1 -fsyntax-only -verify %s

// Reachability tests have to come first because they get suppressed
// if any errors have occurred.
namespace test5 {
  struct A {
    __attribute__((noreturn)) void fail();
    void nofail();
  } a;

  int &test1() {
    a.nofail();
  } // expected-warning {{control reaches end of non-void function}}

  int &test2() {
    a.fail();
  }
}

// PR5620
void f0() __attribute__((__noreturn__));
void f1(void (*)()); 
void f2() { f1(f0); }

// Taking the address of a noreturn function
void test_f0a() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// Taking the address of an overloaded noreturn function 
void f0(int) __attribute__((__noreturn__));

void test_f0b() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// No-returned function pointers
typedef void (* noreturn_fp)() __attribute__((noreturn));

void f3(noreturn_fp); // expected-note{{candidate function}}

void test_f3() {
  f3(f0); // okay
  f3(f2); // expected-error{{no matching function for call}}
}


class xpto {
  int blah() __attribute__((noreturn));
};

int xpto::blah() {
  return 3; // expected-warning {{function 'blah' declared 'noreturn' should not return}}
}

// PR12948

namespace PR12948 {
  template<int>
  void foo() __attribute__((__noreturn__));

  template<int>
  void foo() {
    while (1) continue;
  }

  void bar() __attribute__((__noreturn__));

  void bar() {
    foo<0>();
  }


  void baz() __attribute__((__noreturn__));
  typedef void voidfn();
  voidfn baz;

  template<typename> void wibble()  __attribute__((__noreturn__));
  template<typename> voidfn wibble;
}

// PR15291
// Overload resolution per over.over should allow implicit noreturn adjustment.
namespace PR15291 {
  __attribute__((noreturn)) void foo(int) {}
  __attribute__((noreturn)) void foo(double) {}

  template <typename T>
  __attribute__((noreturn)) void bar(T) {}

  void baz(int) {}
  void baz(double) {}

  template <typename T>
  void qux(T) {}

  // expected-note@+5 {{candidate function [with T = void (*)(int) __attribute__((noreturn))] not viable: no overload of 'baz' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+4 {{candidate function [with T = void (*)(int) __attribute__((noreturn))] not viable: no overload of 'qux' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+3 {{candidate function [with T = void (*)(int) __attribute__((noreturn))] not viable: no overload of 'bar' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+2 {{candidate function [with T = void (*)(int)] not viable: no overload of 'bar' matching 'void (*)(int)' for 1st argument}}
  // expected-note@+1 {{candidate function [with T = void (int)] not viable: no overload of 'bar' matching 'void (*)(int)' for 1st argument}}
  template <typename T> void accept_T(T) {}

  // expected-note@+1 {{candidate function not viable: no overload of 'bar' matching 'void (*)(int)' for 1st argument}}
  void accept_fptr(void (*f)(int)) {
    f(42);
  }

  // expected-note@+2 {{candidate function not viable: no overload of 'baz' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  // expected-note@+1 {{candidate function not viable: no overload of 'qux' matching 'void (*)(int) __attribute__((noreturn))' for 1st argument}}
  void accept_noreturn_fptr(void __attribute__((noreturn)) (*f)(int)) {
    f(42);
  }

  typedef void (*fptr_t)(int);
  typedef void __attribute__((noreturn)) (*fptr_noreturn_t)(int);

  // expected-note@+1 {{candidate function not viable: no overload of 'bar' matching 'fptr_t' (aka 'void (*)(int)') for 1st argument}}
  void accept_fptr_t(fptr_t f) {
    f(42);
  }

  // expected-note@+2 {{candidate function not viable: no overload of 'baz' matching 'fptr_noreturn_t' (aka 'void (*)(int) __attribute__((noreturn))') for 1st argument}}
  // expected-note@+1 {{candidate function not viable: no overload of 'qux' matching 'fptr_noreturn_t' (aka 'void (*)(int) __attribute__((noreturn))') for 1st argument}}
  void accept_fptr_noreturn_t(fptr_noreturn_t f) {
    f(42);
  }

  // Stripping noreturn should work if everything else is correct.
  void strip_noreturn() {
    accept_fptr(foo);
    accept_fptr(bar<int>);
    accept_fptr(bar<double>); // expected-error {{no matching function for call to 'accept_fptr'}}

    accept_fptr_t(foo);
    accept_fptr_t(bar<int>);
    accept_fptr_t(bar<double>); // expected-error {{no matching function for call to 'accept_fptr_t'}}

    accept_T<void __attribute__((noreturn)) (*)(int)>(foo);
    accept_T<void __attribute__((noreturn)) (*)(int)>(bar<int>);
    accept_T<void __attribute__((noreturn)) (*)(int)>(bar<double>); // expected-error {{no matching function for call to 'accept_T'}}

    accept_T<void (*)(int)>(foo);
    accept_T<void (*)(int)>(bar<int>);
    accept_T<void (*)(int)>(bar<double>); // expected-error {{no matching function for call to 'accept_T'}}

    accept_T<void (int)>(foo);
    accept_T<void (int)>(bar<int>);
    accept_T<void (int)>(bar<double>); // expected-error {{no matching function for call to 'accept_T'}}
  }

  // Introducing noreturn should not work.
  void introduce_noreturn() {
    accept_noreturn_fptr(baz); // expected-error {{no matching function for call to 'accept_noreturn_fptr'}}
    accept_noreturn_fptr(qux<int>); // expected-error {{no matching function for call to 'accept_noreturn_fptr'}}

    accept_fptr_noreturn_t(baz); // expected-error {{no matching function for call to 'accept_fptr_noreturn_t'}}
    accept_fptr_noreturn_t(qux<int>); // expected-error {{no matching function for call to 'accept_fptr_noreturn_t'}}

    accept_T<void __attribute__((noreturn)) (*)(int)>(baz); // expected-error {{no matching function for call to 'accept_T'}}
    accept_T<void __attribute__((noreturn)) (*)(int)>(qux<int>); // expected-error {{no matching function for call to 'accept_T'}}
  }
}
