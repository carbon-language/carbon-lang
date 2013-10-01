// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++11-extensions %s
//
// FIXME: This file is overflow from test/SemaCXX/typo-correction.cpp due to a
// hard-coded limit of 20 different typo corrections Sema::CorrectTypo will
// attempt within a single file (which is to avoid having very broken files take
// minutes to finally be rejected by the parser).

namespace bogus_keyword_suggestion {
void test() {
   status = "OK";  // expected-error-re {{use of undeclared identifier 'status'$}}
   return status;  // expected-error-re {{use of undeclared identifier 'status'$}}
 }
}

namespace PR13387 {
struct A {
  void CreateFoo(float, float);
  void CreateBar(float, float);
};
struct B : A {
  using A::CreateFoo;
  void CreateFoo(int, int);
};
void f(B &x) {
  x.Createfoo(0,0);  // expected-error {{no member named 'Createfoo' in 'PR13387::B'; did you mean 'CreateFoo'?}}
}
}

struct DataStruct {void foo();};
struct T {
 DataStruct data_struct;
 void f();
};
// should be void T::f();
void f() {
 data_struct->foo();  // expected-error-re{{use of undeclared identifier 'data_struct'$}}
}

namespace PR12287 {
class zif {
  void nab(int);
};
void nab();  // expected-note{{'::PR12287::nab' declared here}}
void zif::nab(int) {
  nab();  // expected-error{{too few arguments to function call, expected 1, have 0; did you mean '::PR12287::nab'?}}
}
}

namespace TemplateFunction {
template <class T>
void A(T) { }  // expected-note {{'::TemplateFunction::A' declared here}}

template <class T>
void B(T) { }  // expected-note {{'::TemplateFunction::B' declared here}}

class Foo {
 public:
  void A(int, int) {}
  void B() {}
};

void test(Foo F, int num) {
  F.A(num);  // expected-error {{too few arguments to function call, expected 2, have 1; did you mean '::TemplateFunction::A'?}}
  F.B(num);  // expected-error {{too many arguments to function call, expected 0, have 1; did you mean '::TemplateFunction::B'?}}
}
}
namespace using_suggestion_val_dropped_specifier {
void FFF() {} // expected-note {{'::using_suggestion_val_dropped_specifier::FFF' declared here}}
namespace N { }
using N::FFF; // expected-error {{no member named 'FFF' in namespace 'using_suggestion_val_dropped_specifier::N'; did you mean '::using_suggestion_val_dropped_specifier::FFF'?}}
}

namespace class_member_typo_corrections {
class Outer {
public:
  class Inner {};  // expected-note {{'Outer::Inner' declared here}}
  Inner MyMethod(Inner arg);
};

Inner Outer::MyMethod(Inner arg) {  // expected-error {{unknown type name 'Inner'; did you mean 'Outer::Inner'?}}
  return Inner();
}

class Result {
public:
  enum ResultType {
    ENTITY,  // expected-note {{'Result::ENTITY' declared here}}
    PREDICATE,  // expected-note {{'Result::PREDICATE' declared here}}
    LITERAL  // expected-note {{'Result::LITERAL' declared here}}
  };

  ResultType type();
};

void test() {
  Result result_cell;
  switch (result_cell.type()) {
  case ENTITY:  // expected-error {{use of undeclared identifier 'ENTITY'; did you mean 'Result::ENTITY'?}}
  case LITERAL:  // expected-error {{use of undeclared identifier 'LITERAL'; did you mean 'Result::LITERAL'?}}
  case PREDICAT:  // expected-error {{use of undeclared identifier 'PREDICAT'; did you mean 'Result::PREDICATE'?}}
    break;
  }
}

class Figure {
  enum ResultType {
    SQUARE,
    TRIANGLE,
    CIRCLE
  };

public:
  ResultType type();
};

void testAccess() {
  Figure obj;
  switch (obj.type()) {  // expected-warning {{enumeration values 'SQUARE', 'TRIANGLE', and 'CIRCLE' not handled in switch}}
  case SQUARE:  // expected-error-re {{use of undeclared identifier 'SQUARE'$}}
  case TRIANGLE:  // expected-error-re {{use of undeclared identifier 'TRIANGLE'$}}
  case CIRCE:  // expected-error-re {{use of undeclared identifier 'CIRCE'$}}
    break;
  }
}
}

long readline(const char *, char *, unsigned long);
void assign_to_unknown_var() {
    deadline_ = 1;  // expected-error-re {{use of undeclared identifier 'deadline_'$}}
}

namespace no_ns_before_dot {
namespace re2 {}
void test() {
    req.set_check(false);  // expected-error-re {{use of undeclared identifier 'req'$}}
}
}

namespace PR17394 {
  class A {
  protected:
    long zzzzzzzzzz;
  };
  class B : private A {};
  B zzzzzzzzzy<>; // expected-error {{expected ';' after top level declarator}}{}
}
