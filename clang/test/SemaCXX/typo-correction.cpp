// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++11-extensions %s
//
// WARNING: Do not add more typo correction test cases to this file lest you run
// afoul the hard-coded limit (escape hatch) of 20 different typos whose
// correction was attempted by Sema::CorrectTypo

struct errc {
  int v_;
  operator int() const {return v_;}
};

class error_condition
{
  int _val_;
public:
  error_condition() : _val_(0) {}

  error_condition(int _val)
    : _val_(_val) {}

  template <class E>
  error_condition(E _e) {
    // make_error_condition must not be typo corrected to error_condition
    // even though the first declaration of make_error_condition has not
    // yet been encountered. This was a bug in the first version of the type
    // name typo correction patch that wasn't noticed until building LLVM with
    // Clang failed.
    *this = make_error_condition(_e);
  }

};

inline error_condition make_error_condition(errc _e) {
  return error_condition(static_cast<int>(_e));
}


// Prior to the introduction of a callback object to further filter possible
// typo corrections, this example would not trigger a suggestion as "base_type"
// is a closer match to "basetype" than is "BaseType" but "base_type" does not
// refer to a base class or non-static data member.
struct BaseType { };
struct Derived : public BaseType { // expected-note {{base class 'BaseType' specified here}}
  static int base_type; // expected-note {{'base_type' declared here}}
  Derived() : basetype() {} // expected-error{{initializer 'basetype' does not name a non-static data member or base class; did you mean the base class 'BaseType'?}}
};

// Test the improvement from passing a callback object to CorrectTypo in
// the helper function LookupMemberExprInRecord.
int get_type(struct Derived *st) {
  return st->Base_Type; // expected-error{{no member named 'Base_Type' in 'Derived'; did you mean 'base_type'?}}
}

// In this example, somename should not be corrected to the cached correction
// "some_name" since "some_name" is a class and a namespace name is needed.
class some_name {}; // expected-note {{'some_name' declared here}}
somename Foo; // expected-error {{unknown type name 'somename'; did you mean 'some_name'?}}
namespace SomeName {} // expected-note {{namespace 'SomeName' defined here}}
using namespace somename; // expected-error {{no namespace named 'somename'; did you mean 'SomeName'?}}


// Without the callback object, CorrectTypo would choose "field1" as the
// correction for "fielda" as it is closer than "FieldA", but that correction
// would be later discarded by the caller and no suggestion would be given.
struct st {
  struct {
    int field1;
  };
  double FieldA; // expected-note{{'FieldA' declared here}}
};
st var = { .fielda = 0.0 }; // expected-error{{field designator 'fielda' does not refer to any field in type 'st'; did you mean 'FieldA'?}}

// Test the improvement from passing a callback object to CorrectTypo in
// Sema::BuildCXXNestedNameSpecifier. And also for the improvement by doing
// so in Sema::getTypeName.
typedef char* another_str; // expected-note{{'another_str' declared here}}
namespace AnotherStd { // expected-note{{'AnotherStd' declared here}}
  class string {};
}
another_std::string str; // expected-error{{use of undeclared identifier 'another_std'; did you mean 'AnotherStd'?}}
another_str *cstr = new AnotherStr; // expected-error{{unknown type name 'AnotherStr'; did you mean 'another_str'?}}

// Test the improvement from passing a callback object to CorrectTypo in
// Sema::ActOnSizeofParameterPackExpr.
char* TireNames;
template<typename ...TypeNames> struct count { // expected-note{{parameter pack 'TypeNames' declared here}}
  static const unsigned value = sizeof...(TyreNames); // expected-error{{'TyreNames' does not refer to the name of a parameter pack; did you mean 'TypeNames'?}}
};

// Test the typo-correction callback in Sema::DiagnoseUnknownTypeName.
namespace unknown_type_test {
  class StreamOut {}; // expected-note 2 {{'StreamOut' declared here}}
  long stream_count; // expected-note 2 {{'stream_count' declared here}}
};
unknown_type_test::stream_out out; // expected-error{{no type named 'stream_out' in namespace 'unknown_type_test'; did you mean 'StreamOut'?}}

// Demonstrate a case where using only the cached value returns the wrong thing
// when the cached value was the result of a previous callback object that only
// accepts a subset of the current callback object.
namespace {
using namespace unknown_type_test;
void bar(long i);
void before_caching_classname() {
  bar((stream_out)); // expected-error{{use of undeclared identifier 'stream_out'; did you mean 'stream_count'?}}
}
stream_out out; // expected-error{{unknown type name 'stream_out'; did you mean 'StreamOut'?}}
void after_caching_classname() {
  bar((stream_out)); // expected-error{{use of undeclared identifier 'stream_out'; did you mean 'stream_count'?}}
}
}

// Test the typo-correction callback in Sema::DiagnoseInvalidRedeclaration.
struct BaseDecl {
  void add_in(int i);
};
struct TestRedecl : public BaseDecl {
  void add_it(int i); // expected-note{{'add_it' declared here}}
};
void TestRedecl::add_in(int i) {} // expected-error{{out-of-line definition of 'add_in' does not match any declaration in 'TestRedecl'; did you mean 'add_it'?}}

// Test the improved typo correction for the Parser::ParseCastExpr =>
// Sema::ActOnIdExpression => Sema::DiagnoseEmptyLookup call path.
class SomeNetMessage; // expected-note 2{{'SomeNetMessage'}}
class Message {};
void foo(Message&);
void foo(SomeNetMessage&);
void doit(void *data) {
  Message somenetmsg; // expected-note{{'somenetmsg' declared here}}
  foo(somenetmessage); // expected-error{{use of undeclared identifier 'somenetmessage'; did you mean 'somenetmsg'?}}
  foo((somenetmessage)data); // expected-error{{unknown type name 'somenetmessage'; did you mean 'SomeNetMessage'?}} expected-error{{incomplete type}}
}

// Test the typo-correction callback in BuildRecoveryCallExpr.
// Solves the main issue in PR 9320 of suggesting corrections that take the
// wrong number of arguments.
void revoke(const char*); // expected-note 2{{'revoke' declared here}}
void Test() {
  Invoke(); // expected-error{{use of undeclared identifier 'Invoke'}}
  Invoke("foo"); // expected-error{{use of undeclared identifier 'Invoke'; did you mean 'revoke'?}}
  Invoke("foo", "bar"); // expected-error{{use of undeclared identifier 'Invoke'}}
}
void Test2(void (*invoke)(const char *, int)) { // expected-note{{'invoke' declared here}}
  Invoke(); // expected-error{{use of undeclared identifier 'Invoke'}}
  Invoke("foo"); // expected-error{{use of undeclared identifier 'Invoke'; did you mean 'revoke'?}}
  Invoke("foo", 7); // expected-error{{use of undeclared identifier 'Invoke'; did you mean 'invoke'?}}
  Invoke("foo", 7, 22); // expected-error{{use of undeclared identifier 'Invoke'}}
}

void provoke(const char *x, bool y=false) {} // expected-note 2{{'provoke' declared here}}
void Test3() {
  Provoke(); // expected-error{{use of undeclared identifier 'Provoke'}}
  Provoke("foo"); // expected-error{{use of undeclared identifier 'Provoke'; did you mean 'provoke'?}}
  Provoke("foo", true); // expected-error{{use of undeclared identifier 'Provoke'; did you mean 'provoke'?}}
  Provoke("foo", 7, 22); // expected-error{{use of undeclared identifier 'Provoke'}}
}

// PR 11737 - Don't try to typo-correct the implicit 'begin' and 'end' in a
// C++11 for-range statement.
struct R {};
bool begun(R);
void RangeTest() {
  for (auto b : R()) {} // expected-error {{invalid range expression of type 'R'}}
}

// PR 12019 - Avoid infinite mutual recursion in DiagnoseInvalidRedeclaration
// by not trying to typo-correct a method redeclaration to declarations not
// in the current record.
class Parent {
 void set_types(int index, int value);
 void add_types(int value);
};
class Child: public Parent {};
void Child::add_types(int value) {} // expected-error{{out-of-line definition of 'add_types' does not match any declaration in 'Child'}}

// Fix the callback based filtering of typo corrections within
// Sema::ActOnIdExpression by Parser::ParseCastExpression to allow type names as
// potential corrections for template arguments.
namespace clash {
class ConstructExpr {}; // expected-note 2{{'clash::ConstructExpr' declared here}}
}
class ClashTool {
  bool HaveConstructExpr();
  template <class T> T* getExprAs();

  void test() {
    ConstructExpr *expr = // expected-error{{unknown type name 'ConstructExpr'; did you mean 'clash::ConstructExpr'?}}
        getExprAs<ConstructExpr>(); // expected-error{{unknown type name 'ConstructExpr'; did you mean 'clash::ConstructExpr'?}}
  }
};

namespace test1 {
  struct S {
    struct Foobar *f;  // expected-note{{'Foobar' declared here}}
  };
  test1::FooBar *b;  // expected-error{{no type named 'FooBar' in namespace 'test1'; did you mean 'Foobar'?}}
}

namespace ImplicitInt {
  void f(int, unsinged); // expected-error{{did you mean 'unsigned'}}
  struct S {
    unsinged : 4; // expected-error{{did you mean 'unsigned'}}
  };
}

namespace PR12951 {
// If there are two corrections that have the same identifier and edit distance
// and only differ by their namespaces, don't suggest either as a correction
// since both are equally likely corrections.
namespace foobar { struct Thing {}; }
namespace bazquux { struct Thing {}; }
void f() { Thing t; } // expected-error{{unknown type name 'Thing'}}
}

namespace PR13051 {
  template<typename T> struct S {
    template<typename U> void f();
    operator bool() const;
  };

  void f() {
    f(&S<int>::tempalte f<int>); // expected-error{{did you mean 'template'?}}
    f(&S<int>::opeartor bool); // expected-error{{did you mean 'operator'?}}
    f(&S<int>::foo); // expected-error-re{{no member named 'foo' in 'PR13051::S<int>'$}}
  }
}

inf f(doulbe); // expected-error{{'int'}} expected-error{{'double'}}

namespace PR6325 {
class foo { }; // expected-note{{'foo' declared here}}
// Note that for this example (pulled from the PR), if keywords are not excluded
// as correction candidates then no suggestion would be given; correcting
// 'boo' to 'bool' is the same edit distance as correcting 'boo' to 'foo'.
class bar : boo { }; // expected-error{{unknown class name 'boo'; did you mean 'foo'?}}
}

namespace outer {
  void somefunc();  // expected-note{{'::outer::somefunc' declared here}}
  void somefunc(int, int);  // expected-note{{'::outer::somefunc' declared here}}

  namespace inner {
    void somefunc(int) {
      someFunc();  // expected-error{{use of undeclared identifier 'someFunc'; did you mean '::outer::somefunc'?}}
      someFunc(1, 2);  // expected-error{{use of undeclared identifier 'someFunc'; did you mean '::outer::somefunc'?}}
    }
  }
}

namespace bogus_keyword_suggestion {
void test() {
   status = "OK"; // expected-error-re{{use of undeclared identifier 'status'$}}
   return status; // expected-error-re{{use of undeclared identifier 'status'$}}
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
  x.Createfoo(0,0); // expected-error {{no member named 'Createfoo' in 'PR13387::B'; did you mean 'CreateFoo'?}}
}
}

struct DataStruct {void foo();};
struct T {
 DataStruct data_struct;
 void f();
};
// should be void T::f();
void f() {
 data_struct->foo(); // expected-error-re{{use of undeclared identifier 'data_struct'$}}
}

namespace b6956809_test1 {
  struct A {};
  struct B {};

  struct S1 {
    void method(A*);  // no note here
    void method(B*);
  };

  void test1() {
    B b;
    S1 s;
    s.methodd(&b);  // expected-error{{no member named 'methodd' in 'b6956809_test1::S1'; did you mean 'method'}}
  }

  struct S2 {
    S2();
    void method(A*) const;  // expected-note{{candidate function not viable}}
   private:
    void method(B*);  // expected-note{{candidate function not viable}}
  };

  void test2() {
    B b;
    const S2 s;
    s.methodd(&b);  // expected-error{{no member named 'methodd' in 'b6956809_test1::S2'; did you mean 'method'}}  expected-error{{no matching member function for call to 'method'}}
  }
}

namespace b6956809_test2 {
  template<typename T> struct Err { typename T::error n; };  // expected-error{{type 'void *' cannot be used prior to '::' because it has no members}}
  struct S {
    template<typename T> typename Err<T>::type method(T);  // expected-note{{in instantiation of template class 'b6956809_test2::Err<void *>' requested here}}  expected-note{{while substituting deduced template arguments into function template 'method' [with T = void *]}}
    template<typename T> int method(T *);
  };

  void test() {
    S s;
    int k = s.methodd((void*)0);  // expected-error{{no member named 'methodd' in 'b6956809_test2::S'; did you mean 'method'?}}
  }
}

namespace CorrectTypo_has_reached_its_limit {
int flibberdy();  // no note here
int no_correction() {
  return gibberdy();  // expected-error-re{{use of undeclared identifier 'gibberdy'$}}
};
}
