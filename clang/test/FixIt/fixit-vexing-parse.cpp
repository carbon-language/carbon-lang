// RUN: %clang_cc1 -verify -x c++ %s
// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -x c++ %s 2>&1 | FileCheck %s

struct S {
  int n;
};

struct T {
  T();
  int n;
};

struct U {
  ~U();
  int n;
};

struct V {
  ~V();
};

struct W : V {
};

struct X : U {
};

int F1();
S F2();

namespace N {
  void test() {
    // CHECK: fix-it:"{{.*}}":{34:9-34:11}:" = {}"
    S s1(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{38:9-38:10}:";"
    // CHECK: fix-it:"{{.*}}":{39:7-39:9}:" = {}"
    S s2, // expected-note {{change this ',' to a ';' to call 'F2'}}
    F2(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{43:9-43:11}:""
    // CHECK: fix-it:"{{.*}}":{44:9-44:11}:""
    T t1(), // expected-warning {{function declaration}} expected-note {{remove parentheses}}
      t2(); // expected-warning {{function declaration}} expected-note {{remove parentheses}}

    // CHECK: fix-it:"{{.*}}":{47:8-47:10}:" = {}"
    U u(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{50:8-50:10}:""
    V v(); // expected-warning {{function declaration}} expected-note {{remove parentheses}}

    // CHECK: fix-it:"{{.*}}":{53:8-53:10}:""
    W w(); // expected-warning {{function declaration}} expected-note {{remove parentheses}}

    // TODO: Removing the parens here would not initialize U::n.
    // Maybe suggest an " = X()" initializer for this case?
    // Maybe suggest removing the parens anyway?
    X x(); // expected-warning {{function declaration}}

    // CHECK: fix-it:"{{.*}}":{61:11-61:13}:" = 0"
    int n1(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{65:11-65:12}:";"
    // CHECK: fix-it:"{{.*}}":{66:7-66:9}:" = 0"
    int n2, // expected-note {{change this ',' to a ';' to call 'F1'}}
    F1(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{69:13-69:15}:" = 0.0"
    double d(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    typedef void *Ptr;

    // CHECK: fix-it:"{{.*}}":{74:10-74:12}:" = 0"
    Ptr p(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

#define NULL 0
    // CHECK: fix-it:"{{.*}}":{78:10-78:12}:" = NULL"
    Ptr p(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{81:11-81:13}:" = false"
    bool b(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{84:11-84:13}:" = '\\0'"
    char c(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{87:15-87:17}:" = L'\\0'"
    wchar_t wc(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}
  }
}
