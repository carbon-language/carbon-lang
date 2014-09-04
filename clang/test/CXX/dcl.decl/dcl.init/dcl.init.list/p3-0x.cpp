// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace std {
  typedef decltype(sizeof(int)) size_t;

  template <typename E>
  struct initializer_list
  {
    const E *p;
    size_t n;
    initializer_list(const E *p, size_t n) : p(p), n(n) {}
  };

  struct string {
    string(const char *);
  };

  template<typename A, typename B>
  struct pair {
    pair(const A&, const B&);
  };
}

namespace bullet1 {
  double ad[] = { 1, 2.0 };
  int ai[] = { 1, 2.0 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}

  struct S2 {
    int m1;
    double m2, m3;
  };

  S2 s21 = { 1, 2, 3.0 };
  S2 s22 { 1.0, 2, 3 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}
  S2 s23 { };
}

namespace bullet4_example1 {
  struct S {
    S(std::initializer_list<double> d) {}
    S(std::initializer_list<int> i) {}
    S() {}
  };

  S s1 = { 1.0, 2.0, 3.0 };
  S s2 = { 1, 2, 3 };
  S s3 = { };
}

namespace bullet4_example2 {
  struct Map {
    Map(std::initializer_list<std::pair<std::string,int>>) {}
  };

  Map ship = {{"Sophie",14}, {"Surprise",28}};
}

namespace bullet4_example3 {
  struct S {
    S(int, double, double) {}
    S() {}
  };

  S s1 = { 1, 2, 3.0 };
  S s2 { 1.0, 2, 3 }; // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}
  S s3 {};
}

namespace bullet5 {
  int x1 {2};
  int x2 {2.0};  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}
}

namespace bullet6 {
  struct S {
    S(std::initializer_list<double>) {}
    S(const std::string &) {}
  };

  const S& r1 = { 1, 2, 3.0 };
  const S& r2 = { "Spinach" };
  S& r3 = { 1, 2, 3 };  // expected-error {{non-const lvalue reference to type 'bullet6::S' cannot bind to an initializer list temporary}}
  const int& i1 = { 1 };
  const int& i2 = { 1.1 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}} expected-warning {{implicit conversion}}
  const int (&iar)[2] = { 1, 2 };
}

namespace bullet7 {
  int** pp {};
}

namespace bullet8 {
  struct A { int i; int j; };
  A a1 { 1, 2 };
  A a2 { 1.2 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}} expected-warning {{implicit conversion}}

  struct B {
    B(std::initializer_list<int> i) {}
  };
  B b1 { 1, 2 };
  B b2 { 1, 2.0 }; // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}

  struct C {
    C(int i, double j) {}
  };
  C c1 = { 1, 2.2 };
  // FIXME: Suppress the narrowing warning in the cases where we issue a narrowing error.
  C c2 = { 1.1, 2 }; // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}} expected-warning {{implicit conversion}}

  int j { 1 };
  int k { };
}

namespace rdar13395022 {
  struct MoveOnly { // expected-note {{candidate}}
    MoveOnly(MoveOnly&&); // expected-note 2{{copy constructor is implicitly deleted because}} expected-note {{candidate}}
  };

  void test(MoveOnly mo) {
    auto &&list1 = {mo}; // expected-error {{call to implicitly-deleted copy constructor}} expected-note {{in initialization of temporary of type 'std::initializer_list}}
    MoveOnly (&&list2)[1] = {mo}; // expected-error {{call to implicitly-deleted copy constructor}} expected-note {{in initialization of temporary of type 'rdar13395022::MoveOnly [1]'}}
    std::initializer_list<MoveOnly> &&list3 = {};
    MoveOnly (&&list4)[1] = {}; // expected-error {{no matching constructor}}
    // expected-note@-1 {{in implicit initialization of array element 0 with omitted initializer}}
    // expected-note@-2 {{in initialization of temporary of type 'rdar13395022::MoveOnly [1]' created to list-initialize this reference}}
  }
}
