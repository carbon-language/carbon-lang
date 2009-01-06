// RUN: clang -fsyntax-only -verify %s 

namespace Ns {
  int f(); // expected-note{{previous declaration is here}}

  enum E {
    Enumerator
  };
}
namespace Ns {
  double f(); // expected-error{{functions that differ only in their return type cannot be overloaded}}

  int x = Enumerator;
}

namespace Ns2 {
  float f();
}

int y = Ns::Enumerator;

namespace Ns2 {
  float f(int); // expected-note{{previous declaration is here}}
}

namespace Ns2 {
  double f(int); // expected-error{{functions that differ only in their return type cannot be overloaded}}
}

namespace N {
  int& f1();
}

namespace N {
  struct f1 {
    static int member;
  };

  void test_f1() {
    int &i1 = f1();
  }
}

namespace N {
  float& f1(int);

  struct f2 {
    static int member;
  };
  void f2();
}

int i1 = N::f1::member;
typedef struct N::f1 type1;
int i2 = N::f2::member;
typedef struct N::f2 type2;

void test_f1(int i) {
  int &v1 = N::f1();
  float &v2 = N::f1(i);
  int v3 = ::i1;
  int v4 = N::f1::member;
}

typedef int f2_type;
namespace a {
  typedef int f2_type(int, int);

  void test_f2() {
    ::f2_type(1, 2); // expected-error {{function-style cast to a builtin type can only take one argument}}
  }
}



