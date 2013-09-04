// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -use-auto %t.cpp -- -std=c++11
// RUN: FileCheck -input-file=%t.cpp %s

class MyType {
};

class MyDerivedType : public MyType {
};

int main(int argc, char **argv) {
  MyType *a = new MyType();
  // CHECK: auto a = new MyType();

  static MyType *a_static = new MyType();
  // CHECK: static auto a_static = new MyType();

  MyType *b = new MyDerivedType();
  // CHECK: MyType *b = new MyDerivedType();

  void *c = new MyType();
  // CHECK: void *c = new MyType();

  // CV-qualifier tests.
  //
  // NOTE : the form "type const" is expected here because of a deficiency in
  // TypeLoc where CV qualifiers are not considered part of the type location
  // info. That is, all that is being replaced in each case is "MyType *" and
  // not "MyType * const".
  {
    static MyType * const d_static = new MyType();
    // CHECK: static auto const d_static = new MyType();

    MyType * const d3 = new MyType();
    // CHECK: auto const d3 = new MyType();

    MyType * volatile d4 = new MyType();
    // CHECK: auto volatile d4 = new MyType();
  }

  int (**func)(int, int) = new (int(*[5])(int,int));
  // CHECK: int (**func)(int, int) = new (int(*[5])(int,int));

  int *e = new int[5];
  // CHECK: auto e = new int[5];

  MyType *f(new MyType);
  // CHECK: auto f(new MyType);

  MyType *g{new MyType};
  // CHECK: MyType *g{new MyType};

  // Test for declaration lists.
  {
    MyType *a = new MyType(), *b = new MyType(), *c = new MyType();
    // CHECK: auto a = new MyType(), b = new MyType(), c = new MyType();

    // Non-initialized declaration should not be transformed.
    MyType *d = new MyType(), *e;
    // CHECK: MyType *d = new MyType(), *e;

    MyType **f = new MyType*(), **g = new MyType*();
    // CHECK: auto f = new MyType*(), g = new MyType*();

    // Mismatching types in declaration lists should not be transformed.
    MyType *h = new MyType(), **i = new MyType*();
    // CHECK: MyType *h = new MyType(), **i = new MyType*();

    // '*' shouldn't be removed in case of mismatching types with multiple
    // declarations.
    MyType *j = new MyType(), *k = new MyType(), **l = new MyType*();
    // CHECK: MyType *j = new MyType(), *k = new MyType(), **l = new MyType*();
  }

  // Test for typedefs.
  {
    typedef int * int_p;

    int_p a = new int;
    // CHECK: auto a = new int;
    int_p *b = new int*;
    // CHECK: auto b = new int*;

    // Test for typedefs in declarations lists.
    int_p c = new int, d = new int;
    // CHECK: auto c = new int, d = new int;

    // Different types should not be transformed.
    int_p e = new int, *f = new int_p;
    // CHECK: int_p e = new int, *f = new int_p;

    int_p *g = new int*, *h = new int_p;
    // CHECK: auto g = new int*, h = new int_p;
  }

  return 0;
}
