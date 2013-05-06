// RUN: %clang_cc1 -fsyntax-only -std=c++03 -verify -ast-dump %s > %t-03
// RUN: FileCheck --input-file=%t-03 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -ast-dump %s > %t-11
// RUN: FileCheck --input-file=%t-11 %s
// RUN: FileCheck --input-file=%t-11 %s --check-prefix=CHECK-CXX11

// http://llvm.org/PR7905
namespace PR7905 {
struct S; // expected-note {{forward declaration}}
void foo1() {
  (void)(S[]) {{3}}; // expected-error {{array has incomplete element type}}
}

template <typename T> struct M { T m; };
void foo2() {
  (void)(M<short> []) {{3}};
}
}

// Check compound literals mixed with C++11 list-initialization.
namespace brace_initializers {
  struct POD {
    int x, y;
  };
  struct HasCtor {
    HasCtor(int x, int y);
  };
  struct HasDtor {
    int x, y;
    ~HasDtor();
  };
  struct HasCtorDtor {
    HasCtorDtor(int x, int y);
    ~HasCtorDtor();
  };

  void test() {
    (void)(POD){1, 2};
    // CHECK-NOT: CXXBindTemporaryExpr {{.*}} 'struct brace_initializers::POD'
    // CHECK: CompoundLiteralExpr {{.*}} 'struct brace_initializers::POD'
    // CHECK-NEXT: InitListExpr {{.*}} 'struct brace_initializers::POD'
    // CHECK-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-NEXT: IntegerLiteral {{.*}} 2{{$}}

    (void)(HasDtor){1, 2};
    // CHECK: CXXBindTemporaryExpr {{.*}} 'struct brace_initializers::HasDtor'
    // CHECK-NEXT: CompoundLiteralExpr {{.*}} 'struct brace_initializers::HasDtor'
    // CHECK-NEXT: InitListExpr {{.*}} 'struct brace_initializers::HasDtor'
    // CHECK-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-NEXT: IntegerLiteral {{.*}} 2{{$}}

#if __cplusplus >= 201103L
    (void)(HasCtor){1, 2};
    // CHECK-CXX11-NOT: CXXBindTemporaryExpr {{.*}} 'struct brace_initializers::HasCtor'
    // CHECK-CXX11: CompoundLiteralExpr {{.*}} 'struct brace_initializers::HasCtor'
    // CHECK-CXX11-NEXT: CXXTemporaryObjectExpr {{.*}} 'struct brace_initializers::HasCtor'
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 2{{$}}

    (void)(HasCtorDtor){1, 2};
    // CHECK-CXX11: CXXBindTemporaryExpr {{.*}} 'struct brace_initializers::HasCtorDtor'
    // CHECK-CXX11-NEXT: CompoundLiteralExpr {{.*}} 'struct brace_initializers::HasCtorDtor'
    // CHECK-CXX11-NEXT: CXXTemporaryObjectExpr {{.*}} 'struct brace_initializers::HasCtorDtor'
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 2{{$}}
#endif
  }

  struct PrivateDtor {
    int x, y;
  private:
    ~PrivateDtor(); // expected-note {{declared private here}}
  };

  void testPrivateDtor() {
    (void)(PrivateDtor){1, 2}; // expected-error {{temporary of type 'brace_initializers::PrivateDtor' has private destructor}}
  }
}
