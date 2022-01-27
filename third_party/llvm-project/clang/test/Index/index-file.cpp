using MyTypeAlias = int;

extern "C" {
  template < typename T > *Allocate() { }
}

// rdar://14063074
namespace rdar14063074 {
template <typename T>
struct TS {};
struct TS<int> {};

template <typename T>
void tfoo() {}
void tfoo<int>() {}
}

namespace crash1 {
template<typename T> class A {
  A(A &) = delete;
  void meth();
};
template <> void A<int>::meth();
template class A<int>;
}

class B {
  mutable int x_;
  int y_;

  B() = default;
  B(int);
  explicit B(double);
  B(const B&);
  B(B&&);
};

class C {
  explicit C(const C&);
};

// RUN: c-index-test -index-file %s > %t
// RUN: FileCheck %s -input-file=%t

// CHECK: [indexDeclaration]: kind: type-alias | name: MyTypeAlias | {{.*}} | loc: 1:7
// CHECK: [indexDeclaration]: kind: struct-template-spec | name: TS | {{.*}} | loc: 11:8
// CHECK: [indexDeclaration]: kind: function-template-spec | name: tfoo | {{.*}} | loc: 15:6
// CHECK: [indexDeclaration]: kind: c++-instance-method | name: meth | {{.*}} | loc: 23:26
// CHECK: [indexDeclaration]: kind: field | name: x_ | USR: c:@S@B@FI@x_ | lang: C++ | cursor: FieldDecl=x_:28:15 (Definition) (mutable) | loc: 28:15 | semantic-container: [B:27:7] | lexical-container: [B:27:7] | isRedecl: 0 | isDef: 1 | isContainer: 0 | isImplicit: 0
// CHECK: [indexDeclaration]: kind: field | name: y_ | USR: c:@S@B@FI@y_ | lang: C++ | cursor: FieldDecl=y_:29:7 (Definition) | loc: 29:7 | semantic-container: [B:27:7] | lexical-container: [B:27:7] | isRedecl: 0 | isDef: 1 | isContainer: 0 | isImplicit: 0
// CHECK: [indexDeclaration]: kind: constructor | name: B | {{.*}} (default constructor) (defaulted) | loc: 31:3
// CHECK: [indexDeclaration]: kind: constructor | name: B | {{.*}} (converting constructor) | loc: 32:3
// CHECK: [indexDeclaration]: kind: constructor | name: B | {{.*}} | loc: 33:12
// CHECK: [indexDeclaration]: kind: constructor | name: B | {{.*}} (copy constructor) (converting constructor) | loc: 34:3
// CHECK: [indexDeclaration]: kind: constructor | name: B | {{.*}} (move constructor) (converting constructor) | loc: 35:3
// CHECK: [indexDeclaration]: kind: constructor | name: C | {{.*}} (copy constructor) | loc: 39:12
