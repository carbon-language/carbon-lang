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

// RUN: c-index-test -index-file %s > %t
// RUN: FileCheck %s -input-file=%t

// CHECK: [indexDeclaration]: kind: type-alias | name: MyTypeAlias | {{.*}} | loc: 1:7
// CHECK: [indexDeclaration]: kind: struct-template-spec | name: TS | {{.*}} | loc: 11:8
// CHECK: [indexDeclaration]: kind: function-template-spec | name: tfoo | {{.*}} | loc: 15:6
