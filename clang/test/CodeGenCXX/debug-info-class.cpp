// RUN: %clang  -emit-llvm -g -S %s -o - | FileCheck %s
struct foo;
void func(foo *f) { // CHECK: DW_TAG_structure_type
}
class bar;
void func(bar *f) { // CHECK: DW_TAG_class_type
}
union baz;
void func(baz *f) { // CHECK: DW_TAG_union_type
}
class B { // CHECK: DW_TAG_class_type
public:
  virtual ~B();
// CHECK: metadata !"_vptr$B", {{.*}}, i32 64, metadata !{{.*}}} ; [ DW_TAG_member ]
};
struct A { // CHECK: DW_TAG_structure_type
  int one;
  static const int HdrSize = 52; // CHECK: HdrSize
  int two;
  A() {
    int x = 1;
  }
};


int main() {
  A a;
  B b;
}
