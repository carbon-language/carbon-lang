// RUN: %clang  -emit-llvm -g -S %s -o - | FileCheck %s
struct foo;
void func(foo *f) {
}
class bar;
void func(bar *f) {
}
union baz;
void func(baz *f) {
}
class B {
public:
  virtual ~B();
};
struct A {
  int one;
  static const int HdrSize = 52;
  int two;
  A() {
    int x = 1;
  }
};


int main(int argc, char **argv) {
  B b;
  if (argc) {
    A a;
  }
  return 0;
}

// CHECK: unwind label %terminate.lpad, !dbg ![[EXCEPTLOC:.*]]
// CHECK: store i32 0, i32* %retval, !dbg ![[RETLOC:.*]]
// CHECK: DW_TAG_structure_type ] [foo]
// CHECK: DW_TAG_class_type ] [bar]
// CHECK: DW_TAG_union_type ] [baz]
// CHECK: DW_TAG_structure_type ] [A]
// CHECK: HdrSize
// CHECK: DW_TAG_class_type ] [B]
// CHECK: metadata !"_vptr$B", {{.*}}, i32 64, metadata !{{.*}}} ; [ DW_TAG_member ]
// CHECK: ![[EXCEPTLOC]] = metadata !{i32 31,
// CHECK: ![[RETLOC]] = metadata !{i32 30,
