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

struct C {
  virtual void func();
  struct inner {
    int j;
  };
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
  C::inner c_i;
  if (argc) {
    A a;
  }
  return 0;
}

// RUN: %clang -target x86_64-unknown_unknown -emit-llvm -g -S %s -o - | FileCheck %s
// RUN: %clang -target i686-cygwin -emit-llvm -g -S %s -o - | FileCheck %s
// RUN: %clang -target armv7l-unknown-linux-gnueabihf -emit-llvm -g -S %s -o - | FileCheck %s

// CHECK: invoke {{.+}} @_ZN1BD1Ev(%class.B* %b)
// CHECK-NEXT: unwind label %{{.+}}, !dbg ![[EXCEPTLOC:.*]]
// CHECK: store i32 0, i32* %{{.+}}, !dbg ![[RETLOC:.*]]
// CHECK: DW_TAG_structure_type ] [foo]
// CHECK: DW_TAG_class_type ] [bar]
// CHECK: DW_TAG_union_type ] [baz]
// CHECK: DW_TAG_structure_type ] [A]
// CHECK: HdrSize
// CHECK: DW_TAG_class_type ] [B]
// CHECK: metadata !"_vptr$B", {{.*}}, i32 64, metadata !{{.*}}} ; [ DW_TAG_member ]
// CHECK: metadata [[C_INNER_MEM:![0-9]*]], i32 0, null, null} ; [ DW_TAG_structure_type ] [inner] {{.*}} [def]
// Context chains (in Clang -flimit-debug-info and in GCC generally) contain
// definitions without members (& without a vbase 'containing type'):
// CHECK: null, i32 0, null, null} ; [ DW_TAG_structure_type ] [C] {{.*}} [def]
// CHECK: [[C_INNER_MEM]] = metadata !{metadata [[C_INNER_I:![0-9]*]]}
// CHECK: [[C_INNER_I]] = {{.*}} ; [ DW_TAG_member ] [j] {{.*}} [from int]
// CHECK: ![[EXCEPTLOC]] = metadata !{i32 40,
// CHECK: ![[RETLOC]] = metadata !{i32 39,
