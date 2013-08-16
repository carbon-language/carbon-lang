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

B::~B() {
}

struct C {
  static int s;
  virtual ~C();
};

C::~C() {
}

struct D {
  D();
  virtual ~D();
  void func() {
  }
};

struct E {
  E();
  virtual ~E();
  virtual void func() {
  }
};

struct A {
  int one;
  static const int HdrSize = 52;
  int two;
  A() {
    int x = 1;
  }
};

void f1() {
  D x;
  x.func();
  E y;
}

int main(int argc, char **argv) {
  B b;
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
// CHECK: DW_TAG_class_type ] [B]
// CHECK: metadata !"_vptr$B", {{.*}}, i32 64, metadata !{{.*}}} ; [ DW_TAG_member ]

// CHECK: [[C:![0-9]*]] = {{.*}} metadata [[C_MEM:![0-9]*]], i32 0, metadata [[C]], null} ; [ DW_TAG_structure_type ] [C] {{.*}} [def]
// CHECK: [[C_MEM]] = metadata !{metadata [[C_VPTR:![0-9]*]], metadata [[C_S:![0-9]*]], metadata [[C_DTOR:![0-9]*]]}
// CHECK: [[C_VPTR]] = {{.*}} ; [ DW_TAG_member ] [_vptr$C] {{.*}} [artificial]
// CHECK: [[C_S]] = {{.*}} ; [ DW_TAG_member ] [s] {{.*}} [static] [from int]
// CHECK: [[C_DTOR]] = {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [~C]

// CHECK: ; [ DW_TAG_structure_type ] [A]
// CHECK: HdrSize
// CHECK: metadata [[D_MEM:![0-9]*]], i32 0, null} ; [ DW_TAG_structure_type ] [D] {{.*}} [decl]
// CHECK: [[D_MEM]] = metadata !{metadata [[D_FUNC:![0-9]*]]}
// CHECK: [[D_FUNC]] = {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [func]
// CHECK: null, i32 0, null} ; [ DW_TAG_structure_type ] [E] {{.*}} [decl]
// CHECK: ![[EXCEPTLOC]] = metadata !{i32 62,
// CHECK: ![[RETLOC]] = metadata !{i32 61,
