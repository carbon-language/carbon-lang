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

struct F {
  struct inner {
  };
  static const int i = 2;
  virtual ~F();
};

struct G {
  virtual void func();
  struct inner {
    int j;
  };
};

struct H {};
struct I : virtual H {};
struct J : I {};
J j;

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
  int i = F::i;
  F::inner z;
}

int main(int argc, char **argv) {
  B b;
  G::inner c_i;
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

// CHECK: ; [ DW_TAG_structure_type ] [I] {{.*}} [def]

// CHECK: [[F_I_DEF:![0-9]*]] = {{.*}}, metadata [[F_I:![0-9]*]]} ; [ DW_TAG_variable ] [i]
// CHECK: [[F_I]] = {{.*}} ; [ DW_TAG_member ] [i]
// CHECK: [[F:![0-9]*]] = {{.*}} metadata [[F_MEM:![0-9]*]], i32 0, null, null} ; [ DW_TAG_structure_type ] [F] {{.*}} [def]
// CHECK: [[F_MEM]] = metadata !{metadata [[F_I]]}

// CHECK: null, i32 0, null} ; [ DW_TAG_structure_type ] [E] {{.*}} [decl]

// CHECK: metadata [[G_INNER_MEM:![0-9]*]], i32 0, null, null} ; [ DW_TAG_structure_type ] [inner] [line 50, {{.*}} [def]
// Context chains (in Clang -flimit-debug-info and in GCC generally) contain
// definitions without members (& without a vbase 'containing type'):
// CHECK: null, i32 0, null, null} ; [ DW_TAG_structure_type ] [G] {{.*}} [def]
// CHECK: [[G_INNER_MEM]] = metadata !{metadata [[G_INNER_I:![0-9]*]]}
// CHECK: [[G_INNER_I]] = {{.*}} ; [ DW_TAG_member ] [j] {{.*}} [from int]

// CHECK: ![[EXCEPTLOC]] = metadata !{i32 84,
// CHECK: ![[RETLOC]] = metadata !{i32 83,
