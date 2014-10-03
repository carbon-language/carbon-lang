// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -emit-llvm -g %s -o - | FileCheck %s --check-prefix=MSVC

template<typename T> struct Identity {
  typedef T Type;
};

void f(Identity<int>::Type a) {}
void f(Identity<int> a) {}
void f(int& a) { }

template<typename T> struct A {
  A<T> *next;
};
void f(A<int>) { }

struct B { };

void f() {
  int B::*a = 0;
  void (B::*b)() = 0;
}

namespace EmptyNameCrash {
  struct A { A(); };
  typedef struct { A x; } B;
  B x;
}

// PR4890
namespace PR4890 {
  struct X {
    ~X();
  };

  X::~X() { }
}

namespace VirtualDtor {
  struct Y {
    virtual ~Y();
  };
  
  Y::~Y() { }
}

namespace VirtualBase {
  struct A { int a; };
  struct B : virtual A { int b; };

  void f() {
    B b;
  }
}

// MSVC: [[VBASE_B:![0-9]+]] = metadata !{metadata !"0x13\00B\00{{[0-9]+}}\0096\0032\000\000\000", {{.*}}, null, metadata [[VBASE_B_DEF:![0-9]+]], {{.*}}} ; [ DW_TAG_structure_type ] [B] [line 49, size 96, align 32, offset 0] [def] [from ]
// MSVC: [[VBASE_B_DEF]] = metadata !{metadata [[VBASE_A_IN_B:![0-9]+]],
//
// Look for the vbtable offset of A, which should be 4.
// MSVC: [[VBASE_A_IN_B]] = metadata !{metadata !"0x1c\00\000\000\000\004\0032", null, metadata [[VBASE_B]], metadata !{{[0-9]*}}} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 4] [from A]

// CHECK: metadata !{metadata !"0x13\00B\00{{[0-9]+}}\00128\0064\000\000\000", {{.*}}, null, metadata [[VBASE_B_DEF:![0-9]+]], {{.*}}} ; [ DW_TAG_structure_type ] [B] [line 49, size 128, align 64, offset 0] [def] [from ]
// CHECK: [[VBASE_B_DEF]] = metadata !{metadata [[VBASE_A_IN_B:![0-9]+]],
//
// Look for the vtable offset offset, which should be -24.
// CHECK: [[VBASE_A_IN_B]] = metadata !{metadata !"0x1c\00\000\000\000\0024\0032", null, metadata !"_ZTSN11VirtualBase1BE", metadata !"_ZTSN11VirtualBase1AE"} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 24] [from _ZTSN11VirtualBase1AE]
namespace b5249287 {
template <typename T> class A {
  struct B;
};

class Cls {
  template <typename T> friend class A<T>::B;
};

Cls obj;
}

namespace pr14763 {
struct foo {
  foo(const foo&);
};

foo func(foo f) {
  return f; // reference 'f' for now because otherwise we hit another bug
}

// CHECK: metadata !{metadata !"0x13\00{{.*}}", metadata !{{[0-9]*}}, metadata [[PR14763:![0-9]*]], {{.*}}, metadata !"[[FOO:.*]]"} ; [ DW_TAG_structure_type ] [foo]
// CHECK: [[PR14763]] = {{.*}} ; [ DW_TAG_namespace ] [pr14763]
// CHECK: [[INCTYPE:![0-9]*]] = {{.*}} ; [ DW_TAG_structure_type ] [incomplete]{{.*}} [decl]
// CHECK: metadata [[A_MEM:![0-9]*]], null, null, metadata !"_ZTSN7pr162141aE"} ; [ DW_TAG_structure_type ] [a]
// CHECK: [[A_MEM]] = metadata !{metadata [[A_I:![0-9]*]]}
// CHECK: [[A_I]] = {{.*}} ; [ DW_TAG_member ] [i] {{.*}} [from int]
// CHECK: ; [ DW_TAG_structure_type ] [b] {{.*}}[decl]

// CHECK: [[FUNC:![0-9]*]] = metadata !{metadata !"0x2e\00func\00func\00_ZN7pr147634funcENS_3fooE\00{{.*}}"{{, [^,]+, [^,]+}}, metadata [[FUNC_TYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
}

void foo() {
  const wchar_t c = L'x';
  wchar_t d = c;
}

// CHECK-NOT: ; [ DW_TAG_variable ] [c]

namespace pr9608 { // also pr9600
struct incomplete;
incomplete (*x)[3];
// CHECK: metadata [[INCARRAYPTR:![0-9]*]], [3 x i8]** @_ZN6pr96081xE, null} ; [ DW_TAG_variable ] [x]
// CHECK: [[INCARRAYPTR]] = {{.*}}metadata [[INCARRAY:![0-9]*]]} ; [ DW_TAG_pointer_type ]
// CHECK: [[INCARRAY]] = metadata !{metadata !"0x1\00\000\000\000\000\000\000", null, null, metadata !"_ZTSN6pr960810incompleteE", metadata {{![0-9]+}}, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 0, offset 0] [from _ZTSN6pr960810incompleteE]
}

// For some reason function arguments ended up down here
// CHECK: = metadata !{metadata !"0x101\00f\00{{.*}}\008192", metadata [[FUNC]], metadata {{![0-9]+}}, metadata !"[[FOO]]"} ; [ DW_TAG_arg_variable ] [f]

// CHECK: ; [ DW_TAG_auto_variable ] [c]

namespace pr16214 {
struct a {
  int i;
};

typedef a at;

struct b {
};

typedef b bt;

void func() {
  at a_inst;
  bt *b_ptr_inst;
  const bt *b_cnst_ptr_inst;
}

}
