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

// MSVC: [[VBASE_B:![0-9]+]] = metadata !{i32 {{.*}}, metadata !"B", i32 {{[0-9]*}}, i64 96, i64 32, i32 0, i32 0, null, metadata [[VBASE_B_DEF:![0-9]+]], i32 0, {{.*}}} ; [ DW_TAG_structure_type ] [B] [line 49, size 96, align 32, offset 0] [def] [from ]
// MSVC: [[VBASE_B_DEF]] = metadata !{metadata [[VBASE_A_IN_B:![0-9]+]],
//
// Look for the vbtable offset of A, which should be 4.
// MSVC: [[VBASE_A_IN_B]] = metadata !{i32 786460, null, metadata [[VBASE_B]], null, i32 0, i64 0, i64 0, i64 4, i32 32, metadata !{{[0-9]*}}} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 4] [from A]

// CHECK: metadata !{{{.*}}, metadata !"B", i32 {{[0-9]*}}, i64 128, i64 64, i32 0, i32 0, null, metadata [[VBASE_B_DEF:![0-9]+]], i32 0, {{.*}}} ; [ DW_TAG_structure_type ] [B] [line 49, size 128, align 64, offset 0] [def] [from ]
// CHECK: [[VBASE_B_DEF]] = metadata !{metadata [[VBASE_A_IN_B:![0-9]+]],
//
// Look for the vtable offset offset, which should be -24.
// CHECK: [[VBASE_A_IN_B]] = metadata !{i32 786460, null, metadata !"_ZTSN11VirtualBase1BE", null, i32 0, i64 0, i64 0, i64 24, i32 32, metadata !"_ZTSN11VirtualBase1AE"} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 24] [from _ZTSN11VirtualBase1AE]
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

// CHECK: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata [[PR14763:![0-9]*]], {{.*}}, metadata !"[[FOO:.*]]"} ; [ DW_TAG_structure_type ] [foo]
// CHECK: [[PR14763]] = {{.*}} ; [ DW_TAG_namespace ] [pr14763]
// CHECK: [[INCTYPE:![0-9]*]] = {{.*}} ; [ DW_TAG_structure_type ] [incomplete]{{.*}} [decl]
// CHECK: metadata [[A_MEM:![0-9]*]], i32 0, null, null, metadata !"_ZTSN7pr162141aE"} ; [ DW_TAG_structure_type ] [a]
// CHECK: [[A_MEM]] = metadata !{metadata [[A_I:![0-9]*]]}
// CHECK: [[A_I]] = {{.*}} ; [ DW_TAG_member ] [i] {{.*}} [from int]
// CHECK: ; [ DW_TAG_structure_type ] [b] {{.*}}[decl]

// CHECK: [[FUNC:![0-9]*]] = {{.*}} metadata !"_ZN7pr147634funcENS_3fooE", i32 {{[0-9]*}}, metadata [[FUNC_TYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
}

void foo() {
  const wchar_t c = L'x';
  wchar_t d = c;
}

// CHECK-NOT: ; [ DW_TAG_variable ] [c]

namespace pr9608 { // also pr9600
struct incomplete;
incomplete (*x)[3];
// CHECK: metadata [[INCARRAYPTR:![0-9]*]], i32 0, i32 1, [3 x i8]** @_ZN6pr96081xE, null} ; [ DW_TAG_variable ] [x]
// CHECK: [[INCARRAYPTR]] = {{.*}}metadata [[INCARRAY:![0-9]*]]} ; [ DW_TAG_pointer_type ]
// CHECK: [[INCARRAY]] = {{.*}}metadata !"_ZTSN6pr960810incompleteE", metadata {{![0-9]*}}, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 0, offset 0] [from _ZTSN6pr960810incompleteE]
}

// For some reason function arguments ended up down here
// CHECK: = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], {{.*}}, metadata !"[[FOO]]", i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [f]

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
