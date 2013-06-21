// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s
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
  struct A { };
  struct B : virtual A { };

  void f() {
    B b;
  }
}

void foo() {
  const wchar_t c = L'x';
  wchar_t d = c;
}

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

// CHECK: [[FUNC:![0-9]*]] = {{.*}} metadata !"_ZN7pr147634funcENS_3fooE", i32 {{[0-9]*}}, metadata [[FUNC_TYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
// CHECK: [[PR14763:![0-9]*]] = {{.*}} ; [ DW_TAG_namespace ] [pr14763]
// CHECK: [[FOO:![0-9]*]] = metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata [[PR14763]], {{.*}} ; [ DW_TAG_structure_type ] [foo]
}

namespace pr9608 { // also pr9600
struct incomplete;
incomplete (*x)[3];
// CHECK: metadata [[INCARRAYPTR:![0-9]*]], i32 0, i32 1, [3 x i8]** @_ZN6pr96081xE, null} ; [ DW_TAG_variable ] [x]
// CHECK: [[INCARRAYPTR]] = {{.*}}metadata [[INCARRAY:![0-9]*]]} ; [ DW_TAG_pointer_type ]
// CHECK: [[INCARRAY]] = {{.*}}metadata [[INCTYPE:![0-9]*]], metadata {{![0-9]*}}, i32 0, i32 0} ; [ DW_TAG_array_type ] [line 0, size 0, align 0, offset 0] [from incomplete]
// CHECK: [[INCTYPE]] = {{.*}} ; [ DW_TAG_structure_type ] [incomplete]{{.*}} [decl]
}

// For some reason the argument for PR14763 ended up all the way down here
// CHECK: = metadata !{i32 {{[0-9]*}}, metadata [[FUNC]], {{.*}}, metadata [[FOO]], i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [f]

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

// CHECK: metadata [[A_MEM:![0-9]*]], i32 0, null, null} ; [ DW_TAG_structure_type ] [a]
// CHECK: [[A_MEM]] = metadata !{metadata [[A_I:![0-9]*]], metadata !{{[0-9]*}}}
// CHECK: [[A_I]] = {{.*}} ; [ DW_TAG_member ] [i] {{.*}} [from int]
// CHECK: ; [ DW_TAG_structure_type ] [b] {{.*}}[decl]
}
