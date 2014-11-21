// RUN: %clangxx -target x86_64-unknown-unknown -g %s -emit-llvm -S -o - | FileCheck %s
// PR14471

enum X {
  Y
};
class C
{
  static int a;
  const static bool const_a = true;
protected:
  static int b;
  const static float const_b = 3.14;
public:
  static int c;
  const static int const_c = 18;
  int d;
  static X x_a;
};

int C::a = 4;
int C::b = 2;
int C::c = 1;

int main()
{
        C instance_C;
        instance_C.d = 8;
        return C::c;
}

// The definition of C::a drives the emission of class C, which is
// why the definition of "a" comes before the declarations while
// "b" and "c" come after.

// CHECK: metadata !"_ZTS1X"} ; [ DW_TAG_enumeration_type ] [X]
// CHECK: metadata !"_ZTS1C"} ; [ DW_TAG_class_type ] [C]
// CHECK: ![[DECL_A:[0-9]+]] = metadata {{.*}} [ DW_TAG_member ] [a] [line {{.*}}, size 0, align 0, offset 0] [static]
// CHECK: metadata !"0xd\00const_a\00{{.*}}", {{.*}}, i1 true} ; [ DW_TAG_member ] [const_a] [line {{.*}}, size 0, align 0, offset 0] [static]
// CHECK: ![[DECL_B:[0-9]+]] = metadata !{metadata !"0xd\00b\00{{.*}}", {{.*}} [ DW_TAG_member ] [b] [line {{.*}}, size 0, align 0, offset 0] [protected] [static]
// CHECK: metadata !"0xd\00const_b\00{{.*}}", {{.*}}, float 0x{{.*}}} ; [ DW_TAG_member ] [const_b] [line {{.*}}, size 0, align 0, offset 0] [protected] [static]
// CHECK: ![[DECL_C:[0-9]+]] = metadata !{metadata !"0xd\00c\00{{.*}}", {{.*}} [ DW_TAG_member ] [c] [line {{.*}}, size 0, align 0, offset 0] [public] [static]
// CHECK: metadata !"0xd\00const_c\00{{.*}}", {{.*}} [ DW_TAG_member ] [const_c] [line {{.*}}, size 0, align 0, offset 0] [public] [static]
// CHECK: metadata !"0xd\00x_a\00{{.*}}", {{.*}} [ DW_TAG_member ] [x_a] {{.*}} [public] [static]

// CHECK: ; [ DW_TAG_structure_type ] [static_decl_templ<int>] {{.*}} [def]
// CHECK: ; [ DW_TAG_member ] [static_decl_templ_var]

// CHECK: [[NS_X:![0-9]+]] = {{.*}} ; [ DW_TAG_namespace ] [x]

// Test this in an anonymous namespace to ensure the type is retained even when
// it doesn't get automatically retained by the string type reference machinery.
namespace {
struct anon_static_decl_struct {
  static const int anon_static_decl_var = 117;
};
}


// CHECK: ; [ DW_TAG_structure_type ] [anon_static_decl_struct] {{.*}} [def]
// CHECK: ; [ DW_TAG_member ] [anon_static_decl_var]

int ref() {
  return anon_static_decl_struct::anon_static_decl_var;
}

template<typename T>
struct static_decl_templ {
  static const int static_decl_templ_var = 7;
};

template<typename T>
const int static_decl_templ<T>::static_decl_templ_var;

int static_decl_templ_ref() {
  return static_decl_templ<int>::static_decl_templ_var;
}

// CHECK: metadata !{metadata !"0x34\00a\00{{.*}}", null, {{.*}} @_ZN1C1aE, metadata ![[DECL_A]]} ; [ DW_TAG_variable ] [a] {{.*}} [def]
// CHECK: metadata !{metadata !"0x34\00b\00{{.*}}", null, {{.*}} @_ZN1C1bE, metadata ![[DECL_B]]} ; [ DW_TAG_variable ] [b] {{.*}} [def]
// CHECK: metadata !{metadata !"0x34\00c\00{{.*}}", null, {{.*}} @_ZN1C1cE, metadata ![[DECL_C]]} ; [ DW_TAG_variable ] [c] {{.*}} [def]

// CHECK-NOT: ; [ DW_TAG_variable ] [anon_static_decl_var]

// Verify that even when a static member declaration is created lazily when
// creating the definition, the declaration line is that of the canonical
// declaration, not the definition. Also, since we look at the canonical
// definition, we should also correctly emit the constant value (42) into the
// debug info.
struct V {
  virtual ~V(); // cause the definition of 'V' to be omitted by no-standalone-debug optimization
  static const int const_va = 42;
};
// CHECK: i32 42} ; [ DW_TAG_member ] [const_va] [line [[@LINE-2]],
const int V::const_va;

namespace x {
struct y {
  static int z;
};
int y::z;
}

// CHECK: metadata !{metadata !"0x34\00z\00{{.*}}", metadata [[NS_X]], {{.*}} ; [ DW_TAG_variable ] [z] {{.*}} [def]
