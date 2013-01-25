// RUN: %clangxx -target x86_64-unknown-unknown -g -O0 %s -emit-llvm -S -o - | FileCheck %s
// PR14471


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

// CHECK: metadata !"a", {{.*}} @_ZN1C1aE, metadata ![[DECL_A:[0-9]+]]} ; [ DW_TAG_variable ] [a] {{.*}} [def]
// CHECK: ![[DECL_A]] = metadata {{.*}} [ DW_TAG_member ] [a] [line {{.*}}, size 0, align 0, offset 0] [private] [static]
// CHECK: metadata !"const_a", {{.*}}, i1 true} ; [ DW_TAG_member ] [const_a] [line {{.*}}, size 0, align 0, offset 0] [private] [static]
// CHECK: ![[DECL_B:[0-9]+]] {{.*}} metadata !"b", {{.*}} [ DW_TAG_member ] [b] [line {{.*}}, size 0, align 0, offset 0] [protected] [static]
// CHECK: metadata !"const_b", {{.*}}, float 0x{{.*}}} ; [ DW_TAG_member ] [const_b] [line {{.*}}, size 0, align 0, offset 0] [protected] [static]
// CHECK: ![[DECL_C:[0-9]+]] {{.*}} metadata !"c", {{.*}} [ DW_TAG_member ] [c] [line {{.*}}, size 0, align 0, offset 0] [static]
// CHECK: metadata !"const_c", {{.*}} [ DW_TAG_member ] [const_c] [line {{.*}}, size 0, align 0, offset 0] [static]
// CHECK: metadata !"b", {{.*}} @_ZN1C1bE, metadata ![[DECL_B]]} ; [ DW_TAG_variable ] [b] {{.*}} [def]
// CHECK: metadata !"c", {{.*}} @_ZN1C1cE, metadata ![[DECL_C]]} ; [ DW_TAG_variable ] [c] {{.*}} [def]
