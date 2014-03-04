// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -g %s -o - -fno-standalone-debug | FileCheck %s

// Run again with -gline-tables-only and verify we don't crash.  We won't output
// type info at all.
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -g %s -o - -gline-tables-only | FileCheck %s -check-prefix LINES-ONLY

// LINES-ONLY-NOT: DW_TAG_structure_type

template <typename T>
struct a {
};
extern template class a<int>;
// CHECK-NOT: ; [ DW_TAG_structure_type ] [a<int>]

template <typename T>
struct b {
};
extern template class b<int>;
b<int> bi;
// CHECK: ; [ DW_TAG_structure_type ] [b<int>] {{.*}} [def]

template <typename T>
struct c {
  void f() {}
};
extern template class c<int>;
c<int> ci;
// CHECK: ; [ DW_TAG_structure_type ] [c<int>] {{.*}} [decl]

template <typename T>
struct d {
  void f();
};
extern template class d<int>;
d<int> di;
// CHECK: ; [ DW_TAG_structure_type ] [d<int>] {{.*}} [def]

template <typename T>
struct e {
  void f();
};
template <typename T>
void e<T>::f() {
}
extern template class e<int>;
e<int> ei;
// There's no guarantee that the out of line definition will appear before the
// explicit template instantiation definition, so conservatively emit the type
// definition here.
// CHECK: ; [ DW_TAG_structure_type ] [e<int>] {{.*}} [def]

template <typename T>
struct f {
  void g();
};
extern template class f<int>;
template <typename T>
void f<T>::g() {
}
f<int> fi;
// CHECK: ; [ DW_TAG_structure_type ] [f<int>] {{.*}} [def]

template <typename T>
struct g {
  void f();
};
template <>
void g<int>::f();
extern template class g<int>;
g<int> gi;
// CHECK: ; [ DW_TAG_structure_type ] [g<int>] {{.*}} [def]

template <typename T>
struct h {
};
template class h<int>;
// CHECK: ; [ DW_TAG_structure_type ] [h<int>] {{.*}} [def]

template <typename T>
struct i {
  void f() {}
};
template<> void i<int>::f();
extern template class i<int>;
i<int> ii;
// CHECK: ; [ DW_TAG_structure_type ] [i<int>] {{.*}} [def]
