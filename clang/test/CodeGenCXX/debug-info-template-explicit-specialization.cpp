// RUN: %clang_cc1 -S -emit-llvm -g %s -o - | FileCheck %s

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
// CHECK: ; [ DW_TAG_structure_type ] [e<int>] {{.*}} [decl]

template <typename T>
struct f {
  void g();
};
extern template class f<int>;
template <typename T>
void f<T>::g() {
}
f<int> fi;
// Is this right? We don't seem to emit a def for 'f<int>::g' (even if it is
// called in this translation unit) so I guess if we're relying on its
// definition to be wherever the explicit instantiation definition is, we can do
// the same for the debug info.
// CHECK: ; [ DW_TAG_structure_type ] [f<int>] {{.*}} [decl]

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
