// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -g %s -o - -fstandalone-debug | FileCheck %s
namespace __pointer_type_imp
{
  template <class _Tp, class _Dp, bool > struct __pointer_type1 {};

  // CHECK: metadata ![[PARAMS:[0-9]+]], metadata !"_ZTSN18__pointer_type_imp15__pointer_type1I1C14default_deleteIS1_ELb0EEE"} ; [ DW_TAG_structure_type ] [__pointer_type1<C, default_delete<C>, false>] [line [[@LINE+1]], size 8, align 8, offset 0] [def] [from ]
  template <class _Tp, class _Dp> struct __pointer_type1<_Tp, _Dp, false>
  {
    typedef _Tp* type;
  };
}
template <class _Tp, class _Dp>
struct __pointer_type2
{
  // Test that the bool template type parameter is emitted.
  //
  // CHECK: ![[PARAMS]] = metadata !{metadata !{{.*}}, metadata !{{.*}}, metadata ![[FALSE:[0-9]+]]}
  // CHECK: ![[FALSE]] = {{.*}} i8 0, {{.*}}} ; [ DW_TAG_template_value_parameter ]
  typedef typename __pointer_type_imp::__pointer_type1<_Tp, _Dp, false>::type type;
};
template <class _Tp> struct default_delete {};
template <class _Tp, class _Dp = default_delete<_Tp> > class unique_ptr
{
  typedef typename __pointer_type2<_Tp, _Dp>::type pointer;
  unique_ptr(pointer __p, _Dp __d) {}
};
class C {
  unique_ptr<C> Ptr;
};
void foo(C &c) {
}
