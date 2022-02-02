// Test without serialization:
// RUN: %clang_cc1 -fsyntax-only %s -ast-dump -std=c++17 | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

namespace PR46111 {
template <typename>
struct S;

template <typename T>
struct HasDeductionGuide {
  typedef PR46111::S<T> STy;
  HasDeductionGuide(typename STy::Child);
};

// This causes deduction guides to be generated for all constructors.
HasDeductionGuide()->HasDeductionGuide<int>;

template <typename T>
struct HasDeductionGuideTypeAlias {
  using STy = PR46111::S<T>;
  HasDeductionGuideTypeAlias(typename STy::Child);
};

// This causes deduction guides to be generated for all constructors.
HasDeductionGuideTypeAlias()->HasDeductionGuideTypeAlias<int>;

// The parameter to this one shouldn't be an elaborated type.
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuide> 'auto (typename STy::Child) -> HasDeductionGuide<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuide> 'auto (HasDeductionGuide<T>) -> HasDeductionGuide<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} <deduction guide for HasDeductionGuide> 'auto () -> HasDeductionGuide<int>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuideTypeAlias> 'auto (typename STy::Child) -> HasDeductionGuideTypeAlias<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuideTypeAlias> 'auto (HasDeductionGuideTypeAlias<T>) -> HasDeductionGuideTypeAlias<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} <deduction guide for HasDeductionGuideTypeAlias> 'auto () -> HasDeductionGuideTypeAlias<int>'
} // namespace PR46111


namespace PR48177 {
  template <class A> struct Base {
    using type_alias = A;
  };
  template<class T, int S, class A>
  struct Derived : Base<A> {
    using type_alias = typename Derived::type_alias;
    Derived(Derived &&, typename Derived::type_alias const&);
    Derived(T);
  };

  template<class T, class A>
  Derived(T, A) -> Derived<T, 1, A>;

  void init() {
    Derived d {1,2};
  }
} // namespace PR48177

// CHECK: CXXRecordDecl {{.*}} struct Derived
// CHECK: TypeAliasDecl {{.*}} type_alias 'typename Derived<T, S, A>::type_alias'
// CHECK-NEXT: DependentNameType {{.*}} 'typename Derived<T, S, A>::type_alias' dependent

// CHECK: CXXRecordDecl {{.*}} struct Derived
// CHECK: TypeAliasDecl {{.*}} type_alias 'typename Derived<int, 1, int>::type_alias':'int'
// CHECK-NEXT: ElaboratedType {{.*}} 'typename Derived<int, 1, int>::type_alias' sugar
// CHECK-NEXT: TypedefType {{.*}} 'PR48177::Base<int>::type_alias' sugar
// CHECK-NEXT: TypeAlias {{.*}} 'type_alias'
// CHECK-NEXT: SubstTemplateTypeParmType {{.*}} 'int' sugar
// CHECK-NEXT: TemplateTypeParmType {{.*}} 'A'
// CHECK-NEXT: TemplateTypeParm {{.*}} 'A'
// CHECK-NEXT: BuiltinType {{.*}} 'int'

// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for Derived> 'auto (Derived<T, S, A> &&, const typename Derived<T, S, A>::type_alias &) -> Derived<T, S, A>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for Derived> 'auto (T) -> Derived<T, S, A>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for Derived> 'auto (Derived<T, S, A>) -> Derived<T, S, A>'
// CHECK: CXXDeductionGuideDecl {{.*}} <deduction guide for Derived> 'auto (T, A) -> Derived<T, 1, A>'
// CHECK: CXXDeductionGuideDecl {{.*}} <deduction guide for Derived> 'auto (int, int) -> Derived<int, 1, int>'
