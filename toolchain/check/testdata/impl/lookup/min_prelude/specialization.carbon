// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/specialization.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/specialization.carbon

// --- specialized_self_first.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}
// CHECK:STDERR: specialized_self_first.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C as Z(C) where .X = C {}
// CHECK:STDERR:                      ^~
// CHECK:STDERR: specialized_self_first.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z(C) where .X = C {}

// CHECK:STDERR: specialized_self_first.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: specialized_self_first.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: specialized_self_first.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C.(Z(C).X) = {} as C;
  // CHECK:STDERR:             ^~~~~~
  // CHECK:STDERR: specialized_self_first.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C.(Z(C).X) = {} as C;
}

// --- specialized_self_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

// CHECK:STDERR: specialized_self_second.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: specialized_self_second.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) where .X = () {}

class C {}
// CHECK:STDERR: specialized_self_second.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C as Z(C) where .X = C {}
// CHECK:STDERR:                      ^~
// CHECK:STDERR: specialized_self_second.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z(C) where .X = C {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: specialized_self_second.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C.(Z(C).X) = {} as C;
  // CHECK:STDERR:             ^~~~~~
  // CHECK:STDERR: specialized_self_second.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C.(Z(C).X) = {} as C;
}

// --- specialized_constraint_first.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

// CHECK:STDERR: specialized_constraint_first.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C as Z(C) where .X = C {}
// CHECK:STDERR:                      ^~
// CHECK:STDERR: specialized_constraint_first.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z(C) where .X = C {}
// CHECK:STDERR: specialized_constraint_first.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C as Z(T) where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: specialized_constraint_first.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as Z(T) where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: specialized_constraint_first.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C.(Z(C).X) = {} as C;
  // CHECK:STDERR:             ^~~~~~
  // CHECK:STDERR: specialized_constraint_first.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C.(Z(C).X) = {} as C;
}

// --- specialized_constraint_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

// CHECK:STDERR: specialized_constraint_second.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C as Z(T) where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: specialized_constraint_second.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as Z(T) where .X = () {}
// CHECK:STDERR: specialized_constraint_second.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C as Z(C) where .X = C {}
// CHECK:STDERR:                      ^~
// CHECK:STDERR: specialized_constraint_second.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z(C) where .X = C {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: specialized_constraint_second.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C.(Z(C).X) = {} as C;
  // CHECK:STDERR:             ^~~~~~
  // CHECK:STDERR: specialized_constraint_second.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C.(Z(C).X) = {} as C;
}

// --- specialized_self_vs_constraint_self_first.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C(T:! type) {}

// CHECK:STDERR: specialized_self_vs_constraint_self_first.carbon:[[@LINE+7]]:44: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C(()) as Z(T) where .X = C(()) {}
// CHECK:STDERR:                                            ^~
// CHECK:STDERR: specialized_self_vs_constraint_self_first.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C(()) as Z(T) where .X = C(()) {}
// CHECK:STDERR: specialized_self_vs_constraint_self_first.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Z(C(()))` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C(T) as Z(C(())) where .X = () {}
// CHECK:STDERR:                                               ^~
// CHECK:STDERR: specialized_self_vs_constraint_self_first.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C(T) as Z(C(())) where .X = () {}

fn F() {
  // The specialization of `C(())` should match in preference to the blanket impl
  // of `C(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: specialized_self_vs_constraint_self_first.carbon:[[@LINE+7]]:17: error: member access into undefined facet type `Z(C(()))` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(()).(Z(C(())).X) = {} as C(());
  // CHECK:STDERR:                 ^~~~~~~~~~
  // CHECK:STDERR: specialized_self_vs_constraint_self_first.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(C(())).X) = {} as C(());
}

// --- specialized_self_vs_constraint_self_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C(T:! type) {}

// CHECK:STDERR: specialized_self_vs_constraint_self_second.carbon:[[@LINE+7]]:47: error: member access into undefined facet type `Z(C(()))` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C(T) as Z(C(())) where .X = () {}
// CHECK:STDERR:                                               ^~
// CHECK:STDERR: specialized_self_vs_constraint_self_second.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C(T) as Z(C(())) where .X = () {}
// CHECK:STDERR: specialized_self_vs_constraint_self_second.carbon:[[@LINE+7]]:44: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C(()) as Z(T) where .X = C(()) {}
// CHECK:STDERR:                                            ^~
// CHECK:STDERR: specialized_self_vs_constraint_self_second.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C(()) as Z(T) where .X = C(()) {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: specialized_self_vs_constraint_self_second.carbon:[[@LINE+7]]:17: error: member access into undefined facet type `Z(C(()))` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(()).(Z(C(())).X) = {} as C(());
  // CHECK:STDERR:                 ^~~~~~~~~~
  // CHECK:STDERR: specialized_self_vs_constraint_self_second.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(C(())).X) = {} as C(());
}

// --- generic_class_with_fully_specified_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

// CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) where .X = {.a: ()} {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) where .X = {.a: ()} {}

class C(T:! type) {}
// CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE+7]]:27: error: member access into undefined facet type `Z(())` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(()) as Z(()) where .X = C(()) {}
// CHECK:STDERR:                           ^~
// CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(()) as Z(()) where .X = C(()) {}

// CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE+7]]:43: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C(T) as Z(T) where .X = {.b: ()} {}
// CHECK:STDERR:                                           ^~
// CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE-26]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C(T) as Z(T) where .X = {.b: ()} {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket
  // impls of `Z(T)`. If a blanket impl is chosen, then `a` will have a struct
  // type which will fail to typecheck here when constructed from a `C` value.
  // CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE+7]]:17: error: member access into undefined facet type `Z(())` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(()).(Z(()).X) = {} as C(());
  // CHECK:STDERR:                 ^~~~~~~
  // CHECK:STDERR: generic_class_with_fully_specified_impl.carbon:[[@LINE-39]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(()).X) = {} as C(());
}

// --- generic_class_with_blanket_impl_first.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C(T:! type) {}

// CHECK:STDERR: generic_class_with_blanket_impl_first.carbon:[[@LINE+7]]:43: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C(T) as Z(T) where .X = C(()) {}
// CHECK:STDERR:                                           ^~
// CHECK:STDERR: generic_class_with_blanket_impl_first.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C(T) as Z(T) where .X = C(()) {}

// CHECK:STDERR: generic_class_with_blanket_impl_first.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: generic_class_with_blanket_impl_first.carbon:[[@LINE-18]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket
  // impls of `Z(T)`. If a blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here when constructed from a `C` value.
  // CHECK:STDERR: generic_class_with_blanket_impl_first.carbon:[[@LINE+7]]:17: error: member access into undefined facet type `Z(())` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(()).(Z(()).X) = {} as C(());
  // CHECK:STDERR:                 ^~~~~~~
  // CHECK:STDERR: generic_class_with_blanket_impl_first.carbon:[[@LINE-31]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(()).X) = {} as C(());
}

// --- generic_class_with_blanket_impl_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

// CHECK:STDERR: generic_class_with_blanket_impl_second.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: generic_class_with_blanket_impl_second.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) where .X = () {}

class C(T:! type) {}

// CHECK:STDERR: generic_class_with_blanket_impl_second.carbon:[[@LINE+7]]:43: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] C(T) as Z(T) where .X = C(()) {}
// CHECK:STDERR:                                           ^~
// CHECK:STDERR: generic_class_with_blanket_impl_second.carbon:[[@LINE-18]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C(T) as Z(T) where .X = C(()) {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket
  // impls of `Z(T)`. If a blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here when constructed from a `C` value.
  //
  // TODO: C(T) is C(()) so this should work?
  // CHECK:STDERR: generic_class_with_blanket_impl_second.carbon:[[@LINE+7]]:17: error: member access into undefined facet type `Z(())` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(()).(Z(()).X) = {} as C(());
  // CHECK:STDERR:                 ^~~~~~~
  // CHECK:STDERR: generic_class_with_blanket_impl_second.carbon:[[@LINE-33]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(()).X) = {} as C(());
}

// --- specialized_class_with_facet_value_param.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

class D(T:! type) {}
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] D(T) as Z where .X = D(()) {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D(T) as Z where .X = D(()) {}

class E {}
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+7]]:19: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl E as Z where .X = () {}
// CHECK:STDERR:                   ^~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-18]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl E as Z where .X = () {}

class C(T:! Z, U:! Z) {}

// This places a FacetValue of type FacetType(Z) at the position of `D` in the
// self type, because the class C requires a facet value satisfying Z. It tests
// that we correctly determine that this FacetType is not symbolic, and look at
// the parametes of D for symbolic references.
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+17]]:21: error: member access into undefined facet type `Core.ImplicitAs(Z)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(T, D(E)) as Z where .X = () {}
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-12]]:16: note: initializing generic parameter `U` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z, U:! Z) {}
// CHECK:STDERR:                ^
// CHECK:STDERR:
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+7]]:43: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(T, D(E)) as Z where .X = () {}
// CHECK:STDERR:                                           ^~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-43]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! Z] C(T, D(E)) as Z where .X = () {}
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+17]]:21: error: member access into undefined facet type `Core.ImplicitAs(Z)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(D(T), T) as Z where .X = () {}
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-30]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z, U:! Z) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+7]]:43: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(D(T), T) as Z where .X = () {}
// CHECK:STDERR:                                           ^~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-61]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! Z] C(D(T), T) as Z where .X = () {}
// This is the best match, `T` is in the last position compared to the others.
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+27]]:21: error: member access into undefined facet type `Core.ImplicitAs(Z)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(D(E), T) as Z where .X = C(E, E) {}
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-49]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z, U:! Z) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+17]]:43: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(D(E), T) as Z where .X = C(E, E) {}
// CHECK:STDERR:                                           ^~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-80]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+10]]:48: error: member access into undefined facet type `Core.ImplicitAs(Z)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(D(E), T) as Z where .X = C(E, E) {}
// CHECK:STDERR:                                                ^~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-66]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! Z, U:! Z) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl forall [T:! Z] C(D(E), T) as Z where .X = C(E, E) {}
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(T, T) as Z where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-98]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! Z] C(T, T) as Z where .X = () {}

fn F() {
  // CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+27]]:10: error: member access into undefined facet type `Core.ImplicitAs(Z)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(D(E), D(E)).(Z.X) = {} as C(E, E);
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-87]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z, U:! Z) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  // CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+17]]:25: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(D(E), D(E)).(Z.X) = {} as C(E, E);
  // CHECK:STDERR:                         ^~~
  // CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-118]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE+10]]:38: error: member access into undefined facet type `Core.ImplicitAs(Z)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(D(E), D(E)).(Z.X) = {} as C(E, E);
  // CHECK:STDERR:                                      ^~~~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: specialized_class_with_facet_value_param.carbon:[[@LINE-104]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(T:! Z, U:! Z) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  let a: C(D(E), D(E)).(Z.X) = {} as C(E, E);
}

// --- specialized_class_with_symbolic_facet_value_param.carbon

interface Z {
  let X:! type;
}

// CHECK:STDERR: specialized_class_with_symbolic_facet_value_param.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z where .X = T {}
// CHECK:STDERR:                                     ^~
// CHECK:STDERR: specialized_class_with_symbolic_facet_value_param.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z where .X = T {}

interface Y {}
class C(T:! Y) {}

class D {}
impl D as Y {}

// D can be either a concrete or symbolic FacetValue, depending on what the
// caller has.
// CHECK:STDERR: specialized_class_with_symbolic_facet_value_param.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [D:! Y] C(D) as Z where .X = () {}
// CHECK:STDERR:                                     ^~
// CHECK:STDERR: specialized_class_with_symbolic_facet_value_param.carbon:[[@LINE-24]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [D:! Y] C(D) as Z where .X = () {}

fn F[D:! Y](d: D) {
  // The FacetValue deduced for the param of `C` will be a symbolic FacetValue
  // because we are in a generic where `D` is an unknown type, which will cause
  // the query and impl self type to be C(FacetValue) for a symbolic FacetValue.
  //
  // TODO: The type of `.X` should actually be symbolic here, unless the impl is
  // effectively final (since the lookup query uses a symbolic `D` in its
  // input). But we don't have support for either of those yet.
  // CHECK:STDERR: specialized_class_with_symbolic_facet_value_param.carbon:[[@LINE+7]]:16: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(D).(Z.X) = ();
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: specialized_class_with_symbolic_facet_value_param.carbon:[[@LINE-41]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(D).(Z.X) = ();
}

// --- pointer_specialization_first.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

class C {}
// CHECK:STDERR: pointer_specialization_first.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C* as Z where .X = C {}
// CHECK:STDERR:                    ^~
// CHECK:STDERR: pointer_specialization_first.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C* as Z where .X = C {}

// CHECK:STDERR: pointer_specialization_first.carbon:[[@LINE+7]]:38: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T* as Z where .X = () {}
// CHECK:STDERR:                                      ^~
// CHECK:STDERR: pointer_specialization_first.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T* as Z where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: pointer_specialization_first.carbon:[[@LINE+7]]:16: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: (C*).(Z.X) = {} as C;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: pointer_specialization_first.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: (C*).(Z.X) = {} as C;
}

// --- pointer_specialization_second.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// CHECK:STDERR: pointer_specialization_second.carbon:[[@LINE+7]]:38: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T* as Z where .X = () {}
// CHECK:STDERR:                                      ^~
// CHECK:STDERR: pointer_specialization_second.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T* as Z where .X = () {}

class C {}
// CHECK:STDERR: pointer_specialization_second.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C* as Z where .X = C {}
// CHECK:STDERR:                    ^~
// CHECK:STDERR: pointer_specialization_second.carbon:[[@LINE-17]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C* as Z where .X = C {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  // CHECK:STDERR: pointer_specialization_second.carbon:[[@LINE+7]]:16: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: (C*).(Z.X) = {} as C;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: pointer_specialization_second.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: (C*).(Z.X) = {} as C;
}

// --- cycle_in_deduce_avoided_by_specialization.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

class C(T:! type) {}

// This impl makes a cycle, but it's not considered at all for `C(())` since
// there is another impl with a better type structure, so no diagnostic is
// emitted.
// CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE+7]]:34: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] T as Z where .X = () {}
// CHECK:STDERR:                                  ^~
// CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE-12]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! Z] T as Z where .X = () {}
// Also a cycle, and also a worse match for `C(())`.
// CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE+7]]:37: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Z] C(T) as Z where .X = () {}
// CHECK:STDERR:                                     ^~
// CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE-21]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! Z] C(T) as Z where .X = () {}

// CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE+7]]:23: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(()) as Z where .X = C(()) {}
// CHECK:STDERR:                       ^~
// CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE-30]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C(()) as Z where .X = C(()) {}

fn F() {
  // CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE+7]]:17: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C(()).(Z.X) = {} as C(());
  // CHECK:STDERR:                 ^~~
  // CHECK:STDERR: cycle_in_deduce_avoided_by_specialization.carbon:[[@LINE-40]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z.X) = {} as C(());
}

// --- todo_fail_specialization_written_after_use_is_poisoned.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

// CHECK:STDERR: todo_fail_specialization_written_after_use_is_poisoned.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(T)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(T) where .X = () {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: todo_fail_specialization_written_after_use_is_poisoned.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(T) where .X = () {}

fn F() {
  // CHECK:STDERR: todo_fail_specialization_written_after_use_is_poisoned.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: C.(Z(C).X) = ();
  // CHECK:STDERR:             ^~~~~~
  // CHECK:STDERR: todo_fail_specialization_written_after_use_is_poisoned.carbon:[[@LINE-19]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: C.(Z(C).X) = ();
}

// TODO: This impl changes the result of a previous lookup. The previous lookup
// should have created a poison value that this trips and thus generates a
// diagnostic.
// CHECK:STDERR: todo_fail_specialization_written_after_use_is_poisoned.carbon:[[@LINE+7]]:22: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C as Z(C) where .X = C {}
// CHECK:STDERR:                      ^~
// CHECK:STDERR: todo_fail_specialization_written_after_use_is_poisoned.carbon:[[@LINE-32]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z(C) where .X = C {}

// --- fail_todo_final_specialization_before_generic_use_of_type_constant.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

// CHECK:STDERR: fail_todo_final_specialization_before_generic_use_of_type_constant.carbon:[[@LINE+7]]:50: error: member access into undefined facet type `Z(U)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type, U:! type] T as Z(U) where .X = () {}
// CHECK:STDERR:                                                  ^~
// CHECK:STDERR: fail_todo_final_specialization_before_generic_use_of_type_constant.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type, U:! type] T as Z(U) where .X = () {}

// TODO: Make this `final`.
// CHECK:STDERR: fail_todo_final_specialization_before_generic_use_of_type_constant.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(C) where .X = C {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: fail_todo_final_specialization_before_generic_use_of_type_constant.carbon:[[@LINE-19]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(C) where .X = C {}

fn F[U:! type](T:! Z(C)) {
  // TODO: The value of `.X` can be known to be `C` here when the impl `T as Z(C)` is
  // final.
  // CHECK:STDERR: fail_todo_final_specialization_before_generic_use_of_type_constant.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: T.X = {} as C;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_final_specialization_before_generic_use_of_type_constant.carbon:[[@LINE-31]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: T.X = {} as C;
}

// --- fail_specialization_written_after_generic_use_of_type_constant.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

// CHECK:STDERR: fail_specialization_written_after_generic_use_of_type_constant.carbon:[[@LINE+7]]:50: error: member access into undefined facet type `Z(U)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type, U:! type] T as Z(U) where .X = () {}
// CHECK:STDERR:                                                  ^~
// CHECK:STDERR: fail_specialization_written_after_generic_use_of_type_constant.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type, U:! type] T as Z(U) where .X = () {}

fn F[U:! type](T:! Z(C)) {
  // The value of `.X` is symbolic, it can't be assigned a value of type `C`.
  // CHECK:STDERR: fail_specialization_written_after_generic_use_of_type_constant.carbon:[[@LINE+7]]:10: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   let a: T.X = {} as C;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_specialization_written_after_generic_use_of_type_constant.carbon:[[@LINE-20]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: T.X = {} as C;
}

// TODO: Make this `final`.
// CHECK:STDERR: fail_specialization_written_after_generic_use_of_type_constant.carbon:[[@LINE+7]]:40: error: member access into undefined facet type `Z(C)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! type] T as Z(C) where .X = C {}
// CHECK:STDERR:                                        ^~
// CHECK:STDERR: fail_specialization_written_after_generic_use_of_type_constant.carbon:[[@LINE-31]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z(C) where .X = C {}

// --- fail_specialization_written_after_generic_use.carbon
library "[[@TEST_NAME]]";

interface Z {
  let V:! type;
  fn F() -> V;
}

interface Y {}
// CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE+7]]:34: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [T:! Y] T as Z where .V = () {
// CHECK:STDERR:                                  ^~
// CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! Y] T as Z where .V = () {
  fn F() -> () { return (); }
}

// CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE+7]]:19: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn G(U:! Y) -> U.(Z.V) {
// CHECK:STDERR:                   ^~~
// CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE-20]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn G(U:! Y) -> U.(Z.V) {
  // CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   return U.(Z.F)();
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE-28]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  return U.(Z.F)();
}

class C {
  impl as Y {}
}

// CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE+7]]:19: error: member access into undefined facet type `Z` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C as Z where .V = {} {
// CHECK:STDERR:                   ^~
// CHECK:STDERR: fail_specialization_written_after_generic_use.carbon:[[@LINE-42]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Z {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z where .V = {} {
  fn F() -> {} { return {}; }
}

fn H() {
  let x: {} = G(C);
}
