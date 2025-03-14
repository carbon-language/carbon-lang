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
impl C as Z(C) where .X = C {}

impl forall [T:! type] T as Z(T) where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  let a: C.(Z(C).X) = {} as C;
}

// --- specialized_self_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

impl forall [T:! type] T as Z(T) where .X = () {}

class C {}
impl C as Z(C) where .X = C {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  let a: C.(Z(C).X) = {} as C;
}

// --- specialized_constraint_first.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

impl C as Z(C) where .X = C {}
impl forall [T:! type] C as Z(T) where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  let a: C.(Z(C).X) = {} as C;
}

// --- specialized_constraint_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

impl forall [T:! type] C as Z(T) where .X = () {}
impl C as Z(C) where .X = C {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  let a: C.(Z(C).X) = {} as C;
}

// --- fail_todo_specialized_self_vs_constraint_self_first.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C(T:! type) {}

impl forall [T:! type] C(()) as Z(T) where .X = C(T) {}
impl forall [T:! type] C(T) as Z(C(())) where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  //
  // TODO: C(T) is C(()) so this should work?
  // CHECK:STDERR: fail_todo_specialized_self_vs_constraint_self_first.carbon:[[@LINE+7]]:31: error: cannot implicitly convert value of type `C(())` to `C(T)` [ConversionFailure]
  // CHECK:STDERR:   let a: C(()).(Z(C(())).X) = {} as C(());
  // CHECK:STDERR:                               ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_specialized_self_vs_constraint_self_first.carbon:[[@LINE+4]]:31: note: type `C(())` does not implement interface `Core.ImplicitAs(C(T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: C(()).(Z(C(())).X) = {} as C(());
  // CHECK:STDERR:                               ^~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(C(())).X) = {} as C(());
}

// --- fail_todo_specialized_self_vs_constraint_self_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C(T:! type) {}

impl forall [T:! type] C(T) as Z(C(())) where .X = () {}
impl forall [T:! type] C(()) as Z(T) where .X = C(T) {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket impl
  // of `Z(T)`. If the blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here.
  //
  // TODO: C(T) is C(()) so this should work?
  // CHECK:STDERR: fail_todo_specialized_self_vs_constraint_self_second.carbon:[[@LINE+7]]:31: error: cannot implicitly convert value of type `C(())` to `C(T)` [ConversionFailure]
  // CHECK:STDERR:   let a: C(()).(Z(C(())).X) = {} as C(());
  // CHECK:STDERR:                               ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_specialized_self_vs_constraint_self_second.carbon:[[@LINE+4]]:31: note: type `C(())` does not implement interface `Core.ImplicitAs(C(T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: C(()).(Z(C(())).X) = {} as C(());
  // CHECK:STDERR:                               ^~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(C(())).X) = {} as C(());
}

// --- generic_class_with_fully_specified_impl.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

impl forall [T:! type] T as Z(T) where .X = {.a: ()} {}

class C(T:! type) {}
impl C(()) as Z(()) where .X = C(()) {}

impl forall [T:! type] C(T) as Z(T) where .X = {.b: ()} {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket
  // impls of `Z(T)`. If a blanket impl is chosen, then `a` will have a struct
  // type which will fail to typecheck here when constructed from a `C` value.
  let a: C(()).(Z(()).X) = {} as C(());
}

// --- fail_todo_generic_class_with_blanket_impl_first.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C(T:! type) {}

impl forall [T:! type] C(T) as Z(T) where .X = C(T) {}

impl forall [T:! type] T as Z(T) where .X = () {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket
  // impls of `Z(T)`. If a blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here when constructed from a `C` value.
  //
  // TODO: C(T) is C(()) so this should work?
  // CHECK:STDERR: fail_todo_generic_class_with_blanket_impl_first.carbon:[[@LINE+7]]:28: error: cannot implicitly convert value of type `C(())` to `C(T)` [ConversionFailure]
  // CHECK:STDERR:   let a: C(()).(Z(()).X) = {} as C(());
  // CHECK:STDERR:                            ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_class_with_blanket_impl_first.carbon:[[@LINE+4]]:28: note: type `C(())` does not implement interface `Core.ImplicitAs(C(T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: C(()).(Z(()).X) = {} as C(());
  // CHECK:STDERR:                            ^~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(()).X) = {} as C(());
}

// --- fail_todo_generic_class_with_blanket_impl_second.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

impl forall [T:! type] T as Z(T) where .X = () {}

class C(T:! type) {}

impl forall [T:! type] C(T) as Z(T) where .X = C(T) {}

fn F() {
  // The specialization of `Z(C)` should match in preference to the blanket
  // impls of `Z(T)`. If a blanket impl is chosen, then `a` will have type `()`
  // which will fail to typecheck here when constructed from a `C` value.
  //
  // TODO: C(T) is C(()) so this should work?
  // CHECK:STDERR: fail_todo_generic_class_with_blanket_impl_second.carbon:[[@LINE+7]]:28: error: cannot implicitly convert value of type `C(())` to `C(T)` [ConversionFailure]
  // CHECK:STDERR:   let a: C(()).(Z(()).X) = {} as C(());
  // CHECK:STDERR:                            ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_class_with_blanket_impl_second.carbon:[[@LINE+4]]:28: note: type `C(())` does not implement interface `Core.ImplicitAs(C(T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: C(()).(Z(()).X) = {} as C(());
  // CHECK:STDERR:                            ^~~~~~~~~~~
  // CHECK:STDERR:
  let a: C(()).(Z(()).X) = {} as C(());
}

// --- specialized_class_with_facet_value_param.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

class D(T:! type) {}
impl forall [T:! type] D(T) as Z where .X = D(()) {}

class E {}
impl E as Z where .X = () {}

class C(T:! Z, U:! Z) {}

// This places a FacetValue of type FacetType(Z) at the position of `D` in the
// self type, because the class C requires a facet value satisfying Z. It tests
// that we correctly determine that this FacetType is not symbolic, and look at
// the parametes of D for symbolic references.
impl forall [T:! Z] C(T, D(E)) as Z where .X = () {}
impl forall [T:! Z] C(D(T), T) as Z where .X = () {}
// This is the best match, `T` is in the last position compared to the others.
impl forall [T:! Z] C(D(E), T) as Z where .X = C(E, E) {}
impl forall [T:! Z] C(T, T) as Z where .X = () {}

fn F() {
  let a: C(D(E), D(E)).(Z.X) = {} as C(E, E);
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
impl forall [T:! Z] T as Z where .X = () {}
// Also a cycle, and also a worse match for `C(())`.
impl forall [T:! Z] C(T) as Z where .X = () {}

impl C(()) as Z where .X = C(()) {}

fn F() {
  let a: C(()).(Z.X) = {} as C(());
}
