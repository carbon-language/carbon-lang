// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/eval_fn_in_identify.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/eval_fn_in_identify.carbon

// --- fail_symbolic_type_in_identify_doesnt_find_self_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

eval fn LookupWithType(generic T: type) -> type {
  // CHECK:STDERR:        ╭─┤ fail_symbolic_type_in_identify_doesnt_find_self_impl.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   return T.(I.X);
  // CHECK:STDERR:        ·          ━━━┳━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: cannot convert type `T` into type implementing `I`
  // CHECK:STDERR:        ·                ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return T.(I.X);
}

impl () as I where .X = () and .Y = LookupWithType(.Self) {}

// --- fail_symbolic_type_in_identify_with_generic_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

class A {}
impl forall [T: type] T as I where .X = A and .Y = () {}

eval fn LookupWithType(generic T: type) -> type {
  // This evalautes to contain a symbolic LookupImplWitness instruction
  // because we found the generic impl above. Then, when `T` is replaced by a
  // concrete type, that same witness allows access to its associated
  // constants. That occurs in the identification of the facet type in the
  // `impl as` decl below. A `type` doesn't come with a witness, and we can't
  // find the impl below before it's declared, so this lookup resolves to the
  // impl above. Then the impl below replaces it for the same query which gives
  // a poison error.
  //
  // If LookupWithType could see that it's being called during identify then
  // the re-evaluation of the LookupImplWitness with T would form an
  // IdentifiedWitness instead, that eventually gets replaced with the correct
  // witness for the impl below. But without the impl above, the lookup would
  // fail entirely, leading to a strange dependency on an unrelated impl for
  // this lookup to succeed.
  //
  // CHECK:STDERR:        ╭─┤ fail_symbolic_type_in_identify_with_generic_impl.carbon:[[@LINE+18]]:10
  // CHECK:STDERR:        │ impl forall [T: type] T as I where .X = A and .Y = () {}
  // CHECK:STDERR:        · ───────────────────────────┬───────────────────────────
  // CHECK:STDERR:        ·                            ╰──┤ the use had selected the `impl` here
  // CHECK:STDERR:        ·                               │ [PoisonedImplLookupConcreteResultNotePreviousImpl]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return T.(I.X);
  // CHECK:STDERR:        ·          ┳━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: found `impl` that would change the result of an earlier use
  // CHECK:STDERR:        ·             ┃        of `() as I`
  // CHECK:STDERR:        ·             ┃        [PoisonedImplLookupConcreteResult]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ impl () as I where .X = () and .Y = LookupWithType(.Self) {}
  // CHECK:STDERR:        · ──────────────┬────────────────────────────────────────────
  // CHECK:STDERR:        ·               ╰──┤ the use would select the `impl` here but it was not found yet
  // CHECK:STDERR:        ·                  │ [PoisonedImplLookupConcreteResultNoteBadImpl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return T.(I.X);
}

impl () as I where .X = () and .Y = LookupWithType(.Self) {}

// --- fail_template_type_in_identify_doesnt_find_self_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

eval fn LookupWithType(template T: type) -> type {
  // This performs an impl lookup action once `T` becomes a concrete `type`.
  // But that happens in the identification of the facet type in the `impl as`
  // decl below. A `type` doesn't come with a witness, and we can't find the
  // impl below before it's declared, so this lookup fails.
  return T.(I.X);
}

// CHECK:STDERR:        ╭─┤ fail_template_type_in_identify_doesnt_find_self_impl.carbon:[[@LINE+13]]:1
// CHECK:STDERR:        │   return T.(I.X);
// CHECK:STDERR:        ·          ───┬───
// CHECK:STDERR:        ·             ╰──┤ cannot convert type `()` into type implementing `I`
// CHECK:STDERR:        ·                │ [ConversionFailureTypeToFacet]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: facet type `I where .(I.X) = () and .(I.Y) = LookupWithType(.Self)`
// CHECK:STDERR:        ·    ┃        cannot be identified in `impl as`
// CHECK:STDERR:        ·    ┃        [ImplOfUnidentifiedFacetType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I where .X = () and .Y = LookupWithType(.Self) {}

// --- fail_template_type_in_identify_with_generic_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

class A {}
impl forall [T: type] T as I where .X = A and .Y = () {}

eval fn LookupWithType(template T: type) -> type {
  // This performs an impl lookup action once `T` becomes a concrete `type`.
  // But that happens in the identification of the facet type in the `impl as`
  // decl below. A `type` doesn't come with a witness, and we can't find the
  // impl below before it's declared, so this lookup finds the generic impl
  // above. Then the impl below replaces it for the same query which gives a
  // poison error.
  //
  // CHECK:STDERR:        ╭─┤ fail_template_type_in_identify_with_generic_impl.carbon:[[@LINE+18]]:10
  // CHECK:STDERR:        │ impl forall [T: type] T as I where .X = A and .Y = () {}
  // CHECK:STDERR:        · ───────────────────────────┬───────────────────────────
  // CHECK:STDERR:        ·                            ╰──┤ the use had selected the `impl` here
  // CHECK:STDERR:        ·                               │ [PoisonedImplLookupConcreteResultNotePreviousImpl]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return T.(I.X);
  // CHECK:STDERR:        ·          ┳━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: found `impl` that would change the result of an earlier use
  // CHECK:STDERR:        ·             ┃        of `() as I`
  // CHECK:STDERR:        ·             ┃        [PoisonedImplLookupConcreteResult]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ impl () as I where .X = () and .Y = LookupWithType(.Self) {}
  // CHECK:STDERR:        · ──────────────┬────────────────────────────────────────────
  // CHECK:STDERR:        ·               ╰──┤ the use would select the `impl` here but it was not found yet
  // CHECK:STDERR:        ·                  │ [PoisonedImplLookupConcreteResultNoteBadImpl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return T.(I.X);
}

impl () as I where .X = () and .Y = LookupWithType(.Self) {}

// --- symbolic_facet_in_identify_with_branch.carbon
library "[[@TEST_NAME]]";

interface I {
  let Cond: bool;
  let X: type;
}

eval fn LookupWithType(generic T: I) -> type {
  // We received a facet of type `I` which has a witness pointing to the
  // impl below.
  if (T.(I.Cond)) {
    return ();
  } else {
    return {};
  }
}

impl () as I where .Cond = false and .X = LookupWithType(.Self) {}

// --- template_facet_in_identify_with_branch.carbon
library "[[@TEST_NAME]]";

interface I {
  let Cond: bool;
  let X: type;
}

eval fn LookupWithType(template T: I) -> type {
  // We received a facet of type `I` which has a witness pointing to the
  // impl below.
  if (T.(I.Cond)) {
    return ();
  } else {
    return {};
  }
}

impl () as I where .Cond = false and .X = LookupWithType(.Self) {}

// --- fail_todo_template_facet_with_wrong_order_ids_for_rewrites.carbon
library "[[@TEST_NAME]]";

interface I {
  let A: type;
  let B: type;
}

eval fn LookupWithType(template T: I) -> type {
  return T.(I.A);
}

// This makes the canonical instruction `.Self.(I.B)` have a lower id index
// than `.Self.(I.A)` which then messes up the order of the rewrites in
// the DeclaredFacetType. Then we try to set `.B` in the witness table below
// before `.A` and we fail the evaluation of LookupWithType.
//
// TODO: If we preserved the order of rewrites in the declared facet type then
// this would work though it may pose canonicalization challenges. If we do
// rewrite constraint resolution in identify then could we reorder constraints
// in the IdentifiedFacetType so that dependencies come first, and defer any
// RHS with an eval fn call to be assumed dependent on everything without a
// call?
impl {} as I where .B = () and .A = () {}

// CHECK:STDERR:        ╭─┤ fail_todo_template_facet_with_wrong_order_ids_for_rewrites.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │   return T.(I.A);
// CHECK:STDERR:        ·          ───┬───
// CHECK:STDERR:        ·             ╰──┤ accessing member from impl before it has a defined value
// CHECK:STDERR:        ·                │ [ImplAccessMemberBeforeSet]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as I where .A = () and .B = LookupWithType(.Self) {}
// CHECK:STDERR:        · ━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: unable to monomorphize specific `LookupWithType(() as I)`
// CHECK:STDERR:        ·               ┃        [ResolvingSpecificHere]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I where .A = () and .B = LookupWithType(.Self) {}
