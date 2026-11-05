// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_assoc_const.carbon

// --- success.carbon
library "[[@TEST_NAME]]";

interface I { let T: type; }

impl () as I where .T = {} {}

// --- success_associated.carbon
library "[[@TEST_NAME]]";

interface I { let T: type; let U: type; }

impl () as I where .T = .U and .U = {} {}

// --- redecl.carbon
library "[[@TEST_NAME]]";

interface I2 { let T2: type; }

impl () as I2 where .T2 = {};
impl () as I2 where .T2 = {} {}

// --- fail_redecl_adds_rewrites.carbon
library "[[@TEST_NAME]]";

interface I3 { let T3: type; }

// CHECK:STDERR:        ╭─┤ fail_redecl_adds_rewrites.carbon:[[@LINE+6]]:1
// CHECK:STDERR: ->     │ impl () as I3;
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I3;
impl () as I3 where .T3 = {} {}

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";

interface J { let U: type; }

// CHECK:STDERR:        ╭─┤ fail_mismatch.carbon:[[@LINE+6]]:1
// CHECK:STDERR: ->     │ impl () as J where .U = {};
// CHECK:STDERR:        · ━━━━━━━━━━━━━┳━━━━━━━━━━━━━
// CHECK:STDERR:        ·              ┗━━┫ error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as J where .U = {};
impl () as J where .U = () {}

// --- fail_mismatch_bad_value.carbon
library "[[@TEST_NAME]]";

interface I4 {
  let T4: type;
  let T5: type;
  let T6: type;
}

// CHECK:STDERR:        ╭─┤ fail_mismatch_bad_value.carbon:[[@LINE+16]]:27
// CHECK:STDERR: ->     │ impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:        ·                           ━━┳━
// CHECK:STDERR:        ·                             ┗━━┫ error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_mismatch_bad_value.carbon:[[@LINE+10]]:61
// CHECK:STDERR: ->     │ impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:        ·                                                             ┳━━━
// CHECK:STDERR:        ·                                                             ┗━━┫ error: name
// CHECK:STDERR:        ·                                                                ┃        `BAD2`
// CHECK:STDERR:        ·                                                                ┃        not
// CHECK:STDERR:        ·                                                                ┃        found
// CHECK:STDERR:        ·                                                                ┃        [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;

// CHECK:STDERR:        ╭─┤ fail_mismatch_bad_value.carbon:[[@LINE+17]]:46
// CHECK:STDERR: ->     │ impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:        ·                                              ━━┳━
// CHECK:STDERR:        ·                                                ┗━━┫ error: name `BAD3` not found
// CHECK:STDERR:        ·                                                   ┃        [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_mismatch_bad_value.carbon:[[@LINE+10]]:61
// CHECK:STDERR: ->     │ impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:        ·                                                             ┳━━━
// CHECK:STDERR:        ·                                                             ┗━━┫ error: name
// CHECK:STDERR:        ·                                                                ┃        `BAD4`
// CHECK:STDERR:        ·                                                                ┃        not
// CHECK:STDERR:        ·                                                                ┃        found
// CHECK:STDERR:        ·                                                                ┃        [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";

interface K { let V: type; }

impl () as K where .V = {};
// CHECK:STDERR:        ╭─┤ fail_missing_on_definition.carbon:[[@LINE+18]]:1
// CHECK:STDERR:        │ interface K { let V: type; }
// CHECK:STDERR:        ·                   ───┬───
// CHECK:STDERR:        ·                      ╰──┤ associated constant declared here
// CHECK:STDERR:        ·                         │ [AssociatedConstantHere]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as K {}
// CHECK:STDERR:        · ━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·       ┗━━┫ error: associated constant V not given a value in impl of interface K
// CHECK:STDERR:        ·          ┃        [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_missing_on_definition.carbon:[[@LINE-13]]:1
// CHECK:STDERR: ->     │ impl () as K where .V = {};
// CHECK:STDERR:        · ━━━━━━━━━━━━━┳━━━━━━━━━━━━━
// CHECK:STDERR:        ·              ┗━━┫ error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as K {}

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";

interface L { let W: type; }

// CHECK:STDERR:        ╭─┤ fail_two_different.carbon:[[@LINE+8]]:12
// CHECK:STDERR: ->     │ impl () as L where .W = {} and .W = () {}
// CHECK:STDERR:        ·            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: associated constant `.(L.W)` given two different values
// CHECK:STDERR:        ·               ┃        `{}` and `()`
// CHECK:STDERR:        ·               ┃        [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as L where .W = {} and .W = () {}

// --- fail_two_different_first_associated.carbon
library "[[@TEST_NAME]]";

interface L2 { let W2: type; let X2: type; }

// CHECK:STDERR:        ╭─┤ fail_two_different_first_associated.carbon:[[@LINE+8]]:12
// CHECK:STDERR: ->     │ impl () as L2 where .W2 = .X2 and .W2 = () {}
// CHECK:STDERR:        ·            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: associated constant `.(L2.W2)` given two different values
// CHECK:STDERR:        ·               ┃        `()` and `.(L2.X2)`
// CHECK:STDERR:        ·               ┃        [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as L2 where .W2 = .X2 and .W2 = () {}

// --- fail_two_different_second_associated.carbon
library "[[@TEST_NAME]]";

interface L2 { let W2: type; let X2: type; }

// CHECK:STDERR:        ╭─┤ fail_two_different_second_associated.carbon:[[@LINE+8]]:12
// CHECK:STDERR: ->     │ impl () as L2 where .W2 = () and .W2 = .X2 {}
// CHECK:STDERR:        ·            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: associated constant `.(L2.W2)` given two different values
// CHECK:STDERR:        ·               ┃        `()` and `.(L2.X2)`
// CHECK:STDERR:        ·               ┃        [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as L2 where .W2 = () and .W2 = .X2 {}

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";

interface L2 { let W2: type; }

// CHECK:STDERR:        ╭─┤ fail_two_different_first_bad.carbon:[[@LINE+6]]:27
// CHECK:STDERR: ->     │ impl () as L2 where .W2 = BAD5 and .W2 = () {}
// CHECK:STDERR:        ·                           ━━┳━
// CHECK:STDERR:        ·                             ┗━━┫ error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as L2 where .W2 = BAD5 and .W2 = () {}

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";

interface L3 { let W3: type; }

// CHECK:STDERR:        ╭─┤ fail_two_different_second_bad.carbon:[[@LINE+7]]:40
// CHECK:STDERR: ->     │ impl () as L3 where .W3 = {} and .W3 = BAD6 {}
// CHECK:STDERR:        ·                                        ━━┳━
// CHECK:STDERR:        ·                                          ┗━━┫ error: name `BAD6` not found
// CHECK:STDERR:        ·                                             ┃        [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as L3 where .W3 = {} and .W3 = BAD6 {}

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";

interface L4 { let W4: type; }

// CHECK:STDERR:        ╭─┤ fail_two_different_both_bad.carbon:[[@LINE+13]]:27
// CHECK:STDERR: ->     │ impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:        ·                           ━━┳━
// CHECK:STDERR:        ·                             ┗━━┫ error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_two_different_both_bad.carbon:[[@LINE+7]]:42
// CHECK:STDERR: ->     │ impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:        ·                                          ━━┳━
// CHECK:STDERR:        ·                                            ┗━━┫ error: name `BAD8` not found
// CHECK:STDERR:        ·                                               ┃        [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}

// --- fail_many_different.carbon
library "[[@TEST_NAME]]";

interface L { let W: type; }

// The facet type should have a single rewrite with `<error>` in the RHS (they
// are all errors, and they get deduped).
//
//@dump-sem-ir-begin
// CHECK:STDERR:        ╭─┤ fail_many_different.carbon:[[@LINE+8]]:12
// CHECK:STDERR: ->     │ impl () as L where .W = ((), (), ()) and .W = ({}, (), ()) and .W = ({}, {...
// CHECK:STDERR:        ·            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: associated constant `.(L.W)` given two different values
// CHECK:STDERR:        ·               ┃        `((), (), ())` and `({}, (), ())`
// CHECK:STDERR:        ·               ┃        [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as L where .W = ((), (), ()) and .W = ({}, (), ()) and .W = ({}, {}, ()) and .W = ({}, (), {}) {}
//@dump-sem-ir-end

// --- repeated.carbon
library "[[@TEST_NAME]]";

interface M { let X: type; }

impl () as M where .X = {} and .X = {} {}

// --- repeated_associated.carbon
library "[[@TEST_NAME]]";

interface M { let X: type; let Y: type; }

impl () as M where .X = .Y and .X = .Y and .Y = () {}

// --- repeated_concrete_value_and_associated.carbon
library "[[@TEST_NAME]]";

interface M { let X: type; let Y: type; }

impl () as M where .X = () and .Y = .X and .X = .Y {}

impl {} as M where .X = () and .X = .X and .Y = () {}

// --- fail_repeated_and_different.carbon
library "[[@TEST_NAME]]";

interface M { let X: type; }

// CHECK:STDERR:        ╭─┤ fail_repeated_and_different.carbon:[[@LINE+8]]:12
// CHECK:STDERR: ->     │ impl () as M where .X = {} and .X = () and .X = {} {}
// CHECK:STDERR:        ·            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: associated constant `.(M.X)` given two different values
// CHECK:STDERR:        ·               ┃        `{}` and `()`
// CHECK:STDERR:        ·               ┃        [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as M where .X = {} and .X = () and .X = {} {}

// --- fail_cycle_single.carbon
library "[[@TEST_NAME]]";

interface M { let X: type; }

// This fails because it resolves to `.X = .X` which is cyclical.
//
// CHECK:STDERR:        ╭─┤ fail_cycle_single.carbon:[[@LINE+7]]:12
// CHECK:STDERR: ->     │ impl () as M where .X = .X {}
// CHECK:STDERR:        ·            ━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·                   ┗━━┫ error: found cycle in facet type constraint for `.(M.X)`
// CHECK:STDERR:        ·                      ┃        [FacetTypeConstraintCycle]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as M where .X = .X {}

// Even though `.X = ()` is specified, the rewrites are resolved left to right
// and a cycle `.X = .X` is found first.
//
// CHECK:STDERR:        ╭─┤ fail_cycle_single.carbon:[[@LINE+7]]:12
// CHECK:STDERR: ->     │ impl () as M where .X = .X and .X = () {}
// CHECK:STDERR:        ·            ━━━━━━━━┳━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: found cycle in facet type constraint for `.(M.X)`
// CHECK:STDERR:        ·                       ┃        [FacetTypeConstraintCycle]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as M where .X = .X and .X = () {}

// --- fail_cycle.carbon
library "[[@TEST_NAME]]";

interface M { let X: type; let Y: type; let Z: type; }

// This fails because it resolves to `.X = .X` which is cyclical.
// The value of .X and .Y becomes <error> but .Z is still valid.
//
//@dump-sem-ir-begin
// CHECK:STDERR:        ╭─┤ fail_cycle.carbon:[[@LINE+7]]:12
// CHECK:STDERR: ->     │ impl () as M where .X = .Y and .Y = .X and .Z = () {}
// CHECK:STDERR:        ·            ━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: found cycle in facet type constraint for `.(M.Y)`
// CHECK:STDERR:        ·                       ┃        [FacetTypeConstraintCycle]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as M where .X = .Y and .Y = .X and .Z = () {}
//@dump-sem-ir-end

// --- fail_todo_cycle_between_interfaces.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1: type;
  let X2: type;
}
interface J {
  let X3: type;
}

// TODO: This should fail due to a cyclic definition of each value. But right
// now it's failing due to compound member access in a rewrite constraint not
// working.
//
// CHECK:STDERR:        ╭─┤ fail_todo_cycle_between_interfaces.carbon:[[@LINE+23]]:45
// CHECK:STDERR: ->     │ ...I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}
// CHECK:STDERR:        ·                                     ┳
// CHECK:STDERR:        ·                                     ┗━━┫ error: expected identifier or `Self`
// CHECK:STDERR:        ·                                        ┃        after `.`
// CHECK:STDERR:        ·                                        ┃        [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_cycle_between_interfaces.carbon:[[@LINE+15]]:45
// CHECK:STDERR: ->     │ ...I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}
// CHECK:STDERR:        ·                                     ┳
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_cycle_between_interfaces.carbon:[[@LINE+8]]:71
// CHECK:STDERR: ->     │ ...I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}
// CHECK:STDERR:        ·                                                               ┳
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: expected identifier or `Self` after `.`
// CHECK:STDERR:        · ┃        [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}

// --- non-type.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y: {.a: {}};
}

impl () as N where .Y = {.a = {}} { }

// --- non-type_repeated.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y: {.a: {}};
}

impl () as N where .Y = {.a = {}} and .Y = {.a = {}} { }

// --- fail_non-type_different.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y: {.a: type};
}

// CHECK:STDERR:        ╭─┤ fail_non-type_different.carbon:[[@LINE+8]]:12
// CHECK:STDERR: ->     │ impl () as N where .Y = {.a = {}} and .Y = {.a = ()} {}
// CHECK:STDERR:        ·            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: associated constant `.(N.Y)` given two different values
// CHECK:STDERR:        ·               ┃        `{.a = {}}` and `{.a = ()}`
// CHECK:STDERR:        ·               ┃        [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as N where .Y = {.a = {}} and .Y = {.a = ()} {}

// --- fail_where_rewrite_function.carbon
library "[[@TEST_NAME]]";

interface IF { fn F(); }

class CD { }

// CHECK:STDERR:        ╭─┤ fail_where_rewrite_function.carbon:[[@LINE+7]]:12
// CHECK:STDERR: ->     │ impl CD as IF where .F = 0 {
// CHECK:STDERR:        ·            ━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·                   ┗━━┫ error: rewrite specified for associated function F
// CHECK:STDERR:        ·                      ┃        [RewriteForAssociatedFunction]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl CD as IF where .F = 0 {
  fn F() {}
}

// --- facet_type_in_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface M { let X: type; }
constraint N {
  extend require impls M where .X = {};
}

impl () as M where .X = N {}

// --- fail_todo_period_self_impl_lookup.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
    let Z1: Y;
}

class C;
impl C as Y {}
// Implied constraint: .Self impls Y. Should be verified when finishing the impl
// decl (where we know .Self is C). See:
// https://discord.com/channels/655572317891461132/941071822756143115/1400953094015160370
//
// CHECK:STDERR:        ╭─┤ fail_todo_period_self_impl_lookup.carbon:[[@LINE+8]]:25
// CHECK:STDERR: ->     │ impl C as Z where .Z1 = .Self {}
// CHECK:STDERR:        ·                         ┳━━━━
// CHECK:STDERR:        ·                         ┗━━┫ error: cannot convert type `.Self` that implements
// CHECK:STDERR:        ·                            ┃        `Z` into type implementing `Y`
// CHECK:STDERR:        ·                            ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C as Z where .Z1 = .Self {}

// --- fail_todo_period_self_compared_with_concrete_self.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
    let Z1: Y;
}

class C;
impl C as Y {}
impl C as Z where .Z1 = C {}

// Implied constraint: .Self impls Y. Should be verified against the type of the
// facet value replacing T when calling F.
//
// CHECK:STDERR:        ╭─┤ fail_todo_period_self_compared_with_concrete_self.carbon:[[@LINE+9]]:38
// CHECK:STDERR: ->     │ fn F(unused generic T: Z where .Z1 = .Self) {}
// CHECK:STDERR:        ·                                      ┳━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                         ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                         ┃        implementing `Y`
// CHECK:STDERR:        ·                                         ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(unused generic T: Z where .Z1 = .Self) {}
fn G() {
  F(C);

  // Implied constraint: .Self impls Y. Should be verified against the type of
  // the facet value of C when casting.
  //
  // CHECK:STDERR:        ╭─┤ fail_todo_period_self_compared_with_concrete_self.carbon:[[@LINE+8]]:23
  // CHECK:STDERR: ->     │   C as (Z where .Z1 = .Self);
  // CHECK:STDERR:        ·                       ┳━━━━
  // CHECK:STDERR:        ·                       ┗━━┫ error: cannot convert type `.Self` that implements
  // CHECK:STDERR:        ·                          ┃        `Z` into type implementing `Y`
  // CHECK:STDERR:        ·                          ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  C as (Z where .Z1 = .Self);
}

// --- assoc_const_impls_in_decl.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

impl () as J {}

// Test that we avoid an infinite impl lookup cycle from an LookupImplWitness
// instruction in the impl declaration. When matching this impl during impl
// lookup, we deduce and replace `T` with a specific argument, and resolve that
// specific's decl eval block. Doing so re-evaluates the lookup for `I` inside
// `.X`. If that lookup found this impl (since it's a final impl, eval can find
// it), we would deduce and produce an infinite cycle. Instructions inside the
// impl decl should not be able to find the impl.
final impl forall [T: type] T as I where .X = () and .X impls J {}

fn F(unused generic T: I where .X = ()) {}
fn G() {
  class C;
  F(C);
}

// CHECK:STDOUT: --- fail_many_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %L.assoc_type: type = assoc_entity_type @L [concrete]
// CHECK:STDOUT:   %assoc0: %L.assoc_type = assoc_entity element0, @L.WithSelf.%W [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: %L.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.63f: <witness> = impl_self_witness %.Self.frozen, @L [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2e9: type = impl_witness_access %.63f, element0 [symbolic_self]
// CHECK:STDOUT:   %tuple.type.2d5: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.7e4: %tuple.type.2d5 = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.e5a: type = tuple_type (%empty_struct_type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.b36: %tuple.type.e5a = tuple_value (%empty_struct, %empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.type.d7e: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.8ed: %tuple.type.d7e = tuple_value (%empty_struct, %empty_struct, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.type.bd8: type = tuple_type (%empty_struct_type, %empty_tuple.type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.d90: %tuple.type.bd8 = tuple_value (%empty_struct, %empty_tuple, %empty_struct) [concrete]
// CHECK:STDOUT:   %.Self: %L.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.300: <witness> = impl_self_witness %.Self, @L [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.865: type = impl_witness_access %.300, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     %.Self.frozen: %L.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %L.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc17_20: %L.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc17_20: type = facet_access_type %.Self.ref.loc17_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc17_20: type = converted %.Self.ref.loc17_20, %.Self.as_type.loc17_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %W.ref.loc17_20: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc17_20.1: type = impl_witness_access constants.%.63f, element0 [symbolic_self = constants.%impl.elem0.2e9]
// CHECK:STDOUT:     %.loc17_27: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_31: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_35: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_36.1: %tuple.type.2d5 = tuple_literal (%.loc17_27, %.loc17_31, %.loc17_35) [concrete = constants.%tuple.7e4]
// CHECK:STDOUT:     %.loc17_36.2: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_36.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_36.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_36.5: type = converted %.loc17_36.1, constants.%tuple.type.2d5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %rewrite.loc17_23.1 = requirement_rewrite %impl.elem0.loc17_20.1, %.loc17_36.5 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc17_42: %L.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc17_42: type = facet_access_type %.Self.ref.loc17_42 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc17_42: type = converted %.Self.ref.loc17_42, %.Self.as_type.loc17_42 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %W.ref.loc17_42: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc17_42.1: type = impl_witness_access constants.%.63f, element0 [symbolic_self = constants.%impl.elem0.2e9]
// CHECK:STDOUT:     %impl.elem0.subst.loc17_42.1: type = impl_witness_access_substituted %impl.elem0.loc17_42.1, %.loc17_36.5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %.loc17_49: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_53: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_57: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_58.1: %tuple.type.e5a = tuple_literal (%.loc17_49, %.loc17_53, %.loc17_57) [concrete = constants.%tuple.b36]
// CHECK:STDOUT:     %.loc17_58.2: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc17_58.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_58.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_58.5: type = converted %.loc17_58.1, constants.%tuple.type.e5a [concrete = constants.%tuple.type.e5a]
// CHECK:STDOUT:     %rewrite.loc17_45.1 = requirement_rewrite %impl.elem0.subst.loc17_42.1, %.loc17_58.5 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc17_64: %L.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc17_64: type = facet_access_type %.Self.ref.loc17_64 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc17_64: type = converted %.Self.ref.loc17_64, %.Self.as_type.loc17_64 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %W.ref.loc17_64: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc17_64.1: type = impl_witness_access constants.%.63f, element0 [symbolic_self = constants.%impl.elem0.2e9]
// CHECK:STDOUT:     %impl.elem0.subst.loc17_64.1: type = impl_witness_access_substituted %impl.elem0.loc17_64.1, %.loc17_36.5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %.loc17_71: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_75: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_79: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_80.1: %tuple.type.d7e = tuple_literal (%.loc17_71, %.loc17_75, %.loc17_79) [concrete = constants.%tuple.8ed]
// CHECK:STDOUT:     %.loc17_80.2: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc17_80.3: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc17_80.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_80.5: type = converted %.loc17_80.1, constants.%tuple.type.d7e [concrete = constants.%tuple.type.d7e]
// CHECK:STDOUT:     %rewrite.loc17_67.1 = requirement_rewrite %impl.elem0.subst.loc17_64.1, %.loc17_80.5 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc17_86: %L.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc17_86: type = facet_access_type %.Self.ref.loc17_86 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc17_86: type = converted %.Self.ref.loc17_86, %.Self.as_type.loc17_86 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %W.ref.loc17_86: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc17_86.1: type = impl_witness_access constants.%.63f, element0 [symbolic_self = constants.%impl.elem0.2e9]
// CHECK:STDOUT:     %impl.elem0.subst.loc17_86.1: type = impl_witness_access_substituted %impl.elem0.loc17_86.1, %.loc17_36.5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %.loc17_93: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_97: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_101: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_102.1: %tuple.type.bd8 = tuple_literal (%.loc17_93, %.loc17_97, %.loc17_101) [concrete = constants.%tuple.d90]
// CHECK:STDOUT:     %.loc17_102.2: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc17_102.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_102.4: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc17_102.5: type = converted %.loc17_102.1, constants.%tuple.type.bd8 [concrete = constants.%tuple.type.bd8]
// CHECK:STDOUT:     %rewrite.loc17_89.1 = requirement_rewrite %impl.elem0.subst.loc17_86.1, %.loc17_102.5 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc17_20.2: type = impl_witness_access constants.%.300, element0 [symbolic_self = constants.%impl.elem0.865]
// CHECK:STDOUT:     %rewrite.loc17_23.2 = requirement_rewrite %impl.elem0.loc17_20.2, %.loc17_36.5 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc17_42.2: type = impl_witness_access constants.%.300, element0 [symbolic_self = constants.%impl.elem0.865]
// CHECK:STDOUT:     %impl.elem0.subst.loc17_42.2: type = impl_witness_access_substituted %impl.elem0.loc17_42.2, %.loc17_36.5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %rewrite.loc17_45.2 = requirement_rewrite %impl.elem0.subst.loc17_42.2, %.loc17_58.5 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc17_64.2: type = impl_witness_access constants.%.300, element0 [symbolic_self = constants.%impl.elem0.865]
// CHECK:STDOUT:     %impl.elem0.subst.loc17_64.2: type = impl_witness_access_substituted %impl.elem0.loc17_64.2, %.loc17_36.5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %rewrite.loc17_67.2 = requirement_rewrite %impl.elem0.subst.loc17_64.2, %.loc17_80.5 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc17_86.2: type = impl_witness_access constants.%.300, element0 [symbolic_self = constants.%impl.elem0.865]
// CHECK:STDOUT:     %impl.elem0.subst.loc17_86.2: type = impl_witness_access_substituted %impl.elem0.loc17_86.2, %.loc17_36.5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %rewrite.loc17_89.2 = requirement_rewrite %impl.elem0.subst.loc17_86.2, %.loc17_102.5 [concrete]
// CHECK:STDOUT:     %.loc17_14: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %L.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc17_23.2 = requirement_rewrite %impl.elem0.loc17_20.2, %.loc17_36.5 [concrete]
// CHECK:STDOUT:       %rewrite.loc17_45.2 = requirement_rewrite %impl.elem0.subst.loc17_42.2, %.loc17_58.5 [concrete]
// CHECK:STDOUT:       %rewrite.loc17_67.2 = requirement_rewrite %impl.elem0.subst.loc17_64.2, %.loc17_80.5 [concrete]
// CHECK:STDOUT:       %rewrite.loc17_89.2 = requirement_rewrite %impl.elem0.subst.loc17_86.2, %.loc17_102.5 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.<error>.impl: %.loc17_7.2 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.WithSelf.%X [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.WithSelf.%Y [concrete]
// CHECK:STDOUT:   %assoc2: %M.assoc_type = assoc_entity element2, @M.WithSelf.%Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: %M.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.b7a: <witness> = impl_self_witness %.Self.frozen, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.88c: type = impl_witness_access %.b7a, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.17a: type = impl_witness_access %.b7a, element1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem2.efd: type = impl_witness_access %.b7a, element2 [symbolic_self]
// CHECK:STDOUT:   %.Self: %M.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.f89: <witness> = impl_self_witness %.Self, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.dfb: type = impl_witness_access %.f89, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.bb0: type = impl_witness_access %.f89, element1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem2.d85: type = impl_witness_access %.f89, element2 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %.loc16_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_7.2: type = converted %.loc16_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self.frozen: %M.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %M.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc16_20: %M.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc16_20: type = facet_access_type %.Self.ref.loc16_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_20: type = converted %.Self.ref.loc16_20, %.Self.as_type.loc16_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %X.ref.loc16_20: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc16_20.1: type = impl_witness_access constants.%.b7a, element0 [symbolic_self = constants.%impl.elem0.88c]
// CHECK:STDOUT:     %.Self.ref.loc16_25: %M.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc16_25: type = facet_access_type %.Self.ref.loc16_25 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_25: type = converted %.Self.ref.loc16_25, %.Self.as_type.loc16_25 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Y.ref.loc16_25: %M.assoc_type = name_ref Y, @Y.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc16_25.1: type = impl_witness_access constants.%.b7a, element1 [symbolic_self = constants.%impl.elem1.17a]
// CHECK:STDOUT:     %rewrite.loc16_23.1 = requirement_rewrite %impl.elem0.loc16_20.1, %impl.elem1.loc16_25.1 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc16_32: %M.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc16_32: type = facet_access_type %.Self.ref.loc16_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_32: type = converted %.Self.ref.loc16_32, %.Self.as_type.loc16_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Y.ref.loc16_32: %M.assoc_type = name_ref Y, @Y.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc16_32.1: type = impl_witness_access constants.%.b7a, element1 [symbolic_self = constants.%impl.elem1.17a]
// CHECK:STDOUT:     %.Self.ref.loc16_37: %M.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc16_37: type = facet_access_type %.Self.ref.loc16_37 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_37: type = converted %.Self.ref.loc16_37, %.Self.as_type.loc16_37 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %X.ref.loc16_37: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc16_37.1: type = impl_witness_access constants.%.b7a, element0 [symbolic_self = constants.%impl.elem0.88c]
// CHECK:STDOUT:     %impl.elem0.subst.loc16_37.1: type = impl_witness_access_substituted %impl.elem0.loc16_37.1, %impl.elem1.loc16_25.1 [symbolic_self = constants.%impl.elem1.17a]
// CHECK:STDOUT:     %rewrite.loc16_35.1 = requirement_rewrite %impl.elem1.loc16_32.1, %impl.elem0.subst.loc16_37.1 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc16_44: %M.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc16_44: type = facet_access_type %.Self.ref.loc16_44 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_44: type = converted %.Self.ref.loc16_44, %.Self.as_type.loc16_44 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc16_44.1: type = impl_witness_access constants.%.b7a, element2 [symbolic_self = constants.%impl.elem2.efd]
// CHECK:STDOUT:     %.loc16_50.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_50.2: type = converted %.loc16_50.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc16_47.1 = requirement_rewrite %impl.elem2.loc16_44.1, %.loc16_50.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc16_20.2: type = impl_witness_access constants.%.f89, element0 [symbolic_self = constants.%impl.elem0.dfb]
// CHECK:STDOUT:     %impl.elem1.loc16_25.2: type = impl_witness_access constants.%.f89, element1 [symbolic_self = constants.%impl.elem1.bb0]
// CHECK:STDOUT:     %rewrite.loc16_23.2 = requirement_rewrite %impl.elem0.loc16_20.2, %impl.elem1.loc16_25.2 [concrete]
// CHECK:STDOUT:     %impl.elem1.loc16_32.2: type = impl_witness_access constants.%.f89, element1 [symbolic_self = constants.%impl.elem1.bb0]
// CHECK:STDOUT:     %impl.elem0.loc16_37.2: type = impl_witness_access constants.%.f89, element0 [symbolic_self = constants.%impl.elem0.dfb]
// CHECK:STDOUT:     %impl.elem1.loc16_25.3: type = impl_witness_access constants.%.f89, element1 [symbolic_self = constants.%impl.elem1.bb0]
// CHECK:STDOUT:     %impl.elem0.subst.loc16_37.2: type = impl_witness_access_substituted %impl.elem0.loc16_37.2, %impl.elem1.loc16_25.3 [symbolic_self = constants.%impl.elem1.bb0]
// CHECK:STDOUT:     %rewrite.loc16_35.2 = requirement_rewrite %impl.elem1.loc16_32.2, %impl.elem0.subst.loc16_37.2 [concrete]
// CHECK:STDOUT:     %impl.elem2.loc16_44.2: type = impl_witness_access constants.%.f89, element2 [symbolic_self = constants.%impl.elem2.d85]
// CHECK:STDOUT:     %rewrite.loc16_47.2 = requirement_rewrite %impl.elem2.loc16_44.2, %.loc16_50.2 [concrete]
// CHECK:STDOUT:     %.loc16_14: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %M.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc16_23.2 = requirement_rewrite %impl.elem0.loc16_20.2, %impl.elem1.loc16_25.2 [concrete]
// CHECK:STDOUT:       %rewrite.loc16_35.2 = requirement_rewrite %impl.elem1.loc16_32.2, %impl.elem0.subst.loc16_37.2 [concrete]
// CHECK:STDOUT:       %rewrite.loc16_47.2 = requirement_rewrite %impl.elem2.loc16_44.2, %.loc16_50.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.<error>.impl: %.loc16_7.2 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
