// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_cycle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_cycle.carbon

// --- core.carbon
package Core;

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

// --- fail_impl_simple_cycle.carbon
library "[[@TEST_NAME]]";

interface Z {}

// This creates a dependency cycle with itself.
impl forall [T:! Z] T as Z {}

class Point {
  impl as Z {}
}

fn F() {
  // CHECK:STDERR: fail_impl_simple_cycle.carbon:[[@LINE+7]]:21: error: cycle found in lookup of interface <TODO: interface name> for type `Point` [ImplLookupCycle]
  // CHECK:STDERR:   ({} as Point) as (Point as Z);
  // CHECK:STDERR:                     ^~~~~~~~~~
  // CHECK:STDERR: fail_impl_simple_cycle.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: impl forall [T:! Z] T as Z {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as Point) as (Point as Z);
}

// --- fail_impl_cycle_one_generic_param.carbon
library "[[@TEST_NAME]]";

import Core;

interface ComparableWith(T:! type) {}

// This creates a dependency cycle with itself.
impl forall [U:! type, T:! ComparableWith(U)]
    U as ComparableWith(T) {}

class C {}
class D {}
impl C as ComparableWith(D) {}

fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}

fn F() {
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE+20]]:3: error: cycle found in lookup of interface <TODO: interface name> for type `C` [ImplLookupCycle]
  // CHECK:STDERR:   Compare({} as C, {} as C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE-13]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: impl forall [U:! type, T:! ComparableWith(U)]
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE+10]]:3: error: cannot implicitly convert from `type` to `ComparableWith(C)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   Compare({} as C, {} as C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(ComparableWith(C))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Compare({} as C, {} as C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE-19]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Compare({} as C, {} as C);

  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE+13]]:3: error: cycle found in lookup of interface <TODO: interface name> for type `D` [ImplLookupCycle]
  // CHECK:STDERR:   Compare({} as C, {} as D);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: impl forall [U:! type, T:! ComparableWith(U)]
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE-38]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: impl forall [U:! type, T:! ComparableWith(U)]
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_cycle_one_generic_param.carbon:[[@LINE-34]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Compare({} as C, {} as D);
}

// --- impl_recurse_with_simpler_type_no_cycle.carbon
library "[[@TEST_NAME]]";

import Core;

class Wraps(T:! type) {}

interface Printable {}

// If T is printable, so is Wraps(T).
impl forall [T:! Printable] Wraps(T) as Printable {}

class C {}
impl C as Printable {}

fn F() {
  ({} as C) as (C as Printable);
  ({} as Wraps(C)) as (Wraps(C) as Printable);
}

// --- impl_recurse_with_simpler_type_in_generic_param_no_cycle.carbon
library "[[@TEST_NAME]]";

import Core;

class Wraps(T:! type) {}

interface ComparableTo(T:! type) {}

// If U is comparable to T, then U is comparable to Wraps(T).
impl forall [T:! type, U:! ComparableTo(T)] U as ComparableTo(Wraps(T)) {}
// If U is comparable to T then Wraps(U) is comparable to T.
impl forall [T:! type, U:! ComparableTo(T)] Wraps(U) as ComparableTo(T) {}

class C {}
class D {}
impl C as ComparableTo(D) {}

fn F() {
  ({} as C) as (C as ComparableTo(D));
  ({} as C) as (C as ComparableTo(Wraps(D)));
  ({} as Wraps(C)) as (Wraps(C) as ComparableTo(D));
  ({} as Wraps(C)) as (Wraps(C) as ComparableTo(Wraps(D)));
}

interface ComparableWith(T:! type) {}

impl forall [U:! type] U as ComparableWith(U) {}
impl forall [U:! type, T:! ComparableTo(U)] U as ComparableWith(T) {}
impl forall [U:! type, T:! ComparableTo(U)] T as ComparableWith(U) {}

fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}

fn G() {
  Compare({} as C, {} as C);
  Compare({} as C, {} as D);
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_no_cycle.carbon:[[@LINE+13]]:3: error: cannot implicitly convert from `type` to `ComparableTo(C)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   Compare({} as D, {} as C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_no_cycle.carbon:[[@LINE+10]]:3: note: type `type` does not implement interface `Core.ImplicitAs(ComparableTo(C))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Compare({} as D, {} as C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_no_cycle.carbon:[[@LINE-14]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: impl forall [U:! type, T:! ComparableTo(U)] U as ComparableWith(T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_no_cycle.carbon:[[@LINE-14]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Compare({} as D, {} as C);
}

// --- fail_todo_impl_recurse_with_simpler_type_in_generic_param_bidirectional_no_cycle.carbon
library "[[@TEST_NAME]]";

import Core;

// Implement this for a type in one direction.
interface ComparableTo(T:! type) {}
// Use this as a bound with two types in any direction.
interface ComparableWith(T:! type) {}

class C {}
class D {}
// C is comparable to D. This should imply:
// - C is ComparableWith(C).
// - C is ComparableWith(D).
// - D is ComparableWith(C).
impl C as ComparableTo(D) {}

// T is always ComparableWith(T).
impl forall [T:! type] T as ComparableWith(T) {}
// If U is ComparableTo(T), U is ComparableWith(T).
//
// TODO: When we look for the impl on `D` of `ComparableTo(C)` here, we find
// that it's not. We then go to the next impl which does match, but deduction
// generates an error here regardless.
impl forall [T:! type, U:! ComparableTo(T)] U as ComparableWith(T) {}
// If U is ComparableTo(T), T is ComparableWith(U).
impl forall [T:! type, U:! ComparableTo(T)] T as ComparableWith(U) {}

fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}

fn F() {
  Compare({} as C, {} as C);
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_bidirectional_no_cycle.carbon:[[@LINE+13]]:3: error: cannot implicitly convert from `type` to `ComparableTo(C)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   Compare({} as C, {} as D);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_bidirectional_no_cycle.carbon:[[@LINE+10]]:3: note: type `type` does not implement interface `Core.ImplicitAs(ComparableTo(C))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Compare({} as C, {} as D);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_bidirectional_no_cycle.carbon:[[@LINE-14]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: impl forall [T:! type, U:! ComparableTo(T)] U as ComparableWith(T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: impl_recurse_with_simpler_type_in_generic_param_bidirectional_no_cycle.carbon:[[@LINE-13]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Compare[T:! type, U:! ComparableWith(T)](t: T, u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Compare({} as C, {} as D);
  Compare({} as D, {} as C);
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [template]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [template = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc3_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:     %Dest.param_patt: type = value_param_pattern %Dest.patt.loc3_14.1, runtime_param<none> [symbolic = %Dest.patt.loc3_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %Dest.loc3_14.1: type = bind_symbolic_name Dest, 0, %Dest.param [symbolic = %Dest.loc3_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [template = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc7_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc7_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:     %Dest.param_patt: type = value_param_pattern %Dest.patt.loc7_22.1, runtime_param<none> [symbolic = %Dest.patt.loc7_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %Dest.loc7_22.1: type = bind_symbolic_name Dest, 0, %Dest.param [symbolic = %Dest.loc7_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc3_14.1: type) {
// CHECK:STDOUT:   %Dest.loc3_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc3_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc3_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc3_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc3_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = value_param runtime_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc4_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc4_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type.loc4_20.2 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc7_22.1: type) {
// CHECK:STDOUT:   %Dest.loc7_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc7_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc7_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc7_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc7_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc7_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc8_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc8_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc7_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = value_param runtime_param0
// CHECK:STDOUT:       %.loc8_20.1: type = splice_block %.loc8_20.3 [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc8_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc8_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc8_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc8_20.3: type = converted %Self.ref, %Self.as_type.loc8_20.2 [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc8_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc8_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc3_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc4_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc7_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc8_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc3_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc4_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc3_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc7_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc7_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc8_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc7_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_simple_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [template]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Z.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.8b4: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [template]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [template]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, %impl_witness.1bc [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Point = %Point.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [template = constants.%Z.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc6_14.1: %Z.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %Z.type = value_param_pattern %T.patt.loc6_14.1, runtime_param<none> [symbolic = %T.patt.loc6_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Z.type = name_ref T, %T.loc6_14.1 [symbolic = %T.loc6_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc6_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6: type = converted %T.ref, %T.as_type.loc6_21.1 [symbolic = %T.as_type.loc6_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %Z.ref.loc6_26: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:     %T.param: %Z.type = value_param runtime_param<none>
// CHECK:STDOUT:     %Z.ref.loc6_18: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:     %T.loc6_14.1: %Z.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(constants.%T) [symbolic = @impl.1.%impl_witness (constants.%impl_witness.8b4)]
// CHECK:STDOUT:   %Point.decl: type = class_decl @Point [template = constants.%Point] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%T.loc6_14.1: %Z.type) {
// CHECK:STDOUT:   %T.loc6_14.2: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc6_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_14.2: %Z.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc6_21.2: type = facet_access_type %T.loc6_14.2 [symbolic = %T.as_type.loc6_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T.loc6_14.2) [symbolic = %impl_witness (constants.%impl_witness.8b4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc6 as %Z.ref.loc6_26 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %Z.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @Point.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [template = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [template = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Point.ref.loc20_10: type = name_ref Point, file.%Point.decl [template = constants.%Point]
// CHECK:STDOUT:   %.loc20_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc20_5.3: init %Point = class_init (), %.loc20_5.2 [template = constants.%Point.val]
// CHECK:STDOUT:   %.loc20_5.4: ref %Point = temporary %.loc20_5.2, %.loc20_5.3
// CHECK:STDOUT:   %.loc20_7: ref %Point = converted %.loc20_5.1, %.loc20_5.4
// CHECK:STDOUT:   %Point.ref.loc20_21: type = name_ref Point, file.%Point.decl [template = constants.%Point]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:   %Z.facet.loc20_27.1: %Z.type = facet_value constants.%Point, <error> [template = <error>]
// CHECK:STDOUT:   %.loc20_27.1: %Z.type = converted constants.%Point, %Z.facet.loc20_27.1 [template = <error>]
// CHECK:STDOUT:   %Z.facet.loc20_27.2: %Z.type = facet_value %Point.ref.loc20_21, constants.%impl_witness.1bc [template = constants.%Z.facet]
// CHECK:STDOUT:   %.loc20_27.2: %Z.type = converted %Point.ref.loc20_21, %Z.facet.loc20_27.2 [template = constants.%Z.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc20_27.2 [template = constants.%Point]
// CHECK:STDOUT:   %.loc20_31: type = converted %.loc20_27.2, %as_type [template = constants.%Point]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T.loc6_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_14.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc6_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8b4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T.loc6_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(<error>) {
// CHECK:STDOUT:   %T.loc6_14.2 => <error>
// CHECK:STDOUT:   %T.patt.loc6_14.2 => <error>
// CHECK:STDOUT:   %T.as_type.loc6_21.2 => <error>
// CHECK:STDOUT:   %impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_cycle_one_generic_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.dbb: type = generic_interface_type @ComparableWith [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %ComparableWith.generic: %ComparableWith.type.dbb = struct_value () [template]
// CHECK:STDOUT:   %ComparableWith.type.bb0721.1: type = facet_type <@ComparableWith, @ComparableWith(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.d53: %ComparableWith.type.bb0721.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.e01: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.bb0721.2: type = facet_type <@ComparableWith, @ComparableWith(%U.8b3)> [symbolic]
// CHECK:STDOUT:   %T.d53: %ComparableWith.type.bb0721.2 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.761: %ComparableWith.type.bb0721.2 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.d53 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.d88: type = facet_type <@ComparableWith, @ComparableWith(%T.as_type)> [symbolic]
// CHECK:STDOUT:   %require_complete.958: <witness> = require_complete_type %ComparableWith.type.d88 [symbolic]
// CHECK:STDOUT:   %impl_witness.cb0: <witness> = impl_witness (), @impl.1(%U.8b3, %T.d53) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %ComparableWith.type.c33: type = facet_type <@ComparableWith, @ComparableWith(%D)> [template]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %U.d53: %ComparableWith.type.bb0721.1 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.761: %ComparableWith.type.bb0721.1 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U.d53 [symbolic]
// CHECK:STDOUT:   %Compare.type: type = fn_type @Compare [template]
// CHECK:STDOUT:   %Compare: %Compare.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %require_complete.9df: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ComparableWith.type.68c: type = facet_type <@ComparableWith, @ComparableWith(%C)> [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7b3: type = facet_type <@ImplicitAs, @ImplicitAs(%ComparableWith.type.68c)> [template]
// CHECK:STDOUT:   %Convert.type.862: type = fn_type @Convert, @ImplicitAs(%ComparableWith.type.68c) [template]
// CHECK:STDOUT:   %Convert.b45: %Convert.type.862 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.468: type = assoc_entity_type %ImplicitAs.type.7b3 [template]
// CHECK:STDOUT:   %assoc0.c98: %ImplicitAs.assoc_type.468 = assoc_entity element0, imports.%Core.import_ref.207961.1 [template]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %D.val: %D = struct_value () [template]
// CHECK:STDOUT:   %ComparableWith.facet.3a4: %ComparableWith.type.c33 = facet_value %C, %impl_witness.1bc [template]
// CHECK:STDOUT:   %complete_type.aa9: <witness> = complete_type_witness %ComparableWith.type.68c [template]
// CHECK:STDOUT:   %impl_witness.147: <witness> = impl_witness (), @impl.1(%D, %ComparableWith.facet.3a4) [template]
// CHECK:STDOUT:   %ComparableWith.facet.c23: %ComparableWith.type.68c = facet_value %D, %impl_witness.147 [template]
// CHECK:STDOUT:   %Compare.specific_fn: <specific function> = specific_function %Compare, @Compare(%C, %ComparableWith.facet.c23) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ffd566.1: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst69 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc8_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.ffd566.2: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst69 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc8_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ComparableWith = %ComparableWith.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Compare = %Compare.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ComparableWith.decl: %ComparableWith.type.dbb = interface_decl @ComparableWith [template = constants.%ComparableWith.generic] {
// CHECK:STDOUT:     %T.patt.loc5_26.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_26.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_26.1, runtime_param<none> [symbolic = %T.patt.loc5_26.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc5_26.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc5_26.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %U.patt.loc8_14.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc8_14.1, runtime_param<none> [symbolic = %U.patt.loc8_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %T.patt.loc8_24.1: @impl.1.%ComparableWith.type.loc8_44.2 (%ComparableWith.type.bb0721.2) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc8_24.2 (constants.%T.patt.761)]
// CHECK:STDOUT:     %T.param_patt: @impl.1.%ComparableWith.type.loc8_44.2 (%ComparableWith.type.bb0721.2) = value_param_pattern %T.patt.loc8_24.1, runtime_param<none> [symbolic = %T.patt.loc8_24.2 (constants.%T.patt.761)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc9: type = name_ref U, %U.loc8_14.1 [symbolic = %U.loc8_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ComparableWith.ref.loc9: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %T.ref: @impl.1.%ComparableWith.type.loc8_44.2 (%ComparableWith.type.bb0721.2) = name_ref T, %T.loc8_24.1 [symbolic = %T.loc8_24.2 (constants.%T.d53)]
// CHECK:STDOUT:     %T.as_type.loc9_26.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %T.ref, %T.as_type.loc9_26.1 [symbolic = %T.as_type.loc9_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %ComparableWith.type.loc9_26.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%T.as_type)> [symbolic = %ComparableWith.type.loc9_26.2 (constants.%ComparableWith.type.d88)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %U.loc8_14.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc8_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %T.param: @impl.1.%ComparableWith.type.loc8_44.2 (%ComparableWith.type.bb0721.2) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc8: type = splice_block %ComparableWith.type.loc8_44.1 [symbolic = %ComparableWith.type.loc8_44.2 (constants.%ComparableWith.type.bb0721.2)] {
// CHECK:STDOUT:       %ComparableWith.ref.loc8: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:       %U.ref.loc8: type = name_ref U, %U.loc8_14.1 [symbolic = %U.loc8_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:       %ComparableWith.type.loc8_44.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%U.8b3)> [symbolic = %ComparableWith.type.loc8_44.2 (constants.%ComparableWith.type.bb0721.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_24.1: @impl.1.%ComparableWith.type.loc8_44.2 (%ComparableWith.type.bb0721.2) = bind_symbolic_name T, 1, %T.param [symbolic = %T.loc8_24.2 (constants.%T.d53)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc9: <witness> = impl_witness (), @impl.1(constants.%U.8b3, constants.%T.d53) [symbolic = @impl.1.%impl_witness (constants.%impl_witness.cb0)]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ComparableWith.type: type = facet_type <@ComparableWith, @ComparableWith(constants.%D)> [template = constants.%ComparableWith.type.c33]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness () [template = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %Compare.decl: %Compare.type = fn_decl @Compare [template = constants.%Compare] {
// CHECK:STDOUT:     %T.patt.loc15_12.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_12.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc15_12.1, runtime_param<none> [symbolic = %T.patt.loc15_12.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %U.patt.loc15_22.1: @Compare.%ComparableWith.type.loc15_42.2 (%ComparableWith.type.bb0721.1) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:     %U.param_patt: @Compare.%ComparableWith.type.loc15_42.2 (%ComparableWith.type.bb0721.1) = value_param_pattern %U.patt.loc15_22.1, runtime_param<none> [symbolic = %U.patt.loc15_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:     %t.patt: @Compare.%T.loc15_12.2 (%T.8b3) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @Compare.%T.loc15_12.2 (%T.8b3) = value_param_pattern %t.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: @Compare.%U.as_type.loc15_54.2 (%U.as_type) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @Compare.%U.as_type.loc15_54.2 (%U.as_type) = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc15_12.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc15_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %U.param: @Compare.%ComparableWith.type.loc15_42.2 (%ComparableWith.type.bb0721.1) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc15_42: type = splice_block %ComparableWith.type.loc15_42.1 [symbolic = %ComparableWith.type.loc15_42.2 (constants.%ComparableWith.type.bb0721.1)] {
// CHECK:STDOUT:       %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:       %T.ref.loc15_41: type = name_ref T, %T.loc15_12.1 [symbolic = %T.loc15_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:       %ComparableWith.type.loc15_42.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%T.8b3)> [symbolic = %ComparableWith.type.loc15_42.2 (constants.%ComparableWith.type.bb0721.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc15_22.1: @Compare.%ComparableWith.type.loc15_42.2 (%ComparableWith.type.bb0721.1) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc15_22.2 (constants.%U.d53)]
// CHECK:STDOUT:     %t.param: @Compare.%T.loc15_12.2 (%T.8b3) = value_param runtime_param0
// CHECK:STDOUT:     %T.ref.loc15_48: type = name_ref T, %T.loc15_12.1 [symbolic = %T.loc15_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %t: @Compare.%T.loc15_12.2 (%T.8b3) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @Compare.%U.as_type.loc15_54.2 (%U.as_type) = value_param runtime_param1
// CHECK:STDOUT:     %.loc15_54.1: type = splice_block %.loc15_54.2 [symbolic = %U.as_type.loc15_54.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: @Compare.%ComparableWith.type.loc15_42.2 (%ComparableWith.type.bb0721.1) = name_ref U, %U.loc15_22.1 [symbolic = %U.loc15_22.2 (constants.%U.d53)]
// CHECK:STDOUT:       %U.as_type.loc15_54.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc15_54.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc15_54.2: type = converted %U.ref, %U.as_type.loc15_54.1 [symbolic = %U.as_type.loc15_54.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Compare.%U.as_type.loc15_54.2 (%U.as_type) = bind_name u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ComparableWith(%T.loc5_26.1: type) {
// CHECK:STDOUT:   %T.loc5_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_26.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc5_26.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_26.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type: type = facet_type <@ComparableWith, @ComparableWith(%T.loc5_26.2)> [symbolic = %ComparableWith.type (constants.%ComparableWith.type.bb0721.1)]
// CHECK:STDOUT:   %Self.2: %ComparableWith.type.bb0721.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.d53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ComparableWith.%ComparableWith.type (%ComparableWith.type.bb0721.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.d53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.ffd566.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%U.loc8_14.1: type, %T.loc8_24.1: @impl.1.%ComparableWith.type.loc8_44.2 (%ComparableWith.type.bb0721.2)) {
// CHECK:STDOUT:   %U.loc8_14.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %U.patt.loc8_14.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   %ComparableWith.type.loc8_44.2: type = facet_type <@ComparableWith, @ComparableWith(%U.loc8_14.2)> [symbolic = %ComparableWith.type.loc8_44.2 (constants.%ComparableWith.type.bb0721.2)]
// CHECK:STDOUT:   %T.loc8_24.2: %ComparableWith.type.bb0721.2 = bind_symbolic_name T, 1 [symbolic = %T.loc8_24.2 (constants.%T.d53)]
// CHECK:STDOUT:   %T.patt.loc8_24.2: %ComparableWith.type.bb0721.2 = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc8_24.2 (constants.%T.patt.761)]
// CHECK:STDOUT:   %T.as_type.loc9_26.2: type = facet_access_type %T.loc8_24.2 [symbolic = %T.as_type.loc9_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %ComparableWith.type.loc9_26.2: type = facet_type <@ComparableWith, @ComparableWith(%T.as_type.loc9_26.2)> [symbolic = %ComparableWith.type.loc9_26.2 (constants.%ComparableWith.type.d88)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%ComparableWith.type.loc9_26.2 (%ComparableWith.type.d88) [symbolic = %require_complete (constants.%require_complete.958)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%U.loc8_14.2, %T.loc8_24.2) [symbolic = %impl_witness (constants.%impl_witness.cb0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc9 as %ComparableWith.type.loc9_26.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C.ref as %ComparableWith.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compare(%T.loc15_12.1: type, %U.loc15_22.1: @Compare.%ComparableWith.type.loc15_42.2 (%ComparableWith.type.bb0721.1)) {
// CHECK:STDOUT:   %T.loc15_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc15_12.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_12.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %ComparableWith.type.loc15_42.2: type = facet_type <@ComparableWith, @ComparableWith(%T.loc15_12.2)> [symbolic = %ComparableWith.type.loc15_42.2 (constants.%ComparableWith.type.bb0721.1)]
// CHECK:STDOUT:   %U.loc15_22.2: %ComparableWith.type.bb0721.1 = bind_symbolic_name U, 1 [symbolic = %U.loc15_22.2 (constants.%U.d53)]
// CHECK:STDOUT:   %U.patt.loc15_22.2: %ComparableWith.type.bb0721.1 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc15_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:   %U.as_type.loc15_54.2: type = facet_access_type %U.loc15_22.2 [symbolic = %U.as_type.loc15_54.2 (constants.%U.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_46: <witness> = require_complete_type @Compare.%T.loc15_12.2 (%T.8b3) [symbolic = %require_complete.loc15_46 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc15_52: <witness> = require_complete_type @Compare.%U.as_type.loc15_54.2 (%U.as_type) [symbolic = %require_complete.loc15_52 (constants.%require_complete.9df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %U.param_patt: @Compare.%ComparableWith.type.loc15_42.2 (%ComparableWith.type.bb0721.1)](%t.param_patt: @Compare.%T.loc15_12.2 (%T.8b3), %u.param_patt: @Compare.%U.as_type.loc15_54.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Compare.ref.loc38: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc38_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc38_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc38_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc38_12.3: init %C = class_init (), %.loc38_12.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc38_12.4: ref %C = temporary %.loc38_12.2, %.loc38_12.3
// CHECK:STDOUT:   %.loc38_14.1: ref %C = converted %.loc38_12.1, %.loc38_12.4
// CHECK:STDOUT:   %.loc38_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc38_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc38_21.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc38_21.3: init %C = class_init (), %.loc38_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc38_21.4: ref %C = temporary %.loc38_21.2, %.loc38_21.3
// CHECK:STDOUT:   %.loc38_23: ref %C = converted %.loc38_21.1, %.loc38_21.4
// CHECK:STDOUT:   %ComparableWith.facet.loc38: %ComparableWith.type.68c = facet_value constants.%C, <error> [template = <error>]
// CHECK:STDOUT:   %.loc38_27.1: %ComparableWith.type.68c = converted constants.%C, %ComparableWith.facet.loc38 [template = <error>]
// CHECK:STDOUT:   %.loc38_27.2: %ComparableWith.type.68c = converted constants.%C, <error> [template = <error>]
// CHECK:STDOUT:   %Compare.specific_fn.loc38: <specific function> = specific_function %Compare.ref.loc38, @Compare(constants.%C, <error>) [template = <error>]
// CHECK:STDOUT:   %.loc38_14.2: %C = bind_value %.loc38_14.1
// CHECK:STDOUT:   %Compare.call.loc38: init %empty_tuple.type = call %Compare.specific_fn.loc38(%.loc38_14.2, <error>)
// CHECK:STDOUT:   %Compare.ref.loc53: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc53_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc53: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc53_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc53_12.3: init %C = class_init (), %.loc53_12.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc53_12.4: ref %C = temporary %.loc53_12.2, %.loc53_12.3
// CHECK:STDOUT:   %.loc53_14.1: ref %C = converted %.loc53_12.1, %.loc53_12.4
// CHECK:STDOUT:   %.loc53_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc53_21.2: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc53_21.3: init %D = class_init (), %.loc53_21.2 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc53_21.4: ref %D = temporary %.loc53_21.2, %.loc53_21.3
// CHECK:STDOUT:   %.loc53_23.1: ref %D = converted %.loc53_21.1, %.loc53_21.4
// CHECK:STDOUT:   %ComparableWith.facet.loc53_27.1: %ComparableWith.type.68c = facet_value constants.%D, <error> [template = <error>]
// CHECK:STDOUT:   %.loc53_27.1: %ComparableWith.type.68c = converted constants.%D, %ComparableWith.facet.loc53_27.1 [template = <error>]
// CHECK:STDOUT:   %ComparableWith.facet.loc53_27.2: %ComparableWith.type.c33 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableWith.facet.3a4]
// CHECK:STDOUT:   %.loc53_27.2: %ComparableWith.type.c33 = converted constants.%C, %ComparableWith.facet.loc53_27.2 [template = constants.%ComparableWith.facet.3a4]
// CHECK:STDOUT:   %ComparableWith.facet.loc53_27.3: %ComparableWith.type.68c = facet_value constants.%D, constants.%impl_witness.147 [template = constants.%ComparableWith.facet.c23]
// CHECK:STDOUT:   %.loc53_27.3: %ComparableWith.type.68c = converted constants.%D, %ComparableWith.facet.loc53_27.3 [template = constants.%ComparableWith.facet.c23]
// CHECK:STDOUT:   %Compare.specific_fn.loc53: <specific function> = specific_function %Compare.ref.loc53, @Compare(constants.%C, %.loc53_27.3) [template = constants.%Compare.specific_fn]
// CHECK:STDOUT:   %.loc53_14.2: %C = bind_value %.loc53_14.1
// CHECK:STDOUT:   %.loc53_23.2: %D = bind_value %.loc53_23.1
// CHECK:STDOUT:   %Compare.call.loc53: init %empty_tuple.type = call %Compare.specific_fn.loc53(%.loc53_14.2, %.loc53_23.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.ffd566.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_26.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc5_26.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(%T.loc5_26.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%U.8b3) {
// CHECK:STDOUT:   %T.loc5_26.2 => constants.%U.8b3
// CHECK:STDOUT:   %T.patt.loc5_26.2 => constants.%U.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc5_26.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.patt.loc5_26.2 => constants.%T.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type => constants.%ComparableWith.type.d88
// CHECK:STDOUT:   %Self.2 => constants.%Self.d53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%U.8b3, constants.%T.d53) {
// CHECK:STDOUT:   %U.loc8_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %ComparableWith.type.loc8_44.2 => constants.%ComparableWith.type.bb0721.2
// CHECK:STDOUT:   %T.loc8_24.2 => constants.%T.d53
// CHECK:STDOUT:   %T.patt.loc8_24.2 => constants.%T.d53
// CHECK:STDOUT:   %T.as_type.loc9_26.2 => constants.%T.as_type
// CHECK:STDOUT:   %ComparableWith.type.loc9_26.2 => constants.%ComparableWith.type.d88
// CHECK:STDOUT:   %require_complete => constants.%require_complete.958
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.cb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.1.%U.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.1.%T.as_type.loc9_26.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%U.loc8_14.2, %T.loc8_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%D) {
// CHECK:STDOUT:   %T.loc5_26.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc5_26.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %Self.2 => constants.%Self.d53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%T.8b3, constants.%U.d53) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc15_12.2 => constants.%T.8b3
// CHECK:STDOUT:   %ComparableWith.type.loc15_42.2 => constants.%ComparableWith.type.bb0721.1
// CHECK:STDOUT:   %U.loc15_22.2 => constants.%U.d53
// CHECK:STDOUT:   %U.patt.loc15_22.2 => constants.%U.d53
// CHECK:STDOUT:   %U.as_type.loc15_54.2 => constants.%U.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@Compare.%T.loc15_12.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%C) {
// CHECK:STDOUT:   %T.loc5_26.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc5_26.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%C, @F.%.loc38_27.1) {
// CHECK:STDOUT:   %U.loc8_14.2 => constants.%C
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc8_44.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %T.loc8_24.2 => <error>
// CHECK:STDOUT:   %T.patt.loc8_24.2 => <error>
// CHECK:STDOUT:   %T.as_type.loc9_26.2 => <error>
// CHECK:STDOUT:   %ComparableWith.type.loc9_26.2 => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(<error>) {
// CHECK:STDOUT:   %T.loc5_26.2 => <error>
// CHECK:STDOUT:   %T.patt.loc5_26.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%C, <error>) {
// CHECK:STDOUT:   %U.loc8_14.2 => constants.%C
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc8_44.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %T.loc8_24.2 => <error>
// CHECK:STDOUT:   %T.patt.loc8_24.2 => <error>
// CHECK:STDOUT:   %T.as_type.loc9_26.2 => <error>
// CHECK:STDOUT:   %ComparableWith.type.loc9_26.2 => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%ComparableWith.type.68c) {
// CHECK:STDOUT:   %Dest => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %Dest.patt => constants.%ComparableWith.type.68c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7b3
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.862
// CHECK:STDOUT:   %Convert => constants.%Convert.b45
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.468
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, <error>) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc15_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc15_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc15_22.2 => <error>
// CHECK:STDOUT:   %U.patt.loc15_22.2 => <error>
// CHECK:STDOUT:   %U.as_type.loc15_54.2 => <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_52 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%C, @F.%.loc53_27.1) {
// CHECK:STDOUT:   %U.loc8_14.2 => constants.%C
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc8_44.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %T.loc8_24.2 => <error>
// CHECK:STDOUT:   %T.patt.loc8_24.2 => <error>
// CHECK:STDOUT:   %T.as_type.loc9_26.2 => <error>
// CHECK:STDOUT:   %ComparableWith.type.loc9_26.2 => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%D, @F.%.loc53_27.2) {
// CHECK:STDOUT:   %U.loc8_14.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc8_44.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %T.loc8_24.2 => constants.%ComparableWith.facet.3a4
// CHECK:STDOUT:   %T.patt.loc8_24.2 => constants.%ComparableWith.facet.3a4
// CHECK:STDOUT:   %T.as_type.loc9_26.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc9_26.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.147
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%D, constants.%ComparableWith.facet.3a4) {
// CHECK:STDOUT:   %U.loc8_14.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc8_44.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %T.loc8_24.2 => constants.%ComparableWith.facet.3a4
// CHECK:STDOUT:   %T.patt.loc8_24.2 => constants.%ComparableWith.facet.3a4
// CHECK:STDOUT:   %T.as_type.loc9_26.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc9_26.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, @F.%.loc53_27.3) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc15_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc15_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc15_22.2 => constants.%ComparableWith.facet.c23
// CHECK:STDOUT:   %U.patt.loc15_22.2 => constants.%ComparableWith.facet.c23
// CHECK:STDOUT:   %U.as_type.loc15_54.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_52 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, constants.%ComparableWith.facet.c23) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc15_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc15_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc15_22.2 => constants.%ComparableWith.facet.c23
// CHECK:STDOUT:   %U.patt.loc15_22.2 => constants.%ComparableWith.facet.c23
// CHECK:STDOUT:   %U.as_type.loc15_54.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_recurse_with_simpler_type_no_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Wraps.type: type = generic_class_type @Wraps [template]
// CHECK:STDOUT:   %Wraps.generic: %Wraps.type = struct_value () [template]
// CHECK:STDOUT:   %Wraps.487: type = class_type @Wraps, @Wraps(%T.8b3) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Printable.type: type = facet_type <@Printable> [template]
// CHECK:STDOUT:   %Self: %Printable.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.90e: %Printable.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.734: %Printable.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.90e [symbolic]
// CHECK:STDOUT:   %Wraps.227: type = class_type @Wraps, @Wraps(%T.as_type) [symbolic]
// CHECK:STDOUT:   %impl_witness.b15: <witness> = impl_witness (), @impl.1(%T.90e) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %Printable.facet.5a6: %Printable.type = facet_value %C, %impl_witness.1bc [template]
// CHECK:STDOUT:   %Wraps.299: type = class_type @Wraps, @Wraps(%C) [template]
// CHECK:STDOUT:   %Wraps.val: %Wraps.299 = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.bb5: <witness> = impl_witness (), @impl.1(%Printable.facet.5a6) [template]
// CHECK:STDOUT:   %Printable.facet.d41: %Printable.type = facet_value %Wraps.299, %impl_witness.bb5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Wraps = %Wraps.decl
// CHECK:STDOUT:     .Printable = %Printable.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Wraps.decl: %Wraps.type = class_decl @Wraps [template = constants.%Wraps.generic] {
// CHECK:STDOUT:     %T.patt.loc5_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_13.1, runtime_param<none> [symbolic = %T.patt.loc5_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Printable.decl: type = interface_decl @Printable [template = constants.%Printable.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc10_14.1: %Printable.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.734)]
// CHECK:STDOUT:     %T.param_patt: %Printable.type = value_param_pattern %T.patt.loc10_14.1, runtime_param<none> [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.734)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Wraps.ref: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:     %T.ref: %Printable.type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.90e)]
// CHECK:STDOUT:     %T.as_type.loc10_36.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_36.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10: type = converted %T.ref, %T.as_type.loc10_36.1 [symbolic = %T.as_type.loc10_36.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %Wraps.loc10_36.1: type = class_type @Wraps, @Wraps(constants.%T.as_type) [symbolic = %Wraps.loc10_36.2 (constants.%Wraps.227)]
// CHECK:STDOUT:     %Printable.ref.loc10_41: type = name_ref Printable, file.%Printable.decl [template = constants.%Printable.type]
// CHECK:STDOUT:     %T.param: %Printable.type = value_param runtime_param<none>
// CHECK:STDOUT:     %Printable.ref.loc10_18: type = name_ref Printable, file.%Printable.decl [template = constants.%Printable.type]
// CHECK:STDOUT:     %T.loc10_14.1: %Printable.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_14.2 (constants.%T.90e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (), @impl.1(constants.%T.90e) [symbolic = @impl.1.%impl_witness (constants.%impl_witness.b15)]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Printable.ref: type = name_ref Printable, file.%Printable.decl [template = constants.%Printable.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness () [template = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Printable {
// CHECK:STDOUT:   %Self: %Printable.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%T.loc10_14.1: %Printable.type) {
// CHECK:STDOUT:   %T.loc10_14.2: %Printable.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.90e)]
// CHECK:STDOUT:   %T.patt.loc10_14.2: %Printable.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.734)]
// CHECK:STDOUT:   %T.as_type.loc10_36.2: type = facet_access_type %T.loc10_14.2 [symbolic = %T.as_type.loc10_36.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %Wraps.loc10_36.2: type = class_type @Wraps, @Wraps(%T.as_type.loc10_36.2) [symbolic = %Wraps.loc10_36.2 (constants.%Wraps.227)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T.loc10_14.2) [symbolic = %impl_witness (constants.%impl_witness.b15)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Wraps.loc10_36.1 as %Printable.ref.loc10_41 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C.ref as %Printable.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wraps(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wraps.487
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc16_10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc16_5.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc16_5.3: init %C = class_init (), %.loc16_5.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc16_5.4: ref %C = temporary %.loc16_5.2, %.loc16_5.3
// CHECK:STDOUT:   %.loc16_7: ref %C = converted %.loc16_5.1, %.loc16_5.4
// CHECK:STDOUT:   %C.ref.loc16_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Printable.ref.loc16: type = name_ref Printable, file.%Printable.decl [template = constants.%Printable.type]
// CHECK:STDOUT:   %Printable.facet.loc16: %Printable.type = facet_value %C.ref.loc16_17, constants.%impl_witness.1bc [template = constants.%Printable.facet.5a6]
// CHECK:STDOUT:   %.loc16_19: %Printable.type = converted %C.ref.loc16_17, %Printable.facet.loc16 [template = constants.%Printable.facet.5a6]
// CHECK:STDOUT:   %as_type.loc16: type = facet_access_type %.loc16_19 [template = constants.%C]
// CHECK:STDOUT:   %.loc16_31: type = converted %.loc16_19, %as_type.loc16 [template = constants.%C]
// CHECK:STDOUT:   %.loc17_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Wraps.ref.loc17_10: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %C.ref.loc17_16: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Wraps.loc17_17: type = class_type @Wraps, @Wraps(constants.%C) [template = constants.%Wraps.299]
// CHECK:STDOUT:   %.loc17_5.2: ref %Wraps.299 = temporary_storage
// CHECK:STDOUT:   %.loc17_5.3: init %Wraps.299 = class_init (), %.loc17_5.2 [template = constants.%Wraps.val]
// CHECK:STDOUT:   %.loc17_5.4: ref %Wraps.299 = temporary %.loc17_5.2, %.loc17_5.3
// CHECK:STDOUT:   %.loc17_7: ref %Wraps.299 = converted %.loc17_5.1, %.loc17_5.4
// CHECK:STDOUT:   %Wraps.ref.loc17_24: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %C.ref.loc17_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Wraps.loc17_31: type = class_type @Wraps, @Wraps(constants.%C) [template = constants.%Wraps.299]
// CHECK:STDOUT:   %Printable.ref.loc17: type = name_ref Printable, file.%Printable.decl [template = constants.%Printable.type]
// CHECK:STDOUT:   %Printable.facet.loc17_33.1: %Printable.type = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%Printable.facet.5a6]
// CHECK:STDOUT:   %.loc17_33.1: %Printable.type = converted constants.%C, %Printable.facet.loc17_33.1 [template = constants.%Printable.facet.5a6]
// CHECK:STDOUT:   %Printable.facet.loc17_33.2: %Printable.type = facet_value %Wraps.loc17_31, constants.%impl_witness.bb5 [template = constants.%Printable.facet.d41]
// CHECK:STDOUT:   %.loc17_33.2: %Printable.type = converted %Wraps.loc17_31, %Printable.facet.loc17_33.2 [template = constants.%Printable.facet.d41]
// CHECK:STDOUT:   %as_type.loc17: type = facet_access_type %.loc17_33.2 [template = constants.%Wraps.299]
// CHECK:STDOUT:   %.loc17_45: type = converted %.loc17_33.2, %as_type.loc17 [template = constants.%Wraps.299]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T.90e) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.90e
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T.90e
// CHECK:STDOUT:   %T.as_type.loc10_36.2 => constants.%T.as_type
// CHECK:STDOUT:   %Wraps.loc10_36.2 => constants.%Wraps.227
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(@impl.1.%T.as_type.loc10_36.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(constants.%C) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%Printable.facet.5a6) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%Printable.facet.5a6
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%Printable.facet.5a6
// CHECK:STDOUT:   %T.as_type.loc10_36.2 => constants.%C
// CHECK:STDOUT:   %Wraps.loc10_36.2 => constants.%Wraps.299
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.bb5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_recurse_with_simpler_type_in_generic_param_no_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Wraps.type: type = generic_class_type @Wraps [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Wraps.generic: %Wraps.type = struct_value () [template]
// CHECK:STDOUT:   %Wraps.487: type = class_type @Wraps, @Wraps(%T.8b3) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %ComparableTo.type.706: type = generic_interface_type @ComparableTo [template]
// CHECK:STDOUT:   %ComparableTo.generic: %ComparableTo.type.706 = struct_value () [template]
// CHECK:STDOUT:   %ComparableTo.type.4eeb0e.1: type = facet_type <@ComparableTo, @ComparableTo(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.b15: %ComparableTo.type.4eeb0e.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.b15: %ComparableTo.type.4eeb0e.1 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.16f: %ComparableTo.type.4eeb0e.1 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.b21: type = facet_access_type %U.b15 [symbolic]
// CHECK:STDOUT:   %ComparableTo.type.040: type = facet_type <@ComparableTo, @ComparableTo(%Wraps.487)> [symbolic]
// CHECK:STDOUT:   %require_complete.066: <witness> = require_complete_type %ComparableTo.type.040 [symbolic]
// CHECK:STDOUT:   %impl_witness.206: <witness> = impl_witness (), @impl.1(%T.8b3, %U.b15) [symbolic]
// CHECK:STDOUT:   %Wraps.b9a: type = class_type @Wraps, @Wraps(%U.as_type.b21) [symbolic]
// CHECK:STDOUT:   %require_complete.084: <witness> = require_complete_type %ComparableTo.type.4eeb0e.1 [symbolic]
// CHECK:STDOUT:   %impl_witness.96c: <witness> = impl_witness (), @impl.2(%T.8b3, %U.b15) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %ComparableTo.type.4f3: type = facet_type <@ComparableTo, @ComparableTo(%D)> [template]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ComparableTo.facet.3c8: %ComparableTo.type.4f3 = facet_value %C, %impl_witness.1bc [template]
// CHECK:STDOUT:   %Wraps.537: type = class_type @Wraps, @Wraps(%D) [template]
// CHECK:STDOUT:   %ComparableTo.type.237: type = facet_type <@ComparableTo, @ComparableTo(%Wraps.537)> [template]
// CHECK:STDOUT:   %complete_type.d0b: <witness> = complete_type_witness %ComparableTo.type.237 [template]
// CHECK:STDOUT:   %impl_witness.b06: <witness> = impl_witness (), @impl.1(%D, %ComparableTo.facet.3c8) [template]
// CHECK:STDOUT:   %ComparableTo.facet.26c: %ComparableTo.type.237 = facet_value %C, %impl_witness.b06 [template]
// CHECK:STDOUT:   %Wraps.299: type = class_type @Wraps, @Wraps(%C) [template]
// CHECK:STDOUT:   %Wraps.val: %Wraps.299 = struct_value () [template]
// CHECK:STDOUT:   %complete_type.b71: <witness> = complete_type_witness %ComparableTo.type.4f3 [template]
// CHECK:STDOUT:   %impl_witness.951: <witness> = impl_witness (), @impl.2(%D, %ComparableTo.facet.3c8) [template]
// CHECK:STDOUT:   %ComparableTo.facet.c6f: %ComparableTo.type.4f3 = facet_value %Wraps.299, %impl_witness.951 [template]
// CHECK:STDOUT:   %impl_witness.098: <witness> = impl_witness (), @impl.1(%D, %ComparableTo.facet.c6f) [template]
// CHECK:STDOUT:   %ComparableTo.facet.6db: %ComparableTo.type.237 = facet_value %Wraps.299, %impl_witness.098 [template]
// CHECK:STDOUT:   %ComparableWith.type.dbb: type = generic_interface_type @ComparableWith [template]
// CHECK:STDOUT:   %ComparableWith.generic: %ComparableWith.type.dbb = struct_value () [template]
// CHECK:STDOUT:   %ComparableWith.type.bb0721.1: type = facet_type <@ComparableWith, @ComparableWith(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.d53: %ComparableWith.type.bb0721.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.e01: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.bb0721.2: type = facet_type <@ComparableWith, @ComparableWith(%U.8b3)> [symbolic]
// CHECK:STDOUT:   %require_complete.17f: <witness> = require_complete_type %ComparableWith.type.bb0721.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.497: <witness> = impl_witness (), @impl.4(%U.8b3) [symbolic]
// CHECK:STDOUT:   %ComparableTo.type.4eeb0e.2: type = facet_type <@ComparableTo, @ComparableTo(%U.8b3)> [symbolic]
// CHECK:STDOUT:   %T.b15: %ComparableTo.type.4eeb0e.2 = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.16f: %ComparableTo.type.4eeb0e.2 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.b15 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.73a: type = facet_type <@ComparableWith, @ComparableWith(%T.as_type)> [symbolic]
// CHECK:STDOUT:   %require_complete.86a: <witness> = require_complete_type %ComparableWith.type.73a [symbolic]
// CHECK:STDOUT:   %impl_witness.595: <witness> = impl_witness (), @impl.5(%U.8b3, %T.b15) [symbolic]
// CHECK:STDOUT:   %impl_witness.c25: <witness> = impl_witness (), @impl.6(%U.8b3, %T.b15) [symbolic]
// CHECK:STDOUT:   %U.d53: %ComparableWith.type.bb0721.1 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.761: %ComparableWith.type.bb0721.1 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.2d1: type = facet_access_type %U.d53 [symbolic]
// CHECK:STDOUT:   %Compare.type: type = fn_type @Compare [template]
// CHECK:STDOUT:   %Compare: %Compare.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %require_complete.9df: <witness> = require_complete_type %U.as_type.2d1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %ComparableWith.type.68c: type = facet_type <@ComparableWith, @ComparableWith(%C)> [template]
// CHECK:STDOUT:   %complete_type.aa9: <witness> = complete_type_witness %ComparableWith.type.68c [template]
// CHECK:STDOUT:   %impl_witness.887: <witness> = impl_witness (), @impl.4(%C) [template]
// CHECK:STDOUT:   %ComparableWith.facet.17d: %ComparableWith.type.68c = facet_value %C, %impl_witness.887 [template]
// CHECK:STDOUT:   %Compare.specific_fn.041: <specific function> = specific_function %Compare, @Compare(%C, %ComparableWith.facet.17d) [template]
// CHECK:STDOUT:   %D.val: %D = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.228: <witness> = impl_witness (), @impl.5(%D, %ComparableTo.facet.3c8) [template]
// CHECK:STDOUT:   %ComparableWith.facet.aa3: %ComparableWith.type.68c = facet_value %D, %impl_witness.228 [template]
// CHECK:STDOUT:   %Compare.specific_fn.94d: <specific function> = specific_function %Compare, @Compare(%C, %ComparableWith.facet.aa3) [template]
// CHECK:STDOUT:   %ComparableWith.type.c33: type = facet_type <@ComparableWith, @ComparableWith(%D)> [template]
// CHECK:STDOUT:   %ComparableTo.type.406: type = facet_type <@ComparableTo, @ComparableTo(%C)> [template]
// CHECK:STDOUT:   %complete_type.9db: <witness> = complete_type_witness %ComparableWith.type.c33 [template]
// CHECK:STDOUT:   %impl_witness.126: <witness> = impl_witness (), @impl.4(%D) [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.632: type = facet_type <@ImplicitAs, @ImplicitAs(%ComparableTo.type.406)> [template]
// CHECK:STDOUT:   %Convert.type.cd0: type = fn_type @Convert, @ImplicitAs(%ComparableTo.type.406) [template]
// CHECK:STDOUT:   %Convert.088: %Convert.type.cd0 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.38f: type = assoc_entity_type %ImplicitAs.type.632 [template]
// CHECK:STDOUT:   %assoc0.dca: %ImplicitAs.assoc_type.38f = assoc_entity element0, imports.%Core.import_ref.207961.1 [template]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.7aa: type = facet_type <@ComparableWith, @ComparableWith(type)> [template]
// CHECK:STDOUT:   %complete_type.3dc: <witness> = complete_type_witness %ComparableWith.type.7aa [template]
// CHECK:STDOUT:   %impl_witness.2ae: <witness> = impl_witness (), @impl.4(type) [template]
// CHECK:STDOUT:   %impl_witness.d37: <witness> = impl_witness (), @impl.6(%D, %ComparableTo.facet.3c8) [template]
// CHECK:STDOUT:   %ComparableWith.facet.7c0: %ComparableWith.type.c33 = facet_value %C, %impl_witness.d37 [template]
// CHECK:STDOUT:   %Compare.specific_fn.f4d: <specific function> = specific_function %Compare, @Compare(%D, %ComparableWith.facet.7c0) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ffd566.1: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst69 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc8_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.ffd566.2: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst69 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc8_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Wraps = %Wraps.decl
// CHECK:STDOUT:     .ComparableTo = %ComparableTo.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .ComparableWith = %ComparableWith.decl
// CHECK:STDOUT:     .Compare = %Compare.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Wraps.decl: %Wraps.type = class_decl @Wraps [template = constants.%Wraps.generic] {
// CHECK:STDOUT:     %T.patt.loc5_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_13.1, runtime_param<none> [symbolic = %T.patt.loc5_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ComparableTo.decl: %ComparableTo.type.706 = interface_decl @ComparableTo [template = constants.%ComparableTo.generic] {
// CHECK:STDOUT:     %T.patt.loc7_24.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_24.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_24.1, runtime_param<none> [symbolic = %T.patt.loc7_24.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc7_24.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_24.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc10_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_14.1, runtime_param<none> [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %U.patt.loc10_24.1: @impl.1.%ComparableTo.type.loc10_42.2 (%ComparableTo.type.4eeb0e.1) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc10_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:     %U.param_patt: @impl.1.%ComparableTo.type.loc10_42.2 (%ComparableTo.type.4eeb0e.1) = value_param_pattern %U.patt.loc10_24.1, runtime_param<none> [symbolic = %U.patt.loc10_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: @impl.1.%ComparableTo.type.loc10_42.2 (%ComparableTo.type.4eeb0e.1) = name_ref U, %U.loc10_24.1 [symbolic = %U.loc10_24.2 (constants.%U.b15)]
// CHECK:STDOUT:     %U.as_type.loc10_45.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc10_45.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %.loc10_45: type = converted %U.ref, %U.as_type.loc10_45.1 [symbolic = %U.as_type.loc10_45.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %ComparableTo.ref.loc10_50: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:     %Wraps.ref: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:     %T.ref.loc10_69: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Wraps.loc10_70.1: type = class_type @Wraps, @Wraps(constants.%T.8b3) [symbolic = %Wraps.loc10_70.2 (constants.%Wraps.487)]
// CHECK:STDOUT:     %ComparableTo.type.loc10_71.1: type = facet_type <@ComparableTo, @ComparableTo(constants.%Wraps.487)> [symbolic = %ComparableTo.type.loc10_71.2 (constants.%ComparableTo.type.040)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %U.param: @impl.1.%ComparableTo.type.loc10_42.2 (%ComparableTo.type.4eeb0e.1) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc10_42: type = splice_block %ComparableTo.type.loc10_42.1 [symbolic = %ComparableTo.type.loc10_42.2 (constants.%ComparableTo.type.4eeb0e.1)] {
// CHECK:STDOUT:       %ComparableTo.ref.loc10_28: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:       %T.ref.loc10_41: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:       %ComparableTo.type.loc10_42.1: type = facet_type <@ComparableTo, @ComparableTo(constants.%T.8b3)> [symbolic = %ComparableTo.type.loc10_42.2 (constants.%ComparableTo.type.4eeb0e.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_24.1: @impl.1.%ComparableTo.type.loc10_42.2 (%ComparableTo.type.4eeb0e.1) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc10_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (), @impl.1(constants.%T.8b3, constants.%U.b15) [symbolic = @impl.1.%impl_witness (constants.%impl_witness.206)]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_14.1, runtime_param<none> [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %U.patt.loc12_24.1: @impl.2.%ComparableTo.type.loc12_42.2 (%ComparableTo.type.4eeb0e.1) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc12_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:     %U.param_patt: @impl.2.%ComparableTo.type.loc12_42.2 (%ComparableTo.type.4eeb0e.1) = value_param_pattern %U.patt.loc12_24.1, runtime_param<none> [symbolic = %U.patt.loc12_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Wraps.ref: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:     %U.ref: @impl.2.%ComparableTo.type.loc12_42.2 (%ComparableTo.type.4eeb0e.1) = name_ref U, %U.loc12_24.1 [symbolic = %U.loc12_24.2 (constants.%U.b15)]
// CHECK:STDOUT:     %U.as_type.loc12_52.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc12_52.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %.loc12_52: type = converted %U.ref, %U.as_type.loc12_52.1 [symbolic = %U.as_type.loc12_52.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %Wraps.loc12_52.1: type = class_type @Wraps, @Wraps(constants.%U.as_type.b21) [symbolic = %Wraps.loc12_52.2 (constants.%Wraps.b9a)]
// CHECK:STDOUT:     %ComparableTo.ref.loc12_57: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:     %T.ref.loc12_70: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %ComparableTo.type.loc12_71: type = facet_type <@ComparableTo, @ComparableTo(constants.%T.8b3)> [symbolic = %ComparableTo.type.loc12_42.2 (constants.%ComparableTo.type.4eeb0e.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %U.param: @impl.2.%ComparableTo.type.loc12_42.2 (%ComparableTo.type.4eeb0e.1) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc12_42: type = splice_block %ComparableTo.type.loc12_42.1 [symbolic = %ComparableTo.type.loc12_42.2 (constants.%ComparableTo.type.4eeb0e.1)] {
// CHECK:STDOUT:       %ComparableTo.ref.loc12_28: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:       %T.ref.loc12_41: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:       %ComparableTo.type.loc12_42.1: type = facet_type <@ComparableTo, @ComparableTo(constants.%T.8b3)> [symbolic = %ComparableTo.type.loc12_42.2 (constants.%ComparableTo.type.4eeb0e.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc12_24.1: @impl.2.%ComparableTo.type.loc12_42.2 (%ComparableTo.type.4eeb0e.1) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc12_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness (), @impl.2(constants.%T.8b3, constants.%U.b15) [symbolic = @impl.2.%impl_witness (constants.%impl_witness.96c)]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ComparableTo.ref: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ComparableTo.type: type = facet_type <@ComparableTo, @ComparableTo(constants.%D)> [template = constants.%ComparableTo.type.4f3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness () [template = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %ComparableWith.decl: %ComparableWith.type.dbb = interface_decl @ComparableWith [template = constants.%ComparableWith.generic] {
// CHECK:STDOUT:     %T.patt.loc25_26.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_26.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc25_26.1, runtime_param<none> [symbolic = %T.patt.loc25_26.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc25_26.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc25_26.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %U.patt.loc27_14.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc27_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc27_14.1, runtime_param<none> [symbolic = %U.patt.loc27_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc27_24: type = name_ref U, %U.loc27_14.1 [symbolic = %U.loc27_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %U.ref.loc27_44: type = name_ref U, %U.loc27_14.1 [symbolic = %U.loc27_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ComparableWith.type.loc27_45.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%U.8b3)> [symbolic = %ComparableWith.type.loc27_45.2 (constants.%ComparableWith.type.bb0721.2)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %U.loc27_14.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc27_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc27: <witness> = impl_witness (), @impl.4(constants.%U.8b3) [symbolic = @impl.4.%impl_witness (constants.%impl_witness.497)]
// CHECK:STDOUT:   impl_decl @impl.5 [template] {
// CHECK:STDOUT:     %U.patt.loc28_14.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc28_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc28_14.1, runtime_param<none> [symbolic = %U.patt.loc28_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %T.patt.loc28_24.1: @impl.5.%ComparableTo.type.loc28_42.2 (%ComparableTo.type.4eeb0e.2) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc28_24.2 (constants.%T.patt.16f)]
// CHECK:STDOUT:     %T.param_patt: @impl.5.%ComparableTo.type.loc28_42.2 (%ComparableTo.type.4eeb0e.2) = value_param_pattern %T.patt.loc28_24.1, runtime_param<none> [symbolic = %T.patt.loc28_24.2 (constants.%T.patt.16f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc28_45: type = name_ref U, %U.loc28_14.1 [symbolic = %U.loc28_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %T.ref: @impl.5.%ComparableTo.type.loc28_42.2 (%ComparableTo.type.4eeb0e.2) = name_ref T, %T.loc28_24.1 [symbolic = %T.loc28_24.2 (constants.%T.b15)]
// CHECK:STDOUT:     %T.as_type.loc28_66.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_66.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_66: type = converted %T.ref, %T.as_type.loc28_66.1 [symbolic = %T.as_type.loc28_66.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %ComparableWith.type.loc28_66.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%T.as_type)> [symbolic = %ComparableWith.type.loc28_66.2 (constants.%ComparableWith.type.73a)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %U.loc28_14.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc28_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %T.param: @impl.5.%ComparableTo.type.loc28_42.2 (%ComparableTo.type.4eeb0e.2) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc28_42: type = splice_block %ComparableTo.type.loc28_42.1 [symbolic = %ComparableTo.type.loc28_42.2 (constants.%ComparableTo.type.4eeb0e.2)] {
// CHECK:STDOUT:       %ComparableTo.ref: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:       %U.ref.loc28_41: type = name_ref U, %U.loc28_14.1 [symbolic = %U.loc28_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:       %ComparableTo.type.loc28_42.1: type = facet_type <@ComparableTo, @ComparableTo(constants.%U.8b3)> [symbolic = %ComparableTo.type.loc28_42.2 (constants.%ComparableTo.type.4eeb0e.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc28_24.1: @impl.5.%ComparableTo.type.loc28_42.2 (%ComparableTo.type.4eeb0e.2) = bind_symbolic_name T, 1, %T.param [symbolic = %T.loc28_24.2 (constants.%T.b15)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc28: <witness> = impl_witness (), @impl.5(constants.%U.8b3, constants.%T.b15) [symbolic = @impl.5.%impl_witness (constants.%impl_witness.595)]
// CHECK:STDOUT:   impl_decl @impl.6 [template] {
// CHECK:STDOUT:     %U.patt.loc29_14.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc29_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc29_14.1, runtime_param<none> [symbolic = %U.patt.loc29_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %T.patt.loc29_24.1: @impl.6.%ComparableTo.type.loc29_42.2 (%ComparableTo.type.4eeb0e.2) = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc29_24.2 (constants.%T.patt.16f)]
// CHECK:STDOUT:     %T.param_patt: @impl.6.%ComparableTo.type.loc29_42.2 (%ComparableTo.type.4eeb0e.2) = value_param_pattern %T.patt.loc29_24.1, runtime_param<none> [symbolic = %T.patt.loc29_24.2 (constants.%T.patt.16f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: @impl.6.%ComparableTo.type.loc29_42.2 (%ComparableTo.type.4eeb0e.2) = name_ref T, %T.loc29_24.1 [symbolic = %T.loc29_24.2 (constants.%T.b15)]
// CHECK:STDOUT:     %T.as_type.loc29_45.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc29_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc29_45: type = converted %T.ref, %T.as_type.loc29_45.1 [symbolic = %T.as_type.loc29_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %U.ref.loc29_65: type = name_ref U, %U.loc29_14.1 [symbolic = %U.loc29_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ComparableWith.type.loc29_66.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%U.8b3)> [symbolic = %ComparableWith.type.loc29_66.2 (constants.%ComparableWith.type.bb0721.2)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %U.loc29_14.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc29_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %T.param: @impl.6.%ComparableTo.type.loc29_42.2 (%ComparableTo.type.4eeb0e.2) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc29_42: type = splice_block %ComparableTo.type.loc29_42.1 [symbolic = %ComparableTo.type.loc29_42.2 (constants.%ComparableTo.type.4eeb0e.2)] {
// CHECK:STDOUT:       %ComparableTo.ref: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:       %U.ref.loc29_41: type = name_ref U, %U.loc29_14.1 [symbolic = %U.loc29_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:       %ComparableTo.type.loc29_42.1: type = facet_type <@ComparableTo, @ComparableTo(constants.%U.8b3)> [symbolic = %ComparableTo.type.loc29_42.2 (constants.%ComparableTo.type.4eeb0e.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc29_24.1: @impl.6.%ComparableTo.type.loc29_42.2 (%ComparableTo.type.4eeb0e.2) = bind_symbolic_name T, 1, %T.param [symbolic = %T.loc29_24.2 (constants.%T.b15)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc29: <witness> = impl_witness (), @impl.6(constants.%U.8b3, constants.%T.b15) [symbolic = @impl.6.%impl_witness (constants.%impl_witness.c25)]
// CHECK:STDOUT:   %Compare.decl: %Compare.type = fn_decl @Compare [template = constants.%Compare] {
// CHECK:STDOUT:     %T.patt.loc31_12.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_12.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc31_12.1, runtime_param<none> [symbolic = %T.patt.loc31_12.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %U.patt.loc31_22.1: @Compare.%ComparableWith.type.loc31_42.2 (%ComparableWith.type.bb0721.1) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc31_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:     %U.param_patt: @Compare.%ComparableWith.type.loc31_42.2 (%ComparableWith.type.bb0721.1) = value_param_pattern %U.patt.loc31_22.1, runtime_param<none> [symbolic = %U.patt.loc31_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:     %t.patt: @Compare.%T.loc31_12.2 (%T.8b3) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @Compare.%T.loc31_12.2 (%T.8b3) = value_param_pattern %t.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: @Compare.%U.as_type.loc31_54.2 (%U.as_type.2d1) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @Compare.%U.as_type.loc31_54.2 (%U.as_type.2d1) = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc31_12.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc31_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %U.param: @Compare.%ComparableWith.type.loc31_42.2 (%ComparableWith.type.bb0721.1) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc31_42: type = splice_block %ComparableWith.type.loc31_42.1 [symbolic = %ComparableWith.type.loc31_42.2 (constants.%ComparableWith.type.bb0721.1)] {
// CHECK:STDOUT:       %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:       %T.ref.loc31_41: type = name_ref T, %T.loc31_12.1 [symbolic = %T.loc31_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:       %ComparableWith.type.loc31_42.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%T.8b3)> [symbolic = %ComparableWith.type.loc31_42.2 (constants.%ComparableWith.type.bb0721.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc31_22.1: @Compare.%ComparableWith.type.loc31_42.2 (%ComparableWith.type.bb0721.1) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc31_22.2 (constants.%U.d53)]
// CHECK:STDOUT:     %t.param: @Compare.%T.loc31_12.2 (%T.8b3) = value_param runtime_param0
// CHECK:STDOUT:     %T.ref.loc31_48: type = name_ref T, %T.loc31_12.1 [symbolic = %T.loc31_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %t: @Compare.%T.loc31_12.2 (%T.8b3) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @Compare.%U.as_type.loc31_54.2 (%U.as_type.2d1) = value_param runtime_param1
// CHECK:STDOUT:     %.loc31_54.1: type = splice_block %.loc31_54.2 [symbolic = %U.as_type.loc31_54.2 (constants.%U.as_type.2d1)] {
// CHECK:STDOUT:       %U.ref: @Compare.%ComparableWith.type.loc31_42.2 (%ComparableWith.type.bb0721.1) = name_ref U, %U.loc31_22.1 [symbolic = %U.loc31_22.2 (constants.%U.d53)]
// CHECK:STDOUT:       %U.as_type.loc31_54.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc31_54.2 (constants.%U.as_type.2d1)]
// CHECK:STDOUT:       %.loc31_54.2: type = converted %U.ref, %U.as_type.loc31_54.1 [symbolic = %U.as_type.loc31_54.2 (constants.%U.as_type.2d1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Compare.%U.as_type.loc31_54.2 (%U.as_type.2d1) = bind_name u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ComparableTo(%T.loc7_24.1: type) {
// CHECK:STDOUT:   %T.loc7_24.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_24.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc7_24.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_24.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableTo.type: type = facet_type <@ComparableTo, @ComparableTo(%T.loc7_24.2)> [symbolic = %ComparableTo.type (constants.%ComparableTo.type.4eeb0e.1)]
// CHECK:STDOUT:   %Self.2: %ComparableTo.type.4eeb0e.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ComparableTo.%ComparableTo.type (%ComparableTo.type.4eeb0e.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ComparableWith(%T.loc25_26.1: type) {
// CHECK:STDOUT:   %T.loc25_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc25_26.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc25_26.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_26.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type: type = facet_type <@ComparableWith, @ComparableWith(%T.loc25_26.2)> [symbolic = %ComparableWith.type (constants.%ComparableWith.type.bb0721.1)]
// CHECK:STDOUT:   %Self.2: %ComparableWith.type.bb0721.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.d53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ComparableWith.%ComparableWith.type (%ComparableWith.type.bb0721.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.d53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.ffd566.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%T.loc10_14.1: type, %U.loc10_24.1: @impl.1.%ComparableTo.type.loc10_42.2 (%ComparableTo.type.4eeb0e.1)) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc10_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %ComparableTo.type.loc10_42.2: type = facet_type <@ComparableTo, @ComparableTo(%T.loc10_14.2)> [symbolic = %ComparableTo.type.loc10_42.2 (constants.%ComparableTo.type.4eeb0e.1)]
// CHECK:STDOUT:   %U.loc10_24.2: %ComparableTo.type.4eeb0e.1 = bind_symbolic_name U, 1 [symbolic = %U.loc10_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   %U.patt.loc10_24.2: %ComparableTo.type.4eeb0e.1 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc10_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   %U.as_type.loc10_45.2: type = facet_access_type %U.loc10_24.2 [symbolic = %U.as_type.loc10_45.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:   %Wraps.loc10_70.2: type = class_type @Wraps, @Wraps(%T.loc10_14.2) [symbolic = %Wraps.loc10_70.2 (constants.%Wraps.487)]
// CHECK:STDOUT:   %ComparableTo.type.loc10_71.2: type = facet_type <@ComparableTo, @ComparableTo(%Wraps.loc10_70.2)> [symbolic = %ComparableTo.type.loc10_71.2 (constants.%ComparableTo.type.040)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%ComparableTo.type.loc10_71.2 (%ComparableTo.type.040) [symbolic = %require_complete (constants.%require_complete.066)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T.loc10_14.2, %U.loc10_24.2) [symbolic = %impl_witness (constants.%impl_witness.206)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc10_45 as %ComparableTo.type.loc10_71.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc12_14.1: type, %U.loc12_24.1: @impl.2.%ComparableTo.type.loc12_42.2 (%ComparableTo.type.4eeb0e.1)) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %ComparableTo.type.loc12_42.2: type = facet_type <@ComparableTo, @ComparableTo(%T.loc12_14.2)> [symbolic = %ComparableTo.type.loc12_42.2 (constants.%ComparableTo.type.4eeb0e.1)]
// CHECK:STDOUT:   %U.loc12_24.2: %ComparableTo.type.4eeb0e.1 = bind_symbolic_name U, 1 [symbolic = %U.loc12_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   %U.patt.loc12_24.2: %ComparableTo.type.4eeb0e.1 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc12_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   %U.as_type.loc12_52.2: type = facet_access_type %U.loc12_24.2 [symbolic = %U.as_type.loc12_52.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:   %Wraps.loc12_52.2: type = class_type @Wraps, @Wraps(%U.as_type.loc12_52.2) [symbolic = %Wraps.loc12_52.2 (constants.%Wraps.b9a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%ComparableTo.type.loc12_42.2 (%ComparableTo.type.4eeb0e.1) [symbolic = %require_complete (constants.%require_complete.084)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T.loc12_14.2, %U.loc12_24.2) [symbolic = %impl_witness (constants.%impl_witness.96c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Wraps.loc12_52.1 as %ComparableTo.type.loc12_71 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C.ref as %ComparableTo.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%U.loc27_14.1: type) {
// CHECK:STDOUT:   %U.loc27_14.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc27_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %U.patt.loc27_14.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc27_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   %ComparableWith.type.loc27_45.2: type = facet_type <@ComparableWith, @ComparableWith(%U.loc27_14.2)> [symbolic = %ComparableWith.type.loc27_45.2 (constants.%ComparableWith.type.bb0721.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.4.%ComparableWith.type.loc27_45.2 (%ComparableWith.type.bb0721.2) [symbolic = %require_complete (constants.%require_complete.17f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.4(%U.loc27_14.2) [symbolic = %impl_witness (constants.%impl_witness.497)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc27_24 as %ComparableWith.type.loc27_45.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(%U.loc28_14.1: type, %T.loc28_24.1: @impl.5.%ComparableTo.type.loc28_42.2 (%ComparableTo.type.4eeb0e.2)) {
// CHECK:STDOUT:   %U.loc28_14.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc28_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %U.patt.loc28_14.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc28_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   %ComparableTo.type.loc28_42.2: type = facet_type <@ComparableTo, @ComparableTo(%U.loc28_14.2)> [symbolic = %ComparableTo.type.loc28_42.2 (constants.%ComparableTo.type.4eeb0e.2)]
// CHECK:STDOUT:   %T.loc28_24.2: %ComparableTo.type.4eeb0e.2 = bind_symbolic_name T, 1 [symbolic = %T.loc28_24.2 (constants.%T.b15)]
// CHECK:STDOUT:   %T.patt.loc28_24.2: %ComparableTo.type.4eeb0e.2 = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc28_24.2 (constants.%T.patt.16f)]
// CHECK:STDOUT:   %T.as_type.loc28_66.2: type = facet_access_type %T.loc28_24.2 [symbolic = %T.as_type.loc28_66.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %ComparableWith.type.loc28_66.2: type = facet_type <@ComparableWith, @ComparableWith(%T.as_type.loc28_66.2)> [symbolic = %ComparableWith.type.loc28_66.2 (constants.%ComparableWith.type.73a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.5.%ComparableWith.type.loc28_66.2 (%ComparableWith.type.73a) [symbolic = %require_complete (constants.%require_complete.86a)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.5(%U.loc28_14.2, %T.loc28_24.2) [symbolic = %impl_witness (constants.%impl_witness.595)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc28_45 as %ComparableWith.type.loc28_66.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(%U.loc29_14.1: type, %T.loc29_24.1: @impl.6.%ComparableTo.type.loc29_42.2 (%ComparableTo.type.4eeb0e.2)) {
// CHECK:STDOUT:   %U.loc29_14.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc29_14.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %U.patt.loc29_14.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc29_14.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   %ComparableTo.type.loc29_42.2: type = facet_type <@ComparableTo, @ComparableTo(%U.loc29_14.2)> [symbolic = %ComparableTo.type.loc29_42.2 (constants.%ComparableTo.type.4eeb0e.2)]
// CHECK:STDOUT:   %T.loc29_24.2: %ComparableTo.type.4eeb0e.2 = bind_symbolic_name T, 1 [symbolic = %T.loc29_24.2 (constants.%T.b15)]
// CHECK:STDOUT:   %T.patt.loc29_24.2: %ComparableTo.type.4eeb0e.2 = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc29_24.2 (constants.%T.patt.16f)]
// CHECK:STDOUT:   %T.as_type.loc29_45.2: type = facet_access_type %T.loc29_24.2 [symbolic = %T.as_type.loc29_45.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %ComparableWith.type.loc29_66.2: type = facet_type <@ComparableWith, @ComparableWith(%U.loc29_14.2)> [symbolic = %ComparableWith.type.loc29_66.2 (constants.%ComparableWith.type.bb0721.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.6.%ComparableWith.type.loc29_66.2 (%ComparableWith.type.bb0721.2) [symbolic = %require_complete (constants.%require_complete.17f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.6(%U.loc29_14.2, %T.loc29_24.2) [symbolic = %impl_witness (constants.%impl_witness.c25)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc29_45 as %ComparableWith.type.loc29_66.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc29
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wraps(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wraps.487
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc19_10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc19_5.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_5.3: init %C = class_init (), %.loc19_5.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc19_5.4: ref %C = temporary %.loc19_5.2, %.loc19_5.3
// CHECK:STDOUT:   %.loc19_7: ref %C = converted %.loc19_5.1, %.loc19_5.4
// CHECK:STDOUT:   %C.ref.loc19_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %ComparableTo.ref.loc19: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:   %D.ref.loc19: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %ComparableTo.type.loc19: type = facet_type <@ComparableTo, @ComparableTo(constants.%D)> [template = constants.%ComparableTo.type.4f3]
// CHECK:STDOUT:   %ComparableTo.facet.loc19: %ComparableTo.type.4f3 = facet_value %C.ref.loc19_17, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %.loc19_19: %ComparableTo.type.4f3 = converted %C.ref.loc19_17, %ComparableTo.facet.loc19 [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %as_type.loc19: type = facet_access_type %.loc19_19 [template = constants.%C]
// CHECK:STDOUT:   %.loc19_37: type = converted %.loc19_19, %as_type.loc19 [template = constants.%C]
// CHECK:STDOUT:   %.loc20_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc20_10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc20_5.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc20_5.3: init %C = class_init (), %.loc20_5.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc20_5.4: ref %C = temporary %.loc20_5.2, %.loc20_5.3
// CHECK:STDOUT:   %.loc20_7: ref %C = converted %.loc20_5.1, %.loc20_5.4
// CHECK:STDOUT:   %C.ref.loc20_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %ComparableTo.ref.loc20: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:   %Wraps.ref.loc20: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %D.ref.loc20: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %Wraps.loc20: type = class_type @Wraps, @Wraps(constants.%D) [template = constants.%Wraps.537]
// CHECK:STDOUT:   %ComparableTo.type.loc20: type = facet_type <@ComparableTo, @ComparableTo(constants.%Wraps.537)> [template = constants.%ComparableTo.type.237]
// CHECK:STDOUT:   %ComparableTo.facet.loc20_19.1: %ComparableTo.type.4f3 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %.loc20_19.1: %ComparableTo.type.4f3 = converted constants.%C, %ComparableTo.facet.loc20_19.1 [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %ComparableTo.facet.loc20_19.2: %ComparableTo.type.237 = facet_value %C.ref.loc20_17, constants.%impl_witness.b06 [template = constants.%ComparableTo.facet.26c]
// CHECK:STDOUT:   %.loc20_19.2: %ComparableTo.type.237 = converted %C.ref.loc20_17, %ComparableTo.facet.loc20_19.2 [template = constants.%ComparableTo.facet.26c]
// CHECK:STDOUT:   %as_type.loc20: type = facet_access_type %.loc20_19.2 [template = constants.%C]
// CHECK:STDOUT:   %.loc20_44: type = converted %.loc20_19.2, %as_type.loc20 [template = constants.%C]
// CHECK:STDOUT:   %.loc21_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Wraps.ref.loc21_10: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %C.ref.loc21_16: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Wraps.loc21_17: type = class_type @Wraps, @Wraps(constants.%C) [template = constants.%Wraps.299]
// CHECK:STDOUT:   %.loc21_5.2: ref %Wraps.299 = temporary_storage
// CHECK:STDOUT:   %.loc21_5.3: init %Wraps.299 = class_init (), %.loc21_5.2 [template = constants.%Wraps.val]
// CHECK:STDOUT:   %.loc21_5.4: ref %Wraps.299 = temporary %.loc21_5.2, %.loc21_5.3
// CHECK:STDOUT:   %.loc21_7: ref %Wraps.299 = converted %.loc21_5.1, %.loc21_5.4
// CHECK:STDOUT:   %Wraps.ref.loc21_24: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %C.ref.loc21_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Wraps.loc21_31: type = class_type @Wraps, @Wraps(constants.%C) [template = constants.%Wraps.299]
// CHECK:STDOUT:   %ComparableTo.ref.loc21: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:   %D.ref.loc21: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %ComparableTo.type.loc21: type = facet_type <@ComparableTo, @ComparableTo(constants.%D)> [template = constants.%ComparableTo.type.4f3]
// CHECK:STDOUT:   %ComparableTo.facet.loc21_33.1: %ComparableTo.type.4f3 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %.loc21_33.1: %ComparableTo.type.4f3 = converted constants.%C, %ComparableTo.facet.loc21_33.1 [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %ComparableTo.facet.loc21_33.2: %ComparableTo.type.4f3 = facet_value %Wraps.loc21_31, constants.%impl_witness.951 [template = constants.%ComparableTo.facet.c6f]
// CHECK:STDOUT:   %.loc21_33.2: %ComparableTo.type.4f3 = converted %Wraps.loc21_31, %ComparableTo.facet.loc21_33.2 [template = constants.%ComparableTo.facet.c6f]
// CHECK:STDOUT:   %as_type.loc21: type = facet_access_type %.loc21_33.2 [template = constants.%Wraps.299]
// CHECK:STDOUT:   %.loc21_51: type = converted %.loc21_33.2, %as_type.loc21 [template = constants.%Wraps.299]
// CHECK:STDOUT:   %.loc22_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Wraps.ref.loc22_10: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %C.ref.loc22_16: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Wraps.loc22_17: type = class_type @Wraps, @Wraps(constants.%C) [template = constants.%Wraps.299]
// CHECK:STDOUT:   %.loc22_5.2: ref %Wraps.299 = temporary_storage
// CHECK:STDOUT:   %.loc22_5.3: init %Wraps.299 = class_init (), %.loc22_5.2 [template = constants.%Wraps.val]
// CHECK:STDOUT:   %.loc22_5.4: ref %Wraps.299 = temporary %.loc22_5.2, %.loc22_5.3
// CHECK:STDOUT:   %.loc22_7: ref %Wraps.299 = converted %.loc22_5.1, %.loc22_5.4
// CHECK:STDOUT:   %Wraps.ref.loc22_24: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %C.ref.loc22_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Wraps.loc22_31: type = class_type @Wraps, @Wraps(constants.%C) [template = constants.%Wraps.299]
// CHECK:STDOUT:   %ComparableTo.ref.loc22: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:   %Wraps.ref.loc22_49: %Wraps.type = name_ref Wraps, file.%Wraps.decl [template = constants.%Wraps.generic]
// CHECK:STDOUT:   %D.ref.loc22: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %Wraps.loc22_56: type = class_type @Wraps, @Wraps(constants.%D) [template = constants.%Wraps.537]
// CHECK:STDOUT:   %ComparableTo.type.loc22: type = facet_type <@ComparableTo, @ComparableTo(constants.%Wraps.537)> [template = constants.%ComparableTo.type.237]
// CHECK:STDOUT:   %ComparableTo.facet.loc22_33.1: %ComparableTo.type.4f3 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %.loc22_33.1: %ComparableTo.type.4f3 = converted constants.%C, %ComparableTo.facet.loc22_33.1 [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %ComparableTo.facet.loc22_33.2: %ComparableTo.type.4f3 = facet_value constants.%Wraps.299, constants.%impl_witness.951 [template = constants.%ComparableTo.facet.c6f]
// CHECK:STDOUT:   %.loc22_33.2: %ComparableTo.type.4f3 = converted constants.%Wraps.299, %ComparableTo.facet.loc22_33.2 [template = constants.%ComparableTo.facet.c6f]
// CHECK:STDOUT:   %ComparableTo.facet.loc22_33.3: %ComparableTo.type.237 = facet_value %Wraps.loc22_31, constants.%impl_witness.098 [template = constants.%ComparableTo.facet.6db]
// CHECK:STDOUT:   %.loc22_33.3: %ComparableTo.type.237 = converted %Wraps.loc22_31, %ComparableTo.facet.loc22_33.3 [template = constants.%ComparableTo.facet.6db]
// CHECK:STDOUT:   %as_type.loc22: type = facet_access_type %.loc22_33.3 [template = constants.%Wraps.299]
// CHECK:STDOUT:   %.loc22_58: type = converted %.loc22_33.3, %as_type.loc22 [template = constants.%Wraps.299]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compare(%T.loc31_12.1: type, %U.loc31_22.1: @Compare.%ComparableWith.type.loc31_42.2 (%ComparableWith.type.bb0721.1)) {
// CHECK:STDOUT:   %T.loc31_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc31_12.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc31_12.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc31_12.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2: type = facet_type <@ComparableWith, @ComparableWith(%T.loc31_12.2)> [symbolic = %ComparableWith.type.loc31_42.2 (constants.%ComparableWith.type.bb0721.1)]
// CHECK:STDOUT:   %U.loc31_22.2: %ComparableWith.type.bb0721.1 = bind_symbolic_name U, 1 [symbolic = %U.loc31_22.2 (constants.%U.d53)]
// CHECK:STDOUT:   %U.patt.loc31_22.2: %ComparableWith.type.bb0721.1 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc31_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:   %U.as_type.loc31_54.2: type = facet_access_type %U.loc31_22.2 [symbolic = %U.as_type.loc31_54.2 (constants.%U.as_type.2d1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_46: <witness> = require_complete_type @Compare.%T.loc31_12.2 (%T.8b3) [symbolic = %require_complete.loc31_46 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc31_52: <witness> = require_complete_type @Compare.%U.as_type.loc31_54.2 (%U.as_type.2d1) [symbolic = %require_complete.loc31_52 (constants.%require_complete.9df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %U.param_patt: @Compare.%ComparableWith.type.loc31_42.2 (%ComparableWith.type.bb0721.1)](%t.param_patt: @Compare.%T.loc31_12.2 (%T.8b3), %u.param_patt: @Compare.%U.as_type.loc31_54.2 (%U.as_type.2d1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Compare.ref.loc34: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc34_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc34_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc34_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc34_12.3: init %C = class_init (), %.loc34_12.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc34_12.4: ref %C = temporary %.loc34_12.2, %.loc34_12.3
// CHECK:STDOUT:   %.loc34_14.1: ref %C = converted %.loc34_12.1, %.loc34_12.4
// CHECK:STDOUT:   %.loc34_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc34_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc34_21.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc34_21.3: init %C = class_init (), %.loc34_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc34_21.4: ref %C = temporary %.loc34_21.2, %.loc34_21.3
// CHECK:STDOUT:   %.loc34_23.1: ref %C = converted %.loc34_21.1, %.loc34_21.4
// CHECK:STDOUT:   %ComparableWith.facet.loc34: %ComparableWith.type.68c = facet_value constants.%C, constants.%impl_witness.887 [template = constants.%ComparableWith.facet.17d]
// CHECK:STDOUT:   %.loc34_27: %ComparableWith.type.68c = converted constants.%C, %ComparableWith.facet.loc34 [template = constants.%ComparableWith.facet.17d]
// CHECK:STDOUT:   %Compare.specific_fn.loc34: <specific function> = specific_function %Compare.ref.loc34, @Compare(constants.%C, %.loc34_27) [template = constants.%Compare.specific_fn.041]
// CHECK:STDOUT:   %.loc34_14.2: %C = bind_value %.loc34_14.1
// CHECK:STDOUT:   %.loc34_23.2: %C = bind_value %.loc34_23.1
// CHECK:STDOUT:   %Compare.call.loc34: init %empty_tuple.type = call %Compare.specific_fn.loc34(%.loc34_14.2, %.loc34_23.2)
// CHECK:STDOUT:   %Compare.ref.loc35: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc35_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc35: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc35_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc35_12.3: init %C = class_init (), %.loc35_12.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc35_12.4: ref %C = temporary %.loc35_12.2, %.loc35_12.3
// CHECK:STDOUT:   %.loc35_14.1: ref %C = converted %.loc35_12.1, %.loc35_12.4
// CHECK:STDOUT:   %.loc35_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref.loc35: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc35_21.2: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc35_21.3: init %D = class_init (), %.loc35_21.2 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc35_21.4: ref %D = temporary %.loc35_21.2, %.loc35_21.3
// CHECK:STDOUT:   %.loc35_23.1: ref %D = converted %.loc35_21.1, %.loc35_21.4
// CHECK:STDOUT:   %ComparableTo.facet.loc35: %ComparableTo.type.4f3 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %.loc35_27.1: %ComparableTo.type.4f3 = converted constants.%C, %ComparableTo.facet.loc35 [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %ComparableWith.facet.loc35: %ComparableWith.type.68c = facet_value constants.%D, constants.%impl_witness.228 [template = constants.%ComparableWith.facet.aa3]
// CHECK:STDOUT:   %.loc35_27.2: %ComparableWith.type.68c = converted constants.%D, %ComparableWith.facet.loc35 [template = constants.%ComparableWith.facet.aa3]
// CHECK:STDOUT:   %Compare.specific_fn.loc35: <specific function> = specific_function %Compare.ref.loc35, @Compare(constants.%C, %.loc35_27.2) [template = constants.%Compare.specific_fn.94d]
// CHECK:STDOUT:   %.loc35_14.2: %C = bind_value %.loc35_14.1
// CHECK:STDOUT:   %.loc35_23.2: %D = bind_value %.loc35_23.1
// CHECK:STDOUT:   %Compare.call.loc35: init %empty_tuple.type = call %Compare.specific_fn.loc35(%.loc35_14.2, %.loc35_23.2)
// CHECK:STDOUT:   %Compare.ref.loc49: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc49_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref.loc49: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc49_12.2: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc49_12.3: init %D = class_init (), %.loc49_12.2 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc49_12.4: ref %D = temporary %.loc49_12.2, %.loc49_12.3
// CHECK:STDOUT:   %.loc49_14.1: ref %D = converted %.loc49_12.1, %.loc49_12.4
// CHECK:STDOUT:   %.loc49_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc49: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc49_21.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc49_21.3: init %C = class_init (), %.loc49_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc49_21.4: ref %C = temporary %.loc49_21.2, %.loc49_21.3
// CHECK:STDOUT:   %.loc49_23.1: ref %C = converted %.loc49_21.1, %.loc49_21.4
// CHECK:STDOUT:   %.loc49_27.1: %ComparableTo.type.406 = converted constants.%D, <error> [template = <error>]
// CHECK:STDOUT:   %ComparableTo.facet.loc49: %ComparableTo.type.4f3 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %.loc49_27.2: %ComparableTo.type.4f3 = converted constants.%C, %ComparableTo.facet.loc49 [template = constants.%ComparableTo.facet.3c8]
// CHECK:STDOUT:   %ComparableWith.facet.loc49: %ComparableWith.type.c33 = facet_value constants.%C, constants.%impl_witness.d37 [template = constants.%ComparableWith.facet.7c0]
// CHECK:STDOUT:   %.loc49_27.3: %ComparableWith.type.c33 = converted constants.%C, %ComparableWith.facet.loc49 [template = constants.%ComparableWith.facet.7c0]
// CHECK:STDOUT:   %Compare.specific_fn.loc49: <specific function> = specific_function %Compare.ref.loc49, @Compare(constants.%D, %.loc49_27.3) [template = constants.%Compare.specific_fn.f4d]
// CHECK:STDOUT:   %.loc49_14.2: %D = bind_value %.loc49_14.1
// CHECK:STDOUT:   %.loc49_23.2: %C = bind_value %.loc49_23.1
// CHECK:STDOUT:   %Compare.call.loc49: init %empty_tuple.type = call %Compare.specific_fn.loc49(%.loc49_14.2, %.loc49_23.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.ffd566.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc7_24.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc7_24.2 => constants.%T.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableTo.type => constants.%ComparableTo.type.4eeb0e.1
// CHECK:STDOUT:   %Self.2 => constants.%Self.b15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(%T.loc7_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%Wraps.487) {
// CHECK:STDOUT:   %T.loc7_24.2 => constants.%Wraps.487
// CHECK:STDOUT:   %T.patt.loc7_24.2 => constants.%Wraps.487
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableTo.type => constants.%ComparableTo.type.040
// CHECK:STDOUT:   %Self.2 => constants.%Self.b15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T.8b3, constants.%U.b15) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %ComparableTo.type.loc10_42.2 => constants.%ComparableTo.type.4eeb0e.1
// CHECK:STDOUT:   %U.loc10_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.patt.loc10_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.as_type.loc10_45.2 => constants.%U.as_type.b21
// CHECK:STDOUT:   %Wraps.loc10_70.2 => constants.%Wraps.487
// CHECK:STDOUT:   %ComparableTo.type.loc10_71.2 => constants.%ComparableTo.type.040
// CHECK:STDOUT:   %require_complete => constants.%require_complete.066
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(@impl.1.%T.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(@impl.1.%T.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(@impl.1.%Wraps.loc10_70.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T.loc10_14.2, %U.loc10_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(constants.%U.as_type.b21) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%U.as_type.b21
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%U.as_type.b21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T.8b3, constants.%U.b15) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %ComparableTo.type.loc12_42.2 => constants.%ComparableTo.type.4eeb0e.1
// CHECK:STDOUT:   %U.loc12_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.patt.loc12_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.as_type.loc12_52.2 => constants.%U.as_type.b21
// CHECK:STDOUT:   %Wraps.loc12_52.2 => constants.%Wraps.b9a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.084
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.96c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(@impl.2.%T.loc12_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(@impl.2.%U.as_type.loc12_52.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T.loc12_14.2, %U.loc12_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%D) {
// CHECK:STDOUT:   %T.loc7_24.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc7_24.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableTo.type => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %Self.2 => constants.%Self.b15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(constants.%D) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%Wraps.537) {
// CHECK:STDOUT:   %T.loc7_24.2 => constants.%Wraps.537
// CHECK:STDOUT:   %T.patt.loc7_24.2 => constants.%Wraps.537
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%D, @F.%.loc20_19.1) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc10_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc10_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.patt.loc10_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.as_type.loc10_45.2 => constants.%C
// CHECK:STDOUT:   %Wraps.loc10_70.2 => constants.%Wraps.537
// CHECK:STDOUT:   %ComparableTo.type.loc10_71.2 => constants.%ComparableTo.type.237
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d0b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%D, constants.%ComparableTo.facet.3c8) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc10_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc10_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.patt.loc10_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.as_type.loc10_45.2 => constants.%C
// CHECK:STDOUT:   %Wraps.loc10_70.2 => constants.%Wraps.537
// CHECK:STDOUT:   %ComparableTo.type.loc10_71.2 => constants.%ComparableTo.type.237
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d0b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b06
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wraps(constants.%C) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%D, @F.%.loc21_33.1) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc12_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc12_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.patt.loc12_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.as_type.loc12_52.2 => constants.%C
// CHECK:STDOUT:   %Wraps.loc12_52.2 => constants.%Wraps.299
// CHECK:STDOUT:   %require_complete => constants.%complete_type.b71
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.951
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%D, constants.%ComparableTo.facet.3c8) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc12_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc12_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.patt.loc12_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.as_type.loc12_52.2 => constants.%C
// CHECK:STDOUT:   %Wraps.loc12_52.2 => constants.%Wraps.299
// CHECK:STDOUT:   %require_complete => constants.%complete_type.b71
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.951
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%D, @F.%.loc22_33.1) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc12_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc12_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.patt.loc12_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %U.as_type.loc12_52.2 => constants.%C
// CHECK:STDOUT:   %Wraps.loc12_52.2 => constants.%Wraps.299
// CHECK:STDOUT:   %require_complete => constants.%complete_type.b71
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.951
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%D, @F.%.loc22_33.2) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc10_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc10_24.2 => constants.%ComparableTo.facet.c6f
// CHECK:STDOUT:   %U.patt.loc10_24.2 => constants.%ComparableTo.facet.c6f
// CHECK:STDOUT:   %U.as_type.loc10_45.2 => constants.%Wraps.299
// CHECK:STDOUT:   %Wraps.loc10_70.2 => constants.%Wraps.537
// CHECK:STDOUT:   %ComparableTo.type.loc10_71.2 => constants.%ComparableTo.type.237
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d0b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.098
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%D, constants.%ComparableTo.facet.c6f) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc10_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc10_24.2 => constants.%ComparableTo.facet.c6f
// CHECK:STDOUT:   %U.patt.loc10_24.2 => constants.%ComparableTo.facet.c6f
// CHECK:STDOUT:   %U.as_type.loc10_45.2 => constants.%Wraps.299
// CHECK:STDOUT:   %Wraps.loc10_70.2 => constants.%Wraps.537
// CHECK:STDOUT:   %ComparableTo.type.loc10_71.2 => constants.%ComparableTo.type.237
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d0b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.098
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc25_26.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc25_26.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(%T.loc25_26.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%U.8b3) {
// CHECK:STDOUT:   %T.loc25_26.2 => constants.%U.8b3
// CHECK:STDOUT:   %T.patt.loc25_26.2 => constants.%U.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type => constants.%ComparableWith.type.bb0721.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.d53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc27_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc27_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %ComparableWith.type.loc27_45.2 => constants.%ComparableWith.type.bb0721.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.17f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.497
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.4.%U.loc27_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%U.loc27_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%U.8b3) {
// CHECK:STDOUT:   %T.loc7_24.2 => constants.%U.8b3
// CHECK:STDOUT:   %T.patt.loc7_24.2 => constants.%U.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc25_26.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.patt.loc25_26.2 => constants.%T.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type => constants.%ComparableWith.type.73a
// CHECK:STDOUT:   %Self.2 => constants.%Self.d53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%U.8b3, constants.%T.b15) {
// CHECK:STDOUT:   %U.loc28_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc28_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %ComparableTo.type.loc28_42.2 => constants.%ComparableTo.type.4eeb0e.2
// CHECK:STDOUT:   %T.loc28_24.2 => constants.%T.b15
// CHECK:STDOUT:   %T.patt.loc28_24.2 => constants.%T.b15
// CHECK:STDOUT:   %T.as_type.loc28_66.2 => constants.%T.as_type
// CHECK:STDOUT:   %ComparableWith.type.loc28_66.2 => constants.%ComparableWith.type.73a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.86a
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.595
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(@impl.5.%U.loc28_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.5.%T.as_type.loc28_66.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%U.loc28_14.2, %T.loc28_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%U.8b3, constants.%T.b15) {
// CHECK:STDOUT:   %U.loc29_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc29_14.2 => constants.%U.8b3
// CHECK:STDOUT:   %ComparableTo.type.loc29_42.2 => constants.%ComparableTo.type.4eeb0e.2
// CHECK:STDOUT:   %T.loc29_24.2 => constants.%T.b15
// CHECK:STDOUT:   %T.patt.loc29_24.2 => constants.%T.b15
// CHECK:STDOUT:   %T.as_type.loc29_45.2 => constants.%T.as_type
// CHECK:STDOUT:   %ComparableWith.type.loc29_66.2 => constants.%ComparableWith.type.bb0721.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.17f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.c25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(@impl.6.%U.loc29_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.6.%U.loc29_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%U.loc29_14.2, %T.loc29_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%T.8b3, constants.%U.d53) {
// CHECK:STDOUT:   %T.loc31_12.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc31_12.2 => constants.%T.8b3
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2 => constants.%ComparableWith.type.bb0721.1
// CHECK:STDOUT:   %U.loc31_22.2 => constants.%U.d53
// CHECK:STDOUT:   %U.patt.loc31_22.2 => constants.%U.d53
// CHECK:STDOUT:   %U.as_type.loc31_54.2 => constants.%U.as_type.2d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@Compare.%T.loc31_12.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%C) {
// CHECK:STDOUT:   %T.loc25_26.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc25_26.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%C) {
// CHECK:STDOUT:   %U.loc27_14.2 => constants.%C
// CHECK:STDOUT:   %U.patt.loc27_14.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc27_45.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.887
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, @G.%.loc34_27) {
// CHECK:STDOUT:   %T.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc31_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.patt.loc31_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.as_type.loc31_54.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_52 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, constants.%ComparableWith.facet.17d) {
// CHECK:STDOUT:   %T.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc31_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.patt.loc31_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.as_type.loc31_54.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%D, @G.%.loc35_27.1) {
// CHECK:STDOUT:   %U.loc28_14.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc28_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc28_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %T.loc28_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.patt.loc28_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.as_type.loc28_66.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc28_66.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.228
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%D, constants.%ComparableTo.facet.3c8) {
// CHECK:STDOUT:   %U.loc28_14.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc28_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc28_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %T.loc28_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.patt.loc28_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.as_type.loc28_66.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc28_66.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, @G.%.loc35_27.2) {
// CHECK:STDOUT:   %T.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc31_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.patt.loc31_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.as_type.loc31_54.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_52 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, constants.%ComparableWith.facet.aa3) {
// CHECK:STDOUT:   %T.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc31_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc31_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.patt.loc31_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.as_type.loc31_54.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%D) {
// CHECK:STDOUT:   %T.loc25_26.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc25_26.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%C) {
// CHECK:STDOUT:   %T.loc7_24.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc7_24.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%D) {
// CHECK:STDOUT:   %U.loc27_14.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc27_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc27_45.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9db
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%ComparableTo.type.406) {
// CHECK:STDOUT:   %Dest => constants.%ComparableTo.type.406
// CHECK:STDOUT:   %Dest.patt => constants.%ComparableTo.type.406
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.632
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.cd0
// CHECK:STDOUT:   %Convert => constants.%Convert.088
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.38f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(type) {
// CHECK:STDOUT:   %U.loc27_14.2 => type
// CHECK:STDOUT:   %U.patt.loc27_14.2 => type
// CHECK:STDOUT:   %ComparableWith.type.loc27_45.2 => constants.%ComparableWith.type.7aa
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3dc
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(type) {
// CHECK:STDOUT:   %T.loc25_26.2 => type
// CHECK:STDOUT:   %T.patt.loc25_26.2 => type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%C, <error>) {
// CHECK:STDOUT:   %U.loc28_14.2 => constants.%C
// CHECK:STDOUT:   %U.patt.loc28_14.2 => constants.%C
// CHECK:STDOUT:   %ComparableTo.type.loc28_42.2 => constants.%ComparableTo.type.406
// CHECK:STDOUT:   %T.loc28_24.2 => <error>
// CHECK:STDOUT:   %T.patt.loc28_24.2 => <error>
// CHECK:STDOUT:   %T.as_type.loc28_66.2 => <error>
// CHECK:STDOUT:   %ComparableWith.type.loc28_66.2 => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(<error>) {
// CHECK:STDOUT:   %T.loc25_26.2 => <error>
// CHECK:STDOUT:   %T.patt.loc25_26.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%D, @G.%.loc49_27.2) {
// CHECK:STDOUT:   %U.loc29_14.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc29_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc29_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %T.loc29_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.patt.loc29_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.as_type.loc29_45.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc29_66.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9db
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.d37
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%D, constants.%ComparableTo.facet.3c8) {
// CHECK:STDOUT:   %U.loc29_14.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc29_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc29_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %T.loc29_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.patt.loc29_24.2 => constants.%ComparableTo.facet.3c8
// CHECK:STDOUT:   %T.as_type.loc29_45.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc29_66.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9db
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.d37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%D, @G.%.loc49_27.3) {
// CHECK:STDOUT:   %T.loc31_12.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc31_12.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %U.loc31_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.patt.loc31_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.as_type.loc31_54.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc31_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc31_52 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%D, constants.%ComparableWith.facet.7c0) {
// CHECK:STDOUT:   %T.loc31_12.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc31_12.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc31_42.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %U.loc31_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.patt.loc31_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.as_type.loc31_54.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_recurse_with_simpler_type_in_generic_param_bidirectional_no_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ComparableTo.type.706: type = generic_interface_type @ComparableTo [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %ComparableTo.generic: %ComparableTo.type.706 = struct_value () [template]
// CHECK:STDOUT:   %ComparableTo.type.4ee: type = facet_type <@ComparableTo, @ComparableTo(%T)> [symbolic]
// CHECK:STDOUT:   %Self.b15: %ComparableTo.type.4ee = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.dbb: type = generic_interface_type @ComparableWith [template]
// CHECK:STDOUT:   %ComparableWith.generic: %ComparableWith.type.dbb = struct_value () [template]
// CHECK:STDOUT:   %ComparableWith.type.bb0: type = facet_type <@ComparableWith, @ComparableWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.d53: %ComparableWith.type.bb0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %ComparableTo.type.4f3: type = facet_type <@ComparableTo, @ComparableTo(%D)> [template]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %require_complete.17f: <witness> = require_complete_type %ComparableWith.type.bb0 [symbolic]
// CHECK:STDOUT:   %impl_witness.497: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT:   %U.b15: %ComparableTo.type.4ee = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.16f: %ComparableTo.type.4ee = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.b21: type = facet_access_type %U.b15 [symbolic]
// CHECK:STDOUT:   %impl_witness.c25: <witness> = impl_witness (), @impl.3(%T, %U.b15) [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.73a: type = facet_type <@ComparableWith, @ComparableWith(%U.as_type.b21)> [symbolic]
// CHECK:STDOUT:   %require_complete.86a: <witness> = require_complete_type %ComparableWith.type.73a [symbolic]
// CHECK:STDOUT:   %impl_witness.595: <witness> = impl_witness (), @impl.4(%T, %U.b15) [symbolic]
// CHECK:STDOUT:   %U.d53: %ComparableWith.type.bb0 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.761: %ComparableWith.type.bb0 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.2d1: type = facet_access_type %U.d53 [symbolic]
// CHECK:STDOUT:   %Compare.type: type = fn_type @Compare [template]
// CHECK:STDOUT:   %Compare: %Compare.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.9df: <witness> = require_complete_type %U.as_type.2d1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ComparableWith.type.68c: type = facet_type <@ComparableWith, @ComparableWith(%C)> [template]
// CHECK:STDOUT:   %complete_type.aa9: <witness> = complete_type_witness %ComparableWith.type.68c [template]
// CHECK:STDOUT:   %impl_witness.887: <witness> = impl_witness (), @impl.2(%C) [template]
// CHECK:STDOUT:   %ComparableWith.facet.17d: %ComparableWith.type.68c = facet_value %C, %impl_witness.887 [template]
// CHECK:STDOUT:   %Compare.specific_fn.041: <specific function> = specific_function %Compare, @Compare(%C, %ComparableWith.facet.17d) [template]
// CHECK:STDOUT:   %D.val: %D = struct_value () [template]
// CHECK:STDOUT:   %ComparableTo.type.406: type = facet_type <@ComparableTo, @ComparableTo(%C)> [template]
// CHECK:STDOUT:   %ComparableWith.type.c33: type = facet_type <@ComparableWith, @ComparableWith(%D)> [template]
// CHECK:STDOUT:   %complete_type.9db: <witness> = complete_type_witness %ComparableWith.type.c33 [template]
// CHECK:STDOUT:   %impl_witness.126: <witness> = impl_witness (), @impl.2(%D) [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.632: type = facet_type <@ImplicitAs, @ImplicitAs(%ComparableTo.type.406)> [template]
// CHECK:STDOUT:   %Convert.type.cd0: type = fn_type @Convert, @ImplicitAs(%ComparableTo.type.406) [template]
// CHECK:STDOUT:   %Convert.088: %Convert.type.cd0 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.38f: type = assoc_entity_type %ImplicitAs.type.632 [template]
// CHECK:STDOUT:   %assoc0.dca: %ImplicitAs.assoc_type.38f = assoc_entity element0, imports.%Core.import_ref.207961.1 [template]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %ComparableWith.type.7aa: type = facet_type <@ComparableWith, @ComparableWith(type)> [template]
// CHECK:STDOUT:   %complete_type.3dc: <witness> = complete_type_witness %ComparableWith.type.7aa [template]
// CHECK:STDOUT:   %impl_witness.2ae: <witness> = impl_witness (), @impl.2(type) [template]
// CHECK:STDOUT:   %ComparableTo.facet: %ComparableTo.type.4f3 = facet_value %C, %impl_witness.1bc [template]
// CHECK:STDOUT:   %impl_witness.228: <witness> = impl_witness (), @impl.4(%D, %ComparableTo.facet) [template]
// CHECK:STDOUT:   %ComparableWith.facet.aa3: %ComparableWith.type.68c = facet_value %D, %impl_witness.228 [template]
// CHECK:STDOUT:   %Compare.specific_fn.94d: <specific function> = specific_function %Compare, @Compare(%C, %ComparableWith.facet.aa3) [template]
// CHECK:STDOUT:   %impl_witness.d37: <witness> = impl_witness (), @impl.3(%D, %ComparableTo.facet) [template]
// CHECK:STDOUT:   %ComparableWith.facet.7c0: %ComparableWith.type.c33 = facet_value %C, %impl_witness.d37 [template]
// CHECK:STDOUT:   %Compare.specific_fn.f4d: <specific function> = specific_function %Compare, @Compare(%D, %ComparableWith.facet.7c0) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ffd566.1: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst69 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc8_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.ffd566.2: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst69 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc8_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ComparableTo = %ComparableTo.decl
// CHECK:STDOUT:     .ComparableWith = %ComparableWith.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Compare = %Compare.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ComparableTo.decl: %ComparableTo.type.706 = interface_decl @ComparableTo [template = constants.%ComparableTo.generic] {
// CHECK:STDOUT:     %T.patt.loc6_24.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_24.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_24.1, runtime_param<none> [symbolic = %T.patt.loc6_24.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc6_24.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_24.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ComparableWith.decl: %ComparableWith.type.dbb = interface_decl @ComparableWith [template = constants.%ComparableWith.generic] {
// CHECK:STDOUT:     %T.patt.loc8_26.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_26.1, runtime_param<none> [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc8_26.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ComparableTo.ref: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ComparableTo.type: type = facet_type <@ComparableTo, @ComparableTo(constants.%D)> [template = constants.%ComparableTo.type.4f3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness () [template = constants.%impl_witness.1bc]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt.loc19_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc19_14.1, runtime_param<none> [symbolic = %T.patt.loc19_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19_24: type = name_ref T, %T.loc19_14.1 [symbolic = %T.loc19_14.2 (constants.%T)]
// CHECK:STDOUT:     %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %T.ref.loc19_44: type = name_ref T, %T.loc19_14.1 [symbolic = %T.loc19_14.2 (constants.%T)]
// CHECK:STDOUT:     %ComparableWith.type.loc19_45.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%T)> [symbolic = %ComparableWith.type.loc19_45.2 (constants.%ComparableWith.type.bb0)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc19_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc19_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (), @impl.2(constants.%T) [symbolic = @impl.2.%impl_witness (constants.%impl_witness.497)]
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc25_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc25_14.1, runtime_param<none> [symbolic = %T.patt.loc25_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc25_24.1: @impl.3.%ComparableTo.type.loc25_42.2 (%ComparableTo.type.4ee) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc25_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:     %U.param_patt: @impl.3.%ComparableTo.type.loc25_42.2 (%ComparableTo.type.4ee) = value_param_pattern %U.patt.loc25_24.1, runtime_param<none> [symbolic = %U.patt.loc25_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: @impl.3.%ComparableTo.type.loc25_42.2 (%ComparableTo.type.4ee) = name_ref U, %U.loc25_24.1 [symbolic = %U.loc25_24.2 (constants.%U.b15)]
// CHECK:STDOUT:     %U.as_type.loc25_45.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc25_45.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %.loc25_45: type = converted %U.ref, %U.as_type.loc25_45.1 [symbolic = %U.as_type.loc25_45.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %T.ref.loc25_65: type = name_ref T, %T.loc25_14.1 [symbolic = %T.loc25_14.2 (constants.%T)]
// CHECK:STDOUT:     %ComparableWith.type.loc25_66.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%T)> [symbolic = %ComparableWith.type.loc25_66.2 (constants.%ComparableWith.type.bb0)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc25_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc25_14.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @impl.3.%ComparableTo.type.loc25_42.2 (%ComparableTo.type.4ee) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc25_42: type = splice_block %ComparableTo.type.loc25_42.1 [symbolic = %ComparableTo.type.loc25_42.2 (constants.%ComparableTo.type.4ee)] {
// CHECK:STDOUT:       %ComparableTo.ref: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:       %T.ref.loc25_41: type = name_ref T, %T.loc25_14.1 [symbolic = %T.loc25_14.2 (constants.%T)]
// CHECK:STDOUT:       %ComparableTo.type.loc25_42.1: type = facet_type <@ComparableTo, @ComparableTo(constants.%T)> [symbolic = %ComparableTo.type.loc25_42.2 (constants.%ComparableTo.type.4ee)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc25_24.1: @impl.3.%ComparableTo.type.loc25_42.2 (%ComparableTo.type.4ee) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc25_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness (), @impl.3(constants.%T, constants.%U.b15) [symbolic = @impl.3.%impl_witness (constants.%impl_witness.c25)]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %T.patt.loc27_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc27_14.1, runtime_param<none> [symbolic = %T.patt.loc27_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc27_24.1: @impl.4.%ComparableTo.type.loc27_42.2 (%ComparableTo.type.4ee) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc27_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:     %U.param_patt: @impl.4.%ComparableTo.type.loc27_42.2 (%ComparableTo.type.4ee) = value_param_pattern %U.patt.loc27_24.1, runtime_param<none> [symbolic = %U.patt.loc27_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc27_45: type = name_ref T, %T.loc27_14.1 [symbolic = %T.loc27_14.2 (constants.%T)]
// CHECK:STDOUT:     %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:     %U.ref: @impl.4.%ComparableTo.type.loc27_42.2 (%ComparableTo.type.4ee) = name_ref U, %U.loc27_24.1 [symbolic = %U.loc27_24.2 (constants.%U.b15)]
// CHECK:STDOUT:     %U.as_type.loc27_66.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc27_66.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %.loc27_66: type = converted %U.ref, %U.as_type.loc27_66.1 [symbolic = %U.as_type.loc27_66.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:     %ComparableWith.type.loc27_66.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%U.as_type.b21)> [symbolic = %ComparableWith.type.loc27_66.2 (constants.%ComparableWith.type.73a)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc27_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc27_14.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @impl.4.%ComparableTo.type.loc27_42.2 (%ComparableTo.type.4ee) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc27_42: type = splice_block %ComparableTo.type.loc27_42.1 [symbolic = %ComparableTo.type.loc27_42.2 (constants.%ComparableTo.type.4ee)] {
// CHECK:STDOUT:       %ComparableTo.ref: %ComparableTo.type.706 = name_ref ComparableTo, file.%ComparableTo.decl [template = constants.%ComparableTo.generic]
// CHECK:STDOUT:       %T.ref.loc27_41: type = name_ref T, %T.loc27_14.1 [symbolic = %T.loc27_14.2 (constants.%T)]
// CHECK:STDOUT:       %ComparableTo.type.loc27_42.1: type = facet_type <@ComparableTo, @ComparableTo(constants.%T)> [symbolic = %ComparableTo.type.loc27_42.2 (constants.%ComparableTo.type.4ee)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc27_24.1: @impl.4.%ComparableTo.type.loc27_42.2 (%ComparableTo.type.4ee) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc27_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc27: <witness> = impl_witness (), @impl.4(constants.%T, constants.%U.b15) [symbolic = @impl.4.%impl_witness (constants.%impl_witness.595)]
// CHECK:STDOUT:   %Compare.decl: %Compare.type = fn_decl @Compare [template = constants.%Compare] {
// CHECK:STDOUT:     %T.patt.loc29_12.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc29_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc29_12.1, runtime_param<none> [symbolic = %T.patt.loc29_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc29_22.1: @Compare.%ComparableWith.type.loc29_42.2 (%ComparableWith.type.bb0) = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc29_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:     %U.param_patt: @Compare.%ComparableWith.type.loc29_42.2 (%ComparableWith.type.bb0) = value_param_pattern %U.patt.loc29_22.1, runtime_param<none> [symbolic = %U.patt.loc29_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:     %t.patt: @Compare.%T.loc29_12.2 (%T) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @Compare.%T.loc29_12.2 (%T) = value_param_pattern %t.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: @Compare.%U.as_type.loc29_54.2 (%U.as_type.2d1) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @Compare.%U.as_type.loc29_54.2 (%U.as_type.2d1) = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc29_12.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc29_12.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @Compare.%ComparableWith.type.loc29_42.2 (%ComparableWith.type.bb0) = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc29_42: type = splice_block %ComparableWith.type.loc29_42.1 [symbolic = %ComparableWith.type.loc29_42.2 (constants.%ComparableWith.type.bb0)] {
// CHECK:STDOUT:       %ComparableWith.ref: %ComparableWith.type.dbb = name_ref ComparableWith, file.%ComparableWith.decl [template = constants.%ComparableWith.generic]
// CHECK:STDOUT:       %T.ref.loc29_41: type = name_ref T, %T.loc29_12.1 [symbolic = %T.loc29_12.2 (constants.%T)]
// CHECK:STDOUT:       %ComparableWith.type.loc29_42.1: type = facet_type <@ComparableWith, @ComparableWith(constants.%T)> [symbolic = %ComparableWith.type.loc29_42.2 (constants.%ComparableWith.type.bb0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc29_22.1: @Compare.%ComparableWith.type.loc29_42.2 (%ComparableWith.type.bb0) = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc29_22.2 (constants.%U.d53)]
// CHECK:STDOUT:     %t.param: @Compare.%T.loc29_12.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %T.ref.loc29_48: type = name_ref T, %T.loc29_12.1 [symbolic = %T.loc29_12.2 (constants.%T)]
// CHECK:STDOUT:     %t: @Compare.%T.loc29_12.2 (%T) = bind_name t, %t.param
// CHECK:STDOUT:     %u.param: @Compare.%U.as_type.loc29_54.2 (%U.as_type.2d1) = value_param runtime_param1
// CHECK:STDOUT:     %.loc29_54.1: type = splice_block %.loc29_54.2 [symbolic = %U.as_type.loc29_54.2 (constants.%U.as_type.2d1)] {
// CHECK:STDOUT:       %U.ref: @Compare.%ComparableWith.type.loc29_42.2 (%ComparableWith.type.bb0) = name_ref U, %U.loc29_22.1 [symbolic = %U.loc29_22.2 (constants.%U.d53)]
// CHECK:STDOUT:       %U.as_type.loc29_54.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc29_54.2 (constants.%U.as_type.2d1)]
// CHECK:STDOUT:       %.loc29_54.2: type = converted %U.ref, %U.as_type.loc29_54.1 [symbolic = %U.as_type.loc29_54.2 (constants.%U.as_type.2d1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Compare.%U.as_type.loc29_54.2 (%U.as_type.2d1) = bind_name u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ComparableTo(%T.loc6_24.1: type) {
// CHECK:STDOUT:   %T.loc6_24.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_24.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_24.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_24.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableTo.type: type = facet_type <@ComparableTo, @ComparableTo(%T.loc6_24.2)> [symbolic = %ComparableTo.type (constants.%ComparableTo.type.4ee)]
// CHECK:STDOUT:   %Self.2: %ComparableTo.type.4ee = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ComparableTo.%ComparableTo.type (%ComparableTo.type.4ee) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ComparableWith(%T.loc8_26.1: type) {
// CHECK:STDOUT:   %T.loc8_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_26.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_26.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_26.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type: type = facet_type <@ComparableWith, @ComparableWith(%T.loc8_26.2)> [symbolic = %ComparableWith.type (constants.%ComparableWith.type.bb0)]
// CHECK:STDOUT:   %Self.2: %ComparableWith.type.bb0 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.d53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ComparableWith.%ComparableWith.type (%ComparableWith.type.bb0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.d53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.ffd566.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %ComparableTo.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc19_14.1: type) {
// CHECK:STDOUT:   %T.loc19_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc19_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ComparableWith.type.loc19_45.2: type = facet_type <@ComparableWith, @ComparableWith(%T.loc19_14.2)> [symbolic = %ComparableWith.type.loc19_45.2 (constants.%ComparableWith.type.bb0)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%ComparableWith.type.loc19_45.2 (%ComparableWith.type.bb0) [symbolic = %require_complete (constants.%require_complete.17f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T.loc19_14.2) [symbolic = %impl_witness (constants.%impl_witness.497)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc19_24 as %ComparableWith.type.loc19_45.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc25_14.1: type, %U.loc25_24.1: @impl.3.%ComparableTo.type.loc25_42.2 (%ComparableTo.type.4ee)) {
// CHECK:STDOUT:   %T.loc25_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc25_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc25_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ComparableTo.type.loc25_42.2: type = facet_type <@ComparableTo, @ComparableTo(%T.loc25_14.2)> [symbolic = %ComparableTo.type.loc25_42.2 (constants.%ComparableTo.type.4ee)]
// CHECK:STDOUT:   %U.loc25_24.2: %ComparableTo.type.4ee = bind_symbolic_name U, 1 [symbolic = %U.loc25_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   %U.patt.loc25_24.2: %ComparableTo.type.4ee = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc25_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   %U.as_type.loc25_45.2: type = facet_access_type %U.loc25_24.2 [symbolic = %U.as_type.loc25_45.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:   %ComparableWith.type.loc25_66.2: type = facet_type <@ComparableWith, @ComparableWith(%T.loc25_14.2)> [symbolic = %ComparableWith.type.loc25_66.2 (constants.%ComparableWith.type.bb0)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.3.%ComparableWith.type.loc25_66.2 (%ComparableWith.type.bb0) [symbolic = %require_complete (constants.%require_complete.17f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(%T.loc25_14.2, %U.loc25_24.2) [symbolic = %impl_witness (constants.%impl_witness.c25)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc25_45 as %ComparableWith.type.loc25_66.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%T.loc27_14.1: type, %U.loc27_24.1: @impl.4.%ComparableTo.type.loc27_42.2 (%ComparableTo.type.4ee)) {
// CHECK:STDOUT:   %T.loc27_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc27_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc27_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ComparableTo.type.loc27_42.2: type = facet_type <@ComparableTo, @ComparableTo(%T.loc27_14.2)> [symbolic = %ComparableTo.type.loc27_42.2 (constants.%ComparableTo.type.4ee)]
// CHECK:STDOUT:   %U.loc27_24.2: %ComparableTo.type.4ee = bind_symbolic_name U, 1 [symbolic = %U.loc27_24.2 (constants.%U.b15)]
// CHECK:STDOUT:   %U.patt.loc27_24.2: %ComparableTo.type.4ee = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc27_24.2 (constants.%U.patt.16f)]
// CHECK:STDOUT:   %U.as_type.loc27_66.2: type = facet_access_type %U.loc27_24.2 [symbolic = %U.as_type.loc27_66.2 (constants.%U.as_type.b21)]
// CHECK:STDOUT:   %ComparableWith.type.loc27_66.2: type = facet_type <@ComparableWith, @ComparableWith(%U.as_type.loc27_66.2)> [symbolic = %ComparableWith.type.loc27_66.2 (constants.%ComparableWith.type.73a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.4.%ComparableWith.type.loc27_66.2 (%ComparableWith.type.73a) [symbolic = %require_complete (constants.%require_complete.86a)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.4(%T.loc27_14.2, %U.loc27_24.2) [symbolic = %impl_witness (constants.%impl_witness.595)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc27_45 as %ComparableWith.type.loc27_66.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compare(%T.loc29_12.1: type, %U.loc29_22.1: @Compare.%ComparableWith.type.loc29_42.2 (%ComparableWith.type.bb0)) {
// CHECK:STDOUT:   %T.loc29_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc29_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc29_12.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc29_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2: type = facet_type <@ComparableWith, @ComparableWith(%T.loc29_12.2)> [symbolic = %ComparableWith.type.loc29_42.2 (constants.%ComparableWith.type.bb0)]
// CHECK:STDOUT:   %U.loc29_22.2: %ComparableWith.type.bb0 = bind_symbolic_name U, 1 [symbolic = %U.loc29_22.2 (constants.%U.d53)]
// CHECK:STDOUT:   %U.patt.loc29_22.2: %ComparableWith.type.bb0 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc29_22.2 (constants.%U.patt.761)]
// CHECK:STDOUT:   %U.as_type.loc29_54.2: type = facet_access_type %U.loc29_22.2 [symbolic = %U.as_type.loc29_54.2 (constants.%U.as_type.2d1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc29_46: <witness> = require_complete_type @Compare.%T.loc29_12.2 (%T) [symbolic = %require_complete.loc29_46 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc29_52: <witness> = require_complete_type @Compare.%U.as_type.loc29_54.2 (%U.as_type.2d1) [symbolic = %require_complete.loc29_52 (constants.%require_complete.9df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %U.param_patt: @Compare.%ComparableWith.type.loc29_42.2 (%ComparableWith.type.bb0)](%t.param_patt: @Compare.%T.loc29_12.2 (%T), %u.param_patt: @Compare.%U.as_type.loc29_54.2 (%U.as_type.2d1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Compare.ref.loc32: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc32_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc32_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc32_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc32_12.3: init %C = class_init (), %.loc32_12.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc32_12.4: ref %C = temporary %.loc32_12.2, %.loc32_12.3
// CHECK:STDOUT:   %.loc32_14.1: ref %C = converted %.loc32_12.1, %.loc32_12.4
// CHECK:STDOUT:   %.loc32_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc32_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc32_21.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc32_21.3: init %C = class_init (), %.loc32_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc32_21.4: ref %C = temporary %.loc32_21.2, %.loc32_21.3
// CHECK:STDOUT:   %.loc32_23.1: ref %C = converted %.loc32_21.1, %.loc32_21.4
// CHECK:STDOUT:   %ComparableWith.facet.loc32: %ComparableWith.type.68c = facet_value constants.%C, constants.%impl_witness.887 [template = constants.%ComparableWith.facet.17d]
// CHECK:STDOUT:   %.loc32_27: %ComparableWith.type.68c = converted constants.%C, %ComparableWith.facet.loc32 [template = constants.%ComparableWith.facet.17d]
// CHECK:STDOUT:   %Compare.specific_fn.loc32: <specific function> = specific_function %Compare.ref.loc32, @Compare(constants.%C, %.loc32_27) [template = constants.%Compare.specific_fn.041]
// CHECK:STDOUT:   %.loc32_14.2: %C = bind_value %.loc32_14.1
// CHECK:STDOUT:   %.loc32_23.2: %C = bind_value %.loc32_23.1
// CHECK:STDOUT:   %Compare.call.loc32: init %empty_tuple.type = call %Compare.specific_fn.loc32(%.loc32_14.2, %.loc32_23.2)
// CHECK:STDOUT:   %Compare.ref.loc46: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc46_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc46: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc46_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc46_12.3: init %C = class_init (), %.loc46_12.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc46_12.4: ref %C = temporary %.loc46_12.2, %.loc46_12.3
// CHECK:STDOUT:   %.loc46_14.1: ref %C = converted %.loc46_12.1, %.loc46_12.4
// CHECK:STDOUT:   %.loc46_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref.loc46: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc46_21.2: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc46_21.3: init %D = class_init (), %.loc46_21.2 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc46_21.4: ref %D = temporary %.loc46_21.2, %.loc46_21.3
// CHECK:STDOUT:   %.loc46_23.1: ref %D = converted %.loc46_21.1, %.loc46_21.4
// CHECK:STDOUT:   %.loc46_27.1: %ComparableTo.type.406 = converted constants.%D, <error> [template = <error>]
// CHECK:STDOUT:   %ComparableTo.facet.loc46: %ComparableTo.type.4f3 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet]
// CHECK:STDOUT:   %.loc46_27.2: %ComparableTo.type.4f3 = converted constants.%C, %ComparableTo.facet.loc46 [template = constants.%ComparableTo.facet]
// CHECK:STDOUT:   %ComparableWith.facet.loc46: %ComparableWith.type.68c = facet_value constants.%D, constants.%impl_witness.228 [template = constants.%ComparableWith.facet.aa3]
// CHECK:STDOUT:   %.loc46_27.3: %ComparableWith.type.68c = converted constants.%D, %ComparableWith.facet.loc46 [template = constants.%ComparableWith.facet.aa3]
// CHECK:STDOUT:   %Compare.specific_fn.loc46: <specific function> = specific_function %Compare.ref.loc46, @Compare(constants.%C, %.loc46_27.3) [template = constants.%Compare.specific_fn.94d]
// CHECK:STDOUT:   %.loc46_14.2: %C = bind_value %.loc46_14.1
// CHECK:STDOUT:   %.loc46_23.2: %D = bind_value %.loc46_23.1
// CHECK:STDOUT:   %Compare.call.loc46: init %empty_tuple.type = call %Compare.specific_fn.loc46(%.loc46_14.2, %.loc46_23.2)
// CHECK:STDOUT:   %Compare.ref.loc47: %Compare.type = name_ref Compare, file.%Compare.decl [template = constants.%Compare]
// CHECK:STDOUT:   %.loc47_12.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref.loc47: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc47_12.2: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc47_12.3: init %D = class_init (), %.loc47_12.2 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc47_12.4: ref %D = temporary %.loc47_12.2, %.loc47_12.3
// CHECK:STDOUT:   %.loc47_14.1: ref %D = converted %.loc47_12.1, %.loc47_12.4
// CHECK:STDOUT:   %.loc47_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc47: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc47_21.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc47_21.3: init %C = class_init (), %.loc47_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc47_21.4: ref %C = temporary %.loc47_21.2, %.loc47_21.3
// CHECK:STDOUT:   %.loc47_23.1: ref %C = converted %.loc47_21.1, %.loc47_21.4
// CHECK:STDOUT:   %ComparableTo.facet.loc47: %ComparableTo.type.4f3 = facet_value constants.%C, constants.%impl_witness.1bc [template = constants.%ComparableTo.facet]
// CHECK:STDOUT:   %.loc47_27.1: %ComparableTo.type.4f3 = converted constants.%C, %ComparableTo.facet.loc47 [template = constants.%ComparableTo.facet]
// CHECK:STDOUT:   %ComparableWith.facet.loc47: %ComparableWith.type.c33 = facet_value constants.%C, constants.%impl_witness.d37 [template = constants.%ComparableWith.facet.7c0]
// CHECK:STDOUT:   %.loc47_27.2: %ComparableWith.type.c33 = converted constants.%C, %ComparableWith.facet.loc47 [template = constants.%ComparableWith.facet.7c0]
// CHECK:STDOUT:   %Compare.specific_fn.loc47: <specific function> = specific_function %Compare.ref.loc47, @Compare(constants.%D, %.loc47_27.2) [template = constants.%Compare.specific_fn.f4d]
// CHECK:STDOUT:   %.loc47_14.2: %D = bind_value %.loc47_14.1
// CHECK:STDOUT:   %.loc47_23.2: %C = bind_value %.loc47_23.1
// CHECK:STDOUT:   %Compare.call.loc47: init %empty_tuple.type = call %Compare.specific_fn.loc47(%.loc47_14.2, %.loc47_23.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.ffd566.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%T) {
// CHECK:STDOUT:   %T.loc6_24.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_24.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(%T.loc6_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%T) {
// CHECK:STDOUT:   %T.loc8_26.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_26.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type => constants.%ComparableWith.type.bb0
// CHECK:STDOUT:   %Self.2 => constants.%Self.d53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(%T.loc8_26.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%D) {
// CHECK:STDOUT:   %T.loc6_24.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc6_24.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableTo.type => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %Self.2 => constants.%Self.b15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T.loc19_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc19_14.2 => constants.%T
// CHECK:STDOUT:   %ComparableWith.type.loc19_45.2 => constants.%ComparableWith.type.bb0
// CHECK:STDOUT:   %require_complete => constants.%require_complete.17f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.497
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.2.%T.loc19_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T.loc19_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T, constants.%U.b15) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%T
// CHECK:STDOUT:   %ComparableTo.type.loc25_42.2 => constants.%ComparableTo.type.4ee
// CHECK:STDOUT:   %U.loc25_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.patt.loc25_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.as_type.loc25_45.2 => constants.%U.as_type.b21
// CHECK:STDOUT:   %ComparableWith.type.loc25_66.2 => constants.%ComparableWith.type.bb0
// CHECK:STDOUT:   %require_complete => constants.%require_complete.17f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.c25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(@impl.3.%T.loc25_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.3.%T.loc25_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%T.loc25_14.2, %U.loc25_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%U.as_type.b21) {
// CHECK:STDOUT:   %T.loc8_26.2 => constants.%U.as_type.b21
// CHECK:STDOUT:   %T.patt.loc8_26.2 => constants.%U.as_type.b21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ComparableWith.type => constants.%ComparableWith.type.73a
// CHECK:STDOUT:   %Self.2 => constants.%Self.d53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%T, constants.%U.b15) {
// CHECK:STDOUT:   %T.loc27_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc27_14.2 => constants.%T
// CHECK:STDOUT:   %ComparableTo.type.loc27_42.2 => constants.%ComparableTo.type.4ee
// CHECK:STDOUT:   %U.loc27_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.patt.loc27_24.2 => constants.%U.b15
// CHECK:STDOUT:   %U.as_type.loc27_66.2 => constants.%U.as_type.b21
// CHECK:STDOUT:   %ComparableWith.type.loc27_66.2 => constants.%ComparableWith.type.73a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.86a
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.595
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(@impl.4.%T.loc27_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@impl.4.%U.as_type.loc27_66.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%T.loc27_14.2, %U.loc27_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%T, constants.%U.d53) {
// CHECK:STDOUT:   %T.loc29_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc29_12.2 => constants.%T
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2 => constants.%ComparableWith.type.bb0
// CHECK:STDOUT:   %U.loc29_22.2 => constants.%U.d53
// CHECK:STDOUT:   %U.patt.loc29_22.2 => constants.%U.d53
// CHECK:STDOUT:   %U.as_type.loc29_54.2 => constants.%U.as_type.2d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(@Compare.%T.loc29_12.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%C) {
// CHECK:STDOUT:   %T.loc8_26.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc8_26.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%C) {
// CHECK:STDOUT:   %T.loc19_14.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc19_14.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc19_45.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.887
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, @F.%.loc32_27) {
// CHECK:STDOUT:   %T.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc29_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.patt.loc29_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.as_type.loc29_54.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc29_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc29_52 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, constants.%ComparableWith.facet.17d) {
// CHECK:STDOUT:   %T.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc29_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.patt.loc29_22.2 => constants.%ComparableWith.facet.17d
// CHECK:STDOUT:   %U.as_type.loc29_54.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableTo(constants.%C) {
// CHECK:STDOUT:   %T.loc6_24.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_24.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%D) {
// CHECK:STDOUT:   %T.loc19_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc19_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc19_45.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9db
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(constants.%D) {
// CHECK:STDOUT:   %T.loc8_26.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc8_26.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%ComparableTo.type.406) {
// CHECK:STDOUT:   %Dest => constants.%ComparableTo.type.406
// CHECK:STDOUT:   %Dest.patt => constants.%ComparableTo.type.406
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.632
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.cd0
// CHECK:STDOUT:   %Convert => constants.%Convert.088
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.38f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(type) {
// CHECK:STDOUT:   %T.loc19_14.2 => type
// CHECK:STDOUT:   %T.patt.loc19_14.2 => type
// CHECK:STDOUT:   %ComparableWith.type.loc19_45.2 => constants.%ComparableWith.type.7aa
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3dc
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ComparableWith(type) {
// CHECK:STDOUT:   %T.loc8_26.2 => type
// CHECK:STDOUT:   %T.patt.loc8_26.2 => type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%C, <error>) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%C
// CHECK:STDOUT:   %ComparableTo.type.loc25_42.2 => constants.%ComparableTo.type.406
// CHECK:STDOUT:   %U.loc25_24.2 => <error>
// CHECK:STDOUT:   %U.patt.loc25_24.2 => <error>
// CHECK:STDOUT:   %U.as_type.loc25_45.2 => <error>
// CHECK:STDOUT:   %ComparableWith.type.loc25_66.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%D, @F.%.loc46_27.2) {
// CHECK:STDOUT:   %T.loc27_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc27_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc27_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc27_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.patt.loc27_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.as_type.loc27_66.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc27_66.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.228
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%D, constants.%ComparableTo.facet) {
// CHECK:STDOUT:   %T.loc27_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc27_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc27_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc27_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.patt.loc27_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.as_type.loc27_66.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc27_66.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa9
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, @F.%.loc46_27.3) {
// CHECK:STDOUT:   %T.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc29_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.patt.loc29_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.as_type.loc29_54.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc29_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc29_52 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%C, constants.%ComparableWith.facet.aa3) {
// CHECK:STDOUT:   %T.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc29_12.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2 => constants.%ComparableWith.type.68c
// CHECK:STDOUT:   %U.loc29_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.patt.loc29_22.2 => constants.%ComparableWith.facet.aa3
// CHECK:STDOUT:   %U.as_type.loc29_54.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%D, @F.%.loc47_27.1) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc25_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc25_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.patt.loc25_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.as_type.loc25_45.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc25_66.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9db
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.d37
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%D, constants.%ComparableTo.facet) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%D
// CHECK:STDOUT:   %ComparableTo.type.loc25_42.2 => constants.%ComparableTo.type.4f3
// CHECK:STDOUT:   %U.loc25_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.patt.loc25_24.2 => constants.%ComparableTo.facet
// CHECK:STDOUT:   %U.as_type.loc25_45.2 => constants.%C
// CHECK:STDOUT:   %ComparableWith.type.loc25_66.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9db
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.d37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%D, @F.%.loc47_27.2) {
// CHECK:STDOUT:   %T.loc29_12.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc29_12.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %U.loc29_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.patt.loc29_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.as_type.loc29_54.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc29_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc29_52 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compare(constants.%D, constants.%ComparableWith.facet.7c0) {
// CHECK:STDOUT:   %T.loc29_12.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc29_12.2 => constants.%D
// CHECK:STDOUT:   %ComparableWith.type.loc29_42.2 => constants.%ComparableWith.type.c33
// CHECK:STDOUT:   %U.loc29_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.patt.loc29_22.2 => constants.%ComparableWith.facet.7c0
// CHECK:STDOUT:   %U.as_type.loc29_54.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
