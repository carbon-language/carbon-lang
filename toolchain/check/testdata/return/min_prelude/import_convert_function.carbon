// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/min_prelude/import_convert_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/min_prelude/import_convert_function.carbon
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:12:1: error: duplicate name `Int` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class Int(N:! IntLiteral()) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// --- core.carbon

// A minimized prelude intended to contain as few functions as possible.
package Core;

fn IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDERR: core.carbon:[[@LINE+3]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn Int(N: IntLiteral()) -> type = "int.make_type_signed";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl IntLiteral() as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

// --- library.carbon

package P library "[[@TEST_NAME]]";

import Core;

// CHECK:STDERR: library.carbon:[[@LINE+10]]:13: note: in name lookup for `Int` [InNameLookup]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/operators/as.carbon:11:1: error: duplicate name `ImplicitAs` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: core.carbon:11:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class C(N:! i32) {}
class D { var n: i32; var m: i32; }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:25: note: in name lookup for `ImplicitAs` [InNameLookup]
// CHECK:STDERR: fn Make() -> D { return {.n = 0, .m = 0}; }
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:12:1: error: duplicate name `Int` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class Int(N:! IntLiteral()) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: core.carbon:9:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn Int(N: IntLiteral()) -> type = "int.make_type_signed";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn Make() -> D { return {.n = 0, .m = 0}; }

impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }

// --- user.carbon

import P library "library";
import Core;

// Check that we don't crash if we import functions and reallocate the function
// value store while handling a `return` statement.
// CHECK:STDERR: user.carbon:[[@LINE+10]]:10: note: in name lookup for `Int` [InNameLookup]
// CHECK:STDERR: fn F0(n: i32) -> P.D {
// CHECK:STDERR:          ^~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/operators/as.carbon:11:1: error: duplicate name `ImplicitAs` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface ImplicitAs(Dest:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: core.carbon:11:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn F0(n: i32) -> P.D {
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: in name lookup for `ImplicitAs` [InNameLookup]
  // CHECK:STDERR:   if (false) { return {} as P.C(0); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:16:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(0); }
  if (false) { return {} as P.C(1); }
  if (false) { return {} as P.C(2); }
  if (false) { return {} as P.C(3); }
  if (false) { return {} as P.C(4); }
  if (false) { return {} as P.C(5); }
  if (false) { return {} as P.C(6); }
  if (false) { return {} as P.C(7); }
  return P.Make();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.095: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type.095 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.11a: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.e0a: %ImplicitAs.type.11a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.752: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.fcc: %Convert.type.752 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.1cf: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.3ce: %ImplicitAs.assoc_type.1cf = assoc_entity element0, @ImplicitAs.%Convert.decl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness file.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Convert.type.c2a: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.40d: %Convert.type.c2a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.11a = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_22.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_22.3: type = converted %int_literal.make_type, %.loc9_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc11_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc15_17.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc15_17.2: type = converted %int_literal.make_type, %.loc15_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_signed, %.loc15_36.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.11a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (@impl.%Convert.decl), @impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc11_22.1: type) {
// CHECK:STDOUT:   %T.loc11_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc11_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc11_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T.loc11_22.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.095)]
// CHECK:STDOUT:   %assoc0.loc12_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.095) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc11_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc12_20.2: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc12_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type.loc12_20.2 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.095) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc12_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc15_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.c2a = fn_decl @Convert.2 [concrete = constants.%Convert.40d] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc16_31.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc16_31.2: type = converted %int.make_type_signed, %.loc16_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc15_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc11_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc12_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type)) -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2(%self.param: Core.IntLiteral) -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc11_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_22.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.0f3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc11_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc11_22.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc11_22.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self.2 => constants.%Self.e0a
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.752
// CHECK:STDOUT:   %Convert => constants.%Convert.fcc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.1cf
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.3ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32.builtin, constants.%ImplicitAs.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.116: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32.builtin [concrete]
// CHECK:STDOUT:   %struct_type.n.m.566: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete]
// CHECK:STDOUT:   %complete_type.682: <witness> = complete_type_witness %struct_type.n.m.566 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.61e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.d28: %ImplicitAs.type.61e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.059: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.4d7: %Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.398: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.6fd: %ImplicitAs.assoc_type.398 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (imports.%Core.import_ref.f35), @impl.68b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.07a: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a3d: %ImplicitAs.type.61e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.07a) [concrete]
// CHECK:STDOUT:   %.ec1: type = fn_type_with_self_type %Convert.type.059, %ImplicitAs.facet.a3d [concrete]
// CHECK:STDOUT:   %Convert.type.49f: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.cb5: %Convert.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.04a: <bound method> = bound_method %int_0.5c6, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_0.a54: %i32.builtin = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.a54, %int_0.a54) [concrete]
// CHECK:STDOUT:   %C.808: type = class_type @C, @C(%int_0.a54) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.94e: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %Self.371: %ImplicitAs.type.94e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.010: type = fn_type @Convert.1, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %Convert.d38: %Convert.type.010 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.75e: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %assoc0.579: %ImplicitAs.assoc_type.75e = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4af: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc30 [concrete]
// CHECK:STDOUT:   %Convert.type.fff: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.606: %Convert.type.fff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a19: %ImplicitAs.type.94e = facet_value %C.808, (%ImplicitAs.impl_witness.4af) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.92d: <bound method> = bound_method %int_1.5b8, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %C.8be: type = class_type @C, @C(%int_1.f38) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fcc: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc31 [concrete]
// CHECK:STDOUT:   %Convert.type.89f: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.689: %Convert.type.89f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.340: %ImplicitAs.type.94e = facet_value %C.8be, (%ImplicitAs.impl_witness.fcc) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.1b9: <bound method> = bound_method %int_2.ecc, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %C.c17: type = class_type @C, @C(%int_2.5a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.03e: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc32 [concrete]
// CHECK:STDOUT:   %Convert.type.e90: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.ec9: %Convert.type.e90 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ec0: %ImplicitAs.type.94e = facet_value %C.c17, (%ImplicitAs.impl_witness.03e) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b6b: <bound method> = bound_method %int_3.1ba, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %C.414: type = class_type @C, @C(%int_3.a0f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.dec: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc33 [concrete]
// CHECK:STDOUT:   %Convert.type.5db: type = fn_type @Convert.6 [concrete]
// CHECK:STDOUT:   %Convert.193: %Convert.type.5db = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.999: %ImplicitAs.type.94e = facet_value %C.414, (%ImplicitAs.impl_witness.dec) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.626: <bound method> = bound_method %int_4.0c1, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %C.488: type = class_type @C, @C(%int_4.4f1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.842: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc34 [concrete]
// CHECK:STDOUT:   %Convert.type.4e7: type = fn_type @Convert.7 [concrete]
// CHECK:STDOUT:   %Convert.52a: %Convert.type.4e7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.527: %ImplicitAs.type.94e = facet_value %C.488, (%ImplicitAs.impl_witness.842) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.910: <bound method> = bound_method %int_5.64b, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_5.967: %i32.builtin = int_value 5 [concrete]
// CHECK:STDOUT:   %C.3e2: type = class_type @C, @C(%int_5.967) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.eb5: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc35 [concrete]
// CHECK:STDOUT:   %Convert.type.658: type = fn_type @Convert.8 [concrete]
// CHECK:STDOUT:   %Convert.9b6: %Convert.type.658 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.79a: %ImplicitAs.type.94e = facet_value %C.3e2, (%ImplicitAs.impl_witness.eb5) [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.e3a: <bound method> = bound_method %int_6.462, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_6.ec5: %i32.builtin = int_value 6 [concrete]
// CHECK:STDOUT:   %C.78c: type = class_type @C, @C(%int_6.ec5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.377: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc36 [concrete]
// CHECK:STDOUT:   %Convert.type.623: type = fn_type @Convert.9 [concrete]
// CHECK:STDOUT:   %Convert.8c9: %Convert.type.623 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5ba: %ImplicitAs.type.94e = facet_value %C.78c, (%ImplicitAs.impl_witness.377) [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.06a: <bound method> = bound_method %int_7.29f, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_7.6ae: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %C.6aa: type = class_type @C, @C(%int_7.6ae) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.308: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc37 [concrete]
// CHECK:STDOUT:   %Convert.type.3f8: type = fn_type @Convert.10 [concrete]
// CHECK:STDOUT:   %Convert.71a: %Convert.type.3f8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b5c: %ImplicitAs.type.94e = facet_value %C.6aa, (%ImplicitAs.impl_witness.308) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc11_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst49 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//default, loc12_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc11_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst49 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc12_32, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.c62: <witness> = import_ref Core//default, loc15_38, loaded [concrete = constants.%ImplicitAs.impl_witness.07a]
// CHECK:STDOUT:   %Core.import_ref.872: type = import_ref Core//default, loc15_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.4d9: type = import_ref Core//default, loc15_36, loaded [concrete = constants.%ImplicitAs.type.61e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %N.patt.loc16_9.1: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc16_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_13.1: type = splice_block %.loc16_13.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc16_13.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc16_13.3: type = converted %int.make_type_signed, %.loc16_13.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc16_9.1: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc16_9.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.837 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_0) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:     %.loc30_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_0.a54]
// CHECK:STDOUT:     %.loc30_9.2: %i32.builtin = converted %int_0, %.loc30_9.1 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.a54) [concrete = constants.%C.808]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc30 = impl_witness_table (@impl.837.%Convert.decl), @impl.837 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc30: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc30 [concrete = constants.%ImplicitAs.impl_witness.4af]
// CHECK:STDOUT:   impl_decl @impl.511 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound.92d]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc31_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc31_9.2: %i32.builtin = converted %int_1, %.loc31_9.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.f38) [concrete = constants.%C.8be]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc31 = impl_witness_table (@impl.511.%Convert.decl), @impl.511 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc31: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc31 [concrete = constants.%ImplicitAs.impl_witness.fcc]
// CHECK:STDOUT:   impl_decl @impl.141 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound.1b9]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc32_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc32_9.2: %i32.builtin = converted %int_2, %.loc32_9.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.5a1) [concrete = constants.%C.c17]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc32 = impl_witness_table (@impl.141.%Convert.decl), @impl.141 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc32: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc32 [concrete = constants.%ImplicitAs.impl_witness.03e]
// CHECK:STDOUT:   impl_decl @impl.e6b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Convert.bound.b6b]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_3) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc33_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc33_9.2: %i32.builtin = converted %int_3, %.loc33_9.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.a0f) [concrete = constants.%C.414]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc33 = impl_witness_table (@impl.e6b.%Convert.decl), @impl.e6b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc33: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc33 [concrete = constants.%ImplicitAs.impl_witness.dec]
// CHECK:STDOUT:   impl_decl @impl.056 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Convert.bound.626]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_4) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:     %.loc34_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:     %.loc34_9.2: %i32.builtin = converted %int_4, %.loc34_9.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.4f1) [concrete = constants.%C.488]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc34 = impl_witness_table (@impl.056.%Convert.decl), @impl.056 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc34: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc34 [concrete = constants.%ImplicitAs.impl_witness.842]
// CHECK:STDOUT:   impl_decl @impl.154 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Convert.bound.910]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_5) [concrete = constants.%int_5.967]
// CHECK:STDOUT:     %.loc35_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_5.967]
// CHECK:STDOUT:     %.loc35_9.2: %i32.builtin = converted %int_5, %.loc35_9.1 [concrete = constants.%int_5.967]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.967) [concrete = constants.%C.3e2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc35 = impl_witness_table (@impl.154.%Convert.decl), @impl.154 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc35: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc35 [concrete = constants.%ImplicitAs.impl_witness.eb5]
// CHECK:STDOUT:   impl_decl @impl.a6d [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Convert.bound.e3a]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_6) [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:     %.loc36_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:     %.loc36_9.2: %i32.builtin = converted %int_6, %.loc36_9.1 [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.ec5) [concrete = constants.%C.78c]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc36 = impl_witness_table (@impl.a6d.%Convert.decl), @impl.a6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc36: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc36 [concrete = constants.%ImplicitAs.impl_witness.377]
// CHECK:STDOUT:   impl_decl @impl.ebc [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Convert.bound.06a]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_7) [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:     %.loc37_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:     %.loc37_9.2: %i32.builtin = converted %int_7, %.loc37_9.1 [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.6ae) [concrete = constants.%C.6aa]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc37 = impl_witness_table (@impl.ebc.%Convert.decl), @impl.ebc [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc37: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc37 [concrete = constants.%ImplicitAs.impl_witness.308]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.68b: imports.%Core.import_ref.872 as imports.%Core.import_ref.4d9 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.c62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.837: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.fff = fn_decl @Convert.3 [concrete = constants.%Convert.606] {
// CHECK:STDOUT:     %self.patt: %C.808 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.808 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.808 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.837.%C [concrete = constants.%C.808]
// CHECK:STDOUT:     %self: %C.808 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.511: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.89f = fn_decl @Convert.4 [concrete = constants.%Convert.689] {
// CHECK:STDOUT:     %self.patt: %C.8be = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.8be = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.8be = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.511.%C [concrete = constants.%C.8be]
// CHECK:STDOUT:     %self: %C.8be = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.141: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e90 = fn_decl @Convert.5 [concrete = constants.%Convert.ec9] {
// CHECK:STDOUT:     %self.patt: %C.c17 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.c17 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c17 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.141.%C [concrete = constants.%C.c17]
// CHECK:STDOUT:     %self: %C.c17 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.e6b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.5db = fn_decl @Convert.6 [concrete = constants.%Convert.193] {
// CHECK:STDOUT:     %self.patt: %C.414 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.414 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.414 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.e6b.%C [concrete = constants.%C.414]
// CHECK:STDOUT:     %self: %C.414 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.056: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.4e7 = fn_decl @Convert.7 [concrete = constants.%Convert.52a] {
// CHECK:STDOUT:     %self.patt: %C.488 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.488 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.488 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.056.%C [concrete = constants.%C.488]
// CHECK:STDOUT:     %self: %C.488 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.154: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.658 = fn_decl @Convert.8 [concrete = constants.%Convert.9b6] {
// CHECK:STDOUT:     %self.patt: %C.3e2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.3e2 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.3e2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.154.%C [concrete = constants.%C.3e2]
// CHECK:STDOUT:     %self: %C.3e2 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.a6d: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.623 = fn_decl @Convert.9 [concrete = constants.%Convert.8c9] {
// CHECK:STDOUT:     %self.patt: %C.78c = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.78c = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.78c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.a6d.%C [concrete = constants.%C.78c]
// CHECK:STDOUT:     %self: %C.78c = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.ebc: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.3f8 = fn_decl @Convert.10 [concrete = constants.%Convert.71a] {
// CHECK:STDOUT:     %self.patt: %C.6aa = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.6aa = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.6aa = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.ebc.%C [concrete = constants.%C.6aa]
// CHECK:STDOUT:     %self: %C.6aa = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc16_9.1: %i32.builtin) {
// CHECK:STDOUT:   %N.loc16_9.2: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc16_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc16_9.2: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc16_9.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.116
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %int_32.loc17_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17_18: init type = call constants.%Int(%int_32.loc17_18) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc17_18.1: type = value_of_initializer %int.make_type_signed.loc17_18 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc17_18.2: type = converted %int.make_type_signed.loc17_18, %.loc17_18.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc17_16: %D.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc17_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17_30: init type = call constants.%Int(%int_32.loc17_30) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc17_30.1: type = value_of_initializer %int.make_type_signed.loc17_30 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc17_30.2: type = converted %int.make_type_signed.loc17_30, %.loc17_30.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc17_28: %D.elem = field_decl m, element1 [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete = constants.%struct_type.n.m.566]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.m [concrete = constants.%complete_type.682]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc17_16
// CHECK:STDOUT:   .m = %.loc17_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc28_31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc28_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc28_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc28_31, %int_0.loc28_39)
// CHECK:STDOUT:   %impl.elem0.loc28_40.1: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc28_40.1: <bound method> = bound_method %int_0.loc28_31, %impl.elem0.loc28_40.1 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:   %int.convert_checked.loc28_40.1: init %i32.builtin = call %bound_method.loc28_40.1(%int_0.loc28_31) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc28_40.2: init %i32.builtin = converted %int_0.loc28_31, %int.convert_checked.loc28_40.1 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc28_40.3: ref %i32.builtin = class_element_access %return, element0
// CHECK:STDOUT:   %.loc28_40.4: init %i32.builtin = initialize_from %.loc28_40.2 to %.loc28_40.3 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %impl.elem0.loc28_40.2: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc28_40.2: <bound method> = bound_method %int_0.loc28_39, %impl.elem0.loc28_40.2 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:   %int.convert_checked.loc28_40.2: init %i32.builtin = call %bound_method.loc28_40.2(%int_0.loc28_39) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc28_40.5: init %i32.builtin = converted %int_0.loc28_39, %int.convert_checked.loc28_40.2 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc28_40.6: ref %i32.builtin = class_element_access %return, element1
// CHECK:STDOUT:   %.loc28_40.7: init %i32.builtin = initialize_from %.loc28_40.5 to %.loc28_40.6 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc28_40.8: init %D = class_init (%.loc28_40.4, %.loc28_40.7), %return [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc28_41: init %D = converted %.loc28_40.1, %.loc28_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc28_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2 = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3(%self.param: %C.808) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc30: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc30
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4(%self.param: %C.8be) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc31: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc31
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5(%self.param: %C.c17) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc32: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc32
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6(%self.param: %C.414) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc33: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc33
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7(%self.param: %C.488) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc34: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc34
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8(%self.param: %C.3e2) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc35: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc35
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9(%self.param: %C.78c) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc36: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc36
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10(%self.param: %C.6aa) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc37: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc37
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.61e
// CHECK:STDOUT:   %Self => constants.%Self.d28
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.059
// CHECK:STDOUT:   %Convert => constants.%Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.398
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.a54) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_0.a54
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%Self.371
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.010
// CHECK:STDOUT:   %Convert => constants.%Convert.d38
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.75e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.579
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.a19) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.a19
// CHECK:STDOUT:   %Self.as_type => constants.%C.808
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.f38) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_1.f38
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.340) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.340
// CHECK:STDOUT:   %Self.as_type => constants.%C.8be
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.5a1) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_2.5a1
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.ec0) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.ec0
// CHECK:STDOUT:   %Self.as_type => constants.%C.c17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.a0f) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_3.a0f
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.999) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.999
// CHECK:STDOUT:   %Self.as_type => constants.%C.414
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4f1) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_4.4f1
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.527) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.527
// CHECK:STDOUT:   %Self.as_type => constants.%C.488
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.967) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_5.967
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.79a) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.79a
// CHECK:STDOUT:   %Self.as_type => constants.%C.3e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.ec5) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_6.ec5
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.5ba) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.5ba
// CHECK:STDOUT:   %Self.as_type => constants.%C.78c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.6ae) {
// CHECK:STDOUT:   %N.loc16_9.2 => constants.%int_7.6ae
// CHECK:STDOUT:   %N.patt.loc16_9.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.b5c) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.b5c
// CHECK:STDOUT:   %Self.as_type => constants.%C.6aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete]
// CHECK:STDOUT:   %complete_type.682: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.61e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.d28: %ImplicitAs.type.61e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.059: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.4d7: %Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.398: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.6fd: %ImplicitAs.assoc_type.398 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.78e = impl_witness_table (imports.%Core.import_ref.f35), @impl.68b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.07a: <witness> = impl_witness %ImplicitAs.impl_witness_table.78e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a3d: %ImplicitAs.type.61e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.07a) [concrete]
// CHECK:STDOUT:   %.ec1e: type = fn_type_with_self_type %Convert.type.059, %ImplicitAs.facet.a3d [concrete]
// CHECK:STDOUT:   %Convert.type.49f: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.cb5: %Convert.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.04a: <bound method> = bound_method %int_0.5c6, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_0.a54: %i32.builtin = int_value 0 [concrete]
// CHECK:STDOUT:   %C.76d: type = class_type @C, @C(%int_0.a54) [concrete]
// CHECK:STDOUT:   %C.val.3c3: %C.76d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5f9: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %Self.656: %ImplicitAs.type.5f9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.334: type = fn_type @Convert.1, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %Convert.87c: %Convert.type.334 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.953: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %assoc0.8a3: %ImplicitAs.assoc_type.953 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %C.012: type = class_type @C, @C(%int_1.f38) [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %C.3e6: type = class_type @C, @C(%int_2.5a1) [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %C.4d9: type = class_type @C, @C(%int_3.a0f) [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %C.a67: type = class_type @C, @C(%int_4.4f1) [concrete]
// CHECK:STDOUT:   %int_5.967: %i32.builtin = int_value 5 [concrete]
// CHECK:STDOUT:   %C.65c: type = class_type @C, @C(%int_5.967) [concrete]
// CHECK:STDOUT:   %int_6.ec5: %i32.builtin = int_value 6 [concrete]
// CHECK:STDOUT:   %C.898: type = class_type @C, @C(%int_6.ec5) [concrete]
// CHECK:STDOUT:   %int_7.6ae: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %C.f0a: type = class_type @C, @C(%int_7.6ae) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.497 = impl_witness_table (imports.%P.import_ref.5f3), @impl.b24 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6e5: <witness> = impl_witness %ImplicitAs.impl_witness_table.497 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5c5: %ImplicitAs.type.5f9 = facet_value %C.76d, (%ImplicitAs.impl_witness.6e5) [concrete]
// CHECK:STDOUT:   %.18d: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.5c5 [concrete]
// CHECK:STDOUT:   %Convert.type.c72: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.4f5: %Convert.type.c72 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.92d: <bound method> = bound_method %int_1.5b8, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.172: %C.012 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b5d = impl_witness_table (imports.%P.import_ref.4da), @impl.277 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.70d: <witness> = impl_witness %ImplicitAs.impl_witness_table.b5d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.930: %ImplicitAs.type.5f9 = facet_value %C.012, (%ImplicitAs.impl_witness.70d) [concrete]
// CHECK:STDOUT:   %.d7a: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.930 [concrete]
// CHECK:STDOUT:   %Convert.type.d88: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.ecc: %Convert.type.d88 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.1b9: <bound method> = bound_method %int_2.ecc, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.418: %C.3e6 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ec0 = impl_witness_table (imports.%P.import_ref.ebf), @impl.b5f [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f8e: <witness> = impl_witness %ImplicitAs.impl_witness_table.ec0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.05e: %ImplicitAs.type.5f9 = facet_value %C.3e6, (%ImplicitAs.impl_witness.f8e) [concrete]
// CHECK:STDOUT:   %.e6a: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.05e [concrete]
// CHECK:STDOUT:   %Convert.type.c17: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.c2b: %Convert.type.c17 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b6b: <bound method> = bound_method %int_3.1ba, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.bba: %C.4d9 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.129 = impl_witness_table (imports.%P.import_ref.f13), @impl.90b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bb8: <witness> = impl_witness %ImplicitAs.impl_witness_table.129 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eb1: %ImplicitAs.type.5f9 = facet_value %C.4d9, (%ImplicitAs.impl_witness.bb8) [concrete]
// CHECK:STDOUT:   %.40b: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.eb1 [concrete]
// CHECK:STDOUT:   %Convert.type.70f: type = fn_type @Convert.6 [concrete]
// CHECK:STDOUT:   %Convert.2b5: %Convert.type.70f = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.626: <bound method> = bound_method %int_4.0c1, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.4b6: %C.a67 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fb0 = impl_witness_table (imports.%P.import_ref.97a), @impl.161 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.087: <witness> = impl_witness %ImplicitAs.impl_witness_table.fb0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.53f: %ImplicitAs.type.5f9 = facet_value %C.a67, (%ImplicitAs.impl_witness.087) [concrete]
// CHECK:STDOUT:   %.ec10: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.53f [concrete]
// CHECK:STDOUT:   %Convert.type.782: type = fn_type @Convert.7 [concrete]
// CHECK:STDOUT:   %Convert.625: %Convert.type.782 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.910: <bound method> = bound_method %int_5.64b, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.75e: %C.65c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fd6 = impl_witness_table (imports.%P.import_ref.e04), @impl.54a [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f50: <witness> = impl_witness %ImplicitAs.impl_witness_table.fd6 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3ed: %ImplicitAs.type.5f9 = facet_value %C.65c, (%ImplicitAs.impl_witness.f50) [concrete]
// CHECK:STDOUT:   %.c25: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.3ed [concrete]
// CHECK:STDOUT:   %Convert.type.0e6: type = fn_type @Convert.8 [concrete]
// CHECK:STDOUT:   %Convert.73d: %Convert.type.0e6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.e3a: <bound method> = bound_method %int_6.462, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.02a: %C.898 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.732 = impl_witness_table (imports.%P.import_ref.6aa), @impl.d71 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.eaa: <witness> = impl_witness %ImplicitAs.impl_witness_table.732 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.67f: %ImplicitAs.type.5f9 = facet_value %C.898, (%ImplicitAs.impl_witness.eaa) [concrete]
// CHECK:STDOUT:   %.c5c: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.67f [concrete]
// CHECK:STDOUT:   %Convert.type.9e9: type = fn_type @Convert.9 [concrete]
// CHECK:STDOUT:   %Convert.e8e: %Convert.type.9e9 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.06a: <bound method> = bound_method %int_7.29f, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.654: %C.f0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c20 = impl_witness_table (imports.%P.import_ref.243), @impl.16b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e1e: <witness> = impl_witness %ImplicitAs.impl_witness_table.c20 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ce9: %ImplicitAs.type.5f9 = facet_value %C.f0a, (%ImplicitAs.impl_witness.e1e) [concrete]
// CHECK:STDOUT:   %.909: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.ce9 [concrete]
// CHECK:STDOUT:   %Convert.type.fc1: type = fn_type @Convert.10 [concrete]
// CHECK:STDOUT:   %Convert.430: %Convert.type.fc1 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [concrete] {
// CHECK:STDOUT:     .D = %P.D
// CHECK:STDOUT:     .C = %P.C
// CHECK:STDOUT:     .Make = %P.Make
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.D: type = import_ref P//library, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %P.import_ref.7e5: <witness> = import_ref P//library, loc17_35, loaded [concrete = constants.%complete_type.682]
// CHECK:STDOUT:   %P.import_ref.cab = import_ref P//library, inst49 [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a99 = import_ref P//library, loc17_16, unloaded
// CHECK:STDOUT:   %P.import_ref.9d2 = import_ref P//library, loc17_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.85e: %i32.builtin = import_ref P//library, loc16_9, loaded [symbolic = @C.%N (constants.%N)]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc16_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.d9b = import_ref P//library, inst43 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc11_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst49 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//default, loc12_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc11_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst49 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc12_32, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.c62: <witness> = import_ref Core//default, loc15_38, loaded [concrete = constants.%ImplicitAs.impl_witness.07a]
// CHECK:STDOUT:   %Core.import_ref.872: type = import_ref Core//default, loc15_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.4d9: type = import_ref Core//default, loc15_36, loaded [concrete = constants.%ImplicitAs.type.61e]
// CHECK:STDOUT:   %P.import_ref.503: <witness> = import_ref P//library, loc30_33, loaded [concrete = constants.%ImplicitAs.impl_witness.6e5]
// CHECK:STDOUT:   %P.import_ref.624: type = import_ref P//library, loc30_9, loaded [concrete = constants.%C.76d]
// CHECK:STDOUT:   %P.import_ref.b769fa.1: type = import_ref P//library, loc30_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.7ac: <witness> = import_ref P//library, loc31_33, loaded [concrete = constants.%ImplicitAs.impl_witness.70d]
// CHECK:STDOUT:   %P.import_ref.8ff: type = import_ref P//library, loc31_9, loaded [concrete = constants.%C.012]
// CHECK:STDOUT:   %P.import_ref.b769fa.2: type = import_ref P//library, loc31_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.255: <witness> = import_ref P//library, loc32_33, loaded [concrete = constants.%ImplicitAs.impl_witness.f8e]
// CHECK:STDOUT:   %P.import_ref.f4d: type = import_ref P//library, loc32_9, loaded [concrete = constants.%C.3e6]
// CHECK:STDOUT:   %P.import_ref.b769fa.3: type = import_ref P//library, loc32_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.51a: <witness> = import_ref P//library, loc33_33, loaded [concrete = constants.%ImplicitAs.impl_witness.bb8]
// CHECK:STDOUT:   %P.import_ref.dd2: type = import_ref P//library, loc33_9, loaded [concrete = constants.%C.4d9]
// CHECK:STDOUT:   %P.import_ref.b769fa.4: type = import_ref P//library, loc33_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.7c4: <witness> = import_ref P//library, loc34_33, loaded [concrete = constants.%ImplicitAs.impl_witness.087]
// CHECK:STDOUT:   %P.import_ref.2c7: type = import_ref P//library, loc34_9, loaded [concrete = constants.%C.a67]
// CHECK:STDOUT:   %P.import_ref.b769fa.5: type = import_ref P//library, loc34_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.30f: <witness> = import_ref P//library, loc35_33, loaded [concrete = constants.%ImplicitAs.impl_witness.f50]
// CHECK:STDOUT:   %P.import_ref.759: type = import_ref P//library, loc35_9, loaded [concrete = constants.%C.65c]
// CHECK:STDOUT:   %P.import_ref.b769fa.6: type = import_ref P//library, loc35_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.664: <witness> = import_ref P//library, loc36_33, loaded [concrete = constants.%ImplicitAs.impl_witness.eaa]
// CHECK:STDOUT:   %P.import_ref.56e: type = import_ref P//library, loc36_9, loaded [concrete = constants.%C.898]
// CHECK:STDOUT:   %P.import_ref.b769fa.7: type = import_ref P//library, loc36_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.322: <witness> = import_ref P//library, loc37_33, loaded [concrete = constants.%ImplicitAs.impl_witness.e1e]
// CHECK:STDOUT:   %P.import_ref.8ba: type = import_ref P//library, loc37_9, loaded [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %P.import_ref.b769fa.8: type = import_ref P//library, loc37_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.Make: %Make.type = import_ref P//library, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .F0 = %F0.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %F0.decl: %F0.type = fn_decl @F0 [concrete = constants.%F0] {
// CHECK:STDOUT:     %n.patt: %i32.builtin = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32.builtin = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc17: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %n.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %.loc17_10.1: type = splice_block %.loc17_10.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc17_10.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc17_10.3: type = converted %int.make_type_signed, %.loc17_10.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32.builtin = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.68b: imports.%Core.import_ref.872 as imports.%Core.import_ref.4d9 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.c62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b24: imports.%P.import_ref.624 as imports.%P.import_ref.b769fa.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.503
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.277: imports.%P.import_ref.8ff as imports.%P.import_ref.b769fa.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.7ac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b5f: imports.%P.import_ref.f4d as imports.%P.import_ref.b769fa.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.90b: imports.%P.import_ref.dd2 as imports.%P.import_ref.b769fa.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.51a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.161: imports.%P.import_ref.2c7 as imports.%P.import_ref.b769fa.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.7c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.54a: imports.%P.import_ref.759 as imports.%P.import_ref.b769fa.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.30f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d71: imports.%P.import_ref.56e as imports.%P.import_ref.b769fa.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.664
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.16b: imports.%P.import_ref.8ba as imports.%P.import_ref.b769fa.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.322
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.7e5
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.cab
// CHECK:STDOUT:   .n = imports.%P.import_ref.a99
// CHECK:STDOUT:   .m = imports.%P.import_ref.9d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.85e: %i32.builtin) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.d9b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param: %i32.builtin) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc26: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc26 br !if.then.loc26 else br !if.else.loc26
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc26:
// CHECK:STDOUT:   %.loc26_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc26: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc26: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc26_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc26_34: <bound method> = bound_method %int_0, %impl.elem0.loc26_34 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32.builtin = call %bound_method.loc26_34(%int_0) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc26_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc26 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc26_34.2: %i32.builtin = converted %int_0, %.loc26_34.1 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %C.loc26: type = class_type @C, @C(constants.%int_0.a54) [concrete = constants.%C.76d]
// CHECK:STDOUT:   %.loc26_24.2: ref %C.76d = temporary_storage
// CHECK:STDOUT:   %.loc26_24.3: init %C.76d = class_init (), %.loc26_24.2 [concrete = constants.%C.val.3c3]
// CHECK:STDOUT:   %.loc26_24.4: ref %C.76d = temporary %.loc26_24.2, %.loc26_24.3
// CHECK:STDOUT:   %.loc26_26.1: ref %C.76d = converted %.loc26_24.1, %.loc26_24.4
// CHECK:STDOUT:   %impl.elem0.loc26_35: %.18d = impl_witness_access constants.%ImplicitAs.impl_witness.6e5, element0 [concrete = constants.%Convert.4f5]
// CHECK:STDOUT:   %bound_method.loc26_35: <bound method> = bound_method %.loc26_26.1, %impl.elem0.loc26_35
// CHECK:STDOUT:   %.loc26_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc26_26.2: %C.76d = bind_value %.loc26_26.1
// CHECK:STDOUT:   %Convert.call.loc26: init %D = call %bound_method.loc26_35(%.loc26_26.2) to %.loc26_35.1
// CHECK:STDOUT:   %.loc26_35.2: init %D = converted %.loc26_26.1, %Convert.call.loc26
// CHECK:STDOUT:   return %.loc26_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc26:
// CHECK:STDOUT:   %false.loc27: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc27 br !if.then.loc27 else br !if.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc27:
// CHECK:STDOUT:   %.loc27_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc27: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc27: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc27_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc27_34: <bound method> = bound_method %int_1, %impl.elem0.loc27_34 [concrete = constants.%Convert.bound.92d]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32.builtin = call %bound_method.loc27_34(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %.loc27_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc27 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %.loc27_34.2: %i32.builtin = converted %int_1, %.loc27_34.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %C.loc27: type = class_type @C, @C(constants.%int_1.f38) [concrete = constants.%C.012]
// CHECK:STDOUT:   %.loc27_24.2: ref %C.012 = temporary_storage
// CHECK:STDOUT:   %.loc27_24.3: init %C.012 = class_init (), %.loc27_24.2 [concrete = constants.%C.val.172]
// CHECK:STDOUT:   %.loc27_24.4: ref %C.012 = temporary %.loc27_24.2, %.loc27_24.3
// CHECK:STDOUT:   %.loc27_26.1: ref %C.012 = converted %.loc27_24.1, %.loc27_24.4
// CHECK:STDOUT:   %impl.elem0.loc27_35: %.d7a = impl_witness_access constants.%ImplicitAs.impl_witness.70d, element0 [concrete = constants.%Convert.ecc]
// CHECK:STDOUT:   %bound_method.loc27_35: <bound method> = bound_method %.loc27_26.1, %impl.elem0.loc27_35
// CHECK:STDOUT:   %.loc27_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc27_26.2: %C.012 = bind_value %.loc27_26.1
// CHECK:STDOUT:   %Convert.call.loc27: init %D = call %bound_method.loc27_35(%.loc27_26.2) to %.loc27_35.1
// CHECK:STDOUT:   %.loc27_35.2: init %D = converted %.loc27_26.1, %Convert.call.loc27
// CHECK:STDOUT:   return %.loc27_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc27:
// CHECK:STDOUT:   %false.loc28: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc28 br !if.then.loc28 else br !if.else.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc28:
// CHECK:STDOUT:   %.loc28_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc28: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc28: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc28_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc28_34: <bound method> = bound_method %int_2, %impl.elem0.loc28_34 [concrete = constants.%Convert.bound.1b9]
// CHECK:STDOUT:   %int.convert_checked.loc28: init %i32.builtin = call %bound_method.loc28_34(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc28_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc28 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc28_34.2: %i32.builtin = converted %int_2, %.loc28_34.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %C.loc28: type = class_type @C, @C(constants.%int_2.5a1) [concrete = constants.%C.3e6]
// CHECK:STDOUT:   %.loc28_24.2: ref %C.3e6 = temporary_storage
// CHECK:STDOUT:   %.loc28_24.3: init %C.3e6 = class_init (), %.loc28_24.2 [concrete = constants.%C.val.418]
// CHECK:STDOUT:   %.loc28_24.4: ref %C.3e6 = temporary %.loc28_24.2, %.loc28_24.3
// CHECK:STDOUT:   %.loc28_26.1: ref %C.3e6 = converted %.loc28_24.1, %.loc28_24.4
// CHECK:STDOUT:   %impl.elem0.loc28_35: %.e6a = impl_witness_access constants.%ImplicitAs.impl_witness.f8e, element0 [concrete = constants.%Convert.c2b]
// CHECK:STDOUT:   %bound_method.loc28_35: <bound method> = bound_method %.loc28_26.1, %impl.elem0.loc28_35
// CHECK:STDOUT:   %.loc28_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc28_26.2: %C.3e6 = bind_value %.loc28_26.1
// CHECK:STDOUT:   %Convert.call.loc28: init %D = call %bound_method.loc28_35(%.loc28_26.2) to %.loc28_35.1
// CHECK:STDOUT:   %.loc28_35.2: init %D = converted %.loc28_26.1, %Convert.call.loc28
// CHECK:STDOUT:   return %.loc28_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc28:
// CHECK:STDOUT:   %false.loc29: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc29 br !if.then.loc29 else br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc29:
// CHECK:STDOUT:   %.loc29_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc29: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc29: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc29_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc29_34: <bound method> = bound_method %int_3, %impl.elem0.loc29_34 [concrete = constants.%Convert.bound.b6b]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %i32.builtin = call %bound_method.loc29_34(%int_3) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc29_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc29 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc29_34.2: %i32.builtin = converted %int_3, %.loc29_34.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %C.loc29: type = class_type @C, @C(constants.%int_3.a0f) [concrete = constants.%C.4d9]
// CHECK:STDOUT:   %.loc29_24.2: ref %C.4d9 = temporary_storage
// CHECK:STDOUT:   %.loc29_24.3: init %C.4d9 = class_init (), %.loc29_24.2 [concrete = constants.%C.val.bba]
// CHECK:STDOUT:   %.loc29_24.4: ref %C.4d9 = temporary %.loc29_24.2, %.loc29_24.3
// CHECK:STDOUT:   %.loc29_26.1: ref %C.4d9 = converted %.loc29_24.1, %.loc29_24.4
// CHECK:STDOUT:   %impl.elem0.loc29_35: %.40b = impl_witness_access constants.%ImplicitAs.impl_witness.bb8, element0 [concrete = constants.%Convert.2b5]
// CHECK:STDOUT:   %bound_method.loc29_35: <bound method> = bound_method %.loc29_26.1, %impl.elem0.loc29_35
// CHECK:STDOUT:   %.loc29_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc29_26.2: %C.4d9 = bind_value %.loc29_26.1
// CHECK:STDOUT:   %Convert.call.loc29: init %D = call %bound_method.loc29_35(%.loc29_26.2) to %.loc29_35.1
// CHECK:STDOUT:   %.loc29_35.2: init %D = converted %.loc29_26.1, %Convert.call.loc29
// CHECK:STDOUT:   return %.loc29_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc29:
// CHECK:STDOUT:   %false.loc30: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc30 br !if.then.loc30 else br !if.else.loc30
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc30:
// CHECK:STDOUT:   %.loc30_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc30: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc30: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc30_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc30_34: <bound method> = bound_method %int_4, %impl.elem0.loc30_34 [concrete = constants.%Convert.bound.626]
// CHECK:STDOUT:   %int.convert_checked.loc30: init %i32.builtin = call %bound_method.loc30_34(%int_4) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc30_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc30 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc30_34.2: %i32.builtin = converted %int_4, %.loc30_34.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %C.loc30: type = class_type @C, @C(constants.%int_4.4f1) [concrete = constants.%C.a67]
// CHECK:STDOUT:   %.loc30_24.2: ref %C.a67 = temporary_storage
// CHECK:STDOUT:   %.loc30_24.3: init %C.a67 = class_init (), %.loc30_24.2 [concrete = constants.%C.val.4b6]
// CHECK:STDOUT:   %.loc30_24.4: ref %C.a67 = temporary %.loc30_24.2, %.loc30_24.3
// CHECK:STDOUT:   %.loc30_26.1: ref %C.a67 = converted %.loc30_24.1, %.loc30_24.4
// CHECK:STDOUT:   %impl.elem0.loc30_35: %.ec10 = impl_witness_access constants.%ImplicitAs.impl_witness.087, element0 [concrete = constants.%Convert.625]
// CHECK:STDOUT:   %bound_method.loc30_35: <bound method> = bound_method %.loc30_26.1, %impl.elem0.loc30_35
// CHECK:STDOUT:   %.loc30_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc30_26.2: %C.a67 = bind_value %.loc30_26.1
// CHECK:STDOUT:   %Convert.call.loc30: init %D = call %bound_method.loc30_35(%.loc30_26.2) to %.loc30_35.1
// CHECK:STDOUT:   %.loc30_35.2: init %D = converted %.loc30_26.1, %Convert.call.loc30
// CHECK:STDOUT:   return %.loc30_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc30:
// CHECK:STDOUT:   %false.loc31: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc31 br !if.then.loc31 else br !if.else.loc31
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc31:
// CHECK:STDOUT:   %.loc31_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc31: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc31: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc31_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc31_34: <bound method> = bound_method %int_5, %impl.elem0.loc31_34 [concrete = constants.%Convert.bound.910]
// CHECK:STDOUT:   %int.convert_checked.loc31: init %i32.builtin = call %bound_method.loc31_34(%int_5) [concrete = constants.%int_5.967]
// CHECK:STDOUT:   %.loc31_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc31 [concrete = constants.%int_5.967]
// CHECK:STDOUT:   %.loc31_34.2: %i32.builtin = converted %int_5, %.loc31_34.1 [concrete = constants.%int_5.967]
// CHECK:STDOUT:   %C.loc31: type = class_type @C, @C(constants.%int_5.967) [concrete = constants.%C.65c]
// CHECK:STDOUT:   %.loc31_24.2: ref %C.65c = temporary_storage
// CHECK:STDOUT:   %.loc31_24.3: init %C.65c = class_init (), %.loc31_24.2 [concrete = constants.%C.val.75e]
// CHECK:STDOUT:   %.loc31_24.4: ref %C.65c = temporary %.loc31_24.2, %.loc31_24.3
// CHECK:STDOUT:   %.loc31_26.1: ref %C.65c = converted %.loc31_24.1, %.loc31_24.4
// CHECK:STDOUT:   %impl.elem0.loc31_35: %.c25 = impl_witness_access constants.%ImplicitAs.impl_witness.f50, element0 [concrete = constants.%Convert.73d]
// CHECK:STDOUT:   %bound_method.loc31_35: <bound method> = bound_method %.loc31_26.1, %impl.elem0.loc31_35
// CHECK:STDOUT:   %.loc31_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc31_26.2: %C.65c = bind_value %.loc31_26.1
// CHECK:STDOUT:   %Convert.call.loc31: init %D = call %bound_method.loc31_35(%.loc31_26.2) to %.loc31_35.1
// CHECK:STDOUT:   %.loc31_35.2: init %D = converted %.loc31_26.1, %Convert.call.loc31
// CHECK:STDOUT:   return %.loc31_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc31:
// CHECK:STDOUT:   %false.loc32: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc32 br !if.then.loc32 else br !if.else.loc32
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc32:
// CHECK:STDOUT:   %.loc32_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc32: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc32: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc32_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc32_34: <bound method> = bound_method %int_6, %impl.elem0.loc32_34 [concrete = constants.%Convert.bound.e3a]
// CHECK:STDOUT:   %int.convert_checked.loc32: init %i32.builtin = call %bound_method.loc32_34(%int_6) [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:   %.loc32_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc32 [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:   %.loc32_34.2: %i32.builtin = converted %int_6, %.loc32_34.1 [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:   %C.loc32: type = class_type @C, @C(constants.%int_6.ec5) [concrete = constants.%C.898]
// CHECK:STDOUT:   %.loc32_24.2: ref %C.898 = temporary_storage
// CHECK:STDOUT:   %.loc32_24.3: init %C.898 = class_init (), %.loc32_24.2 [concrete = constants.%C.val.02a]
// CHECK:STDOUT:   %.loc32_24.4: ref %C.898 = temporary %.loc32_24.2, %.loc32_24.3
// CHECK:STDOUT:   %.loc32_26.1: ref %C.898 = converted %.loc32_24.1, %.loc32_24.4
// CHECK:STDOUT:   %impl.elem0.loc32_35: %.c5c = impl_witness_access constants.%ImplicitAs.impl_witness.eaa, element0 [concrete = constants.%Convert.e8e]
// CHECK:STDOUT:   %bound_method.loc32_35: <bound method> = bound_method %.loc32_26.1, %impl.elem0.loc32_35
// CHECK:STDOUT:   %.loc32_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc32_26.2: %C.898 = bind_value %.loc32_26.1
// CHECK:STDOUT:   %Convert.call.loc32: init %D = call %bound_method.loc32_35(%.loc32_26.2) to %.loc32_35.1
// CHECK:STDOUT:   %.loc32_35.2: init %D = converted %.loc32_26.1, %Convert.call.loc32
// CHECK:STDOUT:   return %.loc32_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc32:
// CHECK:STDOUT:   %false.loc33: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc33 br !if.then.loc33 else br !if.else.loc33
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc33:
// CHECK:STDOUT:   %.loc33_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc33: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc33: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %impl.elem0.loc33_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc33_34: <bound method> = bound_method %int_7, %impl.elem0.loc33_34 [concrete = constants.%Convert.bound.06a]
// CHECK:STDOUT:   %int.convert_checked.loc33: init %i32.builtin = call %bound_method.loc33_34(%int_7) [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:   %.loc33_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc33 [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:   %.loc33_34.2: %i32.builtin = converted %int_7, %.loc33_34.1 [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:   %C.loc33: type = class_type @C, @C(constants.%int_7.6ae) [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %.loc33_24.2: ref %C.f0a = temporary_storage
// CHECK:STDOUT:   %.loc33_24.3: init %C.f0a = class_init (), %.loc33_24.2 [concrete = constants.%C.val.654]
// CHECK:STDOUT:   %.loc33_24.4: ref %C.f0a = temporary %.loc33_24.2, %.loc33_24.3
// CHECK:STDOUT:   %.loc33_26.1: ref %C.f0a = converted %.loc33_24.1, %.loc33_24.4
// CHECK:STDOUT:   %impl.elem0.loc33_35: %.909 = impl_witness_access constants.%ImplicitAs.impl_witness.e1e, element0 [concrete = constants.%Convert.430]
// CHECK:STDOUT:   %bound_method.loc33_35: <bound method> = bound_method %.loc33_26.1, %impl.elem0.loc33_35
// CHECK:STDOUT:   %.loc33_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc33_26.2: %C.f0a = bind_value %.loc33_26.1
// CHECK:STDOUT:   %Convert.call.loc33: init %D = call %bound_method.loc33_35(%.loc33_26.2) to %.loc33_35.1
// CHECK:STDOUT:   %.loc33_35.2: init %D = converted %.loc33_26.1, %Convert.call.loc33
// CHECK:STDOUT:   return %.loc33_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc33:
// CHECK:STDOUT:   %P.ref.loc34: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc17_15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2 = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.61e
// CHECK:STDOUT:   %Self => constants.%Self.d28
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.059
// CHECK:STDOUT:   %Convert => constants.%Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.398
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.a54) {
// CHECK:STDOUT:   %N => constants.%int_0.a54
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5f9
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.334
// CHECK:STDOUT:   %Convert => constants.%Convert.87c
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.953
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.f38) {
// CHECK:STDOUT:   %N => constants.%int_1.f38
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.5a1) {
// CHECK:STDOUT:   %N => constants.%int_2.5a1
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.a0f) {
// CHECK:STDOUT:   %N => constants.%int_3.a0f
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4f1) {
// CHECK:STDOUT:   %N => constants.%int_4.4f1
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.967) {
// CHECK:STDOUT:   %N => constants.%int_5.967
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.ec5) {
// CHECK:STDOUT:   %N => constants.%int_6.ec5
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.6ae) {
// CHECK:STDOUT:   %N => constants.%int_7.6ae
// CHECK:STDOUT:   %N.patt => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
