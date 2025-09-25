// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/complete_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/complete_type.carbon

// --- fail_incomplete_in_class.carbon

library "[[@TEST_NAME]]";

class B;

class A(T:! type) {
  // CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+6]]:10: error: `T` evaluates to incomplete type `B` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR:   var v: T;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE-6]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class B;
  // CHECK:STDERR: ^~~~~~~~
  var v: T;
}

// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+11]]:6: note: in `A(B)` used here [ResolvingSpecificHere]
// CHECK:STDERR: fn F(x: A(B)) {}
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+7]]:6: error: parameter has incomplete type `A(B)` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: fn F(x: A(B)) {}
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE-19]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn F(x: A(B)) {}

class B {}

// --- incomplete_in_function.carbon

library "[[@TEST_NAME]]";

class B;

fn F(T:! type) {
  var v: T*;
  *v;
}

// F(B) isn't resolved until the end of the file.
fn G() { F(B); }

class B {}

// --- fail_incomplete_in_function_at_eof.carbon

library "[[@TEST_NAME]]";

class B;

fn F(T:! type) {
  var v: T*;
  // CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE+6]]:3: error: `T` evaluates to incomplete type `B` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR:   *v;
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE-7]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class B;
  // CHECK:STDERR: ^~~~~~~~
  *v;
}

// CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE+4]]:10: note: in `F(B)` used here [ResolvingSpecificHere]
// CHECK:STDERR: fn G() { F(B); }
// CHECK:STDERR:          ^
// CHECK:STDERR:
fn G() { F(B); }

// CHECK:STDOUT: --- fail_incomplete_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.130: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A.elem.1ce: type = unbound_element_type %A.130, %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e00: <witness> = impl_witness @A.%Destroy.impl_witness_table, @A.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.ca9: type = ptr_type %A.130 [symbolic]
// CHECK:STDOUT:   %pattern_type.09b: type = pattern_type %ptr.ca9 [symbolic]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op, @A.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.v.ff1: type = struct_type {.v: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.460: <witness> = complete_type_witness %struct_type.v.ff1 [symbolic]
// CHECK:STDOUT:   %A.1d4: type = class_type @A, @A(%B) [concrete]
// CHECK:STDOUT:   %pattern_type.1b7: type = pattern_type %A.1d4 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem.4ce: type = unbound_element_type %A.1d4, %B [concrete]
// CHECK:STDOUT:   %struct_type.v.d45: type = struct_type {.v: %B} [concrete]
// CHECK:STDOUT:   %complete_type.1df: <witness> = complete_type_witness %struct_type.v.d45 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.041: <witness> = impl_witness @B.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl.loc4
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl.loc4: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.1b7 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1b7 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A.1d4 = value_param call_param0
// CHECK:STDOUT:     %.loc27: type = splice_block %A [concrete = constants.%A.1d4] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl.loc4 [concrete = constants.%B]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%B) [concrete = constants.%A.1d4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %A.1d4 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc29: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as.Destroy.impl(@A.%T.loc6_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @A.%Destroy.impl_witness_table, @A.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.e00)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op, @A.as.Destroy.impl(%T) [symbolic = %A.as.Destroy.impl.Op.type (constants.%A.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: @A.as.Destroy.impl.%A.as.Destroy.impl.Op.type (%A.as.Destroy.impl.Op.type) = struct_value () [symbolic = %A.as.Destroy.impl.Op (constants.%A.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @A.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %A.as.Destroy.impl.Op.decl: @A.as.Destroy.impl.%A.as.Destroy.impl.Op.type (%A.as.Destroy.impl.Op.type) = fn_decl @A.as.Destroy.impl.Op [symbolic = @A.as.Destroy.impl.%A.as.Destroy.impl.Op (constants.%A.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @A.as.Destroy.impl.Op.%pattern_type (%pattern_type.09b) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @A.as.Destroy.impl.Op.%pattern_type (%pattern_type.09b) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_19.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @A.as.Destroy.impl.Op.%ptr (%ptr.ca9) = value_param call_param0
// CHECK:STDOUT:       %.loc6_19.2: type = splice_block %Self.ref [symbolic = %A (constants.%A.130)] {
// CHECK:STDOUT:         %.loc6_19.3: type = specific_constant constants.%A.130, @A(constants.%T) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_19.3 [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @A.as.Destroy.impl.Op.%ptr (%ptr.ca9) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @A.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: @B.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.decl: %B.as.Destroy.impl.Op.type = fn_decl @B.as.Destroy.impl.Op [concrete = constants.%B.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc29: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %B.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @B.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:   impl_decl @B.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@B.as.Destroy.impl.%B.as.Destroy.impl.Op.decl), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.041]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc6_9.2: type) {
// CHECK:STDOUT:   %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc6_9.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc6_9.1) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %T.loc6_9.1 [symbolic = %A.elem (constants.%A.elem.1ce)]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: @A.%T.loc6_9.1 (%T)} [symbolic = %struct_type.v (constants.%struct_type.v.ff1)]
// CHECK:STDOUT:   %complete_type.loc14_1.2: <witness> = complete_type_witness %struct_type.v [symbolic = %complete_type.loc14_1.2 (constants.%complete_type.460)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_9.2 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc13: @A.%A.elem (%A.elem.1ce) = field_decl v, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A.130 [symbolic = @A.as.Destroy.impl.%A (constants.%A.130)]
// CHECK:STDOUT:     impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @A.as.Destroy.impl(constants.%T) [symbolic = @A.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.e00)]
// CHECK:STDOUT:     %complete_type.loc14_1.1: <witness> = complete_type_witness constants.%struct_type.v.ff1 [symbolic = %complete_type.loc14_1.2 (constants.%complete_type.460)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc14_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.130
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .v = %.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.as.Destroy.impl.Op(@A.%T.loc6_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:   %ptr: type = ptr_type %A [symbolic = %ptr (constants.%ptr.ca9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.09b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @A.as.Destroy.impl.Op.%ptr (%ptr.ca9)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %A.1d4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.Destroy.impl.Op(%self.param: %ptr.e79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %A => constants.%A.130
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.e00
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %A => constants.%A.130
// CHECK:STDOUT:   %ptr => constants.%ptr.ca9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.09b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%B) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A => constants.%A.1d4
// CHECK:STDOUT:   %A.elem => constants.%A.elem.4ce
// CHECK:STDOUT:   %struct_type.v => constants.%struct_type.v.d45
// CHECK:STDOUT:   %complete_type.loc14_1.2 => constants.%complete_type.1df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- incomplete_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %PointeeT: type = bind_symbolic_name PointeeT, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.type.eb7d1c.1: type = fn_type @ptr.as.Destroy.impl.Op, @ptr.as.Destroy.impl(%PointeeT) [symbolic]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.60a6c7.1: %ptr.as.Destroy.impl.Op.type.eb7d1c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.a13abe.2: type = ptr_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %require_complete.1328e0.2: <witness> = require_complete_type %ptr.a13abe.2 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3af: <witness> = lookup_impl_witness %ptr.79f131.1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.64e: %Destroy.type = facet_value %ptr.79f131.1, (%Destroy.lookup_impl_witness.3af) [symbolic]
// CHECK:STDOUT:   %.fc8: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.64e [symbolic]
// CHECK:STDOUT:   %impl.elem0.108: %.fc8 = impl_witness_access %Destroy.lookup_impl_witness.3af, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a84: <specific function> = specific_impl_function %impl.elem0.108, @Destroy.Op(%Destroy.facet.64e) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.041: <witness> = impl_witness @B.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %complete_type.3bf: <witness> = complete_type_witness %ptr.e79 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40d: <witness> = impl_witness imports.%Destroy.impl_witness_table.1ab, @ptr.as.Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.type.083: type = fn_type @ptr.as.Destroy.impl.Op, @ptr.as.Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.4a5: %ptr.as.Destroy.impl.Op.type.083 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3fb: type = ptr_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %complete_type.c9f: <witness> = complete_type_witness %ptr.3fb [concrete]
// CHECK:STDOUT:   %Destroy.facet.f4c: %Destroy.type = facet_value %ptr.e79, (%Destroy.impl_witness.40d) [concrete]
// CHECK:STDOUT:   %.f78: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.f4c [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Destroy.impl.Op.4a5, @ptr.as.Destroy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.58c: @ptr.as.Destroy.impl.%ptr.as.Destroy.impl.Op.type (%ptr.as.Destroy.impl.Op.type.eb7d1c.1) = import_ref Core//prelude/parts/destroy, loc23_29, loaded [symbolic = @ptr.as.Destroy.impl.%ptr.as.Destroy.impl.Op (constants.%ptr.as.Destroy.impl.Op.60a6c7.1)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.1ab = impl_witness_table (%Core.import_ref.58c), @ptr.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl.loc4
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl.loc4: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %B.decl.loc14: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.Destroy.impl: @B.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.decl: %B.as.Destroy.impl.Op.type = fn_decl @B.as.Destroy.impl.Op [concrete = constants.%B.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %B.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @B.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:   impl_decl @B.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@B.as.Destroy.impl.%B.as.Destroy.impl.Op.decl), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.041]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc7_11.2: type = ptr_type %T.loc6_6.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %require_complete.loc7_11: <witness> = require_complete_type %ptr.loc7_11.2 [symbolic = %require_complete.loc7_11 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc7_11.2 [symbolic = %pattern_type (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc8 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc7_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.3af)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc7_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.64e)]
// CHECK:STDOUT:   %.loc7_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc7_3.2 (constants.%.fc8)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3.2 (%.fc8) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.108)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.a84)]
// CHECK:STDOUT:   %ptr.loc7_3: type = ptr_type %ptr.loc7_11.2 [symbolic = %ptr.loc7_3 (constants.%ptr.a13abe.2)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr.loc7_3 [symbolic = %require_complete.loc7_3 (constants.%require_complete.1328e0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.afe) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.afe) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%ptr.loc7_11.2 (%ptr.79f131.1) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_11: type = splice_block %ptr.loc7_11.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f131.1)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc7_11.1: type = ptr_type %T.ref [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%ptr.loc7_11.2 (%ptr.79f131.1) = bind_name v, %v.var
// CHECK:STDOUT:     %v.ref: ref @F.%ptr.loc7_11.2 (%ptr.79f131.1) = name_ref v, %v
// CHECK:STDOUT:     %.loc8_4: @F.%ptr.loc7_11.2 (%ptr.79f131.1) = bind_value %v.ref
// CHECK:STDOUT:     %.loc8_3: ref @F.%T.loc6_6.1 (%T) = deref %.loc8_4
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3.2 (%.fc8) = impl_witness_access constants.%Destroy.lookup_impl_witness.3af, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.108)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.Op(constants.%Destroy.facet.64e) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.a84)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %addr: @F.%ptr.loc7_3 (%ptr.a13abe.2) = addr_of %v.var
// CHECK:STDOUT:     %.loc7_3.1: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl.loc4 [concrete = constants.%B]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%B) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.Destroy.impl.Op(%self.param: %ptr.e79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%B) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc7_11.2 => constants.%ptr.e79
// CHECK:STDOUT:   %require_complete.loc7_11 => constants.%complete_type.3bf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.960
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.40d
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.f4c
// CHECK:STDOUT:   %.loc7_3.2 => constants.%.f78
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%ptr.as.Destroy.impl.Op.4a5
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%ptr.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr.loc7_3 => constants.%ptr.3fb
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.c9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_in_function_at_eof.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %PointeeT: type = bind_symbolic_name PointeeT, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.type.eb7d1c.1: type = fn_type @ptr.as.Destroy.impl.Op, @ptr.as.Destroy.impl(%PointeeT) [symbolic]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.60a6c7.1: %ptr.as.Destroy.impl.Op.type.eb7d1c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.a13abe.2: type = ptr_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %require_complete.1328e0.2: <witness> = require_complete_type %ptr.a13abe.2 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3af: <witness> = lookup_impl_witness %ptr.79f131.1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.64e: %Destroy.type = facet_value %ptr.79f131.1, (%Destroy.lookup_impl_witness.3af) [symbolic]
// CHECK:STDOUT:   %.fc8: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.64e [symbolic]
// CHECK:STDOUT:   %impl.elem0.108: %.fc8 = impl_witness_access %Destroy.lookup_impl_witness.3af, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a84: <specific function> = specific_impl_function %impl.elem0.108, @Destroy.Op(%Destroy.facet.64e) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %complete_type.3bf: <witness> = complete_type_witness %ptr.e79 [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40d: <witness> = impl_witness imports.%Destroy.impl_witness_table.1ab, @ptr.as.Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.type.083: type = fn_type @ptr.as.Destroy.impl.Op, @ptr.as.Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.4a5: %ptr.as.Destroy.impl.Op.type.083 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3fb: type = ptr_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %complete_type.c9f: <witness> = complete_type_witness %ptr.3fb [concrete]
// CHECK:STDOUT:   %Destroy.facet.f4c: %Destroy.type = facet_value %ptr.e79, (%Destroy.impl_witness.40d) [concrete]
// CHECK:STDOUT:   %.f78: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.f4c [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Destroy.impl.Op.4a5, @ptr.as.Destroy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.58c: @ptr.as.Destroy.impl.%ptr.as.Destroy.impl.Op.type (%ptr.as.Destroy.impl.Op.type.eb7d1c.1) = import_ref Core//prelude/parts/destroy, loc23_29, loaded [symbolic = @ptr.as.Destroy.impl.%ptr.as.Destroy.impl.Op (constants.%ptr.as.Destroy.impl.Op.60a6c7.1)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.1ab = impl_witness_table (%Core.import_ref.58c), @ptr.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc7_11.2: type = ptr_type %T.loc6_6.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %require_complete.loc7_11: <witness> = require_complete_type %ptr.loc7_11.2 [symbolic = %require_complete.loc7_11 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc7_11.2 [symbolic = %pattern_type (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc14 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc7_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.3af)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc7_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.64e)]
// CHECK:STDOUT:   %.loc7_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc7_3.2 (constants.%.fc8)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3.2 (%.fc8) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.108)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.a84)]
// CHECK:STDOUT:   %ptr.loc7_3: type = ptr_type %ptr.loc7_11.2 [symbolic = %ptr.loc7_3 (constants.%ptr.a13abe.2)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr.loc7_3 [symbolic = %require_complete.loc7_3 (constants.%require_complete.1328e0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.afe) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.afe) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%ptr.loc7_11.2 (%ptr.79f131.1) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_11: type = splice_block %ptr.loc7_11.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f131.1)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc7_11.1: type = ptr_type %T.ref [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%ptr.loc7_11.2 (%ptr.79f131.1) = bind_name v, %v.var
// CHECK:STDOUT:     %v.ref: ref @F.%ptr.loc7_11.2 (%ptr.79f131.1) = name_ref v, %v
// CHECK:STDOUT:     %.loc14_4: @F.%ptr.loc7_11.2 (%ptr.79f131.1) = bind_value %v.ref
// CHECK:STDOUT:     %.loc14_3: ref @F.%T.loc6_6.1 (%T) = deref %.loc14_4
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3.2 (%.fc8) = impl_witness_access constants.%Destroy.lookup_impl_witness.3af, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.108)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.Op(constants.%Destroy.facet.64e) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.a84)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %addr: @F.%ptr.loc7_3 (%ptr.a13abe.2) = addr_of %v.var
// CHECK:STDOUT:     %.loc7_3.1: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%B) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%B) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc7_11.2 => constants.%ptr.e79
// CHECK:STDOUT:   %require_complete.loc7_11 => constants.%complete_type.3bf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.960
// CHECK:STDOUT:   %require_complete.loc14 => <error>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.40d
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.f4c
// CHECK:STDOUT:   %.loc7_3.2 => constants.%.f78
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%ptr.as.Destroy.impl.Op.4a5
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%ptr.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr.loc7_3 => constants.%ptr.3fb
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.c9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
