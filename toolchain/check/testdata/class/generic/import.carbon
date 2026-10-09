// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/import.carbon

// --- foo.carbon

library "[[@TEST_NAME]]";

class Class(T: type);

class CompleteClass(T: type) {
  var n: i32;
  fn F() -> i32 { return 0; }
}

fn F() -> CompleteClass(i32);

// --- foo.impl.carbon

impl library "[[@TEST_NAME]]";

class Class(T: type) {
  var x: T;
}

fn F() -> CompleteClass(i32) {
  return {.n = 1};
}

// --- use_foo.carbon

library "[[@TEST_NAME]]";
import library "foo";

fn UseMethod() -> i32 {
  var v: CompleteClass(i32) = F();
  return v.F();
}

fn UseField() -> i32 {
  var v: CompleteClass(i32) = F();
  return v.n;
}

// --- fail_generic_arg_mismatch.carbon

library "[[@TEST_NAME]]";
import library "foo";

fn Use() {
  // CHECK:STDERR: fail_generic_arg_mismatch.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `CompleteClass(i32)` to `CompleteClass(i32*)` [ConversionFailure]
  // CHECK:STDERR:   var unused v: CompleteClass(i32*) = F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_generic_arg_mismatch.carbon:[[@LINE+4]]:3: note: type `CompleteClass(i32)` does not implement interface `Core.ImplicitAs(CompleteClass(i32*))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused v: CompleteClass(i32*) = F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused v: CompleteClass(i32*) = F();
}

// --- fail_foo.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_foo.impl.carbon:[[@LINE+8]]:13: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class Class(U: type) {
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_foo.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: foo.carbon:4:13: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Class(T: type);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
class Class(U: type) {
  // CHECK:STDERR: fail_foo.impl.carbon:[[@LINE+4]]:10: error: name `T` not found [NameNotFound]
  // CHECK:STDERR:   var x: T;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  var x: T;
}

// CHECK:STDOUT: --- foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.810: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass.810, %i32 [symbolic]
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F: %CompleteClass.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.26a: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.2b2: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %CompleteClass.3ae: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %.d45: Core.Form = init_form %CompleteClass.3ae [concrete]
// CHECK:STDOUT:   %pattern_type.9da: type = pattern_type %CompleteClass.3ae [concrete]
// CHECK:STDOUT:   %return.param_patt.636: %pattern_type.9da = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.cdd: %pattern_type.9da = return_slot_pattern %return.param_patt.636, %CompleteClass.3ae [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CompleteClass = %CompleteClass.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompleteClass.decl: %CompleteClass.type = class_decl @CompleteClass [concrete = constants.%CompleteClass.generic] {
// CHECK:STDOUT:     %T.patt.loc6_22.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_24.1: type = splice_block %.loc6_24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9da = out_param_pattern [concrete = constants.%return.param_patt.636]
// CHECK:STDOUT:     %return.patt: %pattern_type.9da = return_slot_pattern %return.param_patt, %CompleteClass [concrete = constants.%return.patt.cdd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, file.%CompleteClass.decl [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.3ae]
// CHECK:STDOUT:     %.loc11: Core.Form = init_form %CompleteClass [concrete = constants.%.d45]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.3ae = out_param call_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.3ae = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(%T.loc6_22.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_22.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T.loc6_22.1) [symbolic = %CompleteClass (constants.%CompleteClass.810)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T.loc6_22.1) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %field_decl: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = field_decl n, element0, %i32 in [concrete] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %CompleteClass.F.decl: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type) = fn_decl @CompleteClass.F [symbolic = @CompleteClass.%CompleteClass.F (constants.%CompleteClass.F)] {
// CHECK:STDOUT:       %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:       %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_13: Core.Form = init_form %i32 [concrete = constants.%.55c]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.26a]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CompleteClass.810
// CHECK:STDOUT:     .n = %field_decl
// CHECK:STDOUT:     .F = %CompleteClass.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(@CompleteClass.%T.loc6_22.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc8_27.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_27.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_27.2(%int_0) [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:     %.loc8_27: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:     return %.loc8_27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %CompleteClass.3ae;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_22.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_22.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.810
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc6_22.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_22.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x [symbolic]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.n.2ae: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.26a: <witness> = complete_type_witness %struct_type.n.2ae [concrete]
// CHECK:STDOUT:   %CompleteClass.810: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.type.a23: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.0c1: %CompleteClass.F.type.a23 = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.782: type = unbound_element_type %CompleteClass.810, %i32 [symbolic]
// CHECK:STDOUT:   %CompleteClass.3ae: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %.d45: Core.Form = init_form %CompleteClass.3ae [concrete]
// CHECK:STDOUT:   %pattern_type.9da: type = pattern_type %CompleteClass.3ae [concrete]
// CHECK:STDOUT:   %return.param_patt.636: %pattern_type.9da = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.cdd: %pattern_type.9da = return_slot_pattern %return.param_patt.636, %CompleteClass.3ae [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.981: type = unbound_element_type %CompleteClass.3ae, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.4a7: type = fn_type @CompleteClass.F, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.6f4: %CompleteClass.F.type.4a7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.n.44a = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.8e8: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %CompleteClass.val: %CompleteClass.3ae = struct_value (%int_1.8e8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.620a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Main//foo, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Main.import_ref.620a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//foo, loc4_14, loaded [symbolic = @Class.%T.1 (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//foo, loc6_22, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.16f: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.26a]
// CHECK:STDOUT:   %Main.import_ref.273 = import_ref Main//foo, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.30d = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.e84 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//foo, loc6_22, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_19.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_19.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc4: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4: type = symbolic_binding T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9da = out_param_pattern [concrete = constants.%return.param_patt.636]
// CHECK:STDOUT:     %return.patt: %pattern_type.9da = return_slot_pattern %return.param_patt, %CompleteClass [concrete = constants.%return.patt.cdd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.3ae]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %CompleteClass [concrete = constants.%.d45]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.3ae = out_param call_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.3ae = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(imports.%Main.import_ref.b3bc94.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T.patt.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:   %T.1: type = symbolic_binding T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %field_decl: @Class.%Class.elem (%Class.elem) = field_decl x, element0, %T.ref in [concrete] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.b3bc94.3: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.810)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.782)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type.a23)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.a23) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F.0c1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.16f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.273
// CHECK:STDOUT:     .n = imports.%Main.import_ref.30d
// CHECK:STDOUT:     .F = imports.%Main.import_ref.e84
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(imports.%Main.import_ref.b3bc94.2: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %CompleteClass.3ae [from "foo.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_17.1: %struct_type.n.44a = struct_literal (%int_1) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc9_17.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_17.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_17.2(%int_1) [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc9_17.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc9_17.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc9_17.4: init %i32 to %.loc9_17.3 = in_place_init %.loc9_17.2 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc9_17.5: init %CompleteClass.3ae to %return.param = class_init (%.loc9_17.4) [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   %.loc9_18: init %CompleteClass.3ae = converted %.loc9_17.1, %.loc9_17.5 [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   return %.loc9_18 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.patt.1 => constants.%T.patt
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.810
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.782
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.a23
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.0c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.3ae
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.981
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.4a7
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.6f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %UseMethod.type: type = fn_type @UseMethod [concrete]
// CHECK:STDOUT:   %UseMethod: %UseMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.26a: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.810: type = class_type @CompleteClass, @CompleteClass(%T.67d) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.type.a23: type = fn_type @CompleteClass.F, @CompleteClass(%T.67d) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.0c1: %CompleteClass.F.type.a23 = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.782: type = unbound_element_type %CompleteClass.810, %i32 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.3ae: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.981: type = unbound_element_type %CompleteClass.3ae, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.4a7: type = fn_type @CompleteClass.F, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.6f4: %CompleteClass.F.type.4a7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9da: type = pattern_type %CompleteClass.3ae [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.9da = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.9da = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.F.specific_fn: <specific function> = specific_function %CompleteClass.F.6f4, @CompleteClass.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1b5: type = pattern_type %struct_type.n [concrete]
// CHECK:STDOUT:   %self.param_patt.51a: %pattern_type.1b5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0c0: %pattern_type.1b5 = wrapper_binding_pattern self, %self.param_patt.51a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.6f8: %pattern_type.9da = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.72f: %pattern_type.9da = wrapper_binding_pattern self, %self.param_patt.6f8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc6_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %UseField.type: type = fn_type @UseField [concrete]
// CHECK:STDOUT:   %UseField: %UseField.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//foo, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//foo, loc6_22, loaded [symbolic = @CompleteClass.%T (constants.%T.67d)]
// CHECK:STDOUT:   %Main.import_ref.16f: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.26a]
// CHECK:STDOUT:   %Main.import_ref.273 = import_ref Main//foo, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.870: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.782) = import_ref Main//foo, loc7_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Main.import_ref.35c: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.a23) = import_ref Main//foo, loc8_17, loaded [symbolic = @CompleteClass.%CompleteClass.F (constants.%CompleteClass.F.0c1)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//foo, loc6_22, loaded [symbolic = @CompleteClass.%T (constants.%T.67d)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.e0a = import_ref Main//foo, loc7_10, unloaded
// CHECK:STDOUT:   %field_decl: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.782) = field_decl n, element0, %Main.import_ref.e0a in [concrete] {}
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc6_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b5 = ref_param_pattern [concrete = constants.%self.param_patt.51a]
// CHECK:STDOUT:     %self.patt: %pattern_type.1b5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0c0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.n = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.n = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc6_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b5 = ref_param_pattern [concrete = constants.%self.param_patt.51a]
// CHECK:STDOUT:     %self.patt: %pattern_type.1b5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0c0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.n = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.n = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9da = ref_param_pattern [concrete = constants.%self.param_patt.6f8]
// CHECK:STDOUT:     %self.patt: %pattern_type.9da = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.72f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %CompleteClass.3ae = ref_param call_param0
// CHECK:STDOUT:     %self: ref %CompleteClass.3ae = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc6_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9da = ref_param_pattern [concrete = constants.%self.param_patt.6f8]
// CHECK:STDOUT:     %self.patt: %pattern_type.9da = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.72f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %CompleteClass.3ae = ref_param call_param0
// CHECK:STDOUT:     %self: ref %CompleteClass.3ae = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UseMethod = %UseMethod.decl
// CHECK:STDOUT:     .UseField = %UseField.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %UseMethod.decl: %UseMethod.type = fn_decl @UseMethod [concrete = constants.%UseMethod] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5: Core.Form = init_form %i32.loc5 [concrete = constants.%.55c]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseField.decl: %UseField.type = fn_decl @UseField [concrete = constants.%UseField] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32.loc10 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10: Core.Form = init_form %i32.loc10 [concrete = constants.%.55c]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.b3bc94.2: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.810)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.782)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type.a23)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.a23) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F.0c1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.16f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.273
// CHECK:STDOUT:     .n = imports.%Main.import_ref.870
// CHECK:STDOUT:     .F = imports.%Main.import_ref.35c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseMethod() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %CompleteClass.3ae = var_storage %v.var_patt
// CHECK:STDOUT:   %F.ref.loc6: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6_3: ref %CompleteClass.3ae = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.3ae to %.loc6_3 = call %F.ref.loc6()
// CHECK:STDOUT:   assign %v.var, %F.call
// CHECK:STDOUT:   %.loc6_27: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.3ae] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.3ae]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.3ae = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.9da = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.9da = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.3ae = name_ref v, %v
// CHECK:STDOUT:   %.loc7: %CompleteClass.F.type.4a7 = specific_constant imports.%Main.import_ref.35c, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.F.6f4]
// CHECK:STDOUT:   %F.ref.loc7: %CompleteClass.F.type.4a7 = name_ref F, %.loc7 [concrete = constants.%CompleteClass.F.6f4]
// CHECK:STDOUT:   %CompleteClass.F.specific_fn: <specific function> = specific_function %F.ref.loc7, @CompleteClass.F(constants.%i32) [concrete = constants.%CompleteClass.F.specific_fn]
// CHECK:STDOUT:   %CompleteClass.F.call: init %i32 = call %CompleteClass.F.specific_fn()
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %v.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%v.var)
// CHECK:STDOUT:   return %CompleteClass.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(imports.%Main.import_ref.b3bc94.1: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.2(%self.param: ref %struct_type.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.2(%self.param: ref %struct_type.n) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.2(%self.param: ref %struct_type.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.3(%self.param: ref %CompleteClass.3ae) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %CompleteClass.elem.981 = name_ref n, imports.%Main.import_ref.870 [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc6: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.3(%self.param: ref %CompleteClass.3ae) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.3(%self.param: ref %CompleteClass.3ae) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseField() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %CompleteClass.3ae = var_storage %v.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc11_3: ref %CompleteClass.3ae = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.3ae to %.loc11_3 = call %F.ref()
// CHECK:STDOUT:   assign %v.var, %F.call
// CHECK:STDOUT:   %.loc11_27: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.3ae] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.3ae]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.3ae = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.9da = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.9da = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.3ae = name_ref v, %v
// CHECK:STDOUT:   %n.ref: %CompleteClass.elem.981 = name_ref n, imports.%Main.import_ref.870 [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc12_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc12_11.2: %i32 = acquire_value %.loc12_11.1
// CHECK:STDOUT:   %impl.elem0: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc12_11.1: <bound method> = bound_method %.loc12_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_11.2: <bound method> = bound_method %.loc12_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc12_11.2(%.loc12_11.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %v.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%v.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.810
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.782
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.a23
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.0c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T.67d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.3ae
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.981
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.4a7
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.6f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_arg_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.26a: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.810: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.type.a23: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.0c1: %CompleteClass.F.type.a23 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.782: type = unbound_element_type %CompleteClass.810, %i32 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.1fe: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.f11: type = class_type @CompleteClass, @CompleteClass(%ptr.1fe) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.c0c: type = unbound_element_type %CompleteClass.f11, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.955: type = fn_type @CompleteClass.F, @CompleteClass(%ptr.1fe) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.1c4: %CompleteClass.F.type.955 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6bf: type = pattern_type %CompleteClass.f11 [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.6bf = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.6bf = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.3ae: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.981: type = unbound_element_type %CompleteClass.3ae, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.4a7: type = fn_type @CompleteClass.F, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.6f4: %CompleteClass.F.type.4a7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc13_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc13_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc13_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1b5: type = pattern_type %struct_type.n [concrete]
// CHECK:STDOUT:   %self.param_patt.51a: %pattern_type.1b5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0c0: %pattern_type.1b5 = wrapper_binding_pattern self, %self.param_patt.51a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc13_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc13_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.7ac: %pattern_type.6bf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6cf: %pattern_type.6bf = wrapper_binding_pattern self, %self.param_patt.7ac [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc13_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc13_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc13_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//foo, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//foo, loc6_22, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.16f: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.26a]
// CHECK:STDOUT:   %Main.import_ref.273 = import_ref Main//foo, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.870: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.782) = import_ref Main//foo, loc7_8, loaded [concrete = %field_decl]
// CHECK:STDOUT:   %Main.import_ref.e84 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//foo, loc6_22, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.e0a = import_ref Main//foo, loc7_10, unloaded
// CHECK:STDOUT:   %field_decl: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.782) = field_decl n, element0, %Main.import_ref.e0a in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc13_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc13_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc13_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b5 = ref_param_pattern [concrete = constants.%self.param_patt.51a]
// CHECK:STDOUT:     %self.patt: %pattern_type.1b5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0c0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.n = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.n = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc13_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b5 = ref_param_pattern [concrete = constants.%self.param_patt.51a]
// CHECK:STDOUT:     %self.patt: %pattern_type.1b5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0c0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.n = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.n = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc13_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6bf = ref_param_pattern [concrete = constants.%self.param_patt.7ac]
// CHECK:STDOUT:     %self.patt: %pattern_type.6bf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6cf]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %CompleteClass.f11 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %CompleteClass.f11 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc13_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6bf = ref_param_pattern [concrete = constants.%self.param_patt.7ac]
// CHECK:STDOUT:     %self.patt: %pattern_type.6bf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6cf]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %CompleteClass.f11 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %CompleteClass.f11 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.b3bc94.2: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.810)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.782)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type.a23)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.a23) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F.0c1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.16f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.273
// CHECK:STDOUT:     .n = imports.%Main.import_ref.870
// CHECK:STDOUT:     .F = imports.%Main.import_ref.e84
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %CompleteClass.f11 = var_storage %v.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13_41: ref %CompleteClass.3ae = temporary_storage
// CHECK:STDOUT:   %F.call: init %CompleteClass.3ae to %.loc13_41 = call %F.ref()
// CHECK:STDOUT:   %.loc13_3: %CompleteClass.f11 = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %.loc13_35: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.f11] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%ptr.1fe) [concrete = constants.%CompleteClass.f11]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.f11 = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.6bf = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.6bf = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %v.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%v.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(imports.%Main.import_ref.b3bc94.1: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc13_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc13_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc13_3.2(%self.param: ref %struct_type.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc13, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc13)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_3.2(%self.param: ref %struct_type.n) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc13_3.2(%self.param: ref %struct_type.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc13_3.3(%self.param: ref %CompleteClass.f11) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %CompleteClass.elem.c0c = name_ref n, imports.%Main.import_ref.870 [concrete = imports.%field_decl]
// CHECK:STDOUT:   %.loc13: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc13, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc13)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_3.3(%self.param: ref %CompleteClass.f11) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc13_3.3(%self.param: ref %CompleteClass.f11) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.810
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.782
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.a23
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.0c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%ptr.1fe) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr.1fe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f11
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.c0c
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.955
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.1c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.3ae
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.981
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.4a7
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.6f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Class.type.3256be.1: type = generic_class_type @Class.1 [concrete]
// CHECK:STDOUT:   %Class.generic.4154f7.1: %Class.type.3256be.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type.3256be.2: type = generic_class_type @Class.loc12 [concrete]
// CHECK:STDOUT:   %Class.generic.4154f7.2: %Class.type.3256be.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.a2fa98.2: type = class_type @Class.loc12, @Class.loc12(%U) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class: %Class.type.3256be.1 = import_ref Main//foo, Class, loaded [concrete = constants.%Class.generic.4154f7.1]
// CHECK:STDOUT:   %Main.CompleteClass = import_ref Main//foo, CompleteClass, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//foo, F, unloaded
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.b3b: type = import_ref Main//foo, loc4_14, loaded [symbolic = @Class.1.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_19.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_19.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type.3256be.2 = class_decl @Class.loc12 [concrete = constants.%Class.generic.4154f7.2] {
// CHECK:STDOUT:     %U.patt.loc12_14.1: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_16.1: type = splice_block %.loc12_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc12_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc12_14.2: type = symbolic_binding U, 0 [symbolic = %U.loc12_14.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.1(imports.%Main.import_ref.b3b: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.loc12(%U.loc12_14.2: type) {
// CHECK:STDOUT:   %U.patt.loc12_14.2: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc12_14.1: type = symbolic_binding U, 0 [symbolic = %U.loc12_14.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %field_decl: <error> = field_decl x, element0, <error> in [concrete] {
// CHECK:STDOUT:       %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.a2fa98.2
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.1(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.loc12(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc12_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc12_14.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
