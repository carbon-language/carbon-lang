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

class Class(T:! type);

class CompleteClass(T:! type) {
  var n: i32;
  fn F() -> i32 { return 0; }
}

fn F() -> CompleteClass(i32);

// --- foo.impl.carbon

impl library "[[@TEST_NAME]]";

class Class(T:! type) {
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
  // TODO: Include the generic arguments in the formatted type name.
  // CHECK:STDERR: fail_generic_arg_mismatch.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `CompleteClass(i32)` to `CompleteClass(i32*)` [ConversionFailure]
  // CHECK:STDERR:   var v: CompleteClass(i32*) = F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_generic_arg_mismatch.carbon:[[@LINE+4]]:3: note: type `CompleteClass(i32)` does not implement interface `Core.ImplicitAs(CompleteClass(i32*))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var v: CompleteClass(i32*) = F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: CompleteClass(i32*) = F();
}

// --- fail_foo.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_foo.impl.carbon:[[@LINE+8]]:13: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class Class(U:! type) {
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_foo.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: foo.carbon:4:13: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Class(T:! type);
// CHECK:STDERR:             ^
// CHECK:STDERR:
class Class(U:! type) {
  // CHECK:STDERR: fail_foo.impl.carbon:[[@LINE+4]]:10: error: name `T` not found [NameNotFound]
  // CHECK:STDERR:   var x: T;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  var x: T;
}

// CHECK:STDOUT: --- foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F: %CompleteClass.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ce1: <witness> = impl_witness @CompleteClass.%Destroy.impl_witness_table, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.5b4: type = ptr_type %CompleteClass.f97 [symbolic]
// CHECK:STDOUT:   %pattern_type.1fe: type = pattern_type %ptr.5b4 [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: %CompleteClass.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %CompleteClass.e9e: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.0b0: type = pattern_type %CompleteClass.e9e [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompleteClass.decl: %CompleteClass.type = class_decl @CompleteClass [concrete = constants.%CompleteClass.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:     %T.loc6_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_21.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.0b0 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0b0 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, file.%CompleteClass.decl [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.e9e]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.e9e = out_param call_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.e9e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CompleteClass.as.Destroy.impl(@CompleteClass.%T.loc6_21.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @CompleteClass.%Destroy.impl_witness_table, @CompleteClass.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.ce1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic = %CompleteClass.as.Destroy.impl.Op.type (constants.%CompleteClass.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type) = struct_value () [symbolic = %CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @CompleteClass.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %CompleteClass.as.Destroy.impl.Op.decl: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type) = fn_decl @CompleteClass.as.Destroy.impl.Op [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @CompleteClass.as.Destroy.impl.Op.%pattern_type (%pattern_type.1fe) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @CompleteClass.as.Destroy.impl.Op.%pattern_type (%pattern_type.1fe) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_31.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @CompleteClass.as.Destroy.impl.Op.%ptr (%ptr.5b4) = value_param call_param0
// CHECK:STDOUT:       %.loc6_31.2: type = splice_block %Self.ref [symbolic = %CompleteClass (constants.%CompleteClass.f97)] {
// CHECK:STDOUT:         %.loc6_31.3: type = specific_constant constants.%CompleteClass.f97, @CompleteClass(constants.%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_31.3 [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @CompleteClass.as.Destroy.impl.Op.%ptr (%ptr.5b4) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %CompleteClass.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @CompleteClass.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(%T.loc6_21.2: type) {
// CHECK:STDOUT:   %T.loc6_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_21.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T.loc6_21.1) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T.loc6_21.1) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %CompleteClass.F.decl: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type) = fn_decl @CompleteClass.F [symbolic = @CompleteClass.%CompleteClass.F (constants.%CompleteClass.F)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%CompleteClass.f97 [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:     impl_decl @CompleteClass.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.decl), @CompleteClass.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @CompleteClass.as.Destroy.impl(constants.%T) [symbolic = @CompleteClass.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.ce1)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CompleteClass.f97
// CHECK:STDOUT:     .n = %.loc7
// CHECK:STDOUT:     .F = %CompleteClass.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(@CompleteClass.%T.loc6_21.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc8_27.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_27.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_27.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_27.2: %i32 = converted %int_0, %.loc8_27.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     return %.loc8_27.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.as.Destroy.impl.Op(@CompleteClass.%T.loc6_21.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %ptr: type = ptr_type %CompleteClass [symbolic = %ptr (constants.%ptr.5b4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1fe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @CompleteClass.as.Destroy.impl.Op.%ptr (%ptr.5b4)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %CompleteClass.e9e;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T.loc6_21.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.ce1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %ptr => constants.%ptr.5b4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.loc6_21.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.n.4d6: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a68: <witness> = complete_type_witness %struct_type.n.4d6 [concrete]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.9ef: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.type.14f: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.874: %CompleteClass.F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.07c: <witness> = impl_witness imports.%Destroy.impl_witness_table.439, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: %CompleteClass.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.5b4: type = ptr_type %CompleteClass.f97 [symbolic]
// CHECK:STDOUT:   %pattern_type.1fe: type = pattern_type %ptr.5b4 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.595: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4c1: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e13: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4c1 = struct_value () [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.816: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x [symbolic]
// CHECK:STDOUT:   %CompleteClass.a06: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %CompleteClass.a06 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.426: type = unbound_element_type %CompleteClass.a06, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.0aa: type = fn_type @CompleteClass.F, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.971: %CompleteClass.F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ea1: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.ea0: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.985: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.449, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c4c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.ea1 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.985) [concrete]
// CHECK:STDOUT:   %.f75: type = fn_type_with_self_type %ImplicitAs.Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.c4c [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.c4c, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %CompleteClass.val: %CompleteClass.a06 = struct_value (%int_1.47b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.fbb = import_ref Main//foo, loc6_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.a68]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.051 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.570 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e27: type = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %Main.import_ref.063: type = import_ref Main//foo, inst82 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.58135e.1 = import_ref Main//foo, loc6_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.58135e.2 = import_ref Main//foo, loc6_31, unloaded
// CHECK:STDOUT:   %Destroy.impl_witness_table.439 = impl_witness_table (%Main.import_ref.58135e.2), @CompleteClass.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.4: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3ad: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4c1) = import_ref Main//foo, inst219 [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e13)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.449 = impl_witness_table (%Main.import_ref.3ad), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.5: type = import_ref Main//foo, loc4_13, loaded [symbolic = @Class.%T.1 (constants.%T)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.595 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_19.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_19.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.84b = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.84b = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.a06]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.a06 = out_param call_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.a06 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CompleteClass.as.Destroy.impl(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.439, @CompleteClass.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.07c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic = %CompleteClass.as.Destroy.impl.Op.type (constants.%CompleteClass.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type) = struct_value () [symbolic = %CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.e27 as imports.%Main.import_ref.063 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.58135e.1
// CHECK:STDOUT:     witness = imports.%Main.import_ref.fbb
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc4: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.816)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc4_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc4_23.3: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_23.3 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.9ef)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type.14f)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.14f) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F.874)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.eb1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3c0
// CHECK:STDOUT:     .n = imports.%Main.import_ref.051
// CHECK:STDOUT:     .F = imports.%Main.import_ref.570
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(imports.%Main.import_ref.5ab3ec.5: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Class.%Class.elem (%Class.elem) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.816)]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(imports.%Main.import_ref.5ab3ec.3: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.as.Destroy.impl.Op(imports.%Main.import_ref.5ab3ec.4: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %ptr: type = ptr_type %CompleteClass [symbolic = %ptr (constants.%ptr.5b4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1fe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc4: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %CompleteClass.a06 [from "foo.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_17.1: %struct_type.n.44a = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0: %.f75 = impl_witness_access constants.%ImplicitAs.impl_witness.985, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c4c]
// CHECK:STDOUT:   %bound_method.loc9_17.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_17.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_17.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc9_17.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc9_17.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc9_17.4: init %i32 = initialize_from %.loc9_17.2 to %.loc9_17.3 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc9_17.5: init %CompleteClass.a06 = class_init (%.loc9_17.4), %return [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   %.loc9_18: init %CompleteClass.a06 = converted %.loc9_17.1, %.loc9_17.5 [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   return %.loc9_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9ef
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.14f
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.07c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %ptr => constants.%ptr.5b4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.816
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.a06
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.426
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.0aa
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.971
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %UseMethod.type: type = fn_type @UseMethod [concrete]
// CHECK:STDOUT:   %UseMethod: %UseMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.28a: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.type.14f: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.874: %CompleteClass.F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.e9e: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.7fc: type = unbound_element_type %CompleteClass.e9e, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.1bc: type = fn_type @CompleteClass.F, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.f7c: %CompleteClass.F.type.1bc = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0b0: type = pattern_type %CompleteClass.e9e [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.F.specific_fn: <specific function> = specific_function %CompleteClass.F.f7c, @CompleteClass.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.d93: <witness> = impl_witness imports.%Destroy.impl_witness_table.727, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type.1a6: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.acd: %CompleteClass.as.Destroy.impl.Op.type.1a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.5b4: type = ptr_type %CompleteClass.f97 [symbolic]
// CHECK:STDOUT:   %pattern_type.1fe: type = pattern_type %ptr.5b4 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.bd6: <witness> = impl_witness imports.%Destroy.impl_witness_table.727, @CompleteClass.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type.079: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.5fe: %CompleteClass.as.Destroy.impl.Op.type.079 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.a97: type = ptr_type %CompleteClass.e9e [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.a97 [concrete]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %CompleteClass.as.Destroy.impl.Op.5fe, @CompleteClass.as.Destroy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %UseField.type: type = fn_type @UseField [concrete]
// CHECK:STDOUT:   %UseField: %UseField.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//foo, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e76: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.28a) = import_ref Main//foo, loc7_8, loaded [concrete = %.364]
// CHECK:STDOUT:   %Main.import_ref.a52: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.14f) = import_ref Main//foo, loc8_17, loaded [symbolic = @CompleteClass.%CompleteClass.F (constants.%CompleteClass.F.874)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.a31: <witness> = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.d93)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e27: type = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %Main.import_ref.063: type = import_ref Main//foo, inst82 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.b2b: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type.1a6) = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op.acd)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.727 = impl_witness_table (%Main.import_ref.b2b), @CompleteClass.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.4: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %.364: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.28a) = field_decl n, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .UseMethod = %UseMethod.decl
// CHECK:STDOUT:     .UseField = %UseField.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %UseMethod.decl: %UseMethod.type = fn_decl @UseMethod [concrete = constants.%UseMethod] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseField.decl: %UseField.type = fn_decl @UseField [concrete = constants.%UseField] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CompleteClass.as.Destroy.impl(imports.%Main.import_ref.5ab3ec.3: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.727, @CompleteClass.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.d93)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic = %CompleteClass.as.Destroy.impl.Op.type (constants.%CompleteClass.as.Destroy.impl.Op.type.1a6)]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type.1a6) = struct_value () [symbolic = %CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op.acd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.e27 as imports.%Main.import_ref.063 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.a31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.28a)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type.14f)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.14f) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F.874)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.eb1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3c0
// CHECK:STDOUT:     .n = imports.%Main.import_ref.e76
// CHECK:STDOUT:     .F = imports.%Main.import_ref.a52
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseMethod() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0b0 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0b0 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc6: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6_3: ref %CompleteClass.e9e = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.e9e = call %F.ref.loc6() to %.loc6_3
// CHECK:STDOUT:   assign %v.var, %F.call
// CHECK:STDOUT:   %.loc6_27: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.e9e] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.e9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.e9e = bind_name v, %v.var
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.e9e = name_ref v, %v
// CHECK:STDOUT:   %.loc7_11: %CompleteClass.F.type.1bc = specific_constant imports.%Main.import_ref.a52, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.F.f7c]
// CHECK:STDOUT:   %F.ref.loc7: %CompleteClass.F.type.1bc = name_ref F, %.loc7_11 [concrete = constants.%CompleteClass.F.f7c]
// CHECK:STDOUT:   %CompleteClass.F.specific_fn: <specific function> = specific_function %F.ref.loc7, @CompleteClass.F(constants.%i32) [concrete = constants.%CompleteClass.F.specific_fn]
// CHECK:STDOUT:   %CompleteClass.F.call: init %i32 = call %CompleteClass.F.specific_fn()
// CHECK:STDOUT:   %.loc7_15.1: %i32 = value_of_initializer %CompleteClass.F.call
// CHECK:STDOUT:   %.loc7_15.2: %i32 = converted %CompleteClass.F.call, %.loc7_15.1
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.bound: <bound method> = bound_method %v.var, constants.%CompleteClass.as.Destroy.impl.Op.5fe
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%CompleteClass.as.Destroy.impl.Op.5fe, @CompleteClass.as.Destroy.impl.Op(constants.%i32) [concrete = constants.%CompleteClass.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %v.var, %CompleteClass.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.a97 = addr_of %v.var
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc7_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.as.Destroy.impl.Op(imports.%Main.import_ref.5ab3ec.4: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %ptr: type = ptr_type %CompleteClass [symbolic = %ptr (constants.%ptr.5b4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1fe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseField() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0b0 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0b0 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var %v.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc11_3: ref %CompleteClass.e9e = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.e9e = call %F.ref() to %.loc11_3
// CHECK:STDOUT:   assign %v.var, %F.call
// CHECK:STDOUT:   %.loc11_27: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.e9e] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.e9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.e9e = bind_name v, %v.var
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.e9e = name_ref v, %v
// CHECK:STDOUT:   %n.ref: %CompleteClass.elem.7fc = name_ref n, imports.%Main.import_ref.e76 [concrete = imports.%.364]
// CHECK:STDOUT:   %.loc12_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc12_11.2: %i32 = bind_value %.loc12_11.1
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.bound: <bound method> = bound_method %v.var, constants.%CompleteClass.as.Destroy.impl.Op.5fe
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%CompleteClass.as.Destroy.impl.Op.5fe, @CompleteClass.as.Destroy.impl.Op(constants.%i32) [concrete = constants.%CompleteClass.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %v.var, %CompleteClass.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.a97 = addr_of %v.var
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc12_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.28a
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.14f
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.e9e
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.7fc
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.1bc
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.f7c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.d93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %ptr => constants.%ptr.5b4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.e9e
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.bd6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type => constants.%CompleteClass.as.Destroy.impl.Op.type.079
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op => constants.%CompleteClass.as.Destroy.impl.Op.5fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl.Op(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.e9e
// CHECK:STDOUT:   %ptr => constants.%ptr.a97
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_arg_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:   %complete_type.a68: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.9ef: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.type.14f: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.874: %CompleteClass.F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.9e1: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.0fe: type = class_type @CompleteClass, @CompleteClass(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.d29: type = unbound_element_type %CompleteClass.0fe, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.8cf: type = fn_type @CompleteClass.F, @CompleteClass(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.012: %CompleteClass.F.type.8cf = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.062: type = pattern_type %CompleteClass.0fe [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.a06: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.426: type = unbound_element_type %CompleteClass.a06, %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.F.type.0aa: type = fn_type @CompleteClass.F, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.F.971: %CompleteClass.F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.58c: <witness> = impl_witness imports.%Destroy.impl_witness_table.dd3, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type.4a4: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.edc: %CompleteClass.as.Destroy.impl.Op.type.4a4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.5b4: type = ptr_type %CompleteClass.f97 [symbolic]
// CHECK:STDOUT:   %pattern_type.1fe: type = pattern_type %ptr.5b4 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.25e: <witness> = impl_witness imports.%Destroy.impl_witness_table.dd3, @CompleteClass.as.Destroy.impl(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type.18c: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.741: %CompleteClass.as.Destroy.impl.Op.type.18c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c79: type = ptr_type %CompleteClass.0fe [concrete]
// CHECK:STDOUT:   %pattern_type.cea: type = pattern_type %ptr.c79 [concrete]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %CompleteClass.as.Destroy.impl.Op.741, @CompleteClass.as.Destroy.impl.Op(%ptr.9e1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//foo, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.a68]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.051 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.570 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Main.import_ref.a31: <witness> = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.58c)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e27: type = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %Main.import_ref.063: type = import_ref Main//foo, inst82 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.9f1: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type.4a4) = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op.edc)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.dd3 = impl_witness_table (%Main.import_ref.9f1), @CompleteClass.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.4: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CompleteClass.as.Destroy.impl(imports.%Main.import_ref.5ab3ec.3: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.dd3, @CompleteClass.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.58c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic = %CompleteClass.as.Destroy.impl.Op.type (constants.%CompleteClass.as.Destroy.impl.Op.type.4a4)]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type.4a4) = struct_value () [symbolic = %CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op.edc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.e27 as imports.%Main.import_ref.063 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.a31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.9ef)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type.14f)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type.14f) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F.874)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.eb1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3c0
// CHECK:STDOUT:     .n = imports.%Main.import_ref.051
// CHECK:STDOUT:     .F = imports.%Main.import_ref.570
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.062 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.062 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.0fe = var %v.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc14_34: ref %CompleteClass.a06 = temporary_storage
// CHECK:STDOUT:   %F.call: init %CompleteClass.a06 = call %F.ref() to %.loc14_34
// CHECK:STDOUT:   %.loc14_3: %CompleteClass.0fe = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %.loc14_28: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.0fe] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.9e1]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%ptr.9e1) [concrete = constants.%CompleteClass.0fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.0fe = bind_name v, %v.var
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.bound: <bound method> = bound_method %v.var, constants.%CompleteClass.as.Destroy.impl.Op.741
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%CompleteClass.as.Destroy.impl.Op.741, @CompleteClass.as.Destroy.impl.Op(constants.%ptr.9e1) [concrete = constants.%CompleteClass.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %v.var, %CompleteClass.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.c79 = addr_of %v.var
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.as.Destroy.impl.Op(imports.%Main.import_ref.5ab3ec.4: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %ptr: type = ptr_type %CompleteClass [symbolic = %ptr (constants.%ptr.5b4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1fe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9ef
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.14f
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%ptr.9e1) {
// CHECK:STDOUT:   %T => constants.%ptr.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.0fe
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.d29
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.8cf
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.012
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.a06
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.426
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type.0aa
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F.971
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.58c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %ptr => constants.%ptr.5b4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl(constants.%ptr.9e1) {
// CHECK:STDOUT:   %T => constants.%ptr.9e1
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.0fe
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.25e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type => constants.%CompleteClass.as.Destroy.impl.Op.type.18c
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op => constants.%CompleteClass.as.Destroy.impl.Op.741
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl.Op(constants.%ptr.9e1) {
// CHECK:STDOUT:   %T => constants.%ptr.9e1
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.0fe
// CHECK:STDOUT:   %ptr => constants.%ptr.c79
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a68: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, %i32 [symbolic]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.F: %CompleteClass.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.07c: <witness> = impl_witness imports.%Destroy.impl_witness_table.439, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: %CompleteClass.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.5b4: type = ptr_type %CompleteClass [symbolic]
// CHECK:STDOUT:   %pattern_type.1fe: type = pattern_type %ptr.5b4 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %Class.type.cf06d9.1: type = generic_class_type @Class.1 [concrete]
// CHECK:STDOUT:   %Class.generic.9545f5.1: %Class.type.cf06d9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.type.cf06d9.2: type = generic_class_type @Class.loc12 [concrete]
// CHECK:STDOUT:   %Class.generic.9545f5.2: %Class.type.cf06d9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1b2d.2: type = class_type @Class.loc12, @Class.loc12(%U) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.816: <witness> = impl_witness @Class.loc12.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%U) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class.fe1b2d.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%U) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class: %Class.type.cf06d9.1 = import_ref Main//foo, Class, loaded [concrete = constants.%Class.generic.9545f5.1]
// CHECK:STDOUT:   %Main.CompleteClass = import_ref Main//foo, CompleteClass, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//foo, F, unloaded
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.fbb = import_ref Main//foo, loc6_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.a68]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.051 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.570 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e27: type = import_ref Main//foo, loc6_31, loaded [symbolic = @CompleteClass.as.Destroy.impl.%CompleteClass (constants.%CompleteClass)]
// CHECK:STDOUT:   %Main.import_ref.063: type = import_ref Main//foo, inst82 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.58135e.1 = import_ref Main//foo, loc6_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.58135e.2 = import_ref Main//foo, loc6_31, unloaded
// CHECK:STDOUT:   %Destroy.impl_witness_table.439 = impl_witness_table (%Main.import_ref.58135e.2), @CompleteClass.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.4: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.5: type = import_ref Main//foo, loc4_13, loaded [symbolic = @Class.1.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_19.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_19.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type.cf06d9.2 = class_decl @Class.loc12 [concrete = constants.%Class.generic.9545f5.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:     %U.loc12_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @CompleteClass.as.Destroy.impl(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.439, @CompleteClass.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.07c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op.type: type = fn_type @CompleteClass.as.Destroy.impl.Op, @CompleteClass.as.Destroy.impl(%T) [symbolic = %CompleteClass.as.Destroy.impl.Op.type (constants.%CompleteClass.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %CompleteClass.as.Destroy.impl.Op: @CompleteClass.as.Destroy.impl.%CompleteClass.as.Destroy.impl.Op.type (%CompleteClass.as.Destroy.impl.Op.type) = struct_value () [symbolic = %CompleteClass.as.Destroy.impl.Op (constants.%CompleteClass.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.e27 as imports.%Main.import_ref.063 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = imports.%Main.import_ref.58135e.1
// CHECK:STDOUT:     witness = imports.%Main.import_ref.fbb
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.loc12.%U.loc12_13.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Class: type = class_type @Class.loc12, @Class.loc12(%U) [symbolic = %Class (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.loc12.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%U) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.816)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%U) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.loc12.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc12_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc12_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class.fe1b2d.2)] {
// CHECK:STDOUT:         %.loc12_23.3: type = specific_constant constants.%Class.fe1b2d.2, @Class.loc12(constants.%U) [symbolic = %Class (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_23.3 [symbolic = %Class (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.loc12.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem)]
// CHECK:STDOUT:   %CompleteClass.F.type: type = fn_type @CompleteClass.F, @CompleteClass(%T) [symbolic = %CompleteClass.F.type (constants.%CompleteClass.F.type)]
// CHECK:STDOUT:   %CompleteClass.F: @CompleteClass.%CompleteClass.F.type (%CompleteClass.F.type) = struct_value () [symbolic = %CompleteClass.F (constants.%CompleteClass.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.eb1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3c0
// CHECK:STDOUT:     .n = imports.%Main.import_ref.051
// CHECK:STDOUT:     .F = imports.%Main.import_ref.570
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.1(imports.%Main.import_ref.5ab3ec.5: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.loc12(%U.loc12_13.2: type) {
// CHECK:STDOUT:   %U.loc12_13.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc17: <error> = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class.fe1b2d.2 [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%U) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.816)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1b2d.2
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.F(imports.%Main.import_ref.5ab3ec.3: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CompleteClass.as.Destroy.impl.Op(imports.%Main.import_ref.5ab3ec.4: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass)]
// CHECK:STDOUT:   %ptr: type = ptr_type %CompleteClass [symbolic = %ptr (constants.%ptr.5b4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1fe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.loc12.%U.loc12_13.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Class: type = class_type @Class.loc12, @Class.loc12(%U) [symbolic = %Class (constants.%Class.fe1b2d.2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem
// CHECK:STDOUT:   %CompleteClass.F.type => constants.%CompleteClass.F.type
// CHECK:STDOUT:   %CompleteClass.F => constants.%CompleteClass.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.07c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass
// CHECK:STDOUT:   %ptr => constants.%ptr.5b4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.loc12(constants.%U) {
// CHECK:STDOUT:   %U.loc12_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.2
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.816
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Class => constants.%Class.fe1b2d.2
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
