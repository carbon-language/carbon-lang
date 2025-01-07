// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  // CHECK:STDERR: fail_generic_arg_mismatch.carbon:[[@LINE+7]]:3: error: cannot implicitly convert from `CompleteClass(i32)` to `CompleteClass(i32*)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   var v: CompleteClass(i32*) = F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_generic_arg_mismatch.carbon:[[@LINE+4]]:3: note: type `CompleteClass(i32)` does not implement interface `Core.ImplicitAs(CompleteClass(i32*))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var v: CompleteClass(i32*) = F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: CompleteClass(i32*) = F();
}

// --- fail_foo.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_foo.impl.carbon:[[@LINE+8]]:13: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class Class(U:! type) {
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR: fail_foo.impl.carbon:[[@LINE-5]]:6: in import [InImport]
// CHECK:STDERR: foo.carbon:4:13: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Class(T:! type);
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
class Class(U:! type) {
  // CHECK:STDERR: fail_foo.impl.carbon:[[@LINE+3]]:10: error: name `T` not found [NameNotFound]
  // CHECK:STDERR:   var x: T;
  // CHECK:STDERR:          ^
  var x: T;
}

// CHECK:STDOUT: --- foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [template]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.3321: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass.3321, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.24bc: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.b4ac: %F.type.24bc = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.9b72: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %int_0.5c61: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c61, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61a: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %CompleteClass.b7f9: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.type.b258: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.c419: %F.type.b258 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CompleteClass = %CompleteClass.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompleteClass.decl: %CompleteClass.type = class_decl @CompleteClass [template = constants.%CompleteClass.generic] {
// CHECK:STDOUT:     %T.patt.loc6_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_21.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_21.1, runtime_param<invalid> [symbolic = %T.patt.loc6_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_21.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.b258 = fn_decl @F.2 [template = constants.%F.c419] {
// CHECK:STDOUT:     %return.patt: %CompleteClass.b7f9 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %CompleteClass.b7f9 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, file.%CompleteClass.decl [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [template = constants.%CompleteClass.b7f9]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.b7f9 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.b7f9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(%T.loc6_21.1: type) {
// CHECK:STDOUT:   %T.loc6_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_21.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T.loc6_21.2) [symbolic = %CompleteClass (constants.%CompleteClass.3321)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.3321), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T.loc6_21.2) [symbolic = %F.type (constants.%F.type.24bc)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.24bc) = struct_value () [symbolic = %F (constants.%F.b4ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc7: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = field_decl n, element0 [template]
// CHECK:STDOUT:     %F.decl: @CompleteClass.%F.type (%F.type.24bc) = fn_decl @F.1 [symbolic = @CompleteClass.%F (constants.%F.b4ac)] {
// CHECK:STDOUT:       %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.n [template = constants.%complete_type.9b72]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CompleteClass.3321
// CHECK:STDOUT:     .n = %.loc7
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@CompleteClass.%T.loc6_21.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.f61a]
// CHECK:STDOUT:     %.loc8_27.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.f61a]
// CHECK:STDOUT:     %.loc8_27.2: %i32 = converted %int_0, %.loc8_27.1 [template = constants.%int_0.f61a]
// CHECK:STDOUT:     return %.loc8_27.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.b7f9;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_21.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.3321
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem
// CHECK:STDOUT:   %F.type => constants.%F.type.24bc
// CHECK:STDOUT:   %F => constants.%F.b4ac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T.loc6_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc6_21.2 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.db1e: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60ab: <witness> = complete_type_witness %struct_type.x [symbolic]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [template]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.n.dffb: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.a01f: <witness> = complete_type_witness %struct_type.n.dffb [template]
// CHECK:STDOUT:   %CompleteClass.535e: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.9af9: type = unbound_element_type %CompleteClass.535e, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.7da7: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.2ba8: %F.type.7da7 = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.a69b: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.type.b258: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.c419: %F.type.b258 = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.elem.0305: type = unbound_element_type %CompleteClass.a69b, %i32 [template]
// CHECK:STDOUT:   %F.type.764c: type = fn_type @F.1, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.9cac: %F.type.764c = struct_value () [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.n.44a0: type = struct_type {.n: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.40a6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.764d: <witness> = impl_witness (imports.%import_ref.154f), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.f1ce: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.4b81: %Convert.type.f1ce = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b81, %Convert.4b81 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.0593: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %CompleteClass.val: %CompleteClass.a69b = struct_value (%int_1.0593) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.ff53: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d695
// CHECK:STDOUT:     .ImplicitAs = %import_ref.edd8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.dc0c: <witness> = import_ref Main//foo, loc9_1, loaded [template = constants.%complete_type.a01f]
// CHECK:STDOUT:   %import_ref.8a74 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.b9fb = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %import_ref.3558 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.ff53
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = constants.%T.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T, 0, %T.param [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.b258 = fn_decl @F.2 [template = constants.%F.c419] {
// CHECK:STDOUT:     %return.patt: %CompleteClass.a69b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %CompleteClass.a69b = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%import_ref.ff53 [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [template = constants.%CompleteClass.a69b]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.a69b = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.a69b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(constants.%T: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.1 (%T) [symbolic = %require_complete (constants.%require_complete.db1e)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class), @Class.%T.1 (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Class.%struct_type.x (%struct_type.x) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: @Class.%Class.elem (%Class.elem) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.535e)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.535e), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.9af9)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.7da7)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.7da7) = struct_value () [symbolic = %F (constants.%F.2ba8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8a74
// CHECK:STDOUT:     .n = imports.%import_ref.b9fb
// CHECK:STDOUT:     .F = imports.%import_ref.3558
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.dc0c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %CompleteClass.a69b [from "foo.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %.loc9_17.1: %struct_type.n.44a0 = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.40a6 = impl_witness_access constants.%impl_witness.764d, element0 [template = constants.%Convert.4b81]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.0593]
// CHECK:STDOUT:   %.loc9_17.2: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.0593]
// CHECK:STDOUT:   %.loc9_17.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc9_17.4: init %i32 = initialize_from %.loc9_17.2 to %.loc9_17.3 [template = constants.%int_1.0593]
// CHECK:STDOUT:   %.loc9_17.5: init %CompleteClass.a69b = class_init (%.loc9_17.4), %return [template = constants.%CompleteClass.val]
// CHECK:STDOUT:   %.loc9_18: init %CompleteClass.a69b = converted %.loc9_17.1, %.loc9_17.5 [template = constants.%CompleteClass.val]
// CHECK:STDOUT:   return %.loc9_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.535e
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9af9
// CHECK:STDOUT:   %F.type => constants.%F.type.7da7
// CHECK:STDOUT:   %F => constants.%F.2ba8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.a69b
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.0305
// CHECK:STDOUT:   %F.type => constants.%F.type.764c
// CHECK:STDOUT:   %F => constants.%F.9cac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %UseMethod.type: type = fn_type @UseMethod [template]
// CHECK:STDOUT:   %UseMethod: %UseMethod.type = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [template]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.9b72: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.535e: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.bc22: type = unbound_element_type %CompleteClass.535e, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.7da7: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.2ba8: %F.type.7da7 = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.374f: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %CompleteClass.elem.cc96: type = unbound_element_type %CompleteClass.374f, %i32 [template]
// CHECK:STDOUT:   %F.type.8e13: type = fn_type @F.1, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.3dca: %F.type.8e13 = struct_value () [template]
// CHECK:STDOUT:   %F.type.b258: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.c419: %F.type.b258 = struct_value () [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.3dca, @F.1(%i32) [template]
// CHECK:STDOUT:   %UseField.type: type = fn_type @UseField [template]
// CHECK:STDOUT:   %UseField: %UseField.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.73d1 = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %import_ref.ff53: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %import_ref.7ee6: %F.type.b258 = import_ref Main//foo, F, loaded [template = constants.%F.c419]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.dc0c: <witness> = import_ref Main//foo, loc9_1, loaded [template = constants.%complete_type.9b72]
// CHECK:STDOUT:   %import_ref.8a74 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.78db: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.bc22) = import_ref Main//foo, loc7_8, loaded [template = %.817b]
// CHECK:STDOUT:   %import_ref.4bf5: @CompleteClass.%F.type (%F.type.7da7) = import_ref Main//foo, loc8_17, loaded [symbolic = @CompleteClass.%F (constants.%F.2ba8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = imports.%import_ref.73d1
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.ff53
// CHECK:STDOUT:     .F = imports.%import_ref.7ee6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UseMethod = %UseMethod.decl
// CHECK:STDOUT:     .UseField = %UseField.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UseMethod.decl: %UseMethod.type = fn_decl @UseMethod [template = constants.%UseMethod] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseField.decl: %UseField.type = fn_decl @UseField [template = constants.%UseField] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.535e)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.535e), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.bc22)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.7da7)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.7da7) = struct_value () [symbolic = %F (constants.%F.2ba8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8a74
// CHECK:STDOUT:     .n = imports.%import_ref.78db
// CHECK:STDOUT:     .F = imports.%import_ref.4bf5
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.dc0c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseMethod() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %CompleteClass.374f = var v
// CHECK:STDOUT:   %v: ref %CompleteClass.374f = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc6: %F.type.b258 = name_ref F, imports.%import_ref.7ee6 [template = constants.%F.c419]
// CHECK:STDOUT:   %.loc6: ref %CompleteClass.374f = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc6: init %CompleteClass.374f = call %F.ref.loc6() to %.loc6
// CHECK:STDOUT:   assign %v.var, %F.call.loc6
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.374f = name_ref v, %v
// CHECK:STDOUT:   %.loc7_11: %F.type.8e13 = specific_constant imports.%import_ref.4bf5, @CompleteClass(constants.%i32) [template = constants.%F.3dca]
// CHECK:STDOUT:   %F.ref.loc7: %F.type.8e13 = name_ref F, %.loc7_11 [template = constants.%F.3dca]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref.loc7, @F.1(constants.%i32) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call.loc7: init %i32 = call %F.specific_fn()
// CHECK:STDOUT:   %.loc7_15.1: %i32 = value_of_initializer %F.call.loc7
// CHECK:STDOUT:   %.loc7_15.2: %i32 = converted %F.call.loc7, %.loc7_15.1
// CHECK:STDOUT:   return %.loc7_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.374f [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseField() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %CompleteClass.374f = var v
// CHECK:STDOUT:   %v: ref %CompleteClass.374f = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref: %F.type.b258 = name_ref F, imports.%import_ref.7ee6 [template = constants.%F.c419]
// CHECK:STDOUT:   %.loc11: ref %CompleteClass.374f = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.374f = call %F.ref() to %.loc11
// CHECK:STDOUT:   assign %v.var, %F.call
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.374f = name_ref v, %v
// CHECK:STDOUT:   %n.ref: %CompleteClass.elem.cc96 = name_ref n, imports.%import_ref.78db [template = imports.%.817b]
// CHECK:STDOUT:   %.loc12_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc12_11.2: %i32 = bind_value %.loc12_11.1
// CHECK:STDOUT:   return %.loc12_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.535e
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.bc22
// CHECK:STDOUT:   %F.type => constants.%F.type.7da7
// CHECK:STDOUT:   %F => constants.%F.2ba8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.374f
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.cc96
// CHECK:STDOUT:   %F.type => constants.%F.type.8e13
// CHECK:STDOUT:   %F => constants.%F.3dca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_arg_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [template]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [template]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.a01f: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.535e: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.9af9: type = unbound_element_type %CompleteClass.535e, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.7da7: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.2ba8: %F.type.7da7 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.b050: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %CompleteClass.e967: type = class_type @CompleteClass, @CompleteClass(%ptr.b050) [template]
// CHECK:STDOUT:   %CompleteClass.elem.d75b: type = unbound_element_type %CompleteClass.e967, %i32 [template]
// CHECK:STDOUT:   %F.type.bb4d: type = fn_type @F.1, @CompleteClass(%ptr.b050) [template]
// CHECK:STDOUT:   %F.9ec0: %F.type.bb4d = struct_value () [template]
// CHECK:STDOUT:   %F.type.b258: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.c419: %F.type.b258 = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.a69b: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %CompleteClass.elem.0305: type = unbound_element_type %CompleteClass.a69b, %i32 [template]
// CHECK:STDOUT:   %F.type.764c: type = fn_type @F.1, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.9cac: %F.type.764c = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.73d1 = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %import_ref.ff53: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %import_ref.7ee6: %F.type.b258 = import_ref Main//foo, F, loaded [template = constants.%F.c419]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d695
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.dc0c: <witness> = import_ref Main//foo, loc9_1, loaded [template = constants.%complete_type.a01f]
// CHECK:STDOUT:   %import_ref.8a74 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.b9fb = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %import_ref.3558 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = imports.%import_ref.73d1
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.ff53
// CHECK:STDOUT:     .F = imports.%import_ref.7ee6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [template = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.535e)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.535e), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.9af9)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.7da7)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.7da7) = struct_value () [symbolic = %F (constants.%F.2ba8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8a74
// CHECK:STDOUT:     .n = imports.%import_ref.b9fb
// CHECK:STDOUT:     .F = imports.%import_ref.3558
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.dc0c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e967 = var v
// CHECK:STDOUT:   %v: ref %CompleteClass.e967 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref: %F.type.b258 = name_ref F, imports.%import_ref.7ee6 [template = constants.%F.c419]
// CHECK:STDOUT:   %.loc14_34: ref %CompleteClass.a69b = temporary_storage
// CHECK:STDOUT:   %F.call: init %CompleteClass.a69b = call %F.ref() to %.loc14_34
// CHECK:STDOUT:   %.loc14_35: %CompleteClass.e967 = converted %F.call, <error> [template = <error>]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.a69b [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.535e
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9af9
// CHECK:STDOUT:   %F.type => constants.%F.type.7da7
// CHECK:STDOUT:   %F => constants.%F.2ba8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%ptr.b050) {
// CHECK:STDOUT:   %T => constants.%ptr.b050
// CHECK:STDOUT:   %T.patt => constants.%ptr.b050
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.e967
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.d75b
// CHECK:STDOUT:   %F.type => constants.%F.type.bb4d
// CHECK:STDOUT:   %F => constants.%F.9ec0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.a69b
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.0305
// CHECK:STDOUT:   %F.type => constants.%F.type.764c
// CHECK:STDOUT:   %F => constants.%F.9cac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [template]
// CHECK:STDOUT:   %.fc8f: type = class_type @.1, @.1(%U) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.71cf: %Class.type = import_ref Main//foo, Class, loaded [template = constants.%Class.generic]
// CHECK:STDOUT:   %import_ref.8f03 = import_ref Main//foo, CompleteClass, unloaded
// CHECK:STDOUT:   %import_ref.024f = import_ref Main//foo, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = imports.%import_ref.71cf
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.8f03
// CHECK:STDOUT:     .F = imports.%import_ref.024f
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %U.patt.loc12_13.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_13.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc12_13.1, runtime_param<invalid> [symbolic = %U.patt.loc12_13.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc12_13.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc12_13.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%U.loc12_13.1: type) {
// CHECK:STDOUT:   %U.loc12_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_13.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc12_13.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_13.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc16: <error> = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.fc8f
// CHECK:STDOUT:     .x = %.loc16
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%U) {
// CHECK:STDOUT:   %U.loc12_13.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc12_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
