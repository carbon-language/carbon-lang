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
// CHECK:STDOUT:   %CompleteClass.1: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass.1, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %CompleteClass.2: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %return.patt: %CompleteClass.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %CompleteClass.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, file.%CompleteClass.decl [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [template = constants.%CompleteClass.2]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.2 = return_slot %return.param
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
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T.loc6_21.2) [symbolic = %CompleteClass (constants.%CompleteClass.1)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.1), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T.loc6_21.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = field_decl n, element0 [template]
// CHECK:STDOUT:     %F.decl: @CompleteClass.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @CompleteClass.%F (constants.%F.1)] {
// CHECK:STDOUT:       %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.n [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CompleteClass.1
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
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc8_27.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc8_27.2: %i32 = converted %int_0, %.loc8_27.1 [template = constants.%int_0.2]
// CHECK:STDOUT:     return %.loc8_27.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.2;
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
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.1
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
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
// CHECK:STDOUT:   %require_complete.5: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.x [symbolic]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [template]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.n.1: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %struct_type.n.1 [template]
// CHECK:STDOUT:   %CompleteClass.1: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.1: type = unbound_element_type %CompleteClass.1, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.2: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.elem.2: type = unbound_element_type %CompleteClass.2, %i32 [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.n.2: type = struct_type {.n: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %CompleteClass.val: %CompleteClass.2 = struct_value (%int_1.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.2: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.197
// CHECK:STDOUT:     .ImplicitAs = %import_ref.198
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.193: <witness> = import_ref Main//foo, loc9_1, loaded [template = constants.%complete_type.4]
// CHECK:STDOUT:   %import_ref.194 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.195 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %import_ref.196 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.2
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
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %return.patt: %CompleteClass.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %CompleteClass.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%import_ref.2 [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [template = constants.%CompleteClass.2]
// CHECK:STDOUT:     %return.param: ref %CompleteClass.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %CompleteClass.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(constants.%T: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.1 (%T) [symbolic = %require_complete (constants.%require_complete.5)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class), @Class.%T.1 (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Class.%struct_type.x (%struct_type.x) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Class.%Class.elem (%Class.elem) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.3)]
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
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.1)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.1), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.194
// CHECK:STDOUT:     .n = imports.%import_ref.195
// CHECK:STDOUT:     .F = imports.%import_ref.196
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.193
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %CompleteClass.2 [from "foo.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc9_17.1: %struct_type.n.2 = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.9 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_17.2: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_17.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc9_17.4: init %i32 = initialize_from %.loc9_17.2 to %.loc9_17.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_17.5: init %CompleteClass.2 = class_init (%.loc9_17.4), %return [template = constants.%CompleteClass.val]
// CHECK:STDOUT:   %.loc9_18: init %CompleteClass.2 = converted %.loc9_17.1, %.loc9_17.5 [template = constants.%CompleteClass.val]
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
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.1
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.1
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
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
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.2
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.2
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
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
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.1: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.1: type = unbound_element_type %CompleteClass.1, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.2: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %CompleteClass.elem.2: type = unbound_element_type %CompleteClass.2, %i32 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.2, @F.1(%i32) [template]
// CHECK:STDOUT:   %UseField.type: type = fn_type @UseField [template]
// CHECK:STDOUT:   %UseField: %UseField.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %import_ref.2: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %import_ref.3: %F.type.3 = import_ref Main//foo, F, loaded [template = constants.%F.3]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.4
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8: <witness> = import_ref Main//foo, loc9_1, loaded [template = constants.%complete_type.3]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.10: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.1) = import_ref Main//foo, loc7_8, loaded [template = %.1]
// CHECK:STDOUT:   %import_ref.11: @CompleteClass.%F.type (%F.type.1) = import_ref Main//foo, loc8_17, loaded [symbolic = @CompleteClass.%F (constants.%F.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = imports.%import_ref.1
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.2
// CHECK:STDOUT:     .F = imports.%import_ref.3
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
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseField.decl: %UseField.type = fn_decl @UseField [template = constants.%UseField] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
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
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.1)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.1), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9
// CHECK:STDOUT:     .n = imports.%import_ref.10
// CHECK:STDOUT:     .F = imports.%import_ref.11
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseMethod() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%import_ref.2 [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [template = constants.%CompleteClass.2]
// CHECK:STDOUT:   %v.var: ref %CompleteClass.2 = var v
// CHECK:STDOUT:   %v: ref %CompleteClass.2 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc6: %F.type.3 = name_ref F, imports.%import_ref.3 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc6: ref %CompleteClass.2 = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc6: init %CompleteClass.2 = call %F.ref.loc6() to %.loc6
// CHECK:STDOUT:   assign %v.var, %F.call.loc6
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.2 = name_ref v, %v
// CHECK:STDOUT:   %.loc7_11: %F.type.2 = specific_constant imports.%import_ref.11, @CompleteClass(constants.%i32) [template = constants.%F.2]
// CHECK:STDOUT:   %F.ref.loc7: %F.type.2 = name_ref F, %.loc7_11 [template = constants.%F.2]
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
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.2 [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseField() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%import_ref.2 [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [template = constants.%CompleteClass.2]
// CHECK:STDOUT:   %v.var: ref %CompleteClass.2 = var v
// CHECK:STDOUT:   %v: ref %CompleteClass.2 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref: %F.type.3 = name_ref F, imports.%import_ref.3 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc11: ref %CompleteClass.2 = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.2 = call %F.ref() to %.loc11
// CHECK:STDOUT:   assign %v.var, %F.call
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.2 = name_ref v, %v
// CHECK:STDOUT:   %n.ref: %CompleteClass.elem.2 = name_ref n, imports.%import_ref.10 [template = imports.%.1]
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
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.1
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.1
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
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
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.2
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.2
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
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
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.1: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.1: type = unbound_element_type %CompleteClass.1, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %CompleteClass.2: type = class_type @CompleteClass, @CompleteClass(%ptr.1) [template]
// CHECK:STDOUT:   %CompleteClass.elem.2: type = unbound_element_type %CompleteClass.2, %i32 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @CompleteClass(%ptr.1) [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %CompleteClass.3: type = class_type @CompleteClass, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %CompleteClass.elem.3: type = unbound_element_type %CompleteClass.3, %i32 [template]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.1, @CompleteClass(%i32) [template]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %import_ref.2: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %import_ref.3: %F.type.3 = import_ref Main//foo, F, loaded [template = constants.%F.3]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.11
// CHECK:STDOUT:     .ImplicitAs = %import_ref.12
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: <witness> = import_ref Main//foo, loc9_1, loaded [template = constants.%complete_type.2]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//foo, inst37 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = imports.%import_ref.1
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.2
// CHECK:STDOUT:     .F = imports.%import_ref.3
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
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.1)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type @CompleteClass.%CompleteClass (%CompleteClass.1), %i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8
// CHECK:STDOUT:     .n = imports.%import_ref.9
// CHECK:STDOUT:     .F = imports.%import_ref.10
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%import_ref.2 [template = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [template = constants.%ptr.1]
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%ptr.1) [template = constants.%CompleteClass.2]
// CHECK:STDOUT:   %v.var: ref %CompleteClass.2 = var v
// CHECK:STDOUT:   %v: ref %CompleteClass.2 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref: %F.type.3 = name_ref F, imports.%import_ref.3 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc14_34: ref %CompleteClass.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %CompleteClass.3 = call %F.ref() to %.loc14_34
// CHECK:STDOUT:   %.loc14_35: %CompleteClass.2 = converted %F.call, <error> [template = <error>]
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
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.3 [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.1
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.1
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%ptr.1) {
// CHECK:STDOUT:   %T => constants.%ptr.1
// CHECK:STDOUT:   %T.patt => constants.%ptr.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.2
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.2
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.3
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.3
// CHECK:STDOUT:   %F.type => constants.%F.type.4
// CHECK:STDOUT:   %F => constants.%F.4
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
// CHECK:STDOUT:   %.1: type = class_type @.1, @.1(%U) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Class.type = import_ref Main//foo, Class, loaded [template = constants.%Class.generic]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//foo, CompleteClass, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//foo, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = imports.%import_ref.1
// CHECK:STDOUT:     .CompleteClass = imports.%import_ref.2
// CHECK:STDOUT:     .F = imports.%import_ref.3
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
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [template = <error>]
// CHECK:STDOUT:     %.loc16: <error> = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.1
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
