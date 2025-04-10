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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %CompleteClass.e9e: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompleteClass.decl: %CompleteClass.type = class_decl @CompleteClass [concrete = constants.%CompleteClass.generic] {
// CHECK:STDOUT:     %T.patt.loc6_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %return.patt: %CompleteClass.e9e = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %CompleteClass.e9e = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T.loc6_21.2) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T.loc6_21.2) [symbolic = %F.type (constants.%F.type.14f)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.14f) = struct_value () [symbolic = %F (constants.%F.874)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc7_8: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc7_3: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = var_pattern %.loc7_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = var <none>
// CHECK:STDOUT:     %F.decl: @CompleteClass.%F.type (%F.type.14f) = fn_decl @F.1 [symbolic = @CompleteClass.%F (constants.%F.874)] {
// CHECK:STDOUT:       %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CompleteClass.f97
// CHECK:STDOUT:     .n = %.loc7_8
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@CompleteClass.%T.loc6_21.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc8_27.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_27.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc8_27.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_27.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_27.2: %i32 = converted %int_0, %.loc8_27.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     return %.loc8_27.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.e9e;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_21.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T.loc6_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc6_21.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x [symbolic]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.n.4d6: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a68: <witness> = complete_type_witness %struct_type.n.4d6 [concrete]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.9ef: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.a06: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.426: type = unbound_element_type %CompleteClass.a06, %i32 [concrete]
// CHECK:STDOUT:   %F.type.0aa: type = fn_type @F.1, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.971: %F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.ea0: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.847: <witness> = impl_witness (imports.%Main.import_ref.773), @impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.e14: type = fn_type @Convert.2, @impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.4cb: %Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b9e = facet_value Core.IntLiteral, (%impl_witness.847) [concrete]
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.4cb [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.4cb, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %CompleteClass.val: %CompleteClass.a06 = struct_value (%int_1.47b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc4_13, loaded [symbolic = @Class.%T.1 (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.a68]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst35 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.051 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.570 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_19.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_19.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %return.patt: %CompleteClass.a06 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %CompleteClass.a06 = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT: generic class @Class(imports.%Main.import_ref.5ab3ec.1: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.1 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Class.%struct_type.x (%struct_type.x) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5_8: @Class.%Class.elem (%Class.elem) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc5_3: @Class.%Class.elem (%Class.elem) = var_pattern %.loc5_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Class.%Class.elem (%Class.elem) = var <none>
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.9ef)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.14f)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.14f) = struct_value () [symbolic = %F (constants.%F.874)]
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
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.3: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %CompleteClass.a06 [from "foo.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_17.1: %struct_type.n.44a = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:   %bound_method.loc9_17.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_17.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc9_17.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc9_17.2: init %i32 = converted %int_1, %int.convert_checked [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc9_17.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc9_17.4: init %i32 = initialize_from %.loc9_17.2 to %.loc9_17.3 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc9_17.5: init %CompleteClass.a06 = class_init (%.loc9_17.4), %return [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   %.loc9_18: init %CompleteClass.a06 = converted %.loc9_17.1, %.loc9_17.5 [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   return %.loc9_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9ef
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.a06
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.426
// CHECK:STDOUT:   %F.type => constants.%F.type.0aa
// CHECK:STDOUT:   %F => constants.%F.971
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %UseMethod.type: type = fn_type @UseMethod [concrete]
// CHECK:STDOUT:   %UseMethod: %UseMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.28a: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.e9e: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.7fc: type = unbound_element_type %CompleteClass.e9e, %i32 [concrete]
// CHECK:STDOUT:   %F.type.1bc: type = fn_type @F.1, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.f7c: %F.type.1bc = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.f7c, @F.1(%i32) [concrete]
// CHECK:STDOUT:   %UseField.type: type = fn_type @UseField [concrete]
// CHECK:STDOUT:   %UseField: %UseField.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type.b25 = import_ref Main//foo, F, loaded [concrete = constants.%F.c41]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst35 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e76: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.28a) = import_ref Main//foo, loc7_8, loaded [concrete = %.364]
// CHECK:STDOUT:   %Main.import_ref.a52: @CompleteClass.%F.type (%F.type.14f) = import_ref Main//foo, loc8_17, loaded [symbolic = @CompleteClass.%F (constants.%F.874)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
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
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseField.decl: %UseField.type = fn_decl @UseField [concrete = constants.%UseField] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.28a)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.14f)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.14f) = struct_value () [symbolic = %F (constants.%F.874)]
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
// CHECK:STDOUT:     %v.patt: %CompleteClass.e9e = binding_pattern v
// CHECK:STDOUT:     %.loc6_3.1: %CompleteClass.e9e = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var v
// CHECK:STDOUT:   %F.ref.loc6: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
// CHECK:STDOUT:   %.loc6_3.2: ref %CompleteClass.e9e = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc6: init %CompleteClass.e9e = call %F.ref.loc6() to %.loc6_3.2
// CHECK:STDOUT:   assign %v.var, %F.call.loc6
// CHECK:STDOUT:   %.loc6_27: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.e9e] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.e9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.e9e = bind_name v, %v.var
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.e9e = name_ref v, %v
// CHECK:STDOUT:   %.loc7_11: %F.type.1bc = specific_constant imports.%Main.import_ref.a52, @CompleteClass(constants.%i32) [concrete = constants.%F.f7c]
// CHECK:STDOUT:   %F.ref.loc7: %F.type.1bc = name_ref F, %.loc7_11 [concrete = constants.%F.f7c]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref.loc7, @F.1(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call.loc7: init %i32 = call %F.specific_fn()
// CHECK:STDOUT:   %.loc7_15.1: %i32 = value_of_initializer %F.call.loc7
// CHECK:STDOUT:   %.loc7_15.2: %i32 = converted %F.call.loc7, %.loc7_15.1
// CHECK:STDOUT:   return %.loc7_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.e9e [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseField() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %CompleteClass.e9e = binding_pattern v
// CHECK:STDOUT:     %.loc11_3.1: %CompleteClass.e9e = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var v
// CHECK:STDOUT:   %F.ref: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
// CHECK:STDOUT:   %.loc11_3.2: ref %CompleteClass.e9e = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.e9e = call %F.ref() to %.loc11_3.2
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
// CHECK:STDOUT:   return %.loc12_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.28a
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.e9e
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.7fc
// CHECK:STDOUT:   %F.type => constants.%F.type.1bc
// CHECK:STDOUT:   %F => constants.%F.f7c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_arg_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a68: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.9ef: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.9e1: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.0fe: type = class_type @CompleteClass, @CompleteClass(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.d29: type = unbound_element_type %CompleteClass.0fe, %i32 [concrete]
// CHECK:STDOUT:   %F.type.8cf: type = fn_type @F.1, @CompleteClass(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %F.012: %F.type.8cf = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.a06: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.426: type = unbound_element_type %CompleteClass.a06, %i32 [concrete]
// CHECK:STDOUT:   %F.type.0aa: type = fn_type @F.1, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.971: %F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type.b25 = import_ref Main//foo, F, loaded [concrete = constants.%F.c41]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.a68]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst35 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.051 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.570 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem.9ef)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @CompleteClass(%T) [symbolic = %F.type (constants.%F.type.14f)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.14f) = struct_value () [symbolic = %F (constants.%F.874)]
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
// CHECK:STDOUT:     %v.patt: %CompleteClass.0fe = binding_pattern v
// CHECK:STDOUT:     %.loc14_3.1: %CompleteClass.0fe = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.0fe = var v
// CHECK:STDOUT:   %F.ref: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
// CHECK:STDOUT:   %.loc14_34: ref %CompleteClass.a06 = temporary_storage
// CHECK:STDOUT:   %F.call: init %CompleteClass.a06 = call %F.ref() to %.loc14_34
// CHECK:STDOUT:   %.loc14_3.2: %CompleteClass.0fe = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %.loc14_28: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.0fe] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.9e1]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%ptr.9e1) [concrete = constants.%CompleteClass.0fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.0fe = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %CompleteClass.a06 [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9ef
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%ptr.9e1) {
// CHECK:STDOUT:   %T => constants.%ptr.9e1
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.0fe
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.d29
// CHECK:STDOUT:   %F.type => constants.%F.type.8cf
// CHECK:STDOUT:   %F => constants.%F.012
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.a06
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.426
// CHECK:STDOUT:   %F.type => constants.%F.type.0aa
// CHECK:STDOUT:   %F => constants.%F.971
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %Class.type.cf06d9.1: type = generic_class_type @Class.1 [concrete]
// CHECK:STDOUT:   %Class.generic.9545f5.1: %Class.type.cf06d9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type.cf06d9.2: type = generic_class_type @Class.2 [concrete]
// CHECK:STDOUT:   %Class.generic.9545f5.2: %Class.type.cf06d9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1b2d.2: type = class_type @Class.2, @Class.2(%U) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class: %Class.type.cf06d9.1 = import_ref Main//foo, Class, loaded [concrete = constants.%Class.generic.9545f5.1]
// CHECK:STDOUT:   %Main.CompleteClass = import_ref Main//foo, CompleteClass, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//foo, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab: type = import_ref Main//foo, loc4_13, loaded [symbolic = @Class.1.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = imports.%Main.Class
// CHECK:STDOUT:     .CompleteClass = imports.%Main.CompleteClass
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_19.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_19.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type.cf06d9.2 = class_decl @Class.2 [concrete = constants.%Class.generic.9545f5.2] {
// CHECK:STDOUT:     %U.patt.loc12_13.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_13.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc12_13.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_13.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.1(imports.%Main.import_ref.5ab: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.2(%U.loc12_13.1: type) {
// CHECK:STDOUT:   %U.loc12_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_13.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc12_13.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_13.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc17_8: <error> = field_decl x, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc17_3: <error> = var_pattern %.loc17_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref <error> = var <none>
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1b2d.2
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc17_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.2(constants.%U) {
// CHECK:STDOUT:   %U.loc12_13.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc12_13.2 => constants.%U.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
