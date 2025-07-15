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
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.loc8, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %CompleteClass.e9e: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.0b0: type = pattern_type %CompleteClass.e9e [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.loc11 [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
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
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CompleteClass.decl: %CompleteClass.type = class_decl @CompleteClass [concrete = constants.%CompleteClass.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.loc11 [concrete = constants.%F.c41] {
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
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(%T.loc6_21.1: type) {
// CHECK:STDOUT:   %T.loc6_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_21.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass: type = class_type @CompleteClass, @CompleteClass(%T.loc6_21.2) [symbolic = %CompleteClass (constants.%CompleteClass.f97)]
// CHECK:STDOUT:   %CompleteClass.elem: type = unbound_element_type %CompleteClass, constants.%i32 [symbolic = %CompleteClass.elem (constants.%CompleteClass.elem)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc8, @CompleteClass(%T.loc6_21.2) [symbolic = %F.type (constants.%F.type.14f)]
// CHECK:STDOUT:   %F: @CompleteClass.%F.type (%F.type.14f) = struct_value () [symbolic = %F (constants.%F.874)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %F.decl: @CompleteClass.%F.type (%F.type.14f) = fn_decl @F.loc8 [symbolic = @CompleteClass.%F (constants.%F.874)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %struct_type.n: type = struct_type {.n: %i32} [concrete = constants.%struct_type.n]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%CompleteClass.f97
// CHECK:STDOUT:     .n = %.loc7
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc8(@CompleteClass.%T.loc6_21.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
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
// CHECK:STDOUT: fn @F.loc11() -> %CompleteClass.e9e;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc8(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T.loc6_21.2 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.595: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Convert.type.9a6: type = fn_type @Convert.2, @ImplicitAs.impl.a8d(%To) [symbolic]
// CHECK:STDOUT:   %Convert.458: %Convert.type.9a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x [symbolic]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.n.4d6: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a68: <witness> = complete_type_witness %struct_type.n.4d6 [concrete]
// CHECK:STDOUT:   %CompleteClass.f97: type = class_type @CompleteClass, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %CompleteClass.elem.9ef: type = unbound_element_type %CompleteClass.f97, %i32 [symbolic]
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.a06: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %CompleteClass.a06 [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.loc8 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.426: type = unbound_element_type %CompleteClass.a06, %i32 [concrete]
// CHECK:STDOUT:   %F.type.0aa: type = fn_type @F.1, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.971: %F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.ea0: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1ad, @ImplicitAs.impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.e14: type = fn_type @Convert.2, @ImplicitAs.impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.4cb: %Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b9e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.940: type = fn_type_with_self_type %Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.4cb [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.4cb, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %CompleteClass.val: %CompleteClass.a06 = struct_value (%int_1.47b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.773: @ImplicitAs.impl.a8d.%Convert.type (%Convert.type.9a6) = import_ref Main//foo, inst140 [indirect], loaded [symbolic = @ImplicitAs.impl.a8d.%Convert (constants.%Convert.458)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1ad = impl_witness_table (%Main.import_ref.773), @ImplicitAs.impl.a8d [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc4_13, loaded [symbolic = @Class.%T.1 (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.a68]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst33 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.051 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.570 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
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
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.loc8 [concrete = constants.%F.c41] {
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
// CHECK:STDOUT: generic class @Class(imports.%Main.import_ref.5ab3ec.1: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.x.loc6_1.2: type = struct_type {.x: @Class.%T.1 (%T)} [symbolic = %struct_type.x.loc6_1.2 (constants.%struct_type.x)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x.loc6_1.2 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Class.%Class.elem (%Class.elem) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %struct_type.x.loc6_1.1: type = struct_type {.x: %T} [symbolic = %struct_type.x.loc6_1.2 (constants.%struct_type.x)]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x.loc6_1.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
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
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc8() -> %return.param: %CompleteClass.a06 [from "foo.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_17.1: %struct_type.n.44a = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Convert.4cb]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9ef
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
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
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %CompleteClass.e9e: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.7fc: type = unbound_element_type %CompleteClass.e9e, %i32 [concrete]
// CHECK:STDOUT:   %F.type.1bc: type = fn_type @F.1, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.f7c: %F.type.1bc = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0b0: type = pattern_type %CompleteClass.e9e [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.f7c, @F.1(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.9c5: type = fn_type @Op.3, @Destroy.impl(%CompleteClass.e9e) [concrete]
// CHECK:STDOUT:   %Op.d10: %Op.type.9c5 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.a97: type = ptr_type %CompleteClass.e9e [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.d10, @Op.3(%CompleteClass.e9e) [concrete]
// CHECK:STDOUT:   %UseField.type: type = fn_type @UseField [concrete]
// CHECK:STDOUT:   %UseField: %UseField.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type.b25 = import_ref Main//foo, F, loaded [concrete = constants.%F.c41]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst33 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e76: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.28a) = import_ref Main//foo, loc7_8, loaded [concrete = %.364]
// CHECK:STDOUT:   %Main.import_ref.a52: @CompleteClass.%F.type (%F.type.14f) = import_ref Main//foo, loc8_17, loaded [symbolic = @CompleteClass.%F (constants.%F.874)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
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
// CHECK:STDOUT:     %v.patt: %pattern_type.0b0 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0b0 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc6: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
// CHECK:STDOUT:   %.loc6_3: ref %CompleteClass.e9e = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc6: init %CompleteClass.e9e = call %F.ref.loc6() to %.loc6_3
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
// CHECK:STDOUT:   %Op.bound.loc6_3.1: <bound method> = bound_method %.loc6_3, constants.%Op.d10
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.d10, @Op.3(constants.%CompleteClass.e9e) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_3.1: <bound method> = bound_method %.loc6_3, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc6_3.1: %ptr.a97 = addr_of %.loc6_3
// CHECK:STDOUT:   %no_op.loc6_3.1: init %empty_tuple.type = call %bound_method.loc6_3.1(%addr.loc6_3.1)
// CHECK:STDOUT:   %Op.bound.loc6_3.2: <bound method> = bound_method %v.var, constants.%Op.d10
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.d10, @Op.3(constants.%CompleteClass.e9e) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_3.2: <bound method> = bound_method %v.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc6_3.2: %ptr.a97 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc6_3.2: init %empty_tuple.type = call %bound_method.loc6_3.2(%addr.loc6_3.2)
// CHECK:STDOUT:   return %.loc7_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2 [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseField() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0b0 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0b0 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var %v.var_patt
// CHECK:STDOUT:   %F.ref: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
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
// CHECK:STDOUT:   %Op.bound.loc11_3.1: <bound method> = bound_method %.loc11_3, constants.%Op.d10
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.d10, @Op.3(constants.%CompleteClass.e9e) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_3.1: <bound method> = bound_method %.loc11_3, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc11_3.1: %ptr.a97 = addr_of %.loc11_3
// CHECK:STDOUT:   %no_op.loc11_3.1: init %empty_tuple.type = call %bound_method.loc11_3.1(%addr.loc11_3.1)
// CHECK:STDOUT:   %Op.bound.loc11_3.2: <bound method> = bound_method %v.var, constants.%Op.d10
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.d10, @Op.3(constants.%CompleteClass.e9e) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_3.2: <bound method> = bound_method %v.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc11_3.2: %ptr.a97 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc11_3.2: init %empty_tuple.type = call %bound_method.loc11_3.2(%addr.loc11_3.2)
// CHECK:STDOUT:   return %.loc12_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.28a
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
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
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @CompleteClass(%T) [symbolic]
// CHECK:STDOUT:   %F.874: %F.type.14f = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.9e1: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %CompleteClass.0fe: type = class_type @CompleteClass, @CompleteClass(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.d29: type = unbound_element_type %CompleteClass.0fe, %i32 [concrete]
// CHECK:STDOUT:   %F.type.8cf: type = fn_type @F.1, @CompleteClass(%ptr.9e1) [concrete]
// CHECK:STDOUT:   %F.012: %F.type.8cf = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.062: type = pattern_type %CompleteClass.0fe [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.a06: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %CompleteClass.elem.426: type = unbound_element_type %CompleteClass.a06, %i32 [concrete]
// CHECK:STDOUT:   %F.type.0aa: type = fn_type @F.1, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %F.971: %F.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.370: type = fn_type @Op.3, @Destroy.impl(%CompleteClass.a06) [concrete]
// CHECK:STDOUT:   %Op.867: %Op.type.370 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d29: type = ptr_type %CompleteClass.a06 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.0ab: <specific function> = specific_function %Op.867, @Op.3(%CompleteClass.a06) [concrete]
// CHECK:STDOUT:   %Op.type.310: type = fn_type @Op.3, @Destroy.impl(%CompleteClass.0fe) [concrete]
// CHECK:STDOUT:   %Op.200: %Op.type.310 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c79: type = ptr_type %CompleteClass.0fe [concrete]
// CHECK:STDOUT:   %Op.specific_fn.a4e: <specific function> = specific_function %Op.200, @Op.3(%CompleteClass.0fe) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class = import_ref Main//foo, Class, unloaded
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type.b25 = import_ref Main//foo, F, loaded [concrete = constants.%F.c41]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.eb1: <witness> = import_ref Main//foo, loc9_1, loaded [concrete = constants.%complete_type.a68]
// CHECK:STDOUT:   %Main.import_ref.3c0 = import_ref Main//foo, inst33 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.051 = import_ref Main//foo, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.570 = import_ref Main//foo, loc8_17, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//foo, loc6_21, loaded [symbolic = @CompleteClass.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT: generic class @CompleteClass(imports.%Main.import_ref.5ab3ec.1: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
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
// CHECK:STDOUT:     %v.patt: %pattern_type.062 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.062 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.0fe = var %v.var_patt
// CHECK:STDOUT:   %F.ref: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
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
// CHECK:STDOUT:   %Op.bound.loc14_34: <bound method> = bound_method %.loc14_34, constants.%Op.867
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.867, @Op.3(constants.%CompleteClass.a06) [concrete = constants.%Op.specific_fn.0ab]
// CHECK:STDOUT:   %bound_method.loc14_34: <bound method> = bound_method %.loc14_34, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc14_34: %ptr.d29 = addr_of %.loc14_34
// CHECK:STDOUT:   %no_op.loc14_34: init %empty_tuple.type = call %bound_method.loc14_34(%addr.loc14_34)
// CHECK:STDOUT:   %Op.bound.loc14_3: <bound method> = bound_method %v.var, constants.%Op.200
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.200, @Op.3(constants.%CompleteClass.0fe) [concrete = constants.%Op.specific_fn.a4e]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %v.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc14_3: %ptr.c79 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc14_3: init %empty_tuple.type = call %bound_method.loc14_3(%addr.loc14_3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type) [from "foo.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2 [from "foo.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %CompleteClass => constants.%CompleteClass.f97
// CHECK:STDOUT:   %CompleteClass.elem => constants.%CompleteClass.elem.9ef
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.874
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CompleteClass(constants.%ptr.9e1) {
// CHECK:STDOUT:   %T => constants.%ptr.9e1
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %Class.type.cf06d9.1: type = generic_class_type @Class.1 [concrete]
// CHECK:STDOUT:   %Class.generic.9545f5.1: %Class.type.cf06d9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type.cf06d9.2: type = generic_class_type @Class.loc12 [concrete]
// CHECK:STDOUT:   %Class.generic.9545f5.2: %Class.type.cf06d9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1b2d.2: type = class_type @Class.loc12, @Class.loc12(%U) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Class: %Class.type.cf06d9.1 = import_ref Main//foo, Class, loaded [concrete = constants.%Class.generic.9545f5.1]
// CHECK:STDOUT:   %Main.CompleteClass = import_ref Main//foo, CompleteClass, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//foo, F, unloaded
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_19.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_19.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type.cf06d9.2 = class_decl @Class.loc12 [concrete = constants.%Class.generic.9545f5.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc12_13.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_13.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.1(imports.%Main.import_ref.5ab: type) [from "foo.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class.loc12(%U.loc12_13.1: type) {
// CHECK:STDOUT:   %U.loc12_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_13.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc17: <error> = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %struct_type.x: type = struct_type {.x: <error>} [concrete = <error>]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.x [concrete = <error>]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1b2d.2
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.loc12(constants.%U) {
// CHECK:STDOUT:   %U.loc12_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
