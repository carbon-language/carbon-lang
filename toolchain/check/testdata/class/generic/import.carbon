// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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

//@dump-sem-ir-begin
fn F() -> CompleteClass(i32) {
  return {.n = 1};
}
//@dump-sem-ir-end

// --- use_foo.carbon

library "[[@TEST_NAME]]";
import library "foo";

fn UseMethod() -> i32 {
  //@dump-sem-ir-begin
  var v: CompleteClass(i32) = F();
  return v.F();
  //@dump-sem-ir-end
}

fn UseField() -> i32 {
  //@dump-sem-ir-begin
  var v: CompleteClass(i32) = F();
  return v.n;
  //@dump-sem-ir-end
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

// CHECK:STDOUT: --- foo.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.9a6: type = fn_type @Convert.2, @impl.a8d(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.458: %Convert.type.9a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %CompleteClass.a06: type = class_type @CompleteClass, @CompleteClass(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %CompleteClass.a06 [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.ea0: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1ad, @impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.e14: type = fn_type @Convert.2, @impl.a8d(%int_32) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.773: @impl.a8d.%Convert.type (%Convert.type.9a6) = import_ref Main//foo, inst140 [indirect], loaded [symbolic = @impl.a8d.%Convert (constants.%Convert.458)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1ad = impl_witness_table (%Main.import_ref.773), @impl.a8d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
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
// CHECK:STDOUT: fn @F.2() -> %return.param: %CompleteClass.a06 [from "foo.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_17.1: %struct_type.n.44a = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:   %bound_method.loc10_17.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_17.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc10_17.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc10_17.2: init %i32 = converted %int_1, %int.convert_checked [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc10_17.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc10_17.4: init %i32 = initialize_from %.loc10_17.2 to %.loc10_17.3 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc10_17.5: init %CompleteClass.a06 = class_init (%.loc10_17.4), %return [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   %.loc10_18: init %CompleteClass.a06 = converted %.loc10_17.1, %.loc10_17.5 [concrete = constants.%CompleteClass.val]
// CHECK:STDOUT:   return %.loc10_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_foo.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CompleteClass.type: type = generic_class_type @CompleteClass [concrete]
// CHECK:STDOUT:   %CompleteClass.generic: %CompleteClass.type = struct_value () [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CompleteClass: %CompleteClass.type = import_ref Main//foo, CompleteClass, loaded [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:   %Main.F: %F.type.b25 = import_ref Main//foo, F, loaded [concrete = constants.%F.c41]
// CHECK:STDOUT:   %Main.import_ref.e76: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.28a) = import_ref Main//foo, loc7_8, loaded [concrete = %.364]
// CHECK:STDOUT:   %Main.import_ref.a52: @CompleteClass.%F.type (%F.type.14f) = import_ref Main//foo, loc8_17, loaded [symbolic = @CompleteClass.%F (constants.%F.874)]
// CHECK:STDOUT:   %.364: @CompleteClass.%CompleteClass.elem (%CompleteClass.elem.28a) = field_decl n, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseMethod() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0b0 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0b0 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc7: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
// CHECK:STDOUT:   %.loc7_3: ref %CompleteClass.e9e = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc7: init %CompleteClass.e9e = call %F.ref.loc7() to %.loc7_3
// CHECK:STDOUT:   assign %v.var, %F.call.loc7
// CHECK:STDOUT:   %.loc7_27: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.e9e] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.e9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.e9e = bind_name v, %v.var
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.e9e = name_ref v, %v
// CHECK:STDOUT:   %.loc8_11: %F.type.1bc = specific_constant imports.%Main.import_ref.a52, @CompleteClass(constants.%i32) [concrete = constants.%F.f7c]
// CHECK:STDOUT:   %F.ref.loc8: %F.type.1bc = name_ref F, %.loc8_11 [concrete = constants.%F.f7c]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref.loc8, @F.1(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call.loc8: init %i32 = call %F.specific_fn()
// CHECK:STDOUT:   %.loc8_15.1: %i32 = value_of_initializer %F.call.loc8
// CHECK:STDOUT:   %.loc8_15.2: %i32 = converted %F.call.loc8, %.loc8_15.1
// CHECK:STDOUT:   return %.loc8_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseField() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0b0 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0b0 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %CompleteClass.e9e = var %v.var_patt
// CHECK:STDOUT:   %F.ref: %F.type.b25 = name_ref F, imports.%Main.F [concrete = constants.%F.c41]
// CHECK:STDOUT:   %.loc14_3: ref %CompleteClass.e9e = splice_block %v.var {}
// CHECK:STDOUT:   %F.call: init %CompleteClass.e9e = call %F.ref() to %.loc14_3
// CHECK:STDOUT:   assign %v.var, %F.call
// CHECK:STDOUT:   %.loc14_27: type = splice_block %CompleteClass [concrete = constants.%CompleteClass.e9e] {
// CHECK:STDOUT:     %CompleteClass.ref: %CompleteClass.type = name_ref CompleteClass, imports.%Main.CompleteClass [concrete = constants.%CompleteClass.generic]
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %CompleteClass: type = class_type @CompleteClass, @CompleteClass(constants.%i32) [concrete = constants.%CompleteClass.e9e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %CompleteClass.e9e = bind_name v, %v.var
// CHECK:STDOUT:   %v.ref: ref %CompleteClass.e9e = name_ref v, %v
// CHECK:STDOUT:   %n.ref: %CompleteClass.elem.7fc = name_ref n, imports.%Main.import_ref.e76 [concrete = imports.%.364]
// CHECK:STDOUT:   %.loc15_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc15_11.2: %i32 = bind_value %.loc15_11.1
// CHECK:STDOUT:   return %.loc15_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
