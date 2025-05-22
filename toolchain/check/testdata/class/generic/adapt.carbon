// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/adapt.carbon

// --- adapt_specific_type.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  var x: T;
}

//@dump-sem-ir-begin
class Adapter {
  adapt C(i32);
}
//@dump-sem-ir-end

fn Access(a: Adapter) -> i32 {
  //@dump-sem-ir-begin
  return (a as C(i32)).x;
  //@dump-sem-ir-end
}

// --- import_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

import library "adapt_specific_type";

fn ImportedAccess(a: Adapter) -> i32 {
  //@dump-sem-ir-begin
  return (a as C(i32)).x;
  //@dump-sem-ir-end
}

// --- fail_todo_extend_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  var x: T;
}

class Adapter {
  extend adapt C(i32);
}

fn Access(a: Adapter) -> i32 {
  // TODO: This should presumably work, but the design doesn't say how yet.
  // CHECK:STDERR: fail_todo_extend_adapt_specific_type.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `Adapter` to `C(i32)` [ConversionFailure]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_extend_adapt_specific_type.carbon:[[@LINE+4]]:10: note: type `Adapter` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.x;
}

// --- extend_adapt_specific_type_library.carbon

// TODO: Delete this file and change the next file to instead import the
// previous file once the previous file can be successfully type-checked.

library "[[@TEST_NAME]]";

class C(T:! type) {
  var x: T;
}

//@dump-sem-ir-begin
class Adapter {
  extend adapt C(i32);
}
//@dump-sem-ir-end

// --- fail_todo_import_extend_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

import library "extend_adapt_specific_type_library";

fn ImportedAccess(a: Adapter) -> i32 {
  // TODO: This should presumably work, but the design doesn't say how yet.
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `Adapter` to `C(i32)` [ConversionFailure]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+4]]:10: note: type `Adapter` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.x;
}

// --- adapt_generic_type.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Adapter(T:! type) {
  adapt T;
}

fn Convert(a: Adapter(i32)) -> i32 {
  return a as i32;
}
//@dump-sem-ir-end

// --- import_adapt_generic_type.carbon

library "[[@TEST_NAME]]";

import library "adapt_generic_type";

fn ImportedConvert(a: Adapter(i32)) -> i32 {
  //@dump-sem-ir-begin
  return a as i32;
  //@dump-sem-ir-end
}

class C {
  var n: i32;
}

fn ImportedConvertLocal(a: Adapter(C)) -> i32 {
  //@dump-sem-ir-begin
  return (a as C).n;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %C.elem.476: type = unbound_element_type %C.98a, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%a.param: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   %.loc16_13.1: %C.98a = as_compatible %a.ref
// CHECK:STDOUT:   %.loc16_13.2: %C.98a = converted %a.ref, %.loc16_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.476 = name_ref x, @C.%.loc5 [concrete = @C.%.loc5]
// CHECK:STDOUT:   %.loc16_23.1: ref %i32 = class_element_access %.loc16_13.2, element0
// CHECK:STDOUT:   %.loc16_23.2: %i32 = bind_value %.loc16_23.1
// CHECK:STDOUT:   return %.loc16_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.239: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %C.elem.66c: type = unbound_element_type %C.f2e, %T [symbolic]
// CHECK:STDOUT:   %C.elem.ed6: type = unbound_element_type %C.239, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//adapt_specific_type, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.import_ref.262: @C.%C.elem (%C.elem.66c) = import_ref Main//adapt_specific_type, loc5_8, loaded [concrete = %.22b]
// CHECK:STDOUT:   %.22b: @C.%C.elem (%C.elem.66c) = field_decl x, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.239]
// CHECK:STDOUT:   %.loc8_13.1: %C.239 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc8_13.2: %C.239 = converted %a.ref, %.loc8_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.ed6 = name_ref x, imports.%Main.import_ref.262 [concrete = imports.%.22b]
// CHECK:STDOUT:   %.loc8_23.1: ref %i32 = class_element_access %.loc8_13.2, element0
// CHECK:STDOUT:   %.loc8_23.2: %i32 = bind_value %.loc8_23.1
// CHECK:STDOUT:   return %.loc8_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_adapt_specific_type_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   extend %C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [concrete]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapter.0e3: type = class_type @Adapter, @Adapter(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %complete_type.f87: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Adapter.e4c: type = class_type @Adapter, @Adapter(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.5a0: type = pattern_type %Adapter.e4c [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert [concrete]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %complete_type.1eb: <witness> = complete_type_witness %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Adapter.decl: %Adapter.type = class_decl @Adapter [concrete = constants.%Adapter.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Convert.decl: %Convert.type = fn_decl @Convert [concrete = constants.%Convert] {
// CHECK:STDOUT:     %a.patt: %pattern_type.5a0 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.5a0 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.e4c = value_param call_param0
// CHECK:STDOUT:     %.loc9: type = splice_block %Adapter [concrete = constants.%Adapter.e4c] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:       %int_32.loc9_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [concrete = constants.%Adapter.e4c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.e4c = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(%T.loc5_15.1: type) {
// CHECK:STDOUT:   %T.loc5_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_15.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %complete_type.loc7_1.2: <witness> = complete_type_witness %T.loc5_15.2 [symbolic = %complete_type.loc7_1.2 (constants.%complete_type.f87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_15.1 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:     adapt_decl %T.ref [concrete]
// CHECK:STDOUT:     %complete_type.loc7_1.1: <witness> = complete_type_witness constants.%T [symbolic = %complete_type.loc7_1.2 (constants.%complete_type.f87)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapter.0e3
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%a.param: %Adapter.e4c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.e4c = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc10_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc10_12.2: %i32 = converted %a.ref, %.loc10_12.1
// CHECK:STDOUT:   return %.loc10_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%T) {
// CHECK:STDOUT:   %T.loc5_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%i32) {
// CHECK:STDOUT:   %T.loc5_15.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type.loc7_1.2 => constants.%complete_type.1eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Adapter.e4c: type = class_type @Adapter, @Adapter(%i32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %Adapter.58f: type = class_type @Adapter, @Adapter(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvert(%a.param: %Adapter.e4c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.e4c = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc8_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc8_12.2: %i32 = converted %a.ref, %.loc8_12.1
// CHECK:STDOUT:   return %.loc8_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvertLocal(%a.param: %Adapter.58f) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.58f = name_ref a, %a
// CHECK:STDOUT:   %C.ref.loc18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc18_13.1: %C = as_compatible %a.ref
// CHECK:STDOUT:   %.loc18_13.2: %C = converted %a.ref, %.loc18_13.1
// CHECK:STDOUT:   %n.ref: %C.elem = name_ref n, @C.%.loc13 [concrete = @C.%.loc13]
// CHECK:STDOUT:   %.loc18_18.1: ref %i32 = class_element_access %.loc18_13.2, element0
// CHECK:STDOUT:   %.loc18_18.2: %i32 = bind_value %.loc18_18.1
// CHECK:STDOUT:   return %.loc18_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
