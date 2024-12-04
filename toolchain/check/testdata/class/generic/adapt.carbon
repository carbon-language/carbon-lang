// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

class Adapter {
  adapt C(i32);
}

fn Access(a: Adapter) -> i32 {
  return (a as C(i32)).x;
}

// --- import_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

import library "adapt_specific_type";

fn ImportedAccess(a: Adapter) -> i32 {
  return (a as C(i32)).x;
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
  // CHECK:STDERR: fail_todo_extend_adapt_specific_type.carbon:[[@LINE+7]]:10: error: cannot implicitly convert from `Adapter` to `C(i32)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_extend_adapt_specific_type.carbon:[[@LINE+4]]:10: note: type `Adapter` does not implement interface `ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
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

class Adapter {
  extend adapt C(i32);
}

// --- fail_todo_import_extend_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

import library "extend_adapt_specific_type_library";

fn ImportedAccess(a: Adapter) -> i32 {
  // TODO: This should presumably work, but the design doesn't say how yet.
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+6]]:10: error: cannot implicitly convert from `Adapter` to `C(i32)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+3]]:10: note: type `Adapter` does not implement interface `ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  return a.x;
}

// --- adapt_generic_type.carbon

library "[[@TEST_NAME]]";

class Adapter(T:! type) {
  adapt T;
}

fn Convert(a: Adapter(i32)) -> i32 {
  return a as i32;
}

// --- import_adapt_generic_type.carbon

library "[[@TEST_NAME]]";

import library "adapt_generic_type";

fn ImportedConvert(a: Adapter(i32)) -> i32 {
  return a as i32;
}

class C {
  var n: i32;
}

fn ImportedConvertLocal(a: Adapter(C)) -> i32 {
  return (a as C).n;
}

// CHECK:STDOUT: --- adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.1: type = unbound_element_type %C.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %C.elem.2: type = unbound_element_type %C.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.2 [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<invalid> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [template = constants.%Adapter] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %a.patt: %Adapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [template = constants.%Adapter]
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_26.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_26.2: type = converted %int.make_type_signed.loc12, %.loc12_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %Adapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.2) [symbolic = %C (constants.%C.1)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.1), @C.%T.loc4_9.2 (%T) [symbolic = %C.elem (constants.%C.elem.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_9.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.1)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.1) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_9.1 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.1) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_14.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_14.2: type = converted %int.make_type_signed, %.loc9_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.2]
// CHECK:STDOUT:   adapt_decl %C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.2 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%a.param_patt: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_signed.loc13, %.loc13_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.2]
// CHECK:STDOUT:   %.loc13_13.1: %C.2 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc13_13.2: %C.2 = converted %a.ref, %.loc13_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.2 = name_ref x, @C.%.loc5 [template = @C.%.loc5]
// CHECK:STDOUT:   %.loc13_23.1: ref %i32 = class_element_access %.loc13_13.2, element0
// CHECK:STDOUT:   %.loc13_23.2: %i32 = bind_value %.loc13_23.1
// CHECK:STDOUT:   return %.loc13_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc4_9.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %C.elem => constants.%C.elem.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.2 [template]
// CHECK:STDOUT:   %C.elem.1: type = unbound_element_type %C.1, %T [symbolic]
// CHECK:STDOUT:   %C.elem.2: type = unbound_element_type %C.2, %i32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [template]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %C.type = import_ref Main//adapt_specific_type, inst+9, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//adapt_specific_type, inst+26, loaded [template = constants.%Adapter]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//adapt_specific_type, inst+64, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.10
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: <witness> = import_ref Main//adapt_specific_type, inst+20, loaded [symbolic = @C.%complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//adapt_specific_type, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.6: @C.%C.elem (%C.elem.1) = import_ref Main//adapt_specific_type, inst+18, loaded [template = %.1]
// CHECK:STDOUT:   %import_ref.8: <witness> = import_ref Main//adapt_specific_type, inst+50, loaded [template = constants.%complete_type.2]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//adapt_specific_type, inst+27, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .Adapter = imports.%import_ref.2
// CHECK:STDOUT:     .Access = imports.%import_ref.3
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImportedAccess = %ImportedAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %ImportedAccess.decl: %ImportedAccess.type = fn_decl @ImportedAccess [template = constants.%ImportedAccess] {
// CHECK:STDOUT:     %a.patt: %Adapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%import_ref.2 [template = constants.%Adapter]
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int.make_type_signed.loc6, %.loc6_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %Adapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.1)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.1), @C.%T (%T) [symbolic = %C.elem (constants.%C.elem.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.1)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.1) [symbolic = %complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:     .x = imports.%import_ref.6
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param_patt: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.1 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_21.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_21.2: type = converted %int.make_type_signed.loc7, %.loc7_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.2]
// CHECK:STDOUT:   %.loc7_13.1: %C.2 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_13.2: %C.2 = converted %a.ref, %.loc7_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.2 = name_ref x, imports.%import_ref.6 [template = imports.%.1]
// CHECK:STDOUT:   %.loc7_23.1: ref %i32 = class_element_access %.loc7_13.2, element0
// CHECK:STDOUT:   %.loc7_23.2: %i32 = bind_value %.loc7_23.1
// CHECK:STDOUT:   return %.loc7_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %C.elem => constants.%C.elem.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2
// CHECK:STDOUT:   %complete_type => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.1: type = unbound_element_type %C.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %C.elem.2: type = unbound_element_type %C.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.2 [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<invalid> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [template = constants.%Adapter] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %a.patt: %Adapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [template = constants.%Adapter]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_26.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_26.2: type = converted %int.make_type_signed, %.loc12_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %Adapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.2) [symbolic = %C (constants.%C.1)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.1), @C.%T.loc4_9.2 (%T) [symbolic = %C.elem (constants.%C.elem.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_9.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.1)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.1) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_9.1 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.1) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_21.2: type = converted %int.make_type_signed, %.loc9_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.2]
// CHECK:STDOUT:   adapt_decl %C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.2 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   extend %C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%a.param_patt: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %C.elem.2 = name_ref x, @C.%.loc5 [template = @C.%.loc5]
// CHECK:STDOUT:   %.loc21_11.1: %C.2 = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc21_11.2: %i32 = class_element_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc4_9.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %C.elem => constants.%C.elem.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_adapt_specific_type_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.1: type = unbound_element_type %C.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %C.elem.2: type = unbound_element_type %C.2, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc7_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_9.1, runtime_param<invalid> [symbolic = %T.patt.loc7_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [template = constants.%Adapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc7_9.1: type) {
// CHECK:STDOUT:   %T.loc7_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc7_9.2) [symbolic = %C (constants.%C.1)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.1), @C.%T.loc7_9.2 (%T) [symbolic = %C.elem (constants.%C.elem.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc7_9.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.1)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.1) [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_9.1 [symbolic = %T.loc7_9.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8: @C.%C.elem (%C.elem.1) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc9_1.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     .x = %.loc8
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc9_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %int.make_type_signed, %.loc12_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.2]
// CHECK:STDOUT:   adapt_decl %C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.2 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   extend %C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc7_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc7_9.2) {
// CHECK:STDOUT:   %T.loc7_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc7_9.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc7_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %C.elem => constants.%C.elem.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2
// CHECK:STDOUT:   %complete_type.loc9_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_extend_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.2 [template]
// CHECK:STDOUT:   %C.elem.1: type = unbound_element_type %C.1, %T [symbolic]
// CHECK:STDOUT:   %C.elem.2: type = unbound_element_type %C.2, %i32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [template]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//extend_adapt_specific_type_library, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//extend_adapt_specific_type_library, inst+26, loaded [template = constants.%Adapter]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.10
// CHECK:STDOUT:     .ImplicitAs = %import_ref.11
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: <witness> = import_ref Main//extend_adapt_specific_type_library, inst+20, loaded [symbolic = @C.%complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//extend_adapt_specific_type_library, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.5: @C.%C.elem (%C.elem.1) = import_ref Main//extend_adapt_specific_type_library, inst+18, loaded [template = %.1]
// CHECK:STDOUT:   %import_ref.7: <witness> = import_ref Main//extend_adapt_specific_type_library, inst+50, loaded [template = constants.%complete_type.2]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//extend_adapt_specific_type_library, inst+27, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref Main//extend_adapt_specific_type_library, inst+43, loaded [template = constants.%C.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .Adapter = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImportedAccess = %ImportedAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %ImportedAccess.decl: %ImportedAccess.type = fn_decl @ImportedAccess [template = constants.%ImportedAccess] {
// CHECK:STDOUT:     %a.patt: %Adapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%import_ref.2 [template = constants.%Adapter]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int.make_type_signed, %.loc6_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %Adapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT:   extend imports.%import_ref.9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.1)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.1), @C.%T (%T) [symbolic = %C.elem (constants.%C.elem.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.1)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.1) [symbolic = %complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.4
// CHECK:STDOUT:     .x = imports.%import_ref.5
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param_patt: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %C.elem.2 = name_ref x, imports.%import_ref.5 [template = imports.%.1]
// CHECK:STDOUT:   %.loc14_11.1: %C.2 = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc14_11.2: %i32 = class_element_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %C.elem => constants.%C.elem.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2
// CHECK:STDOUT:   %complete_type => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [template]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [template]
// CHECK:STDOUT:   %Adapter.1: type = class_type @Adapter, @Adapter(%T) [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Adapter.2: type = class_type @Adapter, @Adapter(%i32) [template]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert [template]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:     .Convert = %Convert.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Adapter.decl: %Adapter.type = class_decl @Adapter [template = constants.%Adapter.generic] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_15.1, runtime_param<invalid> [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Convert.decl: %Convert.type = fn_decl @Convert [template = constants.%Convert] {
// CHECK:STDOUT:     %a.patt: %Adapter.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Adapter.ref: %Adapter.type = name_ref Adapter, file.%Adapter.decl [template = constants.%Adapter.generic]
// CHECK:STDOUT:     %int_32.loc8_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc8_23: init type = call constants.%Int(%int_32.loc8_23) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_26.1: type = value_of_initializer %int.make_type_signed.loc8_23 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %int.make_type_signed.loc8_23, %.loc8_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [template = constants.%Adapter.2]
// CHECK:STDOUT:     %int_32.loc8_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc8_32: init type = call constants.%Int(%int_32.loc8_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_32.1: type = value_of_initializer %int.make_type_signed.loc8_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %int.make_type_signed.loc8_32, %.loc8_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %Adapter.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Adapter.%T.loc4_15.2 (%T) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.1 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     adapt_decl %T.ref [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %T [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapter.1
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%a.param_patt: %Adapter.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.2 = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%int_32.loc9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_15.1: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_15.2: type = converted %int.make_type_signed.loc9, %.loc9_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc9_12.2: %i32 = converted %a.ref, %.loc9_12.1
// CHECK:STDOUT:   return %.loc9_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [template]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Adapter.2: type = class_type @Adapter, @Adapter(%i32) [template]
// CHECK:STDOUT:   %ImportedConvert.type: type = fn_type @ImportedConvert [template]
// CHECK:STDOUT:   %ImportedConvert: %ImportedConvert.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %i32 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %Adapter.3: type = class_type @Adapter, @Adapter(%C) [template]
// CHECK:STDOUT:   %ImportedConvertLocal.type: type = fn_type @ImportedConvertLocal [template]
// CHECK:STDOUT:   %ImportedConvertLocal: %ImportedConvertLocal.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Adapter.type = import_ref Main//adapt_generic_type, inst+9, loaded [template = constants.%Adapter.generic]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//adapt_generic_type, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: <witness> = import_ref Main//adapt_generic_type, inst+18, loaded [symbolic = @Adapter.%complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//adapt_generic_type, inst+15, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Adapter = imports.%import_ref.1
// CHECK:STDOUT:     .Convert = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImportedConvert = %ImportedConvert.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ImportedConvertLocal = %ImportedConvertLocal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %ImportedConvert.decl: %ImportedConvert.type = fn_decl @ImportedConvert [template = constants.%ImportedConvert] {
// CHECK:STDOUT:     %a.patt: %Adapter.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%import_ref.1 [template = constants.%Adapter.generic]
// CHECK:STDOUT:     %int_32.loc6_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_31: init type = call constants.%Int(%int_32.loc6_31) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int.make_type_signed.loc6_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int.make_type_signed.loc6_31, %.loc6_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [template = constants.%Adapter.2]
// CHECK:STDOUT:     %int_32.loc6_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_40: init type = call constants.%Int(%int_32.loc6_40) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_40.1: type = value_of_initializer %int.make_type_signed.loc6_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_40.2: type = converted %int.make_type_signed.loc6_40, %.loc6_40.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %Adapter.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %ImportedConvertLocal.decl: %ImportedConvertLocal.type = fn_decl @ImportedConvertLocal [template = constants.%ImportedConvertLocal] {
// CHECK:STDOUT:     %a.patt: %Adapter.3 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter.3 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%import_ref.1 [template = constants.%Adapter.generic]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Adapter: type = class_type @Adapter, @Adapter(constants.%C) [template = constants.%Adapter.3]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_43.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_43.2: type = converted %int.make_type_signed, %.loc14_43.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.3 = value_param runtime_param0
// CHECK:STDOUT:     %a: %Adapter.3 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @Adapter.%T (%T) [symbolic = %complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_10.2: type = converted %int.make_type_signed, %.loc11_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_8: %C.elem = field_decl n, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc11_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvert(%a.param_patt: %Adapter.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.2 = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_15.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_15.2: type = converted %int.make_type_signed.loc7, %.loc7_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_12.2: %i32 = converted %a.ref, %.loc7_12.1
// CHECK:STDOUT:   return %.loc7_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvertLocal(%a.param_patt: %Adapter.3) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.3 = name_ref a, %a
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc15_13.1: %C = as_compatible %a.ref
// CHECK:STDOUT:   %.loc15_13.2: %C = converted %a.ref, %.loc15_13.1
// CHECK:STDOUT:   %n.ref: %C.elem = name_ref n, @C.%.loc11_8 [template = @C.%.loc11_8]
// CHECK:STDOUT:   %.loc15_18.1: ref %i32 = class_element_access %.loc15_13.2, element0
// CHECK:STDOUT:   %.loc15_18.2: %i32 = bind_value %.loc15_18.1
// CHECK:STDOUT:   return %.loc15_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %complete_type => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %complete_type => constants.%complete_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
