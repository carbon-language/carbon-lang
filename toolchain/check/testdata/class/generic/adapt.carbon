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
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+3]]:10: note: type `Adapter` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %C.37e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.db1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.5ab: type = unbound_element_type %C.37e, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.710: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60a: <witness> = complete_type_witness %struct_type.x.710 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.268: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.elem.dd4: type = unbound_element_type %C.268, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.b74: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.f6d: <witness> = complete_type_witness %struct_type.x.b74 [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
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
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [template = constants.%Adapter]
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%T.loc4_9.2 (%T) [symbolic = %require_complete (constants.%require_complete.db1)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.2) [symbolic = %C (constants.%C.37e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.37e), @C.%T.loc4_9.2 (%T) [symbolic = %C.elem (constants.%C.elem.5ab)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_9.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.710)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.710) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.5ab) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x.710 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.37e
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.268]
// CHECK:STDOUT:   adapt_decl %C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.b74 [template = constants.%complete_type.f6d]
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
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.268]
// CHECK:STDOUT:   %.loc13_13.1: %C.268 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc13_13.2: %C.268 = converted %a.ref, %.loc13_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.dd4 = name_ref x, @C.%.loc5 [template = @C.%.loc5]
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
// CHECK:STDOUT: specific @C(%T.loc4_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.268
// CHECK:STDOUT:   %C.elem => constants.%C.elem.dd4
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.b74
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.f6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.710: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60a: <witness> = complete_type_witness %struct_type.x.710 [symbolic]
// CHECK:STDOUT:   %C.df4: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.3f8: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %struct_type.x.6d6: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.555: <witness> = complete_type_witness %struct_type.x.6d6 [template]
// CHECK:STDOUT:   %require_complete.db1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.0d5: type = unbound_element_type %C.df4, %T [symbolic]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.elem.f99: type = unbound_element_type %C.3f8, %i32 [template]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [template]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.26e: %C.type = import_ref Main//adapt_specific_type, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.abc: type = import_ref Main//adapt_specific_type, Adapter, loaded [template = constants.%Adapter]
// CHECK:STDOUT:   %import_ref.92d = import_ref Main//adapt_specific_type, Access, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3ee: <witness> = import_ref Main//adapt_specific_type, loc6_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.60a)]
// CHECK:STDOUT:   %import_ref.495 = import_ref Main//adapt_specific_type, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ec6: @C.%C.elem (%C.elem.0d5) = import_ref Main//adapt_specific_type, loc5_8, loaded [template = %.8d2]
// CHECK:STDOUT:   %import_ref.377: <witness> = import_ref Main//adapt_specific_type, loc10_1, loaded [template = constants.%complete_type.555]
// CHECK:STDOUT:   %import_ref.feb = import_ref Main//adapt_specific_type, inst42 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.26e
// CHECK:STDOUT:     .Adapter = imports.%import_ref.abc
// CHECK:STDOUT:     .Access = imports.%import_ref.92d
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
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%import_ref.abc [template = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter [from "adapt_specific_type.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.feb
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.377
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) [from "adapt_specific_type.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%T (%T) [symbolic = %require_complete (constants.%require_complete.db1)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.df4)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.df4), @C.%T (%T) [symbolic = %C.elem (constants.%C.elem.0d5)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.710)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.710) [symbolic = %complete_type (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.495
// CHECK:STDOUT:     .x = imports.%import_ref.ec6
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.3ee
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param_patt: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.26e [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.3f8]
// CHECK:STDOUT:   %.loc7_13.1: %C.3f8 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_13.2: %C.3f8 = converted %a.ref, %.loc7_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.f99 = name_ref x, imports.%import_ref.ec6 [template = imports.%.8d2]
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.3f8
// CHECK:STDOUT:   %C.elem => constants.%C.elem.f99
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.6d6
// CHECK:STDOUT:   %complete_type => constants.%complete_type.555
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.37e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.db1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.5ab: type = unbound_element_type %C.37e, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.710: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60a: <witness> = complete_type_witness %struct_type.x.710 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.268: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.elem.dd4: type = unbound_element_type %C.268, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.b74: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.f6d: <witness> = complete_type_witness %struct_type.x.b74 [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [template = constants.%Adapter]
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%T.loc4_9.2 (%T) [symbolic = %require_complete (constants.%require_complete.db1)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.2) [symbolic = %C (constants.%C.37e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.37e), @C.%T.loc4_9.2 (%T) [symbolic = %C.elem (constants.%C.elem.5ab)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_9.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.710)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.710) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.5ab) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x.710 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.37e
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.268]
// CHECK:STDOUT:   adapt_decl %C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.b74 [template = constants.%complete_type.f6d]
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
// CHECK:STDOUT:   %x.ref: %C.elem.dd4 = name_ref x, @C.%.loc5 [template = @C.%.loc5]
// CHECK:STDOUT:   %.loc21_11.1: %C.268 = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc21_11.2: %i32 = class_element_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc4_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.268
// CHECK:STDOUT:   %C.elem => constants.%C.elem.dd4
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.b74
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.f6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_adapt_specific_type_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.37e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.db1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.5ab: type = unbound_element_type %C.37e, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.710: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60a: <witness> = complete_type_witness %struct_type.x.710 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.268: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.elem.dd4: type = unbound_element_type %C.268, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.b74: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.f6d: <witness> = complete_type_witness %struct_type.x.b74 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%T.loc7_9.2 (%T) [symbolic = %require_complete (constants.%require_complete.db1)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc7_9.2) [symbolic = %C (constants.%C.37e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.37e), @C.%T.loc7_9.2 (%T) [symbolic = %C.elem (constants.%C.elem.5ab)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc7_9.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.710)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.710) [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc8: @C.%C.elem (%C.elem.5ab) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc9_1.1: <witness> = complete_type_witness %struct_type.x.710 [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.37e
// CHECK:STDOUT:     .x = %.loc8
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc9_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.268]
// CHECK:STDOUT:   adapt_decl %C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.b74 [template = constants.%complete_type.f6d]
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
// CHECK:STDOUT: specific @C(%T.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc7_9.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc7_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.268
// CHECK:STDOUT:   %C.elem => constants.%C.elem.dd4
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.b74
// CHECK:STDOUT:   %complete_type.loc9_1.2 => constants.%complete_type.f6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_extend_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.710: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60a: <witness> = complete_type_witness %struct_type.x.710 [symbolic]
// CHECK:STDOUT:   %C.df4: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.3f8: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %struct_type.x.6d6: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.555: <witness> = complete_type_witness %struct_type.x.6d6 [template]
// CHECK:STDOUT:   %require_complete.db1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.0d5: type = unbound_element_type %C.df4, %T [symbolic]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.elem.f99: type = unbound_element_type %C.3f8, %i32 [template]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [template]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.2081 = import_ref Main//extend_adapt_specific_type_library, C, unloaded
// CHECK:STDOUT:   %import_ref.abc: type = import_ref Main//extend_adapt_specific_type_library, Adapter, loaded [template = constants.%Adapter]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d69
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3ee: <witness> = import_ref Main//extend_adapt_specific_type_library, loc9_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.60a)]
// CHECK:STDOUT:   %import_ref.495 = import_ref Main//extend_adapt_specific_type_library, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ec6: @C.%C.elem (%C.elem.0d5) = import_ref Main//extend_adapt_specific_type_library, loc8_8, loaded [template = %.8d2]
// CHECK:STDOUT:   %import_ref.377: <witness> = import_ref Main//extend_adapt_specific_type_library, loc13_1, loaded [template = constants.%complete_type.555]
// CHECK:STDOUT:   %import_ref.feb = import_ref Main//extend_adapt_specific_type_library, inst42 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.f9d349.2: type = import_ref Main//extend_adapt_specific_type_library, loc12_21, loaded [template = constants.%C.3f8]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.2081
// CHECK:STDOUT:     .Adapter = imports.%import_ref.abc
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param runtime_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%import_ref.abc [template = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter [from "extend_adapt_specific_type_library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.feb
// CHECK:STDOUT:   extend imports.%import_ref.f9d349.2
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.377
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) [from "extend_adapt_specific_type_library.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%T (%T) [symbolic = %require_complete (constants.%require_complete.db1)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.df4)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.df4), @C.%T (%T) [symbolic = %C.elem (constants.%C.elem.0d5)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.710)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @C.%struct_type.x (%struct_type.x.710) [symbolic = %complete_type (constants.%complete_type.60a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.495
// CHECK:STDOUT:     .x = imports.%import_ref.ec6
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.3ee
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param_patt: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %C.elem.f99 = name_ref x, imports.%import_ref.ec6 [template = imports.%.8d2]
// CHECK:STDOUT:   %.loc14_11.1: %C.3f8 = converted %a.ref, <error> [template = <error>]
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.3f8
// CHECK:STDOUT:   %C.elem => constants.%C.elem.f99
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.6d6
// CHECK:STDOUT:   %complete_type => constants.%complete_type.555
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [template]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [template]
// CHECK:STDOUT:   %Adapter.f46: type = class_type @Adapter, @Adapter(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.db1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %complete_type.27d: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Adapter.1d5: type = class_type @Adapter, @Adapter(%i32) [template]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert [template]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %complete_type.d0d: <witness> = complete_type_witness %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
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
// CHECK:STDOUT:     %a.patt: %Adapter.1d5 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter.1d5 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.1d5 = value_param runtime_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %Adapter [template = constants.%Adapter.1d5] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, file.%Adapter.decl [template = constants.%Adapter.generic]
// CHECK:STDOUT:       %int_32.loc8_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_23: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [template = constants.%Adapter.1d5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.1d5 = bind_name a, %a.param
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Adapter.%T.loc4_15.2 (%T) [symbolic = %require_complete (constants.%require_complete.db1)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Adapter.%T.loc4_15.2 (%T) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.27d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.1 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     adapt_decl %T.ref [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %T [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.27d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapter.f46
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%a.param_patt: %Adapter.1d5) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.1d5 = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.d0d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [template]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %complete_type.27d: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %require_complete.db1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Adapter.070: type = class_type @Adapter, @Adapter(%i32) [template]
// CHECK:STDOUT:   %ImportedConvert.type: type = fn_type @ImportedConvert [template]
// CHECK:STDOUT:   %ImportedConvert: %ImportedConvert.type = struct_value () [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %complete_type.d0d: <witness> = complete_type_witness %i32 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.9b7: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %Adapter.e9c: type = class_type @Adapter, @Adapter(%C) [template]
// CHECK:STDOUT:   %ImportedConvertLocal.type: type = fn_type @ImportedConvertLocal [template]
// CHECK:STDOUT:   %ImportedConvertLocal: %ImportedConvertLocal.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type.af2: <witness> = complete_type_witness %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.ff3: %Adapter.type = import_ref Main//adapt_generic_type, Adapter, loaded [template = constants.%Adapter.generic]
// CHECK:STDOUT:   %import_ref.00e = import_ref Main//adapt_generic_type, Convert, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.115: <witness> = import_ref Main//adapt_generic_type, loc6_1, loaded [symbolic = @Adapter.%complete_type (constants.%complete_type.27d)]
// CHECK:STDOUT:   %import_ref.a89 = import_ref Main//adapt_generic_type, inst27 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Adapter = imports.%import_ref.ff3
// CHECK:STDOUT:     .Convert = imports.%import_ref.00e
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImportedConvert = %ImportedConvert.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ImportedConvertLocal = %ImportedConvertLocal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %ImportedConvert.decl: %ImportedConvert.type = fn_decl @ImportedConvert [template = constants.%ImportedConvert] {
// CHECK:STDOUT:     %a.patt: %Adapter.070 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter.070 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_40: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.070 = value_param runtime_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Adapter [template = constants.%Adapter.070] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%import_ref.ff3 [template = constants.%Adapter.generic]
// CHECK:STDOUT:       %int_32.loc6_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_31: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [template = constants.%Adapter.070]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.070 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %ImportedConvertLocal.decl: %ImportedConvertLocal.type = fn_decl @ImportedConvertLocal [template = constants.%ImportedConvertLocal] {
// CHECK:STDOUT:     %a.patt: %Adapter.e9c = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Adapter.e9c = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.e9c = value_param runtime_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %Adapter [template = constants.%Adapter.e9c] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%import_ref.ff3 [template = constants.%Adapter.generic]
// CHECK:STDOUT:       %C.ref.loc14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%C) [template = constants.%Adapter.e9c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.e9c = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(constants.%T: type) [from "adapt_generic_type.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Adapter.%T (%T) [symbolic = %require_complete (constants.%require_complete.db1)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @Adapter.%T (%T) [symbolic = %complete_type (constants.%complete_type.27d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.a89
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.115
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc11: %C.elem = field_decl n, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template = constants.%complete_type.9b7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc11
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvert(%a.param_patt: %Adapter.070) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.070 = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_12.2: %i32 = converted %a.ref, %.loc7_12.1
// CHECK:STDOUT:   return %.loc7_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvertLocal(%a.param_patt: %Adapter.e9c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.e9c = name_ref a, %a
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc15_13.1: %C = as_compatible %a.ref
// CHECK:STDOUT:   %.loc15_13.2: %C = converted %a.ref, %.loc15_13.1
// CHECK:STDOUT:   %n.ref: %C.elem = name_ref n, @C.%.loc11 [template = @C.%.loc11]
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type => constants.%complete_type.d0d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9b7
// CHECK:STDOUT:   %complete_type => constants.%complete_type.af2
// CHECK:STDOUT: }
// CHECK:STDOUT:
