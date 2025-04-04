// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/import.carbon

// --- implicit.carbon

package Implicit;

fn F() -> const i32;

var a_ref: const i32 = F();
var a_ptr_ref: const i32* = &a_ref;

// --- implicit.impl.carbon

impl package Implicit;

// Take a reference to avoid unsupported copy logic. This still validates the
// `const` is handled.
var a: const i32* = &a_ref;
var a_ptr: const i32* = a_ptr_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .a_ptr_ref = %a_ptr_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %const = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const: type = const_type %i32 [concrete = constants.%const]
// CHECK:STDOUT:     %return.param: ref %const = out_param call_param0
// CHECK:STDOUT:     %return: ref %const = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ref.patt: %const = binding_pattern a_ref
// CHECK:STDOUT:     %.loc6_1: %const = var_pattern %a_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref %const = var a_ref
// CHECK:STDOUT:   %.loc6_12: type = splice_block %const.loc6 [concrete = constants.%const] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc6: type = const_type %i32.loc6 [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref: ref %const = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ptr_ref.patt: %ptr = binding_pattern a_ptr_ref
// CHECK:STDOUT:     %.loc7_1: %ptr = var_pattern %a_ptr_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr_ref.var: ref %ptr = var a_ptr_ref
// CHECK:STDOUT:   %.loc7_25: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc7: type = const_type %i32.loc7 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr: type = ptr_type %const.loc7 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr_ref: ref %ptr = bind_name a_ptr_ref, %a_ptr_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %const;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %const = call %F.ref()
// CHECK:STDOUT:   assign file.%a_ref.var, %F.call
// CHECK:STDOUT:   %a_ref.ref: ref %const = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %addr: %ptr = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr_ref.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.F = import_ref Implicit//default, F, unloaded
// CHECK:STDOUT:   %Implicit.a_ref: ref %const = import_ref Implicit//default, a_ref, loaded
// CHECK:STDOUT:   %Implicit.a_ptr_ref: ref %ptr = import_ref Implicit//default, a_ptr_ref, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = imports.%Implicit.F
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .a_ptr_ref = imports.%Implicit.a_ptr_ref
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .a_ptr = %a_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %ptr = binding_pattern a
// CHECK:STDOUT:     %.loc6_1: %ptr = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr = var a
// CHECK:STDOUT:   %.loc6_17: type = splice_block %ptr.loc6 [concrete = constants.%ptr] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc6: type = const_type %i32.loc6 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc6: type = ptr_type %const.loc6 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ptr.patt: %ptr = binding_pattern a_ptr
// CHECK:STDOUT:     %.loc7_1: %ptr = var_pattern %a_ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr.var: ref %ptr = var a_ptr
// CHECK:STDOUT:   %.loc7_21: type = splice_block %ptr.loc7 [concrete = constants.%ptr] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc7: type = const_type %i32.loc7 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %const.loc7 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr: ref %ptr = bind_name a_ptr, %a_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %const = name_ref a_ref, imports.%Implicit.a_ref
// CHECK:STDOUT:   %addr: %ptr = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a.var, %addr
// CHECK:STDOUT:   %a_ptr_ref.ref: ref %ptr = name_ref a_ptr_ref, imports.%Implicit.a_ptr_ref
// CHECK:STDOUT:   %.loc7: %ptr = bind_value %a_ptr_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr.var, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
