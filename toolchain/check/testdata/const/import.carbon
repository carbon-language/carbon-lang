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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %const: type = const_type %i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .a_ptr_ref = %a_ptr_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %const = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_signed, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %const: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:     %return.param: ref %const = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %const = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_12.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_12.2: type = converted %int.make_type_signed.loc6, %.loc6_12.1 [template = constants.%i32]
// CHECK:STDOUT:   %const.loc6: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:   %a_ref.var: ref %const = var a_ref
// CHECK:STDOUT:   %a_ref: ref %const = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_16.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_16.2: type = converted %int.make_type_signed.loc7, %.loc7_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %const.loc7: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [template = constants.%ptr]
// CHECK:STDOUT:   %a_ptr_ref.var: ref %ptr = var a_ptr_ref
// CHECK:STDOUT:   %a_ptr_ref: ref %ptr = bind_name a_ptr_ref, %a_ptr_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %const;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %const: type = const_type %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+24, unloaded
// CHECK:STDOUT:   %import_ref.2: ref %const = import_ref Implicit//default, inst+33, loaded
// CHECK:STDOUT:   %import_ref.3: ref %ptr = import_ref Implicit//default, inst+45, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.4
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = imports.%import_ref.1
// CHECK:STDOUT:     .a_ref = imports.%import_ref.2
// CHECK:STDOUT:     .a_ptr_ref = imports.%import_ref.3
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .a_ptr = %a_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_signed.loc6, %.loc6_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %const.loc6: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:   %ptr.loc6: type = ptr_type %const [template = constants.%ptr]
// CHECK:STDOUT:   %a.var: ref %ptr = var a
// CHECK:STDOUT:   %a: ref %ptr = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_12.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_12.2: type = converted %int.make_type_signed.loc7, %.loc7_12.1 [template = constants.%i32]
// CHECK:STDOUT:   %const.loc7: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:   %ptr.loc7: type = ptr_type %const [template = constants.%ptr]
// CHECK:STDOUT:   %a_ptr.var: ref %ptr = var a_ptr
// CHECK:STDOUT:   %a_ptr: ref %ptr = bind_name a_ptr, %a_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %const = name_ref a_ref, imports.%import_ref.2
// CHECK:STDOUT:   %addr: %ptr = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a.var, %addr
// CHECK:STDOUT:   %a_ptr_ref.ref: ref %ptr = name_ref a_ptr_ref, imports.%import_ref.3
// CHECK:STDOUT:   %.loc7: %ptr = bind_value %a_ptr_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr.var, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
