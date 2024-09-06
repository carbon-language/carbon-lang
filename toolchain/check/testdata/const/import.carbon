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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %.loc4_11.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     @F.%return: ref %.2 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_12.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_12.2: type = converted %int.make_type_32.loc6, %.loc6_12.1 [template = i32]
// CHECK:STDOUT:   %.loc6_12.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref %.2 = var a_ref
// CHECK:STDOUT:   %a_ref: ref %.2 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_16.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_16.2: type = converted %int.make_type_32.loc7, %.loc7_16.1 [template = i32]
// CHECK:STDOUT:   %.loc7_16.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_25: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:   %a_ptr_ref.var: ref %.3 = var a_ptr_ref
// CHECK:STDOUT:   %a_ptr_ref: ref %.3 = bind_name a_ptr_ref, %a_ptr_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %.2 = call %F.ref()
// CHECK:STDOUT:   assign file.%a_ref.var, %F.call
// CHECK:STDOUT:   %a_ref.ref: ref %.2 = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %.loc7: %.3 = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr_ref.var, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Implicit//default, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.2: ref %.2 = import_ref Implicit//default, inst+23, loaded
// CHECK:STDOUT:   %import_ref.3: ref %.3 = import_ref Implicit//default, inst+34, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.4
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_32.loc6, %.loc6_8.1 [template = i32]
// CHECK:STDOUT:   %.loc6_8.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_17: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_12.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_12.2: type = converted %int.make_type_32.loc7, %.loc7_12.1 [template = i32]
// CHECK:STDOUT:   %.loc7_12.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_21: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:   %a_ptr.var: ref %.3 = var a_ptr
// CHECK:STDOUT:   %a_ptr: ref %.3 = bind_name a_ptr, %a_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %.2 = name_ref a_ref, imports.%import_ref.2
// CHECK:STDOUT:   %.loc6: %.3 = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a.var, %.loc6
// CHECK:STDOUT:   %a_ptr_ref.ref: ref %.3 = name_ref a_ptr_ref, imports.%import_ref.3
// CHECK:STDOUT:   %.loc7: %.3 = bind_value %a_ptr_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr.var, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
