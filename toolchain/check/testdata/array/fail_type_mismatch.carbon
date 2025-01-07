// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:19: error: cannot implicitly convert from `String` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: note: type `String` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: [i32; 3] = (1, "Hello", "World");

var t1: (i32, String, String);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:19: error: cannot implicitly convert from `String` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: note: type `String` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
var b: [i32; 3] = t1;

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: error: cannot initialize array of 3 elements from 2 initializers [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR: var c: [i32; 3] = (1, 2);
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
var c: [i32; 3] = (1, 2);

var t2: (i32, i32);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:19: error: cannot initialize array of 3 elements from tuple with 2 elements [ArrayInitFromExprArgCountMismatch]
// CHECK:STDERR: var d: [i32; 3] = t2;
// CHECK:STDERR:                   ^~
var d: [i32; 3] = t2;

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %str.ef1: String = string_literal "Hello" [template]
// CHECK:STDOUT:   %str.abb: String = string_literal "World" [template]
// CHECK:STDOUT:   %tuple.type.b0f: type = tuple_type (Core.IntLiteral, String, String) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.6e2: type = tuple_type (%i32, String, String) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.type.471: type = tuple_type (%i32, %i32) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .t1 = %t1
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .t2 = %t2
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %t1.var: ref %tuple.type.6e2 = var t1
// CHECK:STDOUT:   %t1: ref %tuple.type.6e2 = bind_name t1, %t1.var
// CHECK:STDOUT:   %b.var: ref %array_type = var b
// CHECK:STDOUT:   %b: ref %array_type = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %array_type = var c
// CHECK:STDOUT:   %c: ref %array_type = bind_name c, %c.var
// CHECK:STDOUT:   %t2.var: ref %tuple.type.471 = var t2
// CHECK:STDOUT:   %t2: ref %tuple.type.471 = bind_name t2, %t2.var
// CHECK:STDOUT:   %d.var: ref %array_type = var d
// CHECK:STDOUT:   %d: ref %array_type = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %str.loc18_23: String = string_literal "Hello" [template = constants.%str.ef1]
// CHECK:STDOUT:   %str.loc18_32: String = string_literal "World" [template = constants.%str.abb]
// CHECK:STDOUT:   %.loc18_39.1: %tuple.type.b0f = tuple_literal (%int_1.loc18, %str.loc18_23, %str.loc18_32)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.loc18, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1.loc18) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc18_39.2: init %i32 = converted %int_1.loc18, %int.convert_checked [template = constants.%int_1.c60]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc18_39.3: ref %i32 = array_index file.%a.var, %int_0.loc18
// CHECK:STDOUT:   %.loc18_39.4: init %i32 = initialize_from %.loc18_39.2 to %.loc18_39.3 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc18_39.5: %i32 = converted %str.loc18_23, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %t1.ref: ref %tuple.type.6e2 = name_ref t1, file.%t1
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %t1.ref, element0
// CHECK:STDOUT:   %.loc28_19.1: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %int_0.loc28: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc28_19.2: ref %i32 = array_index file.%b.var, %int_0.loc28
// CHECK:STDOUT:   %.loc28_19.3: init %i32 = initialize_from %.loc28_19.1 to %.loc28_19.2
// CHECK:STDOUT:   %tuple.elem1: ref String = tuple_access %t1.ref, element1
// CHECK:STDOUT:   %.loc28_19.4: %i32 = converted %tuple.elem1, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %int_1.loc34: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc34: %tuple.type.f94 = tuple_literal (%int_1.loc34, %int_2)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %t2.ref: ref %tuple.type.471 = name_ref t2, file.%t2
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
