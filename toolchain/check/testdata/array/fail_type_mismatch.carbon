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
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: note: type `String` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: [i32; 3] = (1, "Hello", "World");

var t1: (i32, String, String);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:19: error: cannot implicitly convert from `String` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: note: type `String` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.7: String = string_literal "Hello" [template]
// CHECK:STDOUT:   %.8: String = string_literal "World" [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, String, String) [template]
// CHECK:STDOUT:   %.9: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%i32, String, String) [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%i32, %i32) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .t1 = %t1
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .t2 = %t2
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc18_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%.loc18_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_14: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_9.2: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_9.3: type = converted %int.make_type_signed.loc18, %.loc18_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_15: type = array_type %.loc18_14, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc20_10: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%.loc20_10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_29.1: %tuple.type.2 = tuple_literal (%int.make_type_signed.loc20, String, String)
// CHECK:STDOUT:   %.loc20_29.2: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_29.3: type = converted %int.make_type_signed.loc20, %.loc20_29.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_29.4: type = converted %.loc20_29.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:   %t1.var: ref %tuple.type.3 = var t1
// CHECK:STDOUT:   %t1: ref %tuple.type.3 = bind_name t1, %t1.var
// CHECK:STDOUT:   %.loc28_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc28: init type = call constants.%Int(%.loc28_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc28_14: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc28_9.2: type = value_of_initializer %int.make_type_signed.loc28 [template = constants.%i32]
// CHECK:STDOUT:   %.loc28_9.3: type = converted %int.make_type_signed.loc28, %.loc28_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc28_15: type = array_type %.loc28_14, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %b.var: ref %.3 = var b
// CHECK:STDOUT:   %b: ref %.3 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc34_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc34: init type = call constants.%Int(%.loc34_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_14: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_9.2: type = value_of_initializer %int.make_type_signed.loc34 [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_9.3: type = converted %int.make_type_signed.loc34, %.loc34_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_15: type = array_type %.loc34_14, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref %.3 = var c
// CHECK:STDOUT:   %c: ref %.3 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc36_10: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc36_10: init type = call constants.%Int(%.loc36_10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_15: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc36_15: init type = call constants.%Int(%.loc36_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_18.1: %tuple.type.6 = tuple_literal (%int.make_type_signed.loc36_10, %int.make_type_signed.loc36_15)
// CHECK:STDOUT:   %.loc36_18.2: type = value_of_initializer %int.make_type_signed.loc36_10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_18.3: type = converted %int.make_type_signed.loc36_10, %.loc36_18.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_18.4: type = value_of_initializer %int.make_type_signed.loc36_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_18.5: type = converted %int.make_type_signed.loc36_15, %.loc36_18.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_18.6: type = converted %.loc36_18.1, constants.%tuple.type.7 [template = constants.%tuple.type.7]
// CHECK:STDOUT:   %t2.var: ref %tuple.type.7 = var t2
// CHECK:STDOUT:   %t2: ref %tuple.type.7 = bind_name t2, %t2.var
// CHECK:STDOUT:   %.loc40_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc40: init type = call constants.%Int(%.loc40_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_14: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc40_9.2: type = value_of_initializer %int.make_type_signed.loc40 [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_9.3: type = converted %int.make_type_signed.loc40, %.loc40_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_15: type = array_type %.loc40_14, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %d.var: ref %.3 = var d
// CHECK:STDOUT:   %d: ref %.3 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_20: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_23: String = string_literal "Hello" [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_32: String = string_literal "World" [template = constants.%.8]
// CHECK:STDOUT:   %.loc18_39.1: %tuple.type.1 = tuple_literal (%.loc18_20, %.loc18_23, %.loc18_32)
// CHECK:STDOUT:   %.loc18_39.2: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_39.3: <bound method> = bound_method %.loc18_20, %.loc18_39.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc18_39.4: <specific function> = specific_function %.loc18_39.3, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc18_39.4(%.loc18_20) [template = constants.%.36]
// CHECK:STDOUT:   %.loc18_39.5: init %i32 = converted %.loc18_20, %int.convert_checked [template = constants.%.36]
// CHECK:STDOUT:   %.loc18_39.6: Core.IntLiteral = int_value 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_39.7: ref %i32 = array_index file.%a.var, %.loc18_39.6
// CHECK:STDOUT:   %.loc18_39.8: init %i32 = initialize_from %.loc18_39.5 to %.loc18_39.7 [template = constants.%.36]
// CHECK:STDOUT:   %.loc18_39.9: %i32 = converted %.loc18_23, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %t1.ref: ref %tuple.type.3 = name_ref t1, file.%t1
// CHECK:STDOUT:   %.loc28_19.1: ref %i32 = tuple_access %t1.ref, element0
// CHECK:STDOUT:   %.loc28_19.2: %i32 = bind_value %.loc28_19.1
// CHECK:STDOUT:   %.loc28_19.3: Core.IntLiteral = int_value 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc28_19.4: ref %i32 = array_index file.%b.var, %.loc28_19.3
// CHECK:STDOUT:   %.loc28_19.5: init %i32 = initialize_from %.loc28_19.2 to %.loc28_19.4
// CHECK:STDOUT:   %.loc28_19.6: ref String = tuple_access %t1.ref, element1
// CHECK:STDOUT:   %.loc28_19.7: %i32 = converted %.loc28_19.6, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc34_20: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc34_23: Core.IntLiteral = int_value 2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc34_24: %tuple.type.5 = tuple_literal (%.loc34_20, %.loc34_23)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %t2.ref: ref %tuple.type.7 = name_ref t2, file.%t2
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
