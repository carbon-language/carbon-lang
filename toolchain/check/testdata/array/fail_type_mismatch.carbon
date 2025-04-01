// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:24: error: cannot implicitly convert expression of type `String` to `i32` [ConversionFailure]
// CHECK:STDERR: var a: array(i32, 3) = (1, "Hello", "World");
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:24: note: type `String` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var a: array(i32, 3) = (1, "Hello", "World");
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: array(i32, 3) = (1, "Hello", "World");

var t1: (i32, String, String);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:24: error: cannot implicitly convert expression of type `String` to `i32` [ConversionFailure]
// CHECK:STDERR: var b: array(i32, 3) = t1;
// CHECK:STDERR:                        ^~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:24: note: type `String` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: array(i32, 3) = t1;
// CHECK:STDERR:                        ^~
// CHECK:STDERR:
var b: array(i32, 3) = t1;

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:24: error: cannot initialize array of 3 elements from 2 initializers [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR: var c: array(i32, 3) = (1, 2);
// CHECK:STDERR:                        ^~~~~~
// CHECK:STDERR:
var c: array(i32, 3) = (1, 2);

var t2: (i32, i32);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:24: error: cannot initialize array of 3 elements from tuple with 2 elements [ArrayInitFromExprArgCountMismatch]
// CHECK:STDERR: var d: array(i32, 3) = t2;
// CHECK:STDERR:                        ^~
// CHECK:STDERR:
var d: array(i32, 3) = t2;

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %str.ef1: String = string_literal "Hello" [concrete]
// CHECK:STDOUT:   %str.abb: String = string_literal "World" [concrete]
// CHECK:STDOUT:   %tuple.type.b0f: type = tuple_type (Core.IntLiteral, String, String) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.9e7: type = tuple_type (%i32, String, String) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .t1 = %t1
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .t2 = %t2
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc18_1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %.loc18_20: type = splice_block %array_type.loc18 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc18: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type.loc18: type = array_type %int_3.loc18, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t1.patt: %tuple.type.9e7 = binding_pattern t1
// CHECK:STDOUT:     %.loc20_1: %tuple.type.9e7 = var_pattern %t1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1.var: ref %tuple.type.9e7 = var t1
// CHECK:STDOUT:   %.loc20_29.1: type = splice_block %.loc20_29.3 [concrete = constants.%tuple.type.9e7] {
// CHECK:STDOUT:     %int_32.loc20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc20_29.2: %tuple.type.ff9 = tuple_literal (%i32.loc20, String, String)
// CHECK:STDOUT:     %.loc20_29.3: type = converted %.loc20_29.2, constants.%tuple.type.9e7 [concrete = constants.%tuple.type.9e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1: ref %tuple.type.9e7 = bind_name t1, %t1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %array_type = binding_pattern b
// CHECK:STDOUT:     %.loc28_1: %array_type = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %array_type = var b
// CHECK:STDOUT:   %.loc28_20: type = splice_block %array_type.loc28 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc28: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type.loc28: type = array_type %int_3.loc28, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %array_type = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %array_type = binding_pattern c
// CHECK:STDOUT:     %.loc34_1: %array_type = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %array_type = var c
// CHECK:STDOUT:   %.loc34_20: type = splice_block %array_type.loc34 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc34: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type.loc34: type = array_type %int_3.loc34, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %array_type = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t2.patt: %tuple.type.d07 = binding_pattern t2
// CHECK:STDOUT:     %.loc36_1: %tuple.type.d07 = var_pattern %t2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t2.var: ref %tuple.type.d07 = var t2
// CHECK:STDOUT:   %.loc36_18.1: type = splice_block %.loc36_18.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc36_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc36_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc36_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc36_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc36_18.2: %tuple.type.24b = tuple_literal (%i32.loc36_10, %i32.loc36_15)
// CHECK:STDOUT:     %.loc36_18.3: type = converted %.loc36_18.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t2: ref %tuple.type.d07 = bind_name t2, %t2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %array_type = binding_pattern d
// CHECK:STDOUT:     %.loc41_1: %array_type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %array_type = var d
// CHECK:STDOUT:   %.loc41_20: type = splice_block %array_type.loc41 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc41: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type.loc41: type = array_type %int_3.loc41, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %array_type = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %str.loc18_28: String = string_literal "Hello" [concrete = constants.%str.ef1]
// CHECK:STDOUT:   %str.loc18_37: String = string_literal "World" [concrete = constants.%str.abb]
// CHECK:STDOUT:   %.loc18_44.1: %tuple.type.b0f = tuple_literal (%int_1.loc18, %str.loc18_28, %str.loc18_37)
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_44.1: <bound method> = bound_method %int_1.loc18, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_44.2: <bound method> = bound_method %int_1.loc18, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc18_44.2(%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_44.2: init %i32 = converted %int_1.loc18, %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc18_44.3: ref %i32 = array_index file.%a.var, %int_0.loc18
// CHECK:STDOUT:   %.loc18_44.4: init %i32 = initialize_from %.loc18_44.2 to %.loc18_44.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_44.5: %i32 = converted %str.loc18_28, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %t1.ref: ref %tuple.type.9e7 = name_ref t1, file.%t1
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %t1.ref, element0
// CHECK:STDOUT:   %.loc28_24.1: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %int_0.loc28: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc28_24.2: ref %i32 = array_index file.%b.var, %int_0.loc28
// CHECK:STDOUT:   %.loc28_24.3: init %i32 = initialize_from %.loc28_24.1 to %.loc28_24.2
// CHECK:STDOUT:   %tuple.elem1: ref String = tuple_access %t1.ref, element1
// CHECK:STDOUT:   %.loc28_24.4: %i32 = converted %tuple.elem1, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %int_1.loc34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc34: %tuple.type.f94 = tuple_literal (%int_1.loc34, %int_2)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %t2.ref: ref %tuple.type.d07 = name_ref t2, file.%t2
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
