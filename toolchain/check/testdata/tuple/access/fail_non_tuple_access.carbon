// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon

fn Main() {
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:3: error: type `Core.IntLiteral` does not support indexing [TypeNotIndexable]
  // CHECK:STDERR:   0[1];
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  0[1];

  var non_tuple: [i32; 2] = (5, 5);
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+3]]:20: error: type `[i32; 2]` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   var first: i32 = non_tuple.0;
  // CHECK:STDERR:                    ^~~~~~~~~~~
  var first: i32 = non_tuple.0;
}

// CHECK:STDOUT: --- fail_non_tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %Int = call %Convert.specific_fn(%int_5) [symbolic]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int.convert_checked, %int.convert_checked) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IndexWith = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.6
// CHECK:STDOUT:     .ImplicitAs = %import_ref.10
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc16: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %non_tuple.var: ref %array_type = var non_tuple
// CHECK:STDOUT:   %non_tuple: ref %array_type = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   %int_5.loc18_30: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_5.loc18_33: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %.loc18_34.1: %tuple.type = tuple_literal (%int_5.loc18_30, %int_5.loc18_33)
// CHECK:STDOUT:   %impl.elem0.loc18_34.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_34.1: <bound method> = bound_method %int_5.loc18_30, %impl.elem0.loc18_34.1 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_34.1: <specific function> = specific_function %Convert.bound.loc18_34.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc18_34.1: init %Int = call %Convert.specific_fn.loc18_34.1(%int_5.loc18_30) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc18_34.2: init %i32 = converted %int_5.loc18_30, %int.convert_checked.loc18_34.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc18_34.3: ref %i32 = array_index %non_tuple.var, %int_0.loc18
// CHECK:STDOUT:   %.loc18_34.4: init %i32 = initialize_from %.loc18_34.2 to %.loc18_34.3 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %impl.elem0.loc18_34.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_34.2: <bound method> = bound_method %int_5.loc18_33, %impl.elem0.loc18_34.2 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_34.2: <specific function> = specific_function %Convert.bound.loc18_34.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc18_34.2: init %Int = call %Convert.specific_fn.loc18_34.2(%int_5.loc18_33) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc18_34.5: init %i32 = converted %int_5.loc18_33, %int.convert_checked.loc18_34.2 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc18_34.6: ref %i32 = array_index %non_tuple.var, %int_1.loc18
// CHECK:STDOUT:   %.loc18_34.7: init %i32 = initialize_from %.loc18_34.5 to %.loc18_34.6 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc18_34.8: init %array_type = array_init (%.loc18_34.4, %.loc18_34.7) to %non_tuple.var [symbolic = constants.%array]
// CHECK:STDOUT:   %.loc18_35: init %array_type = converted %.loc18_34.1, %.loc18_34.8 [symbolic = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc18_35
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %first.var: ref %i32 = var first
// CHECK:STDOUT:   %first: ref %i32 = bind_name first, %first.var
// CHECK:STDOUT:   %non_tuple.ref: ref %array_type = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %int_0.loc22: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
