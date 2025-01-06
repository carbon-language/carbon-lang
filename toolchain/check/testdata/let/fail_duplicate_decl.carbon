// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_duplicate_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_duplicate_decl.carbon

fn F() {
  let a: i32 = 1;
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: error: duplicate name being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR:   let a: i32 = 2;
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:   let a: i32 = 1;
  // CHECK:STDERR:       ^
  let a: i32 = 2;
}

// CHECK:STDOUT: --- fail_duplicate_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.c3: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.bd: <witness> = impl_witness (imports.%import_ref.14), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.2c: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.54: %Convert.type.2c = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.e: <bound method> = bound_method %int_1.5, %Convert.54 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.e, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.e: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.a: <bound method> = bound_method %int_2.e, %Convert.54 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.a, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.3: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.c3 = impl_witness_access constants.%impl_witness.bd, element0 [template = constants.%Convert.54]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_1, %impl.elem0.loc12 [template = constants.%Convert.bound.e]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.specific_fn.loc12(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_17.1: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_17.2: %i32 = converted %int_1, %.loc12_17.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %a.loc12: %i32 = bind_name a, %.loc12_17.2
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.e]
// CHECK:STDOUT:   %impl.elem0.loc19: %Convert.type.c3 = impl_witness_access constants.%impl_witness.bd, element0 [template = constants.%Convert.54]
// CHECK:STDOUT:   %Convert.bound.loc19: <bound method> = bound_method %int_2, %impl.elem0.loc19 [template = constants.%Convert.bound.a]
// CHECK:STDOUT:   %Convert.specific_fn.loc19: <specific function> = specific_function %Convert.bound.loc19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %Convert.specific_fn.loc19(%int_2) [template = constants.%int_2.3]
// CHECK:STDOUT:   %.loc19_17.1: %i32 = value_of_initializer %int.convert_checked.loc19 [template = constants.%int_2.3]
// CHECK:STDOUT:   %.loc19_17.2: %i32 = converted %int_2, %.loc19_17.1 [template = constants.%int_2.3]
// CHECK:STDOUT:   %a.loc19: %i32 = bind_name a, %.loc19_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
