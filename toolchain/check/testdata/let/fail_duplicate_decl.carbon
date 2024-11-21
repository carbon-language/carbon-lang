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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed.loc12, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_16: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_17.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_17.2: <bound method> = bound_method %.loc12_16, %.loc12_17.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_17.3: <specific function> = specific_function %.loc12_17.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_17.3(%.loc12_16) [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_17.4: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_17.5: %i32 = converted %.loc12_16, %.loc12_17.4 [template = constants.%.29]
// CHECK:STDOUT:   %a.loc12: %i32 = bind_name a, %.loc12_17.5
// CHECK:STDOUT:   %.loc19_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%.loc19_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_10.2: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_10.3: type = converted %int.make_type_signed.loc19, %.loc19_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_16: Core.IntLiteral = int_value 2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_17.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_17.2: <bound method> = bound_method %.loc19_16, %.loc19_17.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_17.3: <specific function> = specific_function %.loc19_17.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %.loc19_17.3(%.loc19_16) [template = constants.%.33]
// CHECK:STDOUT:   %.loc19_17.4: %i32 = value_of_initializer %int.convert_checked.loc19 [template = constants.%.33]
// CHECK:STDOUT:   %.loc19_17.5: %i32 = converted %.loc19_16, %.loc19_17.4 [template = constants.%.33]
// CHECK:STDOUT:   %a.loc19: %i32 = bind_name a, %.loc19_17.5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
