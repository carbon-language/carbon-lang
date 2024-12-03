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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.25: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.26: <specific function> = specific_function %.25, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:   %int.32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int.32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed.loc12, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc12_17.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_17.2: <bound method> = bound_method %int.1, %.loc12_17.1 [template = constants.%.25]
// CHECK:STDOUT:   %.loc12_17.3: <specific function> = specific_function %.loc12_17.2, @Convert.2(constants.%int.32) [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_17.3(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc12_17.4: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc12_17.5: %i32 = converted %int.1, %.loc12_17.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %a.loc12: %i32 = bind_name a, %.loc12_17.5
// CHECK:STDOUT:   %int.32.loc19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%int.32.loc19) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_10.1: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_10.2: type = converted %int.make_type_signed.loc19, %.loc19_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc19_17.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_17.2: <bound method> = bound_method %int.2, %.loc19_17.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc19_17.3: <specific function> = specific_function %.loc19_17.2, @Convert.2(constants.%int.32) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %.loc19_17.3(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc19_17.4: %i32 = value_of_initializer %int.convert_checked.loc19 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc19_17.5: %i32 = converted %int.2, %.loc19_17.4 [template = constants.%int.2.2]
// CHECK:STDOUT:   %a.loc19: %i32 = bind_name a, %.loc19_17.5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
