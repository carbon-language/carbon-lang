// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_redundant_compound_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_redundant_compound_access.carbon

fn F() -> i32 { return 0; }

fn Main() {
  var a: i32 = 3;
  // CHECK:STDERR: fail_redundant_compound_access.carbon:[[@LINE+4]]:7: error: member name of type `i32` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   a = a.(a);
  // CHECK:STDERR:       ^~~~~
  // CHECK:STDERR:
  a = a.(a);
  // CHECK:STDERR: fail_redundant_compound_access.carbon:[[@LINE+3]]:7: error: member name of type `<type of F>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   a = a.(F)();
  // CHECK:STDERR:       ^~~~~
  a = a.(F)();
}

// CHECK:STDOUT: --- fail_redundant_compound_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 3 [template]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.3: type = converted %int.make_type_signed, %.loc11_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_24: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_25.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_25.2: <bound method> = bound_method %.loc11_24, %.loc11_25.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_25.3: <specific function> = specific_function %.loc11_25.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc11_25.3(%.loc11_24) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_25.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_25.5: %i32 = converted %.loc11_24, %.loc11_25.4 [template = constants.%.29]
// CHECK:STDOUT:   return %.loc11_25.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc14_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_10.3: type = converted %int.make_type_signed, %.loc14_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc14_16: Core.IntLiteral = int_value 3 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_17.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_17.2: <bound method> = bound_method %.loc14_16, %.loc14_17.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_17.3: <specific function> = specific_function %.loc14_17.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc14_17.3(%.loc14_16) [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_17.4: init %i32 = converted %.loc14_16, %int.convert_checked [template = constants.%.33]
// CHECK:STDOUT:   assign %a.var, %.loc14_17.4
// CHECK:STDOUT:   %a.ref.loc19_3: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc19_7: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc19_10: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc19: %i32 = bind_value %a.ref.loc19_10
// CHECK:STDOUT:   assign %a.ref.loc19_3, %.loc19
// CHECK:STDOUT:   %a.ref.loc23_3: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc23_7: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   assign %a.ref.loc23_3, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
