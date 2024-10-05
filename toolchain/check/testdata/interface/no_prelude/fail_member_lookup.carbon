// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_member_lookup.carbon

interface Interface {
  fn F();

  let T:! type;
}

fn F() {
  // CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+4]]:3: error: value of type `<associated <type of F> in Interface>` is not callable
  // CHECK:STDERR:   Interface.F();
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  Interface.F();

  // CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+3]]:10: error: package `Core` implicitly referenced here, but not found
  // CHECK:STDERR:   var v: Interface.T;
  // CHECK:STDERR:          ^~~~~~~~~~~
  var v: Interface.T;
}

// CHECK:STDOUT: --- fail_member_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %Interface.type, %F.type.1 [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Interface.type, type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element1, @Interface.%T [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc12: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc14: %.4 = assoc_entity element1, %T [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   .T = %.loc14
// CHECK:STDOUT:   witness = (%F.decl, %T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Interface.ref.loc22: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   %F.ref: %.2 = name_ref F, @Interface.%.loc12 [template = constants.%.3]
// CHECK:STDOUT:   %Interface.ref.loc27: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   %T.ref: %.4 = name_ref T, @Interface.%.loc14 [template = constants.%.5]
// CHECK:STDOUT:   %.loc27: type = converted %T.ref, <error> [template = <error>]
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
