// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_out_of_line.carbon

interface Interface {
  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F();

  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn G(a: i32, b: i32) -> i32;
}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+7]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-12]]:3: Name is previously declared here.
// CHECK:STDERR:   default fn F();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.F() {}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-15]]:3: Name is previously declared here.
// CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- fail_todo_define_default_fn_out_of_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Interface, %F.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Interface, %G.type [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element1, @Interface.%G.decl [template]
// CHECK:STDOUT:   %.type.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.7: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = fn_type @.2 [template]
// CHECK:STDOUT:   %.8: %.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.3: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.4: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.5: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.6: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT:   %.decl.loc32: %.type.1 = fn_decl @.1 [template = constants.%.7] {}
// CHECK:STDOUT:   %.decl.loc40: %.type.2 = fn_decl @.2 [template = constants.%.8] {
// CHECK:STDOUT:     %int.make_type_32.loc40_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc40_19.1: type = value_of_initializer %int.make_type_32.loc40_19 [template = i32]
// CHECK:STDOUT:     %.loc40_19.2: type = converted %int.make_type_32.loc40_19, %.loc40_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc40_16.1: i32 = param a
// CHECK:STDOUT:     @.2.%a: i32 = bind_name a, %a.loc40_16.1
// CHECK:STDOUT:     %int.make_type_32.loc40_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc40_27.1: type = value_of_initializer %int.make_type_32.loc40_27 [template = i32]
// CHECK:STDOUT:     %.loc40_27.2: type = converted %int.make_type_32.loc40_27, %.loc40_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc40_24.1: i32 = param b
// CHECK:STDOUT:     @.2.%b: i32 = bind_name b, %b.loc40_24.1
// CHECK:STDOUT:     %int.make_type_32.loc40_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc40_35.1: type = value_of_initializer %int.make_type_32.loc40_35 [template = i32]
// CHECK:STDOUT:     %.loc40_35.2: type = converted %int.make_type_32.loc40_35, %.loc40_35.1 [template = i32]
// CHECK:STDOUT:     @.2.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %.loc16: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc22_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc22_19.1: type = value_of_initializer %int.make_type_32.loc22_19 [template = i32]
// CHECK:STDOUT:     %.loc22_19.2: type = converted %int.make_type_32.loc22_19, %.loc22_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc22_16.1: i32 = param a
// CHECK:STDOUT:     %a.loc22_16.2: i32 = bind_name a, %a.loc22_16.1
// CHECK:STDOUT:     %int.make_type_32.loc22_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc22_27.1: type = value_of_initializer %int.make_type_32.loc22_27 [template = i32]
// CHECK:STDOUT:     %.loc22_27.2: type = converted %int.make_type_32.loc22_27, %.loc22_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc22_24.1: i32 = param b
// CHECK:STDOUT:     %b.loc22_24.2: i32 = bind_name b, %b.loc22_24.1
// CHECK:STDOUT:     %int.make_type_32.loc22_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc22_35.1: type = value_of_initializer %int.make_type_32.loc22_35 [template = i32]
// CHECK:STDOUT:     %.loc22_35.2: type = converted %int.make_type_32.loc22_35, %.loc22_35.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_38: %.5 = assoc_entity element1, %G.decl [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc16
// CHECK:STDOUT:   .G = %.loc22_38
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Interface.%Self: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@Interface.%a.loc22_16.2: i32, @Interface.%b.loc22_24.2: i32) -> i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {}
// CHECK:STDOUT:
