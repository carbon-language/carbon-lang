// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/params_two_comma.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/params_two_comma.carbon

fn Foo(a: i32, b: i32,) {}

// CHECK:STDOUT: --- params_two_comma.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %.2 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.2 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_11: init type = call constants.%Int(%.loc11_11.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_11.2: type = value_of_initializer %int.make_type_signed.loc11_11 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_11.3: type = converted %int.make_type_signed.loc11_11, %.loc11_11.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_19: init type = call constants.%Int(%.loc11_19.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_19.2: type = value_of_initializer %int.make_type_signed.loc11_19 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_19.3: type = converted %int.make_type_signed.loc11_19, %.loc11_19.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %.2 = value_param runtime_param1
// CHECK:STDOUT:     %b: %.2 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a.param_patt: %.2, %b.param_patt: %.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
