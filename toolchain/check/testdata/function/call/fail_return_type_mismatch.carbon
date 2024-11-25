// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/fail_return_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/fail_return_type_mismatch.carbon

fn Foo() -> f64 { return 1.0; }

fn Run() {
  // CHECK:STDERR: fail_return_type_mismatch.carbon:[[@LINE+6]]:3: error: cannot implicitly convert from `f64` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   var x: i32 = Foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_return_type_mismatch.carbon:[[@LINE+3]]:3: note: type `f64` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var x: i32 = Foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  var x: i32 = Foo();
}

// CHECK:STDOUT: --- fail_return_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.2: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_13.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%.loc11_13.1) [template = f64]
// CHECK:STDOUT:     %.loc11_13.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:     %.loc11_13.3: type = converted %float.make_type, %.loc11_13.2 [template = f64]
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_26: f64 = float_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc11_26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_10.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc20_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_10.3: type = converted %int.make_type_signed, %.loc20_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %x.var: ref %i32 = var x
// CHECK:STDOUT:   %x: ref %i32 = bind_name x, %x.var
// CHECK:STDOUT:   %Foo.ref: %Foo.type = name_ref Foo, file.%Foo.decl [template = constants.%Foo]
// CHECK:STDOUT:   %Foo.call: init f64 = call %Foo.ref()
// CHECK:STDOUT:   %.loc20_21: %i32 = converted %Foo.call, <error> [template = <error>]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
