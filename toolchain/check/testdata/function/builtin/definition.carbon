// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/definition.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.2: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: %Int.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Int.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %Int.2 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %Int.2 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %Int.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Int.2 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %Int.loc11_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%Int.2]
// CHECK:STDOUT:     %int_32.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %Int.loc11_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%Int.2]
// CHECK:STDOUT:     %int_32.loc11_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %Int.loc11_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%Int.2]
// CHECK:STDOUT:     %a.param: %Int.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %Int.2 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %Int.2 = value_param runtime_param1
// CHECK:STDOUT:     %b: %Int.2 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %Int.2 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %Int.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param_patt: %Int.2, %b.param_patt: %Int.2) -> %Int.2 = "int.sadd";
// CHECK:STDOUT:
