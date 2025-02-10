// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/core_name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/core_name_poisoning.carbon

// --- fail_implicitly_poison_core.carbon

// CHECK:STDERR: fail_implicitly_poison_core.carbon:[[@LINE+7]]:9: error: `Core.Bool` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn F(x: bool);
// CHECK:STDERR:         ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_implicitly_poison_core.carbon:[[@LINE+3]]:9: error: name used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn F(x: bool);
// CHECK:STDERR:         ^~~~
fn F(x: bool);

// CHECK:STDERR: fail_implicitly_poison_core.carbon:[[@LINE+4]]:1: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class r#Core {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
class r#Core {}

// CHECK:STDOUT: --- fail_implicitly_poison_core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Core: type = class_type @Core [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.decl: type = class_decl @Core [template = constants.%Core] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Core {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Core
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: <error>);
// CHECK:STDOUT:
