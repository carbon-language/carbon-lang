// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/collapse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/collapse.carbon

// OK, `const (const i32)` is the same type as `const i32`.
// CHECK:STDERR: collapse.carbon:[[@LINE+3]]:25: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn F(p: const i32**) -> const (const i32)** {
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~
fn F(p: const i32**) -> const (const i32)** {
  return p;
}

// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %const: type = const_type %i32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %const [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %ptr.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_15: init type = call constants.%Int(%int_32.loc15_15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_9.1: type = value_of_initializer %int.make_type_signed.loc15_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_9.2: type = converted %int.make_type_signed.loc15_15, %.loc15_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %const.loc15_9: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:     %ptr.loc15_18: type = ptr_type %const [template = constants.%ptr.1]
// CHECK:STDOUT:     %ptr.loc15_19: type = ptr_type %ptr.1 [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32.loc15_38: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_38: init type = call constants.%Int(%int_32.loc15_38) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %int.make_type_signed.loc15_38 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %int.make_type_signed.loc15_38, %.loc15_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %const.loc15_32: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:     %const.loc15_25: type = const_type %const [template = constants.%const]
// CHECK:STDOUT:     %ptr.loc15_42: type = ptr_type %const [template = constants.%ptr.1]
// CHECK:STDOUT:     %ptr.loc15_43: type = ptr_type %ptr.1 [template = constants.%ptr.2]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param_patt: %ptr.2) -> %ptr.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
