// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/basic.carbon

fn A(p: const i32**) -> const i32** {
  return p;
}

fn B(p: const (i32*)) -> const (i32*) {
  return p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = const_type %i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.6: type = const_type %.5 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %p.patt: %.4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_15: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_15: init type = call constants.%Int(%.loc11_15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %int.make_type_signed.loc11_15, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_9.3: type = const_type %i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_18: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_19: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %.loc11_31: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_31: init type = call constants.%Int(%.loc11_31) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_signed.loc11_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_signed.loc11_31, %.loc11_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.3: type = const_type %i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_34: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_35: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %p.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.4 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %p.patt: %.6 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.6 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.6 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_16: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc15_16: init type = call constants.%Int(%.loc15_16) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_19.1: type = value_of_initializer %int.make_type_signed.loc15_16 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_19.2: type = converted %int.make_type_signed.loc15_16, %.loc15_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_19.3: type = ptr_type %i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_9: type = const_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     %.loc15_33: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc15_33: init type = call constants.%Int(%.loc15_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_signed.loc15_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_signed.loc15_33, %.loc15_36.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.3: type = ptr_type %i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_26: type = const_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     %p.param: %.6 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.6 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.6 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param_patt: %.4) -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param_patt: %.6) -> %.6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.6 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
