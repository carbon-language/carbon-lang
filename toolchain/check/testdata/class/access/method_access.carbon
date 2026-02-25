// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method_access.carbon

// --- fail_multiple_bindings.carbon

class X {
  fn F[self: Self]();
}

fn G(x: X) {
  // TODO: Produce a better diagnostic for this case.
  // CHECK:STDERR: fail_multiple_bindings.carbon:[[@LINE+4]]:3: error: member name of type `<bound method>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   x.(x.F)();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  x.(x.F)();
}

// CHECK:STDOUT: --- fail_multiple_bindings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %X.F.type: type = fn_type @X.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %X.F: %X.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %X.F.decl: %X.F.type = fn_decl @X.F [concrete = constants.%X.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %X = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %self: %X = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .F = %X.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.F(%self.param: %X);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc12_3: %X = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc12_6: %X = name_ref x, %x
// CHECK:STDOUT:   %F.ref: %X.F.type = name_ref F, @X.%X.F.decl [concrete = constants.%X.F]
// CHECK:STDOUT:   %X.F.bound: <bound method> = bound_method %x.ref.loc12_6, %F.ref
// CHECK:STDOUT:   %X.F.call: init %empty_tuple.type = call %X.F.bound(%x.ref.loc12_6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
