// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon

// --- using_poisoned_name_in_impl.carbon

library "[[@TEST_NAME]]";

interface I {};

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F1(x:! I);

class N.C {
  extend impl as I {
  }
}

// CHECK:STDOUT: --- using_poisoned_name_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %x: %I.type = bind_symbolic_name x, 0 [symbolic]
// CHECK:STDOUT:   %x.patt: %I.type = symbolic_binding_pattern x, 0 [symbolic]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %x.patt.loc8_9.1: %I.type = symbolic_binding_pattern x, 0 [symbolic = %x.patt.loc8_9.2 (constants.%x.patt)]
// CHECK:STDOUT:     %x.param_patt: %I.type = value_param_pattern %x.patt.loc8_9.1, runtime_param<none> [symbolic = %x.patt.loc8_9.2 (constants.%x.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %I.type = value_param runtime_param<none>
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %x.loc8_9.1: %I.type = bind_symbolic_name x, 0, %x.param [symbolic = %x.loc8_9.2 (constants.%x)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   extend @impl.%I.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F1(%x.loc8_9.1: %I.type) {
// CHECK:STDOUT:   %x.loc8_9.2: %I.type = bind_symbolic_name x, 0 [symbolic = %x.loc8_9.2 (constants.%x)]
// CHECK:STDOUT:   %x.patt.loc8_9.2: %I.type = symbolic_binding_pattern x, 0 [symbolic = %x.patt.loc8_9.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: %I.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F1(constants.%x) {
// CHECK:STDOUT:   %x.loc8_9.2 => constants.%x
// CHECK:STDOUT:   %x.patt.loc8_9.2 => constants.%x
// CHECK:STDOUT: }
// CHECK:STDOUT:
