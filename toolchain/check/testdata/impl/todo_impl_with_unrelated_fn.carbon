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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/todo_impl_with_unrelated_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/todo_impl_with_unrelated_fn.carbon

interface A {
  fn B();
}

class X {
  impl as A {
    fn B() {}
    // TODO: This should be diagnosed as an error; there is no `A.F` to
    // implement here.
    fn F() {}
  }
}

// CHECK:STDOUT: --- todo_impl_with_unrelated_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.B.type: type = fn_type @A.B [concrete]
// CHECK:STDOUT:   %A.B: %A.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%A.B.decl [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @X.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %X.as.A.impl.B.type: type = fn_type @X.as.A.impl.B [concrete]
// CHECK:STDOUT:   %X.as.A.impl.B: %X.as.A.impl.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.as.A.impl.F.type: type = fn_type @X.as.A.impl.F [concrete]
// CHECK:STDOUT:   %X.as.A.impl.F: %X.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %X, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %A.B.decl: %A.B.type = fn_decl @A.B [concrete = constants.%A.B] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.B.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = %assoc0
// CHECK:STDOUT:   witness = (%A.B.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.A.impl: %Self.ref as %A.ref {
// CHECK:STDOUT:   %X.as.A.impl.B.decl: %X.as.A.impl.B.type = fn_decl @X.as.A.impl.B [concrete = constants.%X.as.A.impl.B] {} {}
// CHECK:STDOUT:   %X.as.A.impl.F.decl: %X.as.A.impl.F.type = fn_decl @X.as.A.impl.F [concrete = constants.%X.as.A.impl.F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = %X.as.A.impl.B.decl
// CHECK:STDOUT:   .F = %X.as.A.impl.F.decl
// CHECK:STDOUT:   witness = @X.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @X.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@X.as.A.impl.%X.as.A.impl.B.decl), @X.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.B(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.A.impl.B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.A.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.B(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.B(constants.%A.facet) {}
// CHECK:STDOUT:
