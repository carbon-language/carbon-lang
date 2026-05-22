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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/fail_todo_undefined_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/fail_todo_undefined_impl.carbon

interface I {
  fn F();
}

class C {
  // This doesn't match `impl C as I` below under proposal #3763.

  extend impl as I;
}

fn F() {
  // CHECK:STDERR: fail_todo_undefined_impl.carbon:[[@LINE+4]]:3: error: accessing member from impl before it has a defined value [ImplAccessMemberBeforeSet]
  // CHECK:STDERR:   C.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  C.F();
}

// TODO: This should be written `impl C.(Self as I)` or `impl C.(as I)`, once
// that is supported.
impl C as I {
  fn F() {}
}

fn G() {
  // CHECK:STDERR: fail_todo_undefined_impl.carbon:[[@LINE+8]]:3: error: accessing member from impl before it has a defined value [ImplAccessMemberBeforeSet]
  // CHECK:STDERR:   C.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_undefined_impl.carbon:[[@LINE-22]]:3: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:   extend impl as I;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  C.F();
}

// CHECK:STDOUT: --- fail_todo_undefined_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.831: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.ddf: %I.WithSelf.F.type.831 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.impl_witness.398: <witness> = impl_witness @C.as.I.impl.677.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.124: type = fn_type @I.WithSelf.F, @I.WithSelf(%C.type.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.886: %I.WithSelf.F.type.124 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.634: %I.type = facet_value %C, (%I.impl_witness.398) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.f65: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.634) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.cea: %I.WithSelf.F.type.f65 = struct_value () [concrete]
// CHECK:STDOUT:   %.a8d: type = fn_type_with_self_type %I.WithSelf.F.type.f65, %I.facet.634 [concrete]
// CHECK:STDOUT:   %I.impl_witness.ce4: <witness> = impl_witness @C.as.I.impl.87e.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.706: %I.type = facet_value %C, (%I.impl_witness.ce4) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.f6c: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.706) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.2c6: %I.WithSelf.F.type.f6c = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl.87e [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.831) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ddf)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.677: %Self.ref as %I.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.87e: %C.ref as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl), @C.as.I.impl.87e [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.ce4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.I.impl.677 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend @C.as.I.impl.677.%I.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.a8d = impl_witness_access constants.%I.impl_witness.398, element0 [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.a8d = impl_witness_access constants.%I.impl_witness.398, element0 [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.831
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ddf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.124
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.886
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.634) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.634
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.f65
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.cea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.706) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.706
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.f6c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.2c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.706) {}
// CHECK:STDOUT:
