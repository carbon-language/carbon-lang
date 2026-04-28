// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon

// --- fail_todo_unimplemented_operator.carbon

library "[[@TEST_NAME]]";

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent expression.
fn F[template T:! type](x: T) -> i32 {
  // CHECK:STDERR: fail_todo_unimplemented_operator.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.MulWith(Core.IntLiteral)` in type `<dependent type>` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return x.n * 3;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return x.n * 3;
}

// --- fail_todo_unimplemented_value.carbon

library "[[@TEST_NAME]]";

class C {
  var n: i32;
}

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent value where the type is concrete but determined through
// the template dependent value.
fn F(template c:! C) -> i32 {
  // CHECK:STDERR: fail_todo_unimplemented_value.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.MulWith(Core.IntLiteral)` in type `<dependent type>` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return c.n * 3;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return c.n * 3;
}

// --- fail_todo_unimplemented_convert.carbon

library "[[@TEST_NAME]]";

fn F[template T:! Core.Destroy](x: T) {
  // TODO: These diagnostics aren't very good, and we should only produce one error here.
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+8]]:3: error: member name of type `<dependent type>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   var unused v: T = 0;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+4]]:3: error: value of type `<dependent type>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   var unused v: T = 0;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused v: T = 0;
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `T` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var unused w: i32 = x;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+4]]:3: note: type `T` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused w: i32 = x;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused w: i32 = x;
}

// CHECK:STDOUT: --- fail_todo_unimplemented_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %MulWith.type.8b4: type = generic_interface_type @MulWith [concrete]
// CHECK:STDOUT:   %MulWith.generic: %MulWith.type.8b4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .MulWith = %Core.MulWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.MulWith: %MulWith.type.8b4 = import_ref Core//prelude/operators/arithmetic, MulWith, loaded [concrete = constants.%MulWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.51d1c4.1) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_34: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %.loc6_19.1: type = splice_block %.loc6_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc6_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_16.2: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc6_16.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_16.2 [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc6_16.1 (%T) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_16.2: type) {
// CHECK:STDOUT:   %T.loc6_16.1: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc6_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc11_11.3: <instruction> = refine_type_action %x.ref, %T.loc6_16.1 [template]
// CHECK:STDOUT:   %.loc11_11.4: <instruction> = access_member_action %.loc11_11.1, n [template]
// CHECK:STDOUT:   %.loc11_11.5: type = type_of_inst %.loc11_11.4 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc6_16.1 (%T)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc6_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc11_11.1: @F.%T.loc6_16.1 (%T) = splice_inst %.loc11_11.3
// CHECK:STDOUT:     %.loc11_11.2: @F.%.loc11_11.5 (@F.%.loc11_11.5) = splice_inst %.loc11_11.4
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c: %C = symbolic_binding c, 0, template [template]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %MulWith.type.8b4: type = generic_interface_type @MulWith [concrete]
// CHECK:STDOUT:   %MulWith.generic: %MulWith.type.8b4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .MulWith = %Core.MulWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.MulWith: %MulWith.type.8b4 = import_ref Core//prelude/operators/arithmetic, MulWith, loaded [concrete = constants.%MulWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = symbolic_binding_pattern c, 0, template [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_25: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %.loc11_19: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc11_16.2: %C = symbolic_binding c, 0, template [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%c.loc11_16.2: %C) {
// CHECK:STDOUT:   %c.loc11_16.1: %C = symbolic_binding c, 0, template [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc16_11.2: <instruction> = access_member_action %c.ref, n [template]
// CHECK:STDOUT:   %.loc16_11.3: type = type_of_inst %.loc16_11.2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: %C = name_ref c, %c.loc11_16.2 [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:     %.loc16_11.1: @F.%.loc16_11.3 (@F.%.loc16_11.3) = splice_inst %.loc16_11.2
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%c) {
// CHECK:STDOUT:   %c.loc11_16.1 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.cac: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.765: %Destroy.type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %T.as_type.2e9: type = facet_access_type %T.765 [template]
// CHECK:STDOUT:   %pattern_type.eca0e4.2: type = pattern_type %T.as_type.2e9 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6ff: <witness> = require_complete_type %T.as_type.2e9 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9a: type = facet_type <@ImplicitAs, @ImplicitAs(%T.as_type.2e9)> [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc22_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.765, @Destroy [template]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.cb2e47.2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%T.765) [template]
// CHECK:STDOUT:   %.d5f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.cb2e47.2, %T.765 [template]
// CHECK:STDOUT:   %impl.elem0.a6b: %.d5f = impl_witness_access %Destroy.lookup_impl_witness, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn.761: <specific function> = specific_impl_function %impl.elem0.a6b, @Destroy.WithSelf.Op(%T.765) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.cac = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.eca0e4.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.eca0e4.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_23: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.765)]
// CHECK:STDOUT:     %x.param: @F.%T.as_type.loc4_36.1 (%T.as_type.2e9) = value_param call_param0
// CHECK:STDOUT:     %.loc4_36.1: type = splice_block %.loc4_36.2 [template = %T.as_type.loc4_36.1 (constants.%T.as_type.2e9)] {
// CHECK:STDOUT:       %T.ref.loc4: %Destroy.type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T.765)]
// CHECK:STDOUT:       %T.as_type.loc4_36.2: type = facet_access_type %T.ref.loc4 [template = %T.as_type.loc4_36.1 (constants.%T.as_type.2e9)]
// CHECK:STDOUT:       %.loc4_36.2: type = converted %T.ref.loc4, %T.as_type.loc4_36.2 [template = %T.as_type.loc4_36.1 (constants.%T.as_type.2e9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.as_type.loc4_36.1 (%T.as_type.2e9) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_16.2: %Destroy.type) {
// CHECK:STDOUT:   %T.loc4_16.1: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.765)]
// CHECK:STDOUT:   %T.as_type.loc4_36.1: type = facet_access_type %T.loc4_16.1 [template = %T.as_type.loc4_36.1 (constants.%T.as_type.2e9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc4_36.1 [template = %pattern_type (constants.%pattern_type.eca0e4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc4_36.1 [template = %require_complete (constants.%require_complete.6ff)]
// CHECK:STDOUT:   %ImplicitAs.type.loc14_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.as_type.loc4_36.1)> [template = %ImplicitAs.type.loc14_3.2 (constants.%ImplicitAs.type.b9a)]
// CHECK:STDOUT:   %.loc14_3.3: <instruction> = access_member_action %ImplicitAs.type.loc14_3.1, Convert [template]
// CHECK:STDOUT:   %.loc14_3.4: type = type_of_inst %.loc14_3.3 [template]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%T.loc4_16.1) [template = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.cb2e47.2)]
// CHECK:STDOUT:   %.loc14_3.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %T.loc4_16.1 [template = %.loc14_3.5 (constants.%.d5f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_16.1, @Destroy [template = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @F.%.loc14_3.5 (%.d5f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc14_3.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Destroy.WithSelf.Op(%T.loc4_16.1) [template = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.as_type.loc4_36.1 (%T.as_type.2e9)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.eca0e4.2) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.eca0e4.2) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%T.as_type.loc4_36.1 (%T.as_type.2e9) = var %v.var_patt
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %ImplicitAs.type.loc14_3.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%T.as_type.2e9)> [template = %ImplicitAs.type.loc14_3.2 (constants.%ImplicitAs.type.b9a)]
// CHECK:STDOUT:     %.loc14_3.1: @F.%.loc14_3.4 (@F.%.loc14_3.4) = splice_inst %.loc14_3.3
// CHECK:STDOUT:     %.loc14_3.2: @F.%T.as_type.loc4_36.1 (%T.as_type.2e9) = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %v.var, <error>
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.2 [template = %T.as_type.loc4_36.1 (constants.%T.as_type.2e9)] {
// CHECK:STDOUT:       %T.ref.loc14: %Destroy.type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T.765)]
// CHECK:STDOUT:       %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [template = %T.as_type.loc4_36.1 (constants.%T.as_type.2e9)]
// CHECK:STDOUT:       %.loc14_17.2: type = converted %T.ref.loc14, %T.as_type.loc14 [template = %T.as_type.loc4_36.1 (constants.%T.as_type.2e9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%T.as_type.loc4_36.1 (%T.as_type.2e9) = ref_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %pattern_type.7ce = ref_binding_pattern w [concrete]
// CHECK:STDOUT:       %w.var_patt: %pattern_type.7ce = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref %i32 = var %w.var_patt
// CHECK:STDOUT:     %x.ref: @F.%T.as_type.loc4_36.1 (%T.as_type.2e9) = name_ref x, %x
// CHECK:STDOUT:     %.loc22: %i32 = converted %x.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %w.var, <error>
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %w: ref %i32 = ref_binding w, %w.var
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %w.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%w.var)
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @F.%.loc14_3.5 (%.d5f) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc14_3.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Destroy.WithSelf.Op(constants.%T.765) [template = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_3.2(%v.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.765) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.765
// CHECK:STDOUT:   %T.as_type.loc4_36.1 => constants.%T.as_type.2e9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eca0e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
