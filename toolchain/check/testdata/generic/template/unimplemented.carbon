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
  // CHECK:STDERR:   var v: T = 0;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+4]]:3: error: value of type `<dependent type>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   var v: T = 0;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  var v: T = 0;
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `T` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var w: i32 = x;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+4]]:3: note: type `T` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var w: i32 = x;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  var w: i32 = x;
}

// CHECK:STDOUT: --- fail_todo_unimplemented_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
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
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc6_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc6_15.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_15.2 [template = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc6_15.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_15.2: type) {
// CHECK:STDOUT:   %T.loc6_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_15.1 [template = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc6_15.1 [template = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %.loc11_11.3: <instruction> = refine_type_action %x.ref, %T.loc6_15.1 [template]
// CHECK:STDOUT:   %.loc11_11.4: <instruction> = access_member_action %.loc11_11.1, n [template]
// CHECK:STDOUT:   %.loc11_11.5: type = type_of_inst %.loc11_11.4 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc6_15.1 (%T)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc6_15.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc11_11.1: @F.%T.loc6_15.1 (%T) = splice_inst %.loc11_11.3
// CHECK:STDOUT:     %.loc11_11.2: @F.%.loc11_11.5 (@F.%.loc11_11.5) = splice_inst %.loc11_11.4
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
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
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %c: %C = bind_symbolic_name c, 0, template [template]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = symbolic_binding_pattern c, 0, template [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc11_15.2: %C = bind_symbolic_name c, 0, template [template = %c.loc11_15.1 (constants.%c)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%c.loc11_15.2: %C) {
// CHECK:STDOUT:   %c.loc11_15.1: %C = bind_symbolic_name c, 0, template [template = %c.loc11_15.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc16_11.2: <instruction> = access_member_action %c.ref, n [template]
// CHECK:STDOUT:   %.loc16_11.3: type = type_of_inst %.loc16_11.2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: %C = name_ref c, %c.loc11_15.2 [template = %c.loc11_15.1 (constants.%c)]
// CHECK:STDOUT:     %.loc16_11.1: @F.%.loc16_11.3 (@F.%.loc16_11.3) = splice_inst %.loc16_11.2
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%c) {
// CHECK:STDOUT:   %c.loc11_15.1 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.9ff: %Destroy.type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.3ab: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.ef2: type = symbolic_binding_type T, 0, template, %T.9ff [template]
// CHECK:STDOUT:   %pattern_type.2f2: type = pattern_type %T.binding.as_type.ef2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cf3: <witness> = require_complete_type %T.binding.as_type.ef2 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.587: type = facet_type <@ImplicitAs, @ImplicitAs(%T.binding.as_type.ef2)> [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.dc4: <witness> = impl_witness imports.%Copy.impl_witness_table.965, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.dc4) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a12: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a57: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a12 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.a57, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.9ff, @Destroy [template]
// CHECK:STDOUT:   %.8eb: type = fn_type_with_self_type %Destroy.Op.type, %T.9ff [template]
// CHECK:STDOUT:   %impl.elem0.b2e: %.8eb = impl_witness_access %Destroy.lookup_impl_witness, element0 [template]
// CHECK:STDOUT:   %ptr.8654e8.2: type = ptr_type %T.binding.as_type.ef2 [template]
// CHECK:STDOUT:   %specific_impl_fn.b6b: <specific function> = specific_impl_function %impl.elem0.b2e, @Destroy.Op(%T.9ff) [template]
// CHECK:STDOUT:   %require_complete.ebb: <witness> = require_complete_type %ptr.8654e8.2 [template]
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
// CHECK:STDOUT:   %Core.import_ref.2eb = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.965 = impl_witness_table (%Core.import_ref.2eb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3ab = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.2f2) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.2f2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_23: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_15.2: %Destroy.type = bind_symbolic_name T, 0, template [template = %T.loc4_15.1 (constants.%T.9ff)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type.ef2) = value_param call_param0
// CHECK:STDOUT:     %.loc4_36.1: type = splice_block %.loc4_36.2 [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)] {
// CHECK:STDOUT:       %T.ref.loc4: %Destroy.type = name_ref T, %T.loc4_15.2 [template = %T.loc4_15.1 (constants.%T.9ff)]
// CHECK:STDOUT:       %T.as_type.loc4: type = facet_access_type %T.ref.loc4 [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)]
// CHECK:STDOUT:       %.loc4_36.2: type = converted %T.ref.loc4, %T.as_type.loc4 [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type.ef2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.2: %Destroy.type) {
// CHECK:STDOUT:   %T.loc4_15.1: %Destroy.type = bind_symbolic_name T, 0, template [template = %T.loc4_15.1 (constants.%T.9ff)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, template, %T.loc4_15.1 [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [template = %pattern_type (constants.%pattern_type.2f2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4: <witness> = require_complete_type %T.binding.as_type [template = %require_complete.loc4 (constants.%require_complete.cf3)]
// CHECK:STDOUT:   %ImplicitAs.type.loc14_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.binding.as_type)> [template = %ImplicitAs.type.loc14_3.2 (constants.%ImplicitAs.type.587)]
// CHECK:STDOUT:   %.loc14_3.4: <instruction> = access_member_action %ImplicitAs.type.loc14_3.1, Convert [template]
// CHECK:STDOUT:   %.loc14_3.5: type = type_of_inst %.loc14_3.4 [template]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_15.1, @Destroy [template = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc14_3.6: type = fn_type_with_self_type constants.%Destroy.Op.type, %T.loc4_15.1 [template = %.loc14_3.6 (constants.%.8eb)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @F.%.loc14_3.6 (%.8eb) = impl_witness_access %Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc14_3.2 (constants.%impl.elem0.b2e)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Destroy.Op(%T.loc4_15.1) [template = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [template = %ptr (constants.%ptr.8654e8.2)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %ptr [template = %require_complete.loc14 (constants.%require_complete.ebb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type.ef2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.2f2) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.2f2) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%T.binding.as_type (%T.binding.as_type.ef2) = var %v.var_patt
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %ImplicitAs.type.loc14_3.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%T.binding.as_type.ef2)> [template = %ImplicitAs.type.loc14_3.2 (constants.%ImplicitAs.type.587)]
// CHECK:STDOUT:     %.loc14_3.1: @F.%.loc14_3.5 (@F.%.loc14_3.5) = splice_inst %.loc14_3.4
// CHECK:STDOUT:     %.loc14_3.2: @F.%T.binding.as_type (%T.binding.as_type.ef2) = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %v.var, <error>
// CHECK:STDOUT:     %.loc14_10.1: type = splice_block %.loc14_10.2 [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)] {
// CHECK:STDOUT:       %T.ref.loc14: %Destroy.type = name_ref T, %T.loc4_15.2 [template = %T.loc4_15.1 (constants.%T.9ff)]
// CHECK:STDOUT:       %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)]
// CHECK:STDOUT:       %.loc14_10.2: type = converted %T.ref.loc14, %T.as_type.loc14 [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%T.binding.as_type (%T.binding.as_type.ef2) = bind_name v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %pattern_type.7ce = binding_pattern w [concrete]
// CHECK:STDOUT:       %w.var_patt: %pattern_type.7ce = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref %i32 = var %w.var_patt
// CHECK:STDOUT:     %x.ref: @F.%T.binding.as_type (%T.binding.as_type.ef2) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.dc4) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:     %.loc22_3.1: %Copy.type = converted constants.%i32, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:     %as_type.loc22: type = facet_access_type constants.%T.9ff [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)]
// CHECK:STDOUT:     %.loc22_3.2: %i32 = converted %x.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %w.var, <error>
// CHECK:STDOUT:     %.loc22_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:     %facet_value: %type_where = facet_value constants.%i32, () [concrete = constants.%facet_value]
// CHECK:STDOUT:     %.loc22_3.3: %type_where = converted constants.%i32, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %w.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a57
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a57, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc22: <bound method> = bound_method %w.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:     %addr.loc22: %ptr.235 = addr_of %w.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:     %as_type.loc14: type = facet_access_type constants.%T.9ff [template = %T.binding.as_type (constants.%T.binding.as_type.ef2)]
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @F.%.loc14_3.6 (%.8eb) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc14_3.2 (constants.%impl.elem0.b2e)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Destroy.Op(constants.%T.9ff) [template = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %addr.loc14: @F.%ptr (%ptr.8654e8.2) = addr_of %v.var
// CHECK:STDOUT:     %.loc14_3.3: init %empty_tuple.type = call %bound_method.loc14_3.2(%addr.loc14)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.9ff) {
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T.9ff
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.ef2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
