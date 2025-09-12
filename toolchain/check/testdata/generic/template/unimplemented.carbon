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

fn F[template T:! type](x: T) {
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
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
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
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .MulWith = %Core.MulWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc11_15.2: %C = bind_symbolic_name c, 0, template [template = %c.loc11_15.1 (constants.%c)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
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
// CHECK:STDOUT:   %T.8b3d5d.1: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T.8b3d5d.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3d5d.1 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.dc4d8f.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.8b3d5d.1)> [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.3d8: <witness> = impl_witness imports.%Copy.impl_witness_table.000, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.1d5: %Copy.type = facet_value %i32, (%Copy.impl_witness.3d8) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.11b: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.054: %Int.as.Destroy.impl.Op.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Destroy.impl.Op.054, @Int.as.Destroy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.79f131.2: type = ptr_type %T.8b3d5d.1 [template]
// CHECK:STDOUT:   %require_complete.6e5e64.2: <witness> = require_complete_type %ptr.79f131.2 [template]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.8b3d5d.1, @Destroy [template]
// CHECK:STDOUT:   %Destroy.facet.572: %Destroy.type = facet_value %T.8b3d5d.1, (%Destroy.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.f8c: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.572 [template]
// CHECK:STDOUT:   %impl.elem0.796: %.f8c = impl_witness_access %Destroy.lookup_impl_witness, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn.127: <specific function> = specific_impl_function %impl.elem0.796, @Destroy.Op(%Destroy.facet.572) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.46d = import_ref Core//prelude/types/int, loc21_31, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.000 = impl_witness_table (%Core.import_ref.46d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_15.1 (%T.8b3d5d.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, %T.loc4_15.2 [template = %T.loc4_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %x: @F.%T.loc4_15.1 (%T.8b3d5d.1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_15.1 [template = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4: <witness> = require_complete_type %T.loc4_15.1 [template = %require_complete.loc4 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %ImplicitAs.type.loc14_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc4_15.1)> [template = %ImplicitAs.type.loc14_3.2 (constants.%ImplicitAs.type.dc4d8f.2)]
// CHECK:STDOUT:   %.loc14_3.4: <instruction> = access_member_action %ImplicitAs.type.loc14_3.1, Convert [template]
// CHECK:STDOUT:   %.loc14_3.5: type = type_of_inst %.loc14_3.4 [template]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_15.1, @Destroy [template = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T.loc4_15.1, (%Destroy.lookup_impl_witness) [template = %Destroy.facet (constants.%Destroy.facet.572)]
// CHECK:STDOUT:   %.loc14_3.6: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [template = %.loc14_3.6 (constants.%.f8c)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @F.%.loc14_3.6 (%.f8c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc14_3.2 (constants.%impl.elem0.796)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Destroy.Op(%Destroy.facet) [template = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.127)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc4_15.1 [template = %ptr (constants.%ptr.79f131.2)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %ptr [template = %require_complete.loc14 (constants.%require_complete.6e5e64.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc4_15.1 (%T.8b3d5d.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.7dcd0a.1) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%T.loc4_15.1 (%T.8b3d5d.1) = var %v.var_patt
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %ImplicitAs.type.loc14_3.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%T.8b3d5d.1)> [template = %ImplicitAs.type.loc14_3.2 (constants.%ImplicitAs.type.dc4d8f.2)]
// CHECK:STDOUT:     %.loc14_3.1: @F.%.loc14_3.5 (@F.%.loc14_3.5) = splice_inst %.loc14_3.4
// CHECK:STDOUT:     %.loc14_3.2: @F.%T.loc4_15.1 (%T.8b3d5d.1) = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %v.var, <error>
// CHECK:STDOUT:     %T.ref.loc14: type = name_ref T, %T.loc4_15.2 [template = %T.loc4_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %v: ref @F.%T.loc4_15.1 (%T.8b3d5d.1) = bind_name v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %pattern_type.7ce = binding_pattern w [concrete]
// CHECK:STDOUT:       %w.var_patt: %pattern_type.7ce = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref %i32 = var %w.var_patt
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_15.1 (%T.8b3d5d.1) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.3d8) [concrete = constants.%Copy.facet.1d5]
// CHECK:STDOUT:     %.loc22_3.1: %Copy.type = converted constants.%i32, %Copy.facet [concrete = constants.%Copy.facet.1d5]
// CHECK:STDOUT:     %.loc22_3.2: %i32 = converted %x.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %w.var, <error>
// CHECK:STDOUT:     %.loc22_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:     %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %w.var, constants.%Int.as.Destroy.impl.Op.054
// CHECK:STDOUT:     %Int.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%Int.as.Destroy.impl.Op.054, @Int.as.Destroy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc22: <bound method> = bound_method %w.var, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:     %addr.loc22: %ptr.235 = addr_of %w.var
// CHECK:STDOUT:     %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @F.%.loc14_3.6 (%.f8c) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc14_3.2 (constants.%impl.elem0.796)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Destroy.Op(constants.%Destroy.facet.572) [template = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.127)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %addr.loc14: @F.%ptr (%ptr.79f131.2) = addr_of %v.var
// CHECK:STDOUT:     %.loc14_3.3: init %empty_tuple.type = call %bound_method.loc14_3.2(%addr.loc14)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T.8b3d5d.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
