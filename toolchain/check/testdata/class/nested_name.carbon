// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/nested_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/nested_name.carbon

class Outer {
  class Inner {
    var n: i32;
  }
}

fn F(oi: Outer.Inner) -> i32 {
  return oi.n;
}

fn G(o: Outer) {
  var i: o.Inner;
}

// CHECK:STDOUT: --- nested_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Outer: type = class_type @Outer [concrete]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.1dc: <witness> = impl_witness @Inner.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.36a: type = ptr_type %Inner [concrete]
// CHECK:STDOUT:   %pattern_type.27f: type = pattern_type %ptr.36a [concrete]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type: type = fn_type @Inner.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op: %Inner.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.34a: <witness> = impl_witness @Outer.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.5df: type = ptr_type %Outer [concrete]
// CHECK:STDOUT:   %pattern_type.95c: type = pattern_type %ptr.5df [concrete]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: %Outer.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.906: type = pattern_type %Inner [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.e74: type = pattern_type %Outer [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [concrete = constants.%Outer] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %oi.patt: %pattern_type.906 = binding_pattern oi [concrete]
// CHECK:STDOUT:     %oi.param_patt: %pattern_type.906 = value_param_pattern %oi.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %oi.param: %Inner = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %Inner.ref [concrete = constants.%Inner] {
// CHECK:STDOUT:       %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %oi: %Inner = bind_name oi, %oi.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %o.patt: %pattern_type.e74 = binding_pattern o [concrete]
// CHECK:STDOUT:     %o.param_patt: %pattern_type.e74 = value_param_pattern %o.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %o.param: %Outer = value_param call_param0
// CHECK:STDOUT:     %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %o: %Outer = bind_name o, %o.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Inner.as.Destroy.impl: @Inner.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.decl: %Inner.as.Destroy.impl.Op.type = fn_decl @Inner.as.Destroy.impl.Op [concrete = constants.%Inner.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.27f = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.27f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.36a = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:     %self: %ptr.36a = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Inner.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Inner.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Outer.as.Destroy.impl: @Outer.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.decl: %Outer.as.Destroy.impl.Op.type = fn_decl @Outer.as.Destroy.impl.Op [concrete = constants.%Outer.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.95c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.95c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.5df = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Outer [concrete = constants.%Outer]
// CHECK:STDOUT:     %self: %ptr.5df = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Outer.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Outer.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Outer [concrete = constants.%Outer]
// CHECK:STDOUT:   impl_decl @Outer.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.decl), @Outer.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.34a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17: %Inner.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:   impl_decl @Inner.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.decl), @Inner.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.1dc]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .n = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Inner.as.Destroy.impl.Op(%self.param: %ptr.36a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Outer.as.Destroy.impl.Op(%self.param: %ptr.5df) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%oi.param: %Inner) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %oi.ref: %Inner = name_ref oi, %oi
// CHECK:STDOUT:   %n.ref: %Inner.elem = name_ref n, @Inner.%.loc17 [concrete = @Inner.%.loc17]
// CHECK:STDOUT:   %.loc22_12.1: ref %i32 = class_element_access %oi.ref, element0
// CHECK:STDOUT:   %.loc22_12.2: %i32 = bind_value %.loc22_12.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc22_12.1: <bound method> = bound_method %.loc22_12.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_12.2: <bound method> = bound_method %.loc22_12.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_12.2(%.loc22_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%o.param: %Outer) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.906 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.906 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %.loc26: type = splice_block %Inner.ref [concrete = constants.%Inner] {
// CHECK:STDOUT:     %o.ref: %Outer = name_ref o, %o
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.bound: <bound method> = bound_method %i.var, constants.%Inner.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.36a = addr_of %i.var
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Inner.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
