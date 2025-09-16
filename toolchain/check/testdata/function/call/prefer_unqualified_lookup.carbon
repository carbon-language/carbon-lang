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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/prefer_unqualified_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/prefer_unqualified_lookup.carbon

// --- prefer_unqualified_lookup.carbon

library "[[@TEST_NAME]]";

// F in the lexical scope.
class Class(F:! type) {
  class Inner {
    // F in a non-lexical scope.
    fn F() -> i32 { return 0; }
    fn G() -> i32;
  }
}

fn Class(F:! type).Inner.G() -> i32 { return F(); }

// CHECK:STDOUT: --- prefer_unqualified_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %F: type = bind_symbolic_name F, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%F) [symbolic]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%F) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%F) [symbolic]
// CHECK:STDOUT:   %Inner.F: %Inner.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.G.type: type = fn_type @Inner.G, @Inner(%F) [symbolic]
// CHECK:STDOUT:   %Inner.G: %Inner.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.7f2: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%F) [symbolic]
// CHECK:STDOUT:   %ptr.2b5: type = ptr_type %Inner [symbolic]
// CHECK:STDOUT:   %pattern_type.2df: type = pattern_type %ptr.2b5 [symbolic]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type: type = fn_type @Inner.as.Destroy.impl.Op, @Inner.as.Destroy.impl(%F) [symbolic]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op: %Inner.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.8cc: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%F) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%F) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Inner.F.specific_fn: <specific function> = specific_function %Inner.F, @Inner.F(%F) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %F.patt: %pattern_type.98f = symbolic_binding_pattern F, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %F.loc5_13.2: type = bind_symbolic_name F, 0 [symbolic = %F.loc5_13.1 (constants.%F)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.G.decl: %Inner.G.type = fn_decl @Inner.G [symbolic = constants.%Inner.G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %F.loc13_10: type = bind_symbolic_name F, 0 [symbolic = @Class.%F.loc5_13.1 (constants.%F)]
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param.loc13: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return.loc13: ref %i32 = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Inner.as.Destroy.impl(@Class.%F.loc5_13.2: type) {
// CHECK:STDOUT:   %F: type = bind_symbolic_name F, 0 [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%F) [symbolic = %Inner (constants.%Inner)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%F) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.7f2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type: type = fn_type @Inner.as.Destroy.impl.Op, @Inner.as.Destroy.impl(%F) [symbolic = %Inner.as.Destroy.impl.Op.type (constants.%Inner.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op: @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.type (%Inner.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Inner.as.Destroy.impl.Op (constants.%Inner.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Inner.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Inner.as.Destroy.impl.Op.decl: @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.type (%Inner.as.Destroy.impl.Op.type) = fn_decl @Inner.as.Destroy.impl.Op [symbolic = @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op (constants.%Inner.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Inner.as.Destroy.impl.Op.%pattern_type (%pattern_type.2df) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Inner.as.Destroy.impl.Op.%pattern_type (%pattern_type.2df) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_15.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Inner.as.Destroy.impl.Op.%ptr (%ptr.2b5) = value_param call_param0
// CHECK:STDOUT:       %.loc6_15.2: type = splice_block %Self.ref [symbolic = %Inner (constants.%Inner)] {
// CHECK:STDOUT:         %.loc6_15.3: type = specific_constant constants.%Inner, @Inner(constants.%F) [symbolic = %Inner (constants.%Inner)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_15.3 [symbolic = %Inner (constants.%Inner)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.as.Destroy.impl.Op.%ptr (%ptr.2b5) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Inner.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Inner.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%F.loc5_13.2: type) {
// CHECK:STDOUT:   %F: type = bind_symbolic_name F, 0 [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%F) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%F) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.8cc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%F) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc5_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc5_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc5_23.3: type = specific_constant constants.%Class, @Class(constants.%F) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_23.3 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%F.loc5_13.2: type) {
// CHECK:STDOUT:   %F.loc5_13.1: type = bind_symbolic_name F, 0 [symbolic = %F.loc5_13.1 (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%F.loc5_13.1) [symbolic = %Inner (constants.%Inner)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Class.%Inner (constants.%Inner)] {} {}
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%F) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.8cc)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Class.%F.loc5_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F: type = bind_symbolic_name F, 0 [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%F) [symbolic = %Inner.F.type (constants.%Inner.F.type)]
// CHECK:STDOUT:   %Inner.F: @Inner.%Inner.F.type (%Inner.F.type) = struct_value () [symbolic = %Inner.F (constants.%Inner.F)]
// CHECK:STDOUT:   %Inner.G.type: type = fn_type @Inner.G, @Inner(%F) [symbolic = %Inner.G.type (constants.%Inner.G.type)]
// CHECK:STDOUT:   %Inner.G: @Inner.%Inner.G.type (%Inner.G.type) = struct_value () [symbolic = %Inner.G (constants.%Inner.G)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.F.decl: @Inner.%Inner.F.type (%Inner.F.type) = fn_decl @Inner.F [symbolic = @Inner.%Inner.F (constants.%Inner.F)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.G.decl: @Inner.%Inner.G.type (%Inner.G.type) = fn_decl @Inner.G [symbolic = @Inner.%Inner.G (constants.%Inner.G)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %return.param.loc9: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return.loc9: ref %i32 = return_slot %return.param.loc9
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Inner [symbolic = @Inner.as.Destroy.impl.%Inner (constants.%Inner)]
// CHECK:STDOUT:     impl_decl @Inner.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.decl), @Inner.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Inner.as.Destroy.impl(constants.%F) [symbolic = @Inner.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.7f2)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner
// CHECK:STDOUT:     .F = %Inner.F.decl
// CHECK:STDOUT:     .G = %Inner.G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.F(@Class.%F.loc5_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:     %bound_method.loc8_29.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_29.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_29.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     return %.loc8 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.G(@Class.%F.loc5_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.loc13_46: type = bind_symbolic_name F, 0 [symbolic = %F.loc13_46 (constants.%F)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%F.loc13_46) [symbolic = %Inner.F.type (constants.%Inner.F.type)]
// CHECK:STDOUT:   %Inner.F: @Inner.G.%Inner.F.type (%Inner.F.type) = struct_value () [symbolic = %Inner.F (constants.%Inner.F)]
// CHECK:STDOUT:   %Inner.F.specific_fn.loc13_46.2: <specific function> = specific_function %Inner.F, @Inner.F(%F.loc13_46) [symbolic = %Inner.F.specific_fn.loc13_46.2 (constants.%Inner.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc13: @Inner.G.%Inner.F.type (%Inner.F.type) = specific_constant @Inner.%Inner.F.decl, @Inner(constants.%F) [symbolic = %Inner.F (constants.%Inner.F)]
// CHECK:STDOUT:     %F.ref: @Inner.G.%Inner.F.type (%Inner.F.type) = name_ref F, %.loc13 [symbolic = %Inner.F (constants.%Inner.F)]
// CHECK:STDOUT:     %Inner.F.specific_fn.loc13_46.1: <specific function> = specific_function %F.ref, @Inner.F(constants.%F) [symbolic = %Inner.F.specific_fn.loc13_46.2 (constants.%Inner.F.specific_fn)]
// CHECK:STDOUT:     %Inner.F.call: init %i32 = call %Inner.F.specific_fn.loc13_46.1()
// CHECK:STDOUT:     return %Inner.F.call to %return.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.as.Destroy.impl.Op(@Class.%F.loc5_13.2: type) {
// CHECK:STDOUT:   %F: type = bind_symbolic_name F, 0 [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%F) [symbolic = %Inner (constants.%Inner)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Inner [symbolic = %ptr (constants.%ptr.2b5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.2df)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.as.Destroy.impl.Op.%ptr (%ptr.2b5)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%F.loc5_13.2: type) {
// CHECK:STDOUT:   %F: type = bind_symbolic_name F, 0 [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%F) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%F) {
// CHECK:STDOUT:   %F.loc5_13.1 => constants.%F
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%F) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type
// CHECK:STDOUT:   %Inner.F => constants.%Inner.F
// CHECK:STDOUT:   %Inner.G.type => constants.%Inner.G.type
// CHECK:STDOUT:   %Inner.G => constants.%Inner.G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%F) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.G(constants.%F) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl(constants.%F) {
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %Inner => constants.%Inner
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.7f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl.Op(constants.%F) {
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %Inner => constants.%Inner
// CHECK:STDOUT:   %ptr => constants.%ptr.2b5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%F) {
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.8cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%F) {
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
