// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/default_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/default_fn.carbon

// --- parsed_in_class_scope.carbon
library "[[@TEST_NAME]]";

class C {
  //@dump-sem-ir-begin
  interface I {
    // TODO: Use `default` here.
    default fn F(unused self) {
      // I and F are both complete here, and the impl below is in scope.
      var c: C = {};
      c.(I.F)();
    }
  }
  //@dump-sem-ir-end

  // The fact that `F` has a default is visible here.
  //@dump-sem-ir-begin
  impl C as I {}
  //@dump-sem-ir-end
}

// --- generic_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
  //@dump-sem-ir-begin
  default fn G() { Self.F(); }
  //@dump-sem-ir-end
}

//@dump-sem-ir-begin
impl forall [T: type] T as I {
  fn F() {}
}
//@dump-sem-ir-end

fn Call() { ().(I.G)(); }

// CHECK:STDOUT: --- parsed_in_class_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.4e9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.0fa: type = facet_access_type %Self.4e9 [symbolic]
// CHECK:STDOUT:   %pattern_type.492: type = pattern_type %Self.as_type.0fa [symbolic]
// CHECK:STDOUT:   %self.param_patt.55d: %pattern_type.492 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.ce2: %pattern_type.492 = wrapper_binding_pattern self, %self.param_patt.55d [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.933: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.4e9) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.02a: %I.WithSelf.F.type.933 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.08d: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %.c3e: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.684: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ee1: %I.WithSelf.F.type.684 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.0fa [symbolic]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.938: type = fn_type_with_self_type %I.WithSelf.F.type.684, %I.facet [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.36a: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.specific_fn: <specific function> = specific_function %I.WithSelf.F.ee1, @I.WithSelf.F(%I.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.4e9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.933) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.02a)] {
// CHECK:STDOUT:     %self.param_patt.loc7_25.1: @I.WithSelf.F.%pattern_type (%pattern_type.492) = value_param_pattern [symbolic = %self.param_patt.loc7_25.2 (constants.%self.param_patt.55d)]
// CHECK:STDOUT:     %self.patt.loc7_25.1: @I.WithSelf.F.%pattern_type (%pattern_type.492) = wrapper_binding_pattern self, %self.param_patt.loc7_25.1 [symbolic = %self.patt.loc7_25.2 (constants.%self.patt.ce2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.as_type.loc7_25.1 (%Self.as_type.0fa) = value_param call_param0
// CHECK:STDOUT:     %.loc7_25.1: type = splice_block %.loc7_25.2 [symbolic = %Self.as_type.loc7_25.1 (constants.%Self.as_type.0fa)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.4e9)]
// CHECK:STDOUT:       %Self.as_type.loc7_25.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc7_25.1 (constants.%Self.as_type.0fa)]
// CHECK:STDOUT:       %.loc7_25.2: type = converted %Self.ref, %Self.as_type.loc7_25.2 [symbolic = %Self.as_type.loc7_25.1 (constants.%Self.as_type.0fa)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.as_type.loc7_25.1 (%Self.as_type.0fa) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.08d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (constants.%I.WithSelf.F.ee1), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, @C.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I [concrete = constants.%.c3e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.4e9)]
// CHECK:STDOUT:   %Self.as_type.loc7_25.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc7_25.1 (constants.%Self.as_type.0fa)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc7_25.1 [symbolic = %pattern_type (constants.%pattern_type.492)]
// CHECK:STDOUT:   %self.param_patt.loc7_25.2: @I.WithSelf.F.%pattern_type (%pattern_type.492) = value_param_pattern [symbolic = %self.param_patt.loc7_25.2 (constants.%self.param_patt.55d)]
// CHECK:STDOUT:   %self.patt.loc7_25.2: @I.WithSelf.F.%pattern_type (%pattern_type.492) = wrapper_binding_pattern self, %self.param_patt.loc7_25.2 [symbolic = %self.patt.loc7_25.2 (constants.%self.patt.ce2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc7_25.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc7_25.1 (%Self.as_type.0fa)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:     %.loc9_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc9_19.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc9_7: init %C = converted %.loc9_19.1, %.loc9_19.2 [concrete = constants.%C.val]
// CHECK:STDOUT:     assign %c.var, %.loc9_7
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:       %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:     %I.ref: type = name_ref I, @C.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0.08d]
// CHECK:STDOUT:     %impl.elem0: %.938 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%I.WithSelf.F.ee1]
// CHECK:STDOUT:     %bound_method.loc10_8: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:     %I.facet.loc10_15.1: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:     %.loc10_15.1: %I.type = converted constants.%C, %I.facet.loc10_15.1 [concrete = constants.%I.facet]
// CHECK:STDOUT:     %I.facet.loc10_15.2: %I.type = facet_value constants.%C, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:     %.loc10_15.2: %I.type = converted constants.%C, %I.facet.loc10_15.2 [concrete = constants.%I.facet]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @I.WithSelf.F(constants.%I.facet) [concrete = constants.%I.WithSelf.F.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_15: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:     %.loc10_7: %C = acquire_value %c.ref
// CHECK:STDOUT:     %I.WithSelf.F.call: init %empty_tuple.type = call %bound_method.loc10_15(%.loc10_7)
// CHECK:STDOUT:     %Destroy.WithSelf.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.WithSelf.Op.403171.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %Destroy.WithSelf.Op.bound(%c.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_7.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.4e9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.4e9
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.933
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.02a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.4e9) {
// CHECK:STDOUT:   %Self => constants.%Self.4e9
// CHECK:STDOUT:   %Self.as_type.loc7_25.1 => constants.%Self.as_type.0fa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.492
// CHECK:STDOUT:   %self.param_patt.loc7_25.2 => constants.%self.param_patt.55d
// CHECK:STDOUT:   %self.patt.loc7_25.2 => constants.%self.patt.ce2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.684
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ee1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc7_25.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %self.param_patt.loc7_25.2 => constants.%self.param_patt.303
// CHECK:STDOUT:   %self.patt.loc7_25.2 => constants.%self.patt.36a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.dcc: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ec6: %I.WithSelf.F.type.dcc = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.type.d1f: type = fn_type @I.WithSelf.G, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.44c: %I.WithSelf.G.type.d1f = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.WithSelf.%I.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic]
// CHECK:STDOUT:   %.6ad: type = fn_type_with_self_type %I.WithSelf.F.type.dcc, %Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.6ad = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.F(%Self) [symbolic]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %.917: <witness> = impl_self_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.92f: <witness> = impl_witness @T.as.I.impl.%I.impl_witness_table, @T.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.F.type.d46: type = fn_type @T.as.I.impl.F, @T.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.F.ca4: %T.as.I.impl.F.type.d46 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.1b7: %I.type = facet_value %T, (%I.impl_witness.92f) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.424: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.1b7) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.2d3: %I.WithSelf.F.type.424 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.type.33c: type = fn_type @I.WithSelf.G, @I.WithSelf(%I.facet.1b7) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.a95: %I.WithSelf.G.type.33c = struct_value () [symbolic]
// CHECK:STDOUT:   %.2ef: <witness> = impl_self_witness %empty_tuple.type, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.cf7: <witness> = impl_witness @T.as.I.impl.%I.impl_witness_table, @T.as.I.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.I.impl.F.type.3a0: type = fn_type @T.as.I.impl.F, @T.as.I.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.I.impl.F.38f: %T.as.I.impl.F.type.3a0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.e87: %I.type = facet_value %empty_tuple.type, (%I.impl_witness.cf7) [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.type.0af: type = fn_type @I.WithSelf.G, @I.WithSelf(%I.facet.e87) [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.da9: %I.WithSelf.G.type.0af = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.65a: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.e87) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.b11: %I.WithSelf.F.type.65a = struct_value () [concrete]
// CHECK:STDOUT:   %.763: type = fn_type_with_self_type %I.WithSelf.F.type.65a, %I.facet.e87 [concrete]
// CHECK:STDOUT:   %T.as.I.impl.F.specific_fn: <specific function> = specific_function %T.as.I.impl.F.38f, @T.as.I.impl.F(%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @T.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_15.2 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc11_17.1: type = splice_block %.loc11_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc11_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @T.as.I.impl.%T.ref, @I [symbolic = @T.as.I.impl.%.loc11_30 (constants.%.917)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.G.decl: @I.WithSelf.%I.WithSelf.G.type (%I.WithSelf.G.type.d1f) = fn_decl @I.WithSelf.G [symbolic = @I.WithSelf.%I.WithSelf.G (constants.%I.WithSelf.G.44c)] {} {}
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %I.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   .G = @I.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.%I.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl(%T.loc11_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc11_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc11_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:   %.loc11_30: <witness> = impl_self_witness %T.loc11_15.1, @I [symbolic = %.loc11_30 (constants.%.917)]
// CHECK:STDOUT:   %I.impl_witness.loc11_30.2: <witness> = impl_witness %I.impl_witness_table, @T.as.I.impl(%T.loc11_15.1) [symbolic = %I.impl_witness.loc11_30.2 (constants.%I.impl_witness.92f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.F.type: type = fn_type @T.as.I.impl.F, @T.as.I.impl(%T.loc11_15.1) [symbolic = %T.as.I.impl.F.type (constants.%T.as.I.impl.F.type.d46)]
// CHECK:STDOUT:   %T.as.I.impl.F: @T.as.I.impl.%T.as.I.impl.F.type (%T.as.I.impl.F.type.d46) = struct_value () [symbolic = %T.as.I.impl.F (constants.%T.as.I.impl.F.ca4)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %T.as.I.impl.F.decl: @T.as.I.impl.%T.as.I.impl.F.type (%T.as.I.impl.F.type.d46) = fn_decl @T.as.I.impl.F [symbolic = @T.as.I.impl.%T.as.I.impl.F (constants.%T.as.I.impl.F.ca4)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%T.as.I.impl.F.decl, %I.WithSelf.G), @T.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc11_30.1: <witness> = impl_witness %I.impl_witness_table, @T.as.I.impl(constants.%T) [symbolic = %I.impl_witness.loc11_30.2 (constants.%I.impl_witness.92f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as.I.impl.F.decl
// CHECK:STDOUT:     .G = <poisoned>
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc11_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.G(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc6_24.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.dcc)]
// CHECK:STDOUT:   %.loc6_24.2: type = fn_type_with_self_type %I.WithSelf.F.type, %Self [symbolic = %.loc6_24.2 (constants.%.6ad)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc6_24.2: @I.WithSelf.G.%.loc6_24.2 (%.6ad) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_24.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_24.2: <specific function> = specific_impl_function %impl.elem0.loc6_24.2, @I.WithSelf.F(%Self) [symbolic = %specific_impl_fn.loc6_24.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc6_24.1: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc6_24.1: type = converted %Self.ref, %Self.as_type.loc6_24.1 [symbolic = %Self.as_type.loc6_24.2 (constants.%Self.as_type)]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc6_24.1: @I.WithSelf.G.%.loc6_24.2 (%.6ad) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_24.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_24.1: <specific function> = specific_impl_function %impl.elem0.loc6_24.1, @I.WithSelf.F(constants.%Self) [symbolic = %specific_impl_fn.loc6_24.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc6_24.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.F(@T.as.I.impl.%T.loc11_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.dcc
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ec6
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.d1f
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.44c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc11_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc11_15.1 => constants.%T
// CHECK:STDOUT:   %.loc11_30 => constants.%.917
// CHECK:STDOUT:   %I.impl_witness.loc11_30.2 => constants.%I.impl_witness.92f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.F.type => constants.%T.as.I.impl.F.type.d46
// CHECK:STDOUT:   %T.as.I.impl.F => constants.%T.as.I.impl.F.ca4
// CHECK:STDOUT:   %I.facet => constants.%I.facet.1b7
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.33c
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.a95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.1b7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.1b7
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.424
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.2d3
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.33c
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.a95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.patt.loc11_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc11_15.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc11_30 => constants.%.2ef
// CHECK:STDOUT:   %I.impl_witness.loc11_30.2 => constants.%I.impl_witness.cf7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.F.type => constants.%T.as.I.impl.F.type.3a0
// CHECK:STDOUT:   %T.as.I.impl.F => constants.%T.as.I.impl.F.38f
// CHECK:STDOUT:   %I.facet => constants.%I.facet.e87
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.0af
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.da9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.e87) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.e87
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.65a
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.b11
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.0af
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.da9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.G(constants.%I.facet.e87) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.e87
// CHECK:STDOUT:   %Self.as_type.loc6_24.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.65a
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.763
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.cf7
// CHECK:STDOUT:   %impl.elem0.loc6_24.2 => constants.%T.as.I.impl.F.38f
// CHECK:STDOUT:   %specific_impl_fn.loc6_24.2 => constants.%T.as.I.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.F(constants.%empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
