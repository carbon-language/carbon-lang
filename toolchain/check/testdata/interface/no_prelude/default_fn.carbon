// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/default_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/default_fn.carbon

class C {
  interface I {
    // TODO: Use `default` here.
    fn F[self: Self]() {
      // I and F are both complete here, and the impl below is in scope.
      var c: C = {};
      c.(I.F)();
    }
  }

  impl C as I {
    fn F[self: Self]() {}
  }
}

// CHECK:STDOUT: --- default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.0b5: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.20c: %F.type.0b5 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.4a2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.eb2: %F.type.4a2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.56c: type = fn_type_with_self_type %F.type.0b5, %I.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.0b5 = fn_decl @F.1 [concrete = constants.%F.20c] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc14_16.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc14_16.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc14_16.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_16.1: type = splice_block %.loc14_16.2 [symbolic = %Self.as_type.loc14_16.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc14_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc14_16.2: type = converted %Self.ref, %Self.as_type.loc14_16.2 [symbolic = %Self.as_type.loc14_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc14_16.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.4a2 = fn_decl @F.2 [concrete = constants.%F.eb2] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, @C.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc14_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.1.%Self.as_type.loc14_16.1 (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc14_16.1 (%Self.as_type)]() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: %C = binding_pattern c
// CHECK:STDOUT:       %.loc16_7.1: %C = var_pattern %c.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref %C = var c
// CHECK:STDOUT:     %.loc16_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc16_19.2: init %C = class_init (), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc16_7.2: init %C = converted %.loc16_19.1, %.loc16_19.2 [concrete = constants.%C.val]
// CHECK:STDOUT:     assign %c.var, %.loc16_7.2
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:     %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:     %I.ref: type = name_ref I, @C.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: %.56c = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%F.eb2]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:     %.loc17: %C = bind_value %c.ref
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %bound_method(%.loc17)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc14_16.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc14_16.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
