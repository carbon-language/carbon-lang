// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/as_type_of_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/as_type_of_type.carbon

interface Empty {}

class C {}

impl C as Empty {}

fn F(T:! Empty) {
  var x: T*;
}

// CHECK:STDOUT: --- as_type_of_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.e3a: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.071: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness file.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: %Empty.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.639: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %ptr.2b9: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.1f4: <witness> = require_complete_type %ptr.2b9 [symbolic]
// CHECK:STDOUT:   %pattern_type.ae3: type = pattern_type %ptr.2b9 [symbolic]
// CHECK:STDOUT:   %ptr.886: type = ptr_type %ptr.2b9 [symbolic]
// CHECK:STDOUT:   %require_complete.67e: <witness> = require_complete_type %ptr.886 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.85e: <witness> = lookup_impl_witness %ptr.2b9, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.b61: %Destroy.type = facet_value %ptr.2b9, (%Destroy.lookup_impl_witness.85e) [symbolic]
// CHECK:STDOUT:   %.2e5: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.b61 [symbolic]
// CHECK:STDOUT:   %impl.elem0.ceb: %.2e5 = impl_witness_access %Destroy.lookup_impl_witness.85e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.aea: <specific function> = specific_impl_function %impl.elem0.ceb, @Destroy.Op(%Destroy.facet.b61) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Empty.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.impl_witness_table = impl_witness_table (), @C.as.Empty.impl [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness %Empty.impl_witness_table [concrete = constants.%Empty.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.639 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21: type = splice_block %Empty.ref [concrete = constants.%Empty.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_6.2: %Empty.type = bind_symbolic_name T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e3a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT: impl @C.as.Empty.impl: %C.ref as %Empty.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Empty.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.071]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc21_6.2: %Empty.type) {
// CHECK:STDOUT:   %T.loc21_6.1: %Empty.type = bind_symbolic_name T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc22_11.2: type = facet_access_type %T.loc21_6.1 [symbolic = %T.as_type.loc22_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %ptr.loc22_11.2: type = ptr_type %T.as_type.loc22_11.2 [symbolic = %ptr.loc22_11.2 (constants.%ptr.2b9)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %ptr.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.1f4)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc22_11.2 [symbolic = %pattern_type (constants.%pattern_type.ae3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc22_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.85e)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc22_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.b61)]
// CHECK:STDOUT:   %.loc22_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc22_3.2 (constants.%.2e5)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.2: @F.%.loc22_3.2 (%.2e5) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.ceb)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.2: <specific function> = specific_impl_function %impl.elem0.loc22_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.aea)]
// CHECK:STDOUT:   %ptr.loc22_3: type = ptr_type %ptr.loc22_11.2 [symbolic = %ptr.loc22_3 (constants.%ptr.886)]
// CHECK:STDOUT:   %require_complete.loc22_3: <witness> = require_complete_type %ptr.loc22_3 [symbolic = %require_complete.loc22_3 (constants.%require_complete.67e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: @F.%pattern_type (%pattern_type.ae3) = binding_pattern x [concrete]
// CHECK:STDOUT:       %x.var_patt: @F.%pattern_type (%pattern_type.ae3) = var_pattern %x.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref @F.%ptr.loc22_11.2 (%ptr.2b9) = var %x.var_patt
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %ptr.loc22_11.1 [symbolic = %ptr.loc22_11.2 (constants.%ptr.2b9)] {
// CHECK:STDOUT:       %T.ref: %Empty.type = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22_11.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc22_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_11.2: type = converted %T.ref, %T.as_type.loc22_11.1 [symbolic = %T.as_type.loc22_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %ptr.loc22_11.1: type = ptr_type %.loc22_11.2 [symbolic = %ptr.loc22_11.2 (constants.%ptr.2b9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @F.%ptr.loc22_11.2 (%ptr.2b9) = bind_name x, %x.var
// CHECK:STDOUT:     %impl.elem0.loc22_3.1: @F.%.loc22_3.2 (%.2e5) = impl_witness_access constants.%Destroy.lookup_impl_witness.85e, element0 [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.ceb)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %x.var, %impl.elem0.loc22_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.1: <specific function> = specific_impl_function %impl.elem0.loc22_3.1, @Destroy.Op(constants.%Destroy.facet.b61) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.aea)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %x.var, %specific_impl_fn.loc22_3.1
// CHECK:STDOUT:     %addr: @F.%ptr.loc22_3 (%ptr.886) = addr_of %x.var
// CHECK:STDOUT:     %.loc22_3.1: init %empty_tuple.type = call %bound_method.loc22_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc21_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
