// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/as_type_of_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/as_type_of_type.carbon

interface Empty {}

fn F(T:! Empty) {
  var x: T;
}

// CHECK:STDOUT: --- as_type_of_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.e3a: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.e3a: %Empty.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.639: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.e3a [symbolic]
// CHECK:STDOUT:   %require_complete.14a: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.ddf: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %ptr.2b9: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.1f4: <witness> = require_complete_type %ptr.2b9 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.e3a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T.as_type, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.605: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.605 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.Op(%Destroy.facet) [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.639 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17: type = splice_block %Empty.ref [concrete = constants.%Empty.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_6.2: %Empty.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_6.1 (constants.%T.e3a)]
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
// CHECK:STDOUT: generic fn @F(%T.loc17_6.2: %Empty.type) {
// CHECK:STDOUT:   %T.loc17_6.1: %Empty.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_6.1 (constants.%T.e3a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc18_10.2: type = facet_access_type %T.loc17_6.1 [symbolic = %T.as_type.loc18_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete.loc18_10: <witness> = require_complete_type %T.as_type.loc18_10.2 [symbolic = %require_complete.loc18_10 (constants.%require_complete.14a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc18_10.2 [symbolic = %pattern_type (constants.%pattern_type.ddf)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc17_6.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T.as_type.loc18_10.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc18_3.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc18_3.3 (constants.%.605)]
// CHECK:STDOUT:   %impl.elem0.loc18_3.2: @F.%.loc18_3.3 (%.605) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_3.2: <specific function> = specific_impl_function %impl.elem0.loc18_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc18_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc18_10.2 [symbolic = %ptr (constants.%ptr.2b9)]
// CHECK:STDOUT:   %require_complete.loc18_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc18_3 (constants.%require_complete.1f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: @F.%pattern_type (%pattern_type.ddf) = binding_pattern x [concrete]
// CHECK:STDOUT:       %x.var_patt: @F.%pattern_type (%pattern_type.ddf) = var_pattern %x.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref @F.%T.as_type.loc18_10.2 (%T.as_type) = var %x.var_patt
// CHECK:STDOUT:     %.loc18_10.1: type = splice_block %.loc18_10.2 [symbolic = %T.as_type.loc18_10.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Empty.type = name_ref T, %T.loc17_6.2 [symbolic = %T.loc17_6.1 (constants.%T.e3a)]
// CHECK:STDOUT:       %T.as_type.loc18_10.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc18_10.2: type = converted %T.ref, %T.as_type.loc18_10.1 [symbolic = %T.as_type.loc18_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @F.%T.as_type.loc18_10.2 (%T.as_type) = bind_name x, %x.var
// CHECK:STDOUT:     %T.as_type.loc18_3: type = facet_access_type constants.%T.e3a [symbolic = %T.as_type.loc18_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_3.1: type = converted constants.%T.e3a, %T.as_type.loc18_3 [symbolic = %T.as_type.loc18_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc18_3.1: @F.%.loc18_3.3 (%.605) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc18_3.1: <bound method> = bound_method %x.var, %impl.elem0.loc18_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_3.1: <specific function> = specific_impl_function %impl.elem0.loc18_3.1, @Destroy.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc18_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc18_3.2: <bound method> = bound_method %x.var, %specific_impl_fn.loc18_3.1
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.2b9) = addr_of %x.var
// CHECK:STDOUT:     %.loc18_3.2: init %empty_tuple.type = call %bound_method.loc18_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.e3a) {
// CHECK:STDOUT:   %T.loc17_6.1 => constants.%T.e3a
// CHECK:STDOUT: }
// CHECK:STDOUT:
