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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/shadowing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/shadowing.carbon

// --- basics.carbon

namespace NS;

fn Main() {
  var NS: () = ();
  NS = ();

  var unused x: () = ();
  if (true) {
    var x: () = ();

    // TODO: We should reject this use of the shadowed variable `x`.
    x = ();
  }
}

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %NS.patt: %pattern_type.cb1 = ref_binding_pattern NS [concrete]
// CHECK:STDOUT:   %NS.var_patt: %pattern_type.cb1 = var_pattern %NS.patt [concrete]
// CHECK:STDOUT:   %x.patt.06838d.1: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt.28d573.1: %pattern_type.cb1 = var_pattern %x.patt.06838d.1 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %x.patt.06838d.2: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt.28d573.2: %pattern_type.cb1 = var_pattern %x.patt.06838d.2 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %empty_tuple.type, (%DestroyImpl.impl_witness.674, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.var: ref %empty_tuple.type = var_storage %NS.var_patt
// CHECK:STDOUT:   %.loc5_17.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_17.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_3.1: init %empty_tuple.type = converted %.loc5_17.1, %.loc5_17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.var, %.loc5_3.1
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: ref %empty_tuple.type = wrapper_binding NS, %NS.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %NS.patt: %pattern_type.cb1 = ref_binding_pattern NS [concrete = constants.%NS.patt]
// CHECK:STDOUT:     %NS.var_patt: %pattern_type.cb1 = var_pattern %NS.patt [concrete = constants.%NS.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.ref: ref %empty_tuple.type = name_ref NS, %NS
// CHECK:STDOUT:   %.loc6_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_9.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_6: init %empty_tuple.type = converted %.loc6_9.1, %.loc6_9.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.ref, %.loc6_6
// CHECK:STDOUT:   %x.var.loc8: ref %empty_tuple.type = var_storage %x.var_patt.loc8
// CHECK:STDOUT:   %.loc8_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_3.1: init %empty_tuple.type = converted %.loc8_23.1, %.loc8_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc8, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_18.1: type = splice_block %.loc8_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_18.3: type = converted %.loc8_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc8: ref %empty_tuple.type = wrapper_binding x, %x.var.loc8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc8: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt.06838d.1]
// CHECK:STDOUT:     %x.var_patt.loc8: %pattern_type.cb1 = var_pattern %x.patt.loc8 [concrete = constants.%x.var_patt.28d573.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %x.var.loc10: ref %empty_tuple.type = var_storage %x.var_patt.loc10
// CHECK:STDOUT:   %.loc10_18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_18.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_5.1: init %empty_tuple.type = converted %.loc10_18.1, %.loc10_18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc10, %.loc10_5.1
// CHECK:STDOUT:   %.loc10_13.1: type = splice_block %.loc10_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_13.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_13.3: type = converted %.loc10_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc10: ref %empty_tuple.type = wrapper_binding x, %x.var.loc10
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc10: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt.06838d.2]
// CHECK:STDOUT:     %x.var_patt.loc10: %pattern_type.cb1 = var_pattern %x.patt.loc10 [concrete = constants.%x.var_patt.28d573.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, %x.loc10
// CHECK:STDOUT:   %.loc13_10.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_10.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_7: init %empty_tuple.type = converted %.loc13_10.1, %.loc13_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.ref, %.loc13_7
// CHECK:STDOUT:   %facet_value.loc10_5.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_5.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc10_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc10_5.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_5.3: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc10_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%x.var.loc10)
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %facet_value.loc8_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc8_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc8_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.3: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc8_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%x.var.loc8)
// CHECK:STDOUT:   %facet_value.loc5_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc5_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc5_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc5_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc5_3.3: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc5_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.loc5: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%NS.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
