// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/duplicate_name_same_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/duplicate_name_same_line.carbon

fn A() {
  if (true) {
    // This declares `n` in two different scopes, it's just showing the IR
    // behavior of having both on the same line.
    //@dump-sem-ir-begin
    var unused n: () = (); } else { var unused n: () = ();
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %n.patt.785a0d.1: %pattern_type.cb1 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt.961036.1: %pattern_type.cb1 = var_pattern %n.patt.785a0d.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %empty_tuple.type, (%DestroyImpl.impl_witness.674, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %n.patt.785a0d.2: %pattern_type.cb1 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt.961036.2: %pattern_type.cb1 = var_pattern %n.patt.785a0d.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n.var.loc18_5: ref %empty_tuple.type = var_storage %n.var_patt.loc18_5
// CHECK:STDOUT:   %.loc18_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_25.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_5.1: init %empty_tuple.type = converted %.loc18_25.1, %.loc18_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %n.var.loc18_5, %.loc18_5.1
// CHECK:STDOUT:   %.loc18_20.1: type = splice_block %.loc18_20.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_20.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_20.3: type = converted %.loc18_20.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_17: ref %empty_tuple.type = wrapper_binding n, %n.var.loc18_5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_17: %pattern_type.cb1 = ref_binding_pattern n [concrete = constants.%n.patt.785a0d.1]
// CHECK:STDOUT:     %n.var_patt.loc18_5: %pattern_type.cb1 = var_pattern %n.patt.loc18_17 [concrete = constants.%n.var_patt.961036.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc18_5.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc18_5.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc18_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc18_5.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc18_5.3: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc18_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc18_5: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%n.var.loc18_5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %n.var.loc18_37: ref %empty_tuple.type = var_storage %n.var_patt.loc18_37
// CHECK:STDOUT:   %.loc18_57.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_57.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_37.1: init %empty_tuple.type = converted %.loc18_57.1, %.loc18_57.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %n.var.loc18_37, %.loc18_37.1
// CHECK:STDOUT:   %.loc18_52.1: type = splice_block %.loc18_52.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_52.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_52.3: type = converted %.loc18_52.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_49: ref %empty_tuple.type = wrapper_binding n, %n.var.loc18_37
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_49: %pattern_type.cb1 = ref_binding_pattern n [concrete = constants.%n.patt.785a0d.2]
// CHECK:STDOUT:     %n.var_patt.loc18_37: %pattern_type.cb1 = var_pattern %n.patt.loc18_49 [concrete = constants.%n.var_patt.961036.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc18_37.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc18_37.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc18_37.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc18_37.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc18_37.3: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc18_37.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc18_37: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%n.var.loc18_37)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
