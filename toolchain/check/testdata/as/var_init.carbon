// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/var_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/var_init.carbon

// --- var_init.carbon

library "[[@TEST_NAME]]";

class X {
  impl () as Core.ImplicitAs(X) {
    fn Convert(unused self: ()) ->  X { return {}; }
  }
}

fn Convert(unused t: ()) {
  //@dump-sem-ir-begin
  var unused x: X = ();
  //@dump-sem-ir-end
}


// CHECK:STDOUT: --- var_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3d3: type = facet_type <@ImplicitAs, @ImplicitAs(%X)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @empty_tuple.type.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert: %empty_tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3d3 = facet_value %empty_tuple.type, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b95: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%X, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6cb = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.6cb = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %.8a9: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b95, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %empty_tuple, %empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fc5: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e12: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fc5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %X, (%DestroyImpl.impl_witness.e12, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%t.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %X = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc12_22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %impl.elem0: %.8a9 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_22.1, %impl.elem0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %.loc12_3.1: ref %X = splice_block %x.var {}
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_22.2: %empty_tuple.type = converted %.loc12_22.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.call: init %X to %.loc12_3.1 = call %bound_method(%.loc12_22.2)
// CHECK:STDOUT:   %.loc12_3.2: init %X = converted %.loc12_22.1, %empty_tuple.type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   assign %x.var, %.loc12_3.2
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %x: ref %X = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6cb = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6cb = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.e12, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.3: %Destroy.type = converted constants.%X, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.e12, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_3.2(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
