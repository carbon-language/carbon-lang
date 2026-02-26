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
    fn Convert[unused self: ()]() ->  X { return {}; }
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
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.062: type = facet_type <@ImplicitAs, @ImplicitAs(%X)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @empty_tuple.type.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert: %empty_tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.062 = facet_value %empty_tuple.type, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.73b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%X, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1bd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.73b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %empty_tuple, %empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%t.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.05f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.05f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %X = var %x.var_patt
// CHECK:STDOUT:   %.loc12_22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %impl.elem0: %.1bd = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_22.1, %impl.elem0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %.loc12_3.1: ref %X = splice_block %x.var {}
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_22.2: %empty_tuple.type = converted %.loc12_22.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.call: init %X to %.loc12_3.1 = call %bound_method(%.loc12_22.2)
// CHECK:STDOUT:   %.loc12_3.2: init %X = converted %.loc12_22.1, %empty_tuple.type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   assign %x.var, %.loc12_3.2
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %x: ref %X = ref_binding x, %x.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %x.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
