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
    var n: (); } else { var n: ();
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.1dc: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.2, @impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_tuple.type, (%Destroy.impl_witness.1dc) [concrete]
// CHECK:STDOUT:   %.346: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.ea3, @Op.2(%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_9: %pattern_type.cb1 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc18_5: %pattern_type.cb1 = var_pattern %n.patt.loc18_9 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc18_5: ref %empty_tuple.type = var %n.var_patt.loc18_5
// CHECK:STDOUT:   %.loc18_13.1: type = splice_block %.loc18_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_13.3: type = converted %.loc18_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_9: ref %empty_tuple.type = bind_name n, %n.var.loc18_5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_29: %pattern_type.cb1 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt.loc18_25: %pattern_type.cb1 = var_pattern %n.patt.loc18_29 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc18_25: ref %empty_tuple.type = var %n.var_patt.loc18_25
// CHECK:STDOUT:   %.loc18_33.1: type = splice_block %.loc18_33.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_33.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_33.3: type = converted %.loc18_33.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_29: ref %empty_tuple.type = bind_name n, %n.var.loc18_25
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   %impl.elem0.loc18_25: %.346 = impl_witness_access constants.%Destroy.impl_witness.1dc, element0 [concrete = constants.%Op.ea3]
// CHECK:STDOUT:   %bound_method.loc18_25.1: <bound method> = bound_method %n.var.loc18_25, %impl.elem0.loc18_25
// CHECK:STDOUT:   %specific_fn.loc18_25: <specific function> = specific_function %impl.elem0.loc18_25, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_25.2: <bound method> = bound_method %n.var.loc18_25, %specific_fn.loc18_25
// CHECK:STDOUT:   %addr.loc18_25: %ptr.843 = addr_of %n.var.loc18_25
// CHECK:STDOUT:   %no_op.loc18_25: init %empty_tuple.type = call %bound_method.loc18_25.2(%addr.loc18_25)
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.346 = impl_witness_access constants.%Destroy.impl_witness.1dc, element0 [concrete = constants.%Op.ea3]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %n.var.loc18_5, %impl.elem0.loc18_5
// CHECK:STDOUT:   %specific_fn.loc18_5: <specific function> = specific_function %impl.elem0.loc18_5, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %n.var.loc18_5, %specific_fn.loc18_5
// CHECK:STDOUT:   %addr.loc18_5: %ptr.843 = addr_of %n.var.loc18_5
// CHECK:STDOUT:   %no_op.loc18_5: init %empty_tuple.type = call %bound_method.loc18_5.2(%addr.loc18_5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
