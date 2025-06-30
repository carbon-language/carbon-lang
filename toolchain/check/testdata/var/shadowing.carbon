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

  var x: () = ();
  if (true) {
    var x: () = ();

    // TODO: We should reject this use of the shadowed variable `x`.
    x = ();
  }
}

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl [concrete]
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
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %NS.patt: %pattern_type.cb1 = binding_pattern NS [concrete]
// CHECK:STDOUT:     %NS.var_patt: %pattern_type.cb1 = var_pattern %NS.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.var: ref %empty_tuple.type = var %NS.var_patt
// CHECK:STDOUT:   %.loc5_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_17.2: init %empty_tuple.type = tuple_init () to %NS.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_3: init %empty_tuple.type = converted %.loc5_17.1, %.loc5_17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.var, %.loc5_3
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: ref %empty_tuple.type = bind_name NS, %NS.var
// CHECK:STDOUT:   %NS.ref: ref %empty_tuple.type = name_ref NS, %NS
// CHECK:STDOUT:   %.loc6_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: init %empty_tuple.type = tuple_init () to %NS.ref [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_6: init %empty_tuple.type = converted %.loc6_9.1, %.loc6_9.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.ref, %.loc6_6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc8: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc8: %pattern_type.cb1 = var_pattern %x.patt.loc8 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc8: ref %empty_tuple.type = var %x.var_patt.loc8
// CHECK:STDOUT:   %.loc8_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc8 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_3: init %empty_tuple.type = converted %.loc8_16.1, %.loc8_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc8, %.loc8_3
// CHECK:STDOUT:   %.loc8_11.1: type = splice_block %.loc8_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_11.3: type = converted %.loc8_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc8: ref %empty_tuple.type = bind_name x, %x.var.loc8
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc10: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc10: %pattern_type.cb1 = var_pattern %x.patt.loc10 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc10: ref %empty_tuple.type = var %x.var_patt.loc10
// CHECK:STDOUT:   %.loc10_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_18.2: init %empty_tuple.type = tuple_init () to %x.var.loc10 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_5: init %empty_tuple.type = converted %.loc10_18.1, %.loc10_18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc10, %.loc10_5
// CHECK:STDOUT:   %.loc10_13.1: type = splice_block %.loc10_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_13.3: type = converted %.loc10_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc10: ref %empty_tuple.type = bind_name x, %x.var.loc10
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, %x.loc10
// CHECK:STDOUT:   %.loc13_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_10.2: init %empty_tuple.type = tuple_init () to %x.ref [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_7: init %empty_tuple.type = converted %.loc13_10.1, %.loc13_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.ref, %.loc13_7
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %impl.elem0.loc10: %.346 = impl_witness_access constants.%Destroy.impl_witness.1dc, element0 [concrete = constants.%Op.ea3]
// CHECK:STDOUT:   %bound_method.loc10_5.1: <bound method> = bound_method %x.var.loc10, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_5.2: <bound method> = bound_method %x.var.loc10, %specific_fn.loc10
// CHECK:STDOUT:   %addr.loc10: %ptr.843 = addr_of %x.var.loc10
// CHECK:STDOUT:   %no_op.loc10: init %empty_tuple.type = call %bound_method.loc10_5.2(%addr.loc10)
// CHECK:STDOUT:   %impl.elem0.loc8: %.346 = impl_witness_access constants.%Destroy.impl_witness.1dc, element0 [concrete = constants.%Op.ea3]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %x.var.loc8, %impl.elem0.loc8
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %x.var.loc8, %specific_fn.loc8
// CHECK:STDOUT:   %addr.loc8: %ptr.843 = addr_of %x.var.loc8
// CHECK:STDOUT:   %no_op.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%addr.loc8)
// CHECK:STDOUT:   %impl.elem0.loc5: %.346 = impl_witness_access constants.%Destroy.impl_witness.1dc, element0 [concrete = constants.%Op.ea3]
// CHECK:STDOUT:   %bound_method.loc5_3.1: <bound method> = bound_method %NS.var, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_3.2: <bound method> = bound_method %NS.var, %specific_fn.loc5
// CHECK:STDOUT:   %addr.loc5: %ptr.843 = addr_of %NS.var
// CHECK:STDOUT:   %no_op.loc5: init %empty_tuple.type = call %bound_method.loc5_3.2(%addr.loc5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
