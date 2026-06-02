// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init_as.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn F() -> i32 {
  //@dump-sem-ir-begin
  return ({.a = 1, .b = 2} as Class).a;
  //@dump-sem-ir-end
}

fn G() -> i32 {
  //@dump-sem-ir-begin
  var v: Class = {.a = 1, .b = 2} as Class;
  //@dump-sem-ir-end
  return v.a;
}

// CHECK:STDOUT: --- init_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b47: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e5b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.7d5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%int_1.d9c, %int_2.228) [concrete]
// CHECK:STDOUT:   %.685: ref %Class = temporary invalid, %Class.val [concrete]
// CHECK:STDOUT:   %.557: ref %i32 = class_element_access %.685, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.415: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc20_28.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.685, %Destroy.Op.2dd605.4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc20_26.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc20_26.1: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc20_26.1: <bound method> = bound_method %int_1, %impl.elem0.loc20_26.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc20_26.1: <specific function> = specific_function %impl.elem0.loc20_26.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_26.2: <bound method> = bound_method %int_1, %specific_fn.loc20_26.1 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.1: init %i32 = call %bound_method.loc20_26.2(%int_1) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc20_26.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc20_26.3: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc20_26.4: ref %i32 = class_element_access %.loc20_26.3, element0
// CHECK:STDOUT:   %.loc20_26.5: init %i32 to %.loc20_26.4 = in_place_init %.loc20_26.2 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %impl.elem0.loc20_26.2: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc20_26.3: <bound method> = bound_method %int_2, %impl.elem0.loc20_26.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:   %specific_fn.loc20_26.2: <specific function> = specific_function %impl.elem0.loc20_26.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_26.4: <bound method> = bound_method %int_2, %specific_fn.loc20_26.2 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.2: init %i32 = call %bound_method.loc20_26.4(%int_2) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc20_26.6: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20_26.2 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc20_26.7: ref %i32 = class_element_access %.loc20_26.3, element1
// CHECK:STDOUT:   %.loc20_26.8: init %i32 to %.loc20_26.7 = in_place_init %.loc20_26.6 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc20_26.9: init %Class to %.loc20_26.3 = class_init (%.loc20_26.5, %.loc20_26.8) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc20_28.1: init %Class = converted %.loc20_26.1, %.loc20_26.9 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc20_28.2: ref %Class = temporary %.loc20_26.3, %.loc20_28.1 [concrete = constants.%.685]
// CHECK:STDOUT:   %a.ref: %Class.elem = name_ref a, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:   %.loc20_37.1: ref %i32 = class_element_access %.loc20_28.2, element0 [concrete = constants.%.557]
// CHECK:STDOUT:   %.loc20_37.2: %i32 = acquire_value %.loc20_37.1
// CHECK:STDOUT:   %impl.elem0.loc20_37: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc20_37.1: <bound method> = bound_method %.loc20_37.2, %impl.elem0.loc20_37
// CHECK:STDOUT:   %specific_fn.loc20_37: <specific function> = specific_function %impl.elem0.loc20_37, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_37.2: <bound method> = bound_method %.loc20_37.2, %specific_fn.loc20_37
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc20_37.2(%.loc20_37.2)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.685)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_28.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_28.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_28.3(%self.param: ref %struct_type.a.b.b47) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_28.4(%self.param: ref %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.415 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.415 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class = var %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc26_33.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %Class.ref.loc26_38: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc26_33.1: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc26_33.1: <bound method> = bound_method %int_1, %impl.elem0.loc26_33.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc26_33.1: <specific function> = specific_function %impl.elem0.loc26_33.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_33.2: <bound method> = bound_method %int_1, %specific_fn.loc26_33.1 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.1: init %i32 = call %bound_method.loc26_33.2(%int_1) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc26_33.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc26_3: ref %Class = splice_block %v.var {}
// CHECK:STDOUT:   %.loc26_33.3: ref %i32 = class_element_access %.loc26_3, element0
// CHECK:STDOUT:   %.loc26_33.4: init %i32 to %.loc26_33.3 = in_place_init %.loc26_33.2 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %impl.elem0.loc26_33.2: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc26_33.3: <bound method> = bound_method %int_2, %impl.elem0.loc26_33.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:   %specific_fn.loc26_33.2: <specific function> = specific_function %impl.elem0.loc26_33.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_33.4: <bound method> = bound_method %int_2, %specific_fn.loc26_33.2 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.2: init %i32 = call %bound_method.loc26_33.4(%int_2) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc26_33.5: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_33.2 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc26_33.6: ref %i32 = class_element_access %.loc26_3, element1
// CHECK:STDOUT:   %.loc26_33.7: init %i32 to %.loc26_33.6 = in_place_init %.loc26_33.5 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc26_33.8: init %Class to %.loc26_3 = class_init (%.loc26_33.4, %.loc26_33.7) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc26_35: init %Class = converted %.loc26_33.1, %.loc26_33.8 [concrete = constants.%Class.val]
// CHECK:STDOUT:   assign %v.var, %.loc26_35
// CHECK:STDOUT:   %Class.ref.loc26_10: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %v: ref %Class = ref_binding v, %v.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.2dd605.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
