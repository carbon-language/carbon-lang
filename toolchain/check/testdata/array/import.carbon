// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/import.carbon

// --- library.carbon

library "[[@TEST_NAME]]";

fn F() -> array(i32, 42);

// --- user.carbon

import library "library";

fn G(n: i32) -> i32 {
  //@dump-sem-ir-begin
  return F()[n];
  //@dump-sem-ir-end
}

// --- symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let value: array(i32, 1);
}

class C {}

impl C as I where .value = (1,) {}

// --- import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> i32 {
  //@dump-sem-ir-begin
  let a: array(i32, 1) = (C as I).value;
  //@dump-sem-ir-end
  return a[0];
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c07: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.705: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.70d: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt.705 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_12.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc6_12.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_12.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.087: %pattern_type.c07 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b22: %pattern_type.c07 = wrapper_binding_pattern self, %self.param_patt.087 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete]
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete]
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete]
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete]
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete]
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete]
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete]
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete]
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete]
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete]
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete]
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete]
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete]
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete]
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete]
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete]
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc6_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//library, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_12.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc6_12.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_12.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c07 = ref_param_pattern [concrete = constants.%self.param_patt.087]
// CHECK:STDOUT:     %self.patt: %pattern_type.c07 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %array_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %array_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc6_12.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c07 = ref_param_pattern [concrete = constants.%self.param_patt.087]
// CHECK:STDOUT:     %self.patt: %pattern_type.c07 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %array_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %array_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6_12.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call: init %array_type to %.loc6_12.1 = call %F.ref()
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc6_12.2: ref %array_type = temporary %.loc6_12.1, %F.call
// CHECK:STDOUT:   %.loc6_15.1: ref %i32 = array_index %.loc6_12.2, %n.ref
// CHECK:STDOUT:   %.loc6_15.2: %i32 = acquire_value %.loc6_15.1
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc6_15.1: <bound method> = bound_method %.loc6_15.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_15.2: <bound method> = bound_method %.loc6_15.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_15.2(%.loc6_15.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc6_12.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc6_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_12.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_12.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_12.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_12.2(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete = constants.%int_41]
// CHECK:STDOUT:   %.loc6_12.1: ref %i32 = array_index %self.param, %int_41
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.1: <bound method> = bound_method %.loc6_12.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.1(%.loc6_12.1)
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   %.loc6_12.2: ref %i32 = array_index %self.param, %int_40
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.2: <bound method> = bound_method %.loc6_12.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.2(%.loc6_12.2)
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete = constants.%int_39]
// CHECK:STDOUT:   %.loc6_12.3: ref %i32 = array_index %self.param, %int_39
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.3: <bound method> = bound_method %.loc6_12.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.3(%.loc6_12.3)
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete = constants.%int_38]
// CHECK:STDOUT:   %.loc6_12.4: ref %i32 = array_index %self.param, %int_38
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.4: <bound method> = bound_method %.loc6_12.4, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.4(%.loc6_12.4)
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete = constants.%int_37]
// CHECK:STDOUT:   %.loc6_12.5: ref %i32 = array_index %self.param, %int_37
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.5: <bound method> = bound_method %.loc6_12.5, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.5: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.5(%.loc6_12.5)
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete = constants.%int_36]
// CHECK:STDOUT:   %.loc6_12.6: ref %i32 = array_index %self.param, %int_36
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.6: <bound method> = bound_method %.loc6_12.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.6: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.6(%.loc6_12.6)
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35]
// CHECK:STDOUT:   %.loc6_12.7: ref %i32 = array_index %self.param, %int_35
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.7: <bound method> = bound_method %.loc6_12.7, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.7: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.7(%.loc6_12.7)
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34]
// CHECK:STDOUT:   %.loc6_12.8: ref %i32 = array_index %self.param, %int_34
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.8: <bound method> = bound_method %.loc6_12.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.8: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.8(%.loc6_12.8)
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete = constants.%int_33]
// CHECK:STDOUT:   %.loc6_12.9: ref %i32 = array_index %self.param, %int_33
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.9: <bound method> = bound_method %.loc6_12.9, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.9: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.9(%.loc6_12.9)
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc6_12.10: ref %i32 = array_index %self.param, %int_32
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.10: <bound method> = bound_method %.loc6_12.10, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.10: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.10(%.loc6_12.10)
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete = constants.%int_31]
// CHECK:STDOUT:   %.loc6_12.11: ref %i32 = array_index %self.param, %int_31
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.11: <bound method> = bound_method %.loc6_12.11, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.11: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.11(%.loc6_12.11)
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete = constants.%int_30]
// CHECK:STDOUT:   %.loc6_12.12: ref %i32 = array_index %self.param, %int_30
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.12: <bound method> = bound_method %.loc6_12.12, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.12: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.12(%.loc6_12.12)
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete = constants.%int_29]
// CHECK:STDOUT:   %.loc6_12.13: ref %i32 = array_index %self.param, %int_29
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.13: <bound method> = bound_method %.loc6_12.13, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.13: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.13(%.loc6_12.13)
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete = constants.%int_28]
// CHECK:STDOUT:   %.loc6_12.14: ref %i32 = array_index %self.param, %int_28
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.14: <bound method> = bound_method %.loc6_12.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.14: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.14(%.loc6_12.14)
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete = constants.%int_27]
// CHECK:STDOUT:   %.loc6_12.15: ref %i32 = array_index %self.param, %int_27
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.15: <bound method> = bound_method %.loc6_12.15, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.15: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.15(%.loc6_12.15)
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete = constants.%int_26]
// CHECK:STDOUT:   %.loc6_12.16: ref %i32 = array_index %self.param, %int_26
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.16: <bound method> = bound_method %.loc6_12.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.16: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.16(%.loc6_12.16)
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete = constants.%int_25]
// CHECK:STDOUT:   %.loc6_12.17: ref %i32 = array_index %self.param, %int_25
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.17: <bound method> = bound_method %.loc6_12.17, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.17: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.17(%.loc6_12.17)
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24]
// CHECK:STDOUT:   %.loc6_12.18: ref %i32 = array_index %self.param, %int_24
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.18: <bound method> = bound_method %.loc6_12.18, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.18: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.18(%.loc6_12.18)
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete = constants.%int_23]
// CHECK:STDOUT:   %.loc6_12.19: ref %i32 = array_index %self.param, %int_23
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.19: <bound method> = bound_method %.loc6_12.19, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.19(%.loc6_12.19)
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete = constants.%int_22]
// CHECK:STDOUT:   %.loc6_12.20: ref %i32 = array_index %self.param, %int_22
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.20: <bound method> = bound_method %.loc6_12.20, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.20: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.20(%.loc6_12.20)
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete = constants.%int_21]
// CHECK:STDOUT:   %.loc6_12.21: ref %i32 = array_index %self.param, %int_21
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.21: <bound method> = bound_method %.loc6_12.21, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.21: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.21(%.loc6_12.21)
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %.loc6_12.22: ref %i32 = array_index %self.param, %int_20
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.22: <bound method> = bound_method %.loc6_12.22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.22(%.loc6_12.22)
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete = constants.%int_19]
// CHECK:STDOUT:   %.loc6_12.23: ref %i32 = array_index %self.param, %int_19
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.23: <bound method> = bound_method %.loc6_12.23, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.23: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.23(%.loc6_12.23)
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete = constants.%int_18]
// CHECK:STDOUT:   %.loc6_12.24: ref %i32 = array_index %self.param, %int_18
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.24: <bound method> = bound_method %.loc6_12.24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.24(%.loc6_12.24)
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete = constants.%int_17]
// CHECK:STDOUT:   %.loc6_12.25: ref %i32 = array_index %self.param, %int_17
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.25: <bound method> = bound_method %.loc6_12.25, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.25: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.25(%.loc6_12.25)
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %.loc6_12.26: ref %i32 = array_index %self.param, %int_16
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.26: <bound method> = bound_method %.loc6_12.26, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.26: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.26(%.loc6_12.26)
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15]
// CHECK:STDOUT:   %.loc6_12.27: ref %i32 = array_index %self.param, %int_15
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.27: <bound method> = bound_method %.loc6_12.27, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.27: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.27(%.loc6_12.27)
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete = constants.%int_14]
// CHECK:STDOUT:   %.loc6_12.28: ref %i32 = array_index %self.param, %int_14
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.28: <bound method> = bound_method %.loc6_12.28, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.28: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.28(%.loc6_12.28)
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete = constants.%int_13]
// CHECK:STDOUT:   %.loc6_12.29: ref %i32 = array_index %self.param, %int_13
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.29: <bound method> = bound_method %.loc6_12.29, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.29: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.29(%.loc6_12.29)
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc6_12.30: ref %i32 = array_index %self.param, %int_12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.30: <bound method> = bound_method %.loc6_12.30, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.30: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.30(%.loc6_12.30)
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %.loc6_12.31: ref %i32 = array_index %self.param, %int_11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.31: <bound method> = bound_method %.loc6_12.31, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.31: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.31(%.loc6_12.31)
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc6_12.32: ref %i32 = array_index %self.param, %int_10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.32: <bound method> = bound_method %.loc6_12.32, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.32: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.32(%.loc6_12.32)
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc6_12.33: ref %i32 = array_index %self.param, %int_9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.33: <bound method> = bound_method %.loc6_12.33, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.33: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.33(%.loc6_12.33)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc6_12.34: ref %i32 = array_index %self.param, %int_8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.34: <bound method> = bound_method %.loc6_12.34, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.34: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.34(%.loc6_12.34)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc6_12.35: ref %i32 = array_index %self.param, %int_7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.35: <bound method> = bound_method %.loc6_12.35, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.35: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.35(%.loc6_12.35)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc6_12.36: ref %i32 = array_index %self.param, %int_6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.36: <bound method> = bound_method %.loc6_12.36, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.36: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.36(%.loc6_12.36)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc6_12.37: ref %i32 = array_index %self.param, %int_5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.37: <bound method> = bound_method %.loc6_12.37, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.37: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.37(%.loc6_12.37)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc6_12.38: ref %i32 = array_index %self.param, %int_4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.38: <bound method> = bound_method %.loc6_12.38, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.38: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.38(%.loc6_12.38)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc6_12.39: ref %i32 = array_index %self.param, %int_3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.39: <bound method> = bound_method %.loc6_12.39, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.39: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.39(%.loc6_12.39)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc6_12.40: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.40: <bound method> = bound_method %.loc6_12.40, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.40: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.40(%.loc6_12.40)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc6_12.41: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.41: <bound method> = bound_method %.loc6_12.41, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.41: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.41(%.loc6_12.41)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc6_12.42: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_12.42: <bound method> = bound_method %.loc6_12.42, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_12.42: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_12.42(%.loc6_12.42)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_12.2(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_12.2(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.97e: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.97e = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.524: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.98e [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.0c6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.3ab: %I.assoc_type = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = constants.%assoc0.524]
// CHECK:STDOUT:   %Main.import_ref.c09: %array_type = import_ref Main//symbolic_decl, loc10_33, loaded [concrete = constants.%array]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.c09), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.98e: %array_type = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %array_type = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_29: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_29 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_34: type = converted %.loc7_29, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.3ab [concrete = constants.%assoc0.524]
// CHECK:STDOUT:   %impl.elem0.loc7: %array_type = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc7_22: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32.loc7 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %array_type = wrapper_binding a, %impl.elem0.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.97e = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
