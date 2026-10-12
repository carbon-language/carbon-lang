// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
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
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Array.2cc: type = class_type @Array, @Array(%i32, %int_42) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.loc6_12.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc6_12.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_12.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_12.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete]
// CHECK:STDOUT:   %facet_value.911: %facet_type = facet_value %i32, (%custom_witness.2cb555.1, %custom_witness.b80ef6.1) [concrete]
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
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.7e2: %facet_type = facet_value %Array.2cc, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//library, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6_12.1: ref %Array.2cc = temporary_storage
// CHECK:STDOUT:   %F.call: init %Array.2cc to %.loc6_12.1 = call %F.ref()
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc6_12.2: ref %Array.2cc = temporary %.loc6_12.1, %F.call
// CHECK:STDOUT:   %.loc6_15.1: ref %i32 = array_index %.loc6_12.2, %n.ref
// CHECK:STDOUT:   %.loc6_15.2: %i32 = acquire_value %.loc6_15.1
// CHECK:STDOUT:   %impl.elem0: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc6_15.1: <bound method> = bound_method %.loc6_15.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_15.2: <bound method> = bound_method %.loc6_15.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_15.2(%.loc6_15.2)
// CHECK:STDOUT:   %facet_value.loc6_12.1: %facet_type = facet_value constants.%Array.2cc, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.7e2]
// CHECK:STDOUT:   %.loc6_12.3: %facet_type = converted constants.%Array.2cc, %facet_value.loc6_12.1 [concrete = constants.%facet_value.7e2]
// CHECK:STDOUT:   %facet_value.loc6_12.2: %facet_type = facet_value constants.%Array.2cc, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.7e2]
// CHECK:STDOUT:   %.loc6_12.4: %facet_type = converted constants.%Array.2cc, %facet_value.loc6_12.2 [concrete = constants.%facet_value.7e2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc6_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc6_12.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc6_12.2(%self.param: ref %Array.2cc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_12.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_12.2(%self.param: ref %Array.2cc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete = constants.%int_41]
// CHECK:STDOUT:   %.loc6_12.1: ref %i32 = array_index %self.param, %int_41
// CHECK:STDOUT:   %facet_value.loc6_12.1: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.2: %facet_type = converted constants.%i32, %facet_value.loc6_12.1 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.2: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.3: %facet_type = converted constants.%i32, %facet_value.loc6_12.2 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc6_12.1)
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   %.loc6_12.4: ref %i32 = array_index %self.param, %int_40
// CHECK:STDOUT:   %facet_value.loc6_12.3: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.5: %facet_type = converted constants.%i32, %facet_value.loc6_12.3 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.4: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.6: %facet_type = converted constants.%i32, %facet_value.loc6_12.4 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc6_12.4)
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete = constants.%int_39]
// CHECK:STDOUT:   %.loc6_12.7: ref %i32 = array_index %self.param, %int_39
// CHECK:STDOUT:   %facet_value.loc6_12.5: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.8: %facet_type = converted constants.%i32, %facet_value.loc6_12.5 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.6: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.9: %facet_type = converted constants.%i32, %facet_value.loc6_12.6 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc6_12.7)
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete = constants.%int_38]
// CHECK:STDOUT:   %.loc6_12.10: ref %i32 = array_index %self.param, %int_38
// CHECK:STDOUT:   %facet_value.loc6_12.7: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.11: %facet_type = converted constants.%i32, %facet_value.loc6_12.7 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.8: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.12: %facet_type = converted constants.%i32, %facet_value.loc6_12.8 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc6_12.10)
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete = constants.%int_37]
// CHECK:STDOUT:   %.loc6_12.13: ref %i32 = array_index %self.param, %int_37
// CHECK:STDOUT:   %facet_value.loc6_12.9: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.14: %facet_type = converted constants.%i32, %facet_value.loc6_12.9 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.10: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.15: %facet_type = converted constants.%i32, %facet_value.loc6_12.10 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.5: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc6_12.13)
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete = constants.%int_36]
// CHECK:STDOUT:   %.loc6_12.16: ref %i32 = array_index %self.param, %int_36
// CHECK:STDOUT:   %facet_value.loc6_12.11: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.17: %facet_type = converted constants.%i32, %facet_value.loc6_12.11 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.12: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.18: %facet_type = converted constants.%i32, %facet_value.loc6_12.12 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.6: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc6_12.16)
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35]
// CHECK:STDOUT:   %.loc6_12.19: ref %i32 = array_index %self.param, %int_35
// CHECK:STDOUT:   %facet_value.loc6_12.13: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.20: %facet_type = converted constants.%i32, %facet_value.loc6_12.13 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.14: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.21: %facet_type = converted constants.%i32, %facet_value.loc6_12.14 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.7: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc6_12.19)
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34]
// CHECK:STDOUT:   %.loc6_12.22: ref %i32 = array_index %self.param, %int_34
// CHECK:STDOUT:   %facet_value.loc6_12.15: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.23: %facet_type = converted constants.%i32, %facet_value.loc6_12.15 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.16: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.24: %facet_type = converted constants.%i32, %facet_value.loc6_12.16 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.8: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%.loc6_12.22)
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete = constants.%int_33]
// CHECK:STDOUT:   %.loc6_12.25: ref %i32 = array_index %self.param, %int_33
// CHECK:STDOUT:   %facet_value.loc6_12.17: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.26: %facet_type = converted constants.%i32, %facet_value.loc6_12.17 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.18: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.27: %facet_type = converted constants.%i32, %facet_value.loc6_12.18 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.9: init %empty_tuple.type = call %SelfDestruct.specific_fn.9(%.loc6_12.25)
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc6_12.28: ref %i32 = array_index %self.param, %int_32
// CHECK:STDOUT:   %facet_value.loc6_12.19: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.29: %facet_type = converted constants.%i32, %facet_value.loc6_12.19 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.20: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.30: %facet_type = converted constants.%i32, %facet_value.loc6_12.20 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.10: init %empty_tuple.type = call %SelfDestruct.specific_fn.10(%.loc6_12.28)
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete = constants.%int_31]
// CHECK:STDOUT:   %.loc6_12.31: ref %i32 = array_index %self.param, %int_31
// CHECK:STDOUT:   %facet_value.loc6_12.21: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.32: %facet_type = converted constants.%i32, %facet_value.loc6_12.21 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.22: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.33: %facet_type = converted constants.%i32, %facet_value.loc6_12.22 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.11: init %empty_tuple.type = call %SelfDestruct.specific_fn.11(%.loc6_12.31)
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete = constants.%int_30]
// CHECK:STDOUT:   %.loc6_12.34: ref %i32 = array_index %self.param, %int_30
// CHECK:STDOUT:   %facet_value.loc6_12.23: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.35: %facet_type = converted constants.%i32, %facet_value.loc6_12.23 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.24: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.36: %facet_type = converted constants.%i32, %facet_value.loc6_12.24 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.12: init %empty_tuple.type = call %SelfDestruct.specific_fn.12(%.loc6_12.34)
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete = constants.%int_29]
// CHECK:STDOUT:   %.loc6_12.37: ref %i32 = array_index %self.param, %int_29
// CHECK:STDOUT:   %facet_value.loc6_12.25: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.38: %facet_type = converted constants.%i32, %facet_value.loc6_12.25 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.26: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.39: %facet_type = converted constants.%i32, %facet_value.loc6_12.26 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.13: init %empty_tuple.type = call %SelfDestruct.specific_fn.13(%.loc6_12.37)
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete = constants.%int_28]
// CHECK:STDOUT:   %.loc6_12.40: ref %i32 = array_index %self.param, %int_28
// CHECK:STDOUT:   %facet_value.loc6_12.27: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.41: %facet_type = converted constants.%i32, %facet_value.loc6_12.27 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.28: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.42: %facet_type = converted constants.%i32, %facet_value.loc6_12.28 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.14: init %empty_tuple.type = call %SelfDestruct.specific_fn.14(%.loc6_12.40)
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete = constants.%int_27]
// CHECK:STDOUT:   %.loc6_12.43: ref %i32 = array_index %self.param, %int_27
// CHECK:STDOUT:   %facet_value.loc6_12.29: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.44: %facet_type = converted constants.%i32, %facet_value.loc6_12.29 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.30: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.45: %facet_type = converted constants.%i32, %facet_value.loc6_12.30 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.15: init %empty_tuple.type = call %SelfDestruct.specific_fn.15(%.loc6_12.43)
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete = constants.%int_26]
// CHECK:STDOUT:   %.loc6_12.46: ref %i32 = array_index %self.param, %int_26
// CHECK:STDOUT:   %facet_value.loc6_12.31: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.47: %facet_type = converted constants.%i32, %facet_value.loc6_12.31 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.32: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.48: %facet_type = converted constants.%i32, %facet_value.loc6_12.32 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.16: init %empty_tuple.type = call %SelfDestruct.specific_fn.16(%.loc6_12.46)
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete = constants.%int_25]
// CHECK:STDOUT:   %.loc6_12.49: ref %i32 = array_index %self.param, %int_25
// CHECK:STDOUT:   %facet_value.loc6_12.33: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.50: %facet_type = converted constants.%i32, %facet_value.loc6_12.33 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.34: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.51: %facet_type = converted constants.%i32, %facet_value.loc6_12.34 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.17: init %empty_tuple.type = call %SelfDestruct.specific_fn.17(%.loc6_12.49)
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24]
// CHECK:STDOUT:   %.loc6_12.52: ref %i32 = array_index %self.param, %int_24
// CHECK:STDOUT:   %facet_value.loc6_12.35: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.53: %facet_type = converted constants.%i32, %facet_value.loc6_12.35 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.36: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.54: %facet_type = converted constants.%i32, %facet_value.loc6_12.36 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.18: init %empty_tuple.type = call %SelfDestruct.specific_fn.18(%.loc6_12.52)
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete = constants.%int_23]
// CHECK:STDOUT:   %.loc6_12.55: ref %i32 = array_index %self.param, %int_23
// CHECK:STDOUT:   %facet_value.loc6_12.37: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.56: %facet_type = converted constants.%i32, %facet_value.loc6_12.37 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.38: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.57: %facet_type = converted constants.%i32, %facet_value.loc6_12.38 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.19: init %empty_tuple.type = call %SelfDestruct.specific_fn.19(%.loc6_12.55)
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete = constants.%int_22]
// CHECK:STDOUT:   %.loc6_12.58: ref %i32 = array_index %self.param, %int_22
// CHECK:STDOUT:   %facet_value.loc6_12.39: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.59: %facet_type = converted constants.%i32, %facet_value.loc6_12.39 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.40: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.60: %facet_type = converted constants.%i32, %facet_value.loc6_12.40 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.20: init %empty_tuple.type = call %SelfDestruct.specific_fn.20(%.loc6_12.58)
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete = constants.%int_21]
// CHECK:STDOUT:   %.loc6_12.61: ref %i32 = array_index %self.param, %int_21
// CHECK:STDOUT:   %facet_value.loc6_12.41: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.62: %facet_type = converted constants.%i32, %facet_value.loc6_12.41 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.42: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.63: %facet_type = converted constants.%i32, %facet_value.loc6_12.42 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.21: init %empty_tuple.type = call %SelfDestruct.specific_fn.21(%.loc6_12.61)
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %.loc6_12.64: ref %i32 = array_index %self.param, %int_20
// CHECK:STDOUT:   %facet_value.loc6_12.43: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.65: %facet_type = converted constants.%i32, %facet_value.loc6_12.43 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.44: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.66: %facet_type = converted constants.%i32, %facet_value.loc6_12.44 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.22: init %empty_tuple.type = call %SelfDestruct.specific_fn.22(%.loc6_12.64)
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete = constants.%int_19]
// CHECK:STDOUT:   %.loc6_12.67: ref %i32 = array_index %self.param, %int_19
// CHECK:STDOUT:   %facet_value.loc6_12.45: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.68: %facet_type = converted constants.%i32, %facet_value.loc6_12.45 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.46: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.69: %facet_type = converted constants.%i32, %facet_value.loc6_12.46 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.23: init %empty_tuple.type = call %SelfDestruct.specific_fn.23(%.loc6_12.67)
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete = constants.%int_18]
// CHECK:STDOUT:   %.loc6_12.70: ref %i32 = array_index %self.param, %int_18
// CHECK:STDOUT:   %facet_value.loc6_12.47: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.71: %facet_type = converted constants.%i32, %facet_value.loc6_12.47 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.48: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.72: %facet_type = converted constants.%i32, %facet_value.loc6_12.48 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.24: init %empty_tuple.type = call %SelfDestruct.specific_fn.24(%.loc6_12.70)
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete = constants.%int_17]
// CHECK:STDOUT:   %.loc6_12.73: ref %i32 = array_index %self.param, %int_17
// CHECK:STDOUT:   %facet_value.loc6_12.49: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.74: %facet_type = converted constants.%i32, %facet_value.loc6_12.49 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.50: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.75: %facet_type = converted constants.%i32, %facet_value.loc6_12.50 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.25: init %empty_tuple.type = call %SelfDestruct.specific_fn.25(%.loc6_12.73)
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %.loc6_12.76: ref %i32 = array_index %self.param, %int_16
// CHECK:STDOUT:   %facet_value.loc6_12.51: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.77: %facet_type = converted constants.%i32, %facet_value.loc6_12.51 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.52: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.78: %facet_type = converted constants.%i32, %facet_value.loc6_12.52 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.26: init %empty_tuple.type = call %SelfDestruct.specific_fn.26(%.loc6_12.76)
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15]
// CHECK:STDOUT:   %.loc6_12.79: ref %i32 = array_index %self.param, %int_15
// CHECK:STDOUT:   %facet_value.loc6_12.53: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.80: %facet_type = converted constants.%i32, %facet_value.loc6_12.53 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.54: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.81: %facet_type = converted constants.%i32, %facet_value.loc6_12.54 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.27: init %empty_tuple.type = call %SelfDestruct.specific_fn.27(%.loc6_12.79)
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete = constants.%int_14]
// CHECK:STDOUT:   %.loc6_12.82: ref %i32 = array_index %self.param, %int_14
// CHECK:STDOUT:   %facet_value.loc6_12.55: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.83: %facet_type = converted constants.%i32, %facet_value.loc6_12.55 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.56: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.84: %facet_type = converted constants.%i32, %facet_value.loc6_12.56 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.28: init %empty_tuple.type = call %SelfDestruct.specific_fn.28(%.loc6_12.82)
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete = constants.%int_13]
// CHECK:STDOUT:   %.loc6_12.85: ref %i32 = array_index %self.param, %int_13
// CHECK:STDOUT:   %facet_value.loc6_12.57: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.86: %facet_type = converted constants.%i32, %facet_value.loc6_12.57 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.58: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.87: %facet_type = converted constants.%i32, %facet_value.loc6_12.58 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.29: init %empty_tuple.type = call %SelfDestruct.specific_fn.29(%.loc6_12.85)
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc6_12.88: ref %i32 = array_index %self.param, %int_12
// CHECK:STDOUT:   %facet_value.loc6_12.59: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.89: %facet_type = converted constants.%i32, %facet_value.loc6_12.59 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.60: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.90: %facet_type = converted constants.%i32, %facet_value.loc6_12.60 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.30: init %empty_tuple.type = call %SelfDestruct.specific_fn.30(%.loc6_12.88)
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %.loc6_12.91: ref %i32 = array_index %self.param, %int_11
// CHECK:STDOUT:   %facet_value.loc6_12.61: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.92: %facet_type = converted constants.%i32, %facet_value.loc6_12.61 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.62: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.93: %facet_type = converted constants.%i32, %facet_value.loc6_12.62 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.31: init %empty_tuple.type = call %SelfDestruct.specific_fn.31(%.loc6_12.91)
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc6_12.94: ref %i32 = array_index %self.param, %int_10
// CHECK:STDOUT:   %facet_value.loc6_12.63: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.95: %facet_type = converted constants.%i32, %facet_value.loc6_12.63 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.64: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.96: %facet_type = converted constants.%i32, %facet_value.loc6_12.64 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.32: init %empty_tuple.type = call %SelfDestruct.specific_fn.32(%.loc6_12.94)
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc6_12.97: ref %i32 = array_index %self.param, %int_9
// CHECK:STDOUT:   %facet_value.loc6_12.65: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.98: %facet_type = converted constants.%i32, %facet_value.loc6_12.65 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.66: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.99: %facet_type = converted constants.%i32, %facet_value.loc6_12.66 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.33: init %empty_tuple.type = call %SelfDestruct.specific_fn.33(%.loc6_12.97)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc6_12.100: ref %i32 = array_index %self.param, %int_8
// CHECK:STDOUT:   %facet_value.loc6_12.67: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.101: %facet_type = converted constants.%i32, %facet_value.loc6_12.67 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.68: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.102: %facet_type = converted constants.%i32, %facet_value.loc6_12.68 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.34: init %empty_tuple.type = call %SelfDestruct.specific_fn.34(%.loc6_12.100)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc6_12.103: ref %i32 = array_index %self.param, %int_7
// CHECK:STDOUT:   %facet_value.loc6_12.69: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.104: %facet_type = converted constants.%i32, %facet_value.loc6_12.69 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.70: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.105: %facet_type = converted constants.%i32, %facet_value.loc6_12.70 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.35: init %empty_tuple.type = call %SelfDestruct.specific_fn.35(%.loc6_12.103)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc6_12.106: ref %i32 = array_index %self.param, %int_6
// CHECK:STDOUT:   %facet_value.loc6_12.71: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.107: %facet_type = converted constants.%i32, %facet_value.loc6_12.71 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.72: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.108: %facet_type = converted constants.%i32, %facet_value.loc6_12.72 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.36: init %empty_tuple.type = call %SelfDestruct.specific_fn.36(%.loc6_12.106)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc6_12.109: ref %i32 = array_index %self.param, %int_5
// CHECK:STDOUT:   %facet_value.loc6_12.73: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.110: %facet_type = converted constants.%i32, %facet_value.loc6_12.73 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.74: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.111: %facet_type = converted constants.%i32, %facet_value.loc6_12.74 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.37: init %empty_tuple.type = call %SelfDestruct.specific_fn.37(%.loc6_12.109)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc6_12.112: ref %i32 = array_index %self.param, %int_4
// CHECK:STDOUT:   %facet_value.loc6_12.75: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.113: %facet_type = converted constants.%i32, %facet_value.loc6_12.75 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.76: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.114: %facet_type = converted constants.%i32, %facet_value.loc6_12.76 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.38: init %empty_tuple.type = call %SelfDestruct.specific_fn.38(%.loc6_12.112)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc6_12.115: ref %i32 = array_index %self.param, %int_3
// CHECK:STDOUT:   %facet_value.loc6_12.77: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.116: %facet_type = converted constants.%i32, %facet_value.loc6_12.77 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.78: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.117: %facet_type = converted constants.%i32, %facet_value.loc6_12.78 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.39: init %empty_tuple.type = call %SelfDestruct.specific_fn.39(%.loc6_12.115)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc6_12.118: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %facet_value.loc6_12.79: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.119: %facet_type = converted constants.%i32, %facet_value.loc6_12.79 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.80: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.120: %facet_type = converted constants.%i32, %facet_value.loc6_12.80 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.40: init %empty_tuple.type = call %SelfDestruct.specific_fn.40(%.loc6_12.118)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc6_12.121: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %facet_value.loc6_12.81: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.122: %facet_type = converted constants.%i32, %facet_value.loc6_12.81 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.82: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.123: %facet_type = converted constants.%i32, %facet_value.loc6_12.82 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.41: init %empty_tuple.type = call %SelfDestruct.specific_fn.41(%.loc6_12.121)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc6_12.124: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %facet_value.loc6_12.83: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.125: %facet_type = converted constants.%i32, %facet_value.loc6_12.83 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc6_12.84: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc6_12.126: %facet_type = converted constants.%i32, %facet_value.loc6_12.84 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_12.42: init %empty_tuple.type = call %SelfDestruct.specific_fn.42(%.loc6_12.124)
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
// CHECK:STDOUT:   %Array.270: type = class_type @Array, @Array(%i32, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %Array.270 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.faf = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.38c: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.bcc [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.270 = tuple_value (%int_1.874) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.af3: %I.assoc_type = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = constants.%assoc0.38c]
// CHECK:STDOUT:   %Main.import_ref.18d: %Array.270 = import_ref Main//symbolic_decl, loc10_33, loaded [concrete = constants.%array]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.18d), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.bcc: %Array.270 = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %Array.270 = assoc_const_decl @value [concrete] {}
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
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.af3 [concrete = constants.%assoc0.38c]
// CHECK:STDOUT:   %impl.elem0.loc7: %Array.270 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc7_22: type = splice_block %Array [concrete = constants.%Array.270] {
// CHECK:STDOUT:     %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_1.5b8) [concrete = constants.%Array.270]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Array.270 = wrapper_binding a, %impl.elem0.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.faf = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
