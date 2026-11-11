// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/subobject_destroy/arrays.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/subobject_destroy/arrays.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_nontrivial_populated.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasNonTrivialDestruction, 2);
fn Make() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_array_of_array.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasNonTrivialDestruction, 4), 43);
fn Make() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_adapter.carbon

// Not in `adapters.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasNonTrivialDestruction, 2);
}

fn Make() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_populated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.0e1: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_2) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %Array.0e1 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.0b2 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.0b2 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3e1: %SubobjectDestroy.type = facet_value %Array.0e1, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cea: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3e1) [concrete]
// CHECK:STDOUT:   %Destroy.facet.900: %Destroy.type = facet_value %Array.0e1, (%DestroyImpl.impl_witness.cea, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %Array.0e1 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_3.1: ref %Array.0e1 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.0e1 to %.loc11_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref: type = name_ref Array, file.%Array.loc6_7 [concrete = constants.%Array.0e1]
// CHECK:STDOUT:   %x: ref %Array.0e1 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.0b2 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.0b2 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%Array.0e1, (constants.%DestroyImpl.impl_witness.cea, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.900]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%Array.0e1, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.900]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%Array.0e1, (constants.%DestroyImpl.impl_witness.cea, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.900]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%Array.0e1, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.900]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.3(%self.param: ref %Array.0e1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_3.1: ref %HasNonTrivialDestruction = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_3.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_3.4: ref %HasNonTrivialDestruction = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc11_3.3: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.5: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.3 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.4: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.6: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.4 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc11_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_array_of_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.11f: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_4) [concrete]
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete]
// CHECK:STDOUT:   %Array.330: type = class_type @Array, @Array(%Array.11f, %int_43) [concrete]
// CHECK:STDOUT:   %pattern_type.7ee: type = pattern_type %Array.330 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.7ee = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.7ee = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.08d: %SubobjectDestroy.type = facet_value %Array.11f, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c35: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.08d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ea7: %Destroy.type = facet_value %Array.11f, (%DestroyImpl.impl_witness.c35, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete]
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete]
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete]
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete]
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
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
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.300: %SubobjectDestroy.type = facet_value %Array.330, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.456: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.300) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b8b: %Destroy.type = facet_value %Array.330, (%DestroyImpl.impl_witness.456, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %Array.330 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_3.1: ref %Array.330 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.330 to %.loc11_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref: type = name_ref Array, file.%Array.loc6_7 [concrete = constants.%Array.330]
// CHECK:STDOUT:   %x: ref %Array.330 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ee = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.7ee = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%Array.330, (constants.%DestroyImpl.impl_witness.456, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b8b]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%Array.330, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.b8b]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%Array.330, (constants.%DestroyImpl.impl_witness.456, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b8b]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%Array.330, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.b8b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.3(%self.param: ref %Array.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc11_3.1: ref %HasNonTrivialDestruction = array_index %self.param, %int_3
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_3.1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc11_3.4: ref %HasNonTrivialDestruction = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.facet.loc11_3.3: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.5: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.3 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.4: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.6: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.4 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc11_3.4)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_3.7: ref %HasNonTrivialDestruction = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc11_3.5: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.8: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.5 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.6: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.9: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.6 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc11_3.7)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_3.10: ref %HasNonTrivialDestruction = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc11_3.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.11: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.7 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc11_3.12: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc11_3.8 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc11_3.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.4(%self.param: ref %Array.330) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:   %.loc11_3.1: ref %Array.11f = array_index %self.param, %int_42
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_3.1)
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete = constants.%int_41]
// CHECK:STDOUT:   %.loc11_3.4: ref %Array.11f = array_index %self.param, %int_41
// CHECK:STDOUT:   %Destroy.facet.loc11_3.3: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.5: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.3 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.4: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.6: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.4 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc11_3.4)
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   %.loc11_3.7: ref %Array.11f = array_index %self.param, %int_40
// CHECK:STDOUT:   %Destroy.facet.loc11_3.5: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.8: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.5 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.6: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.9: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.6 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc11_3.7)
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete = constants.%int_39]
// CHECK:STDOUT:   %.loc11_3.10: ref %Array.11f = array_index %self.param, %int_39
// CHECK:STDOUT:   %Destroy.facet.loc11_3.7: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.11: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.7 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.8: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.12: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.8 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc11_3.10)
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete = constants.%int_38]
// CHECK:STDOUT:   %.loc11_3.13: ref %Array.11f = array_index %self.param, %int_38
// CHECK:STDOUT:   %Destroy.facet.loc11_3.9: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.14: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.9 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.10: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.15: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.10 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.5: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc11_3.13)
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete = constants.%int_37]
// CHECK:STDOUT:   %.loc11_3.16: ref %Array.11f = array_index %self.param, %int_37
// CHECK:STDOUT:   %Destroy.facet.loc11_3.11: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.17: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.11 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.12: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.18: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.12 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.6: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc11_3.16)
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete = constants.%int_36]
// CHECK:STDOUT:   %.loc11_3.19: ref %Array.11f = array_index %self.param, %int_36
// CHECK:STDOUT:   %Destroy.facet.loc11_3.13: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.20: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.13 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.14: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.21: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.14 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.7: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc11_3.19)
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35]
// CHECK:STDOUT:   %.loc11_3.22: ref %Array.11f = array_index %self.param, %int_35
// CHECK:STDOUT:   %Destroy.facet.loc11_3.15: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.23: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.15 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.16: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.24: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.16 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.8: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%.loc11_3.22)
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34]
// CHECK:STDOUT:   %.loc11_3.25: ref %Array.11f = array_index %self.param, %int_34
// CHECK:STDOUT:   %Destroy.facet.loc11_3.17: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.26: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.17 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.18: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.27: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.18 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.9: init %empty_tuple.type = call %SelfDestruct.specific_fn.9(%.loc11_3.25)
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete = constants.%int_33]
// CHECK:STDOUT:   %.loc11_3.28: ref %Array.11f = array_index %self.param, %int_33
// CHECK:STDOUT:   %Destroy.facet.loc11_3.19: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.29: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.19 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.20: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.30: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.20 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.10: init %empty_tuple.type = call %SelfDestruct.specific_fn.10(%.loc11_3.28)
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc11_3.31: ref %Array.11f = array_index %self.param, %int_32
// CHECK:STDOUT:   %Destroy.facet.loc11_3.21: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.32: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.21 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.22: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.33: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.22 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.11: init %empty_tuple.type = call %SelfDestruct.specific_fn.11(%.loc11_3.31)
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete = constants.%int_31]
// CHECK:STDOUT:   %.loc11_3.34: ref %Array.11f = array_index %self.param, %int_31
// CHECK:STDOUT:   %Destroy.facet.loc11_3.23: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.35: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.23 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.24: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.36: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.24 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.12: init %empty_tuple.type = call %SelfDestruct.specific_fn.12(%.loc11_3.34)
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete = constants.%int_30]
// CHECK:STDOUT:   %.loc11_3.37: ref %Array.11f = array_index %self.param, %int_30
// CHECK:STDOUT:   %Destroy.facet.loc11_3.25: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.38: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.25 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.26: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.39: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.26 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.13: init %empty_tuple.type = call %SelfDestruct.specific_fn.13(%.loc11_3.37)
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete = constants.%int_29]
// CHECK:STDOUT:   %.loc11_3.40: ref %Array.11f = array_index %self.param, %int_29
// CHECK:STDOUT:   %Destroy.facet.loc11_3.27: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.41: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.27 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.28: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.42: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.28 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.14: init %empty_tuple.type = call %SelfDestruct.specific_fn.14(%.loc11_3.40)
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete = constants.%int_28]
// CHECK:STDOUT:   %.loc11_3.43: ref %Array.11f = array_index %self.param, %int_28
// CHECK:STDOUT:   %Destroy.facet.loc11_3.29: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.44: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.29 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.30: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.45: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.30 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.15: init %empty_tuple.type = call %SelfDestruct.specific_fn.15(%.loc11_3.43)
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete = constants.%int_27]
// CHECK:STDOUT:   %.loc11_3.46: ref %Array.11f = array_index %self.param, %int_27
// CHECK:STDOUT:   %Destroy.facet.loc11_3.31: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.47: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.31 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.32: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.48: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.32 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.16: init %empty_tuple.type = call %SelfDestruct.specific_fn.16(%.loc11_3.46)
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete = constants.%int_26]
// CHECK:STDOUT:   %.loc11_3.49: ref %Array.11f = array_index %self.param, %int_26
// CHECK:STDOUT:   %Destroy.facet.loc11_3.33: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.50: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.33 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.34: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.51: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.34 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.17: init %empty_tuple.type = call %SelfDestruct.specific_fn.17(%.loc11_3.49)
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete = constants.%int_25]
// CHECK:STDOUT:   %.loc11_3.52: ref %Array.11f = array_index %self.param, %int_25
// CHECK:STDOUT:   %Destroy.facet.loc11_3.35: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.53: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.35 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.36: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.54: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.36 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.18: init %empty_tuple.type = call %SelfDestruct.specific_fn.18(%.loc11_3.52)
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24]
// CHECK:STDOUT:   %.loc11_3.55: ref %Array.11f = array_index %self.param, %int_24
// CHECK:STDOUT:   %Destroy.facet.loc11_3.37: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.56: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.37 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.38: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.57: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.38 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.19: init %empty_tuple.type = call %SelfDestruct.specific_fn.19(%.loc11_3.55)
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete = constants.%int_23]
// CHECK:STDOUT:   %.loc11_3.58: ref %Array.11f = array_index %self.param, %int_23
// CHECK:STDOUT:   %Destroy.facet.loc11_3.39: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.59: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.39 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.40: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.60: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.40 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.20: init %empty_tuple.type = call %SelfDestruct.specific_fn.20(%.loc11_3.58)
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete = constants.%int_22]
// CHECK:STDOUT:   %.loc11_3.61: ref %Array.11f = array_index %self.param, %int_22
// CHECK:STDOUT:   %Destroy.facet.loc11_3.41: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.62: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.41 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.42: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.63: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.42 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.21: init %empty_tuple.type = call %SelfDestruct.specific_fn.21(%.loc11_3.61)
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete = constants.%int_21]
// CHECK:STDOUT:   %.loc11_3.64: ref %Array.11f = array_index %self.param, %int_21
// CHECK:STDOUT:   %Destroy.facet.loc11_3.43: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.65: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.43 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.44: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.66: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.44 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.22: init %empty_tuple.type = call %SelfDestruct.specific_fn.22(%.loc11_3.64)
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %.loc11_3.67: ref %Array.11f = array_index %self.param, %int_20
// CHECK:STDOUT:   %Destroy.facet.loc11_3.45: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.68: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.45 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.46: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.69: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.46 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.23: init %empty_tuple.type = call %SelfDestruct.specific_fn.23(%.loc11_3.67)
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete = constants.%int_19]
// CHECK:STDOUT:   %.loc11_3.70: ref %Array.11f = array_index %self.param, %int_19
// CHECK:STDOUT:   %Destroy.facet.loc11_3.47: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.71: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.47 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.48: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.72: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.48 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.24: init %empty_tuple.type = call %SelfDestruct.specific_fn.24(%.loc11_3.70)
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete = constants.%int_18]
// CHECK:STDOUT:   %.loc11_3.73: ref %Array.11f = array_index %self.param, %int_18
// CHECK:STDOUT:   %Destroy.facet.loc11_3.49: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.74: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.49 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.50: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.75: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.50 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.25: init %empty_tuple.type = call %SelfDestruct.specific_fn.25(%.loc11_3.73)
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete = constants.%int_17]
// CHECK:STDOUT:   %.loc11_3.76: ref %Array.11f = array_index %self.param, %int_17
// CHECK:STDOUT:   %Destroy.facet.loc11_3.51: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.77: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.51 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.52: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.78: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.52 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.26: init %empty_tuple.type = call %SelfDestruct.specific_fn.26(%.loc11_3.76)
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %.loc11_3.79: ref %Array.11f = array_index %self.param, %int_16
// CHECK:STDOUT:   %Destroy.facet.loc11_3.53: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.80: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.53 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.54: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.81: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.54 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.27: init %empty_tuple.type = call %SelfDestruct.specific_fn.27(%.loc11_3.79)
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15]
// CHECK:STDOUT:   %.loc11_3.82: ref %Array.11f = array_index %self.param, %int_15
// CHECK:STDOUT:   %Destroy.facet.loc11_3.55: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.83: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.55 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.56: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.84: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.56 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.28: init %empty_tuple.type = call %SelfDestruct.specific_fn.28(%.loc11_3.82)
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete = constants.%int_14]
// CHECK:STDOUT:   %.loc11_3.85: ref %Array.11f = array_index %self.param, %int_14
// CHECK:STDOUT:   %Destroy.facet.loc11_3.57: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.86: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.57 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.58: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.87: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.58 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.29: init %empty_tuple.type = call %SelfDestruct.specific_fn.29(%.loc11_3.85)
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete = constants.%int_13]
// CHECK:STDOUT:   %.loc11_3.88: ref %Array.11f = array_index %self.param, %int_13
// CHECK:STDOUT:   %Destroy.facet.loc11_3.59: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.89: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.59 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.60: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.90: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.60 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.30: init %empty_tuple.type = call %SelfDestruct.specific_fn.30(%.loc11_3.88)
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc11_3.91: ref %Array.11f = array_index %self.param, %int_12
// CHECK:STDOUT:   %Destroy.facet.loc11_3.61: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.92: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.61 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.62: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.93: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.62 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.31: init %empty_tuple.type = call %SelfDestruct.specific_fn.31(%.loc11_3.91)
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %.loc11_3.94: ref %Array.11f = array_index %self.param, %int_11
// CHECK:STDOUT:   %Destroy.facet.loc11_3.63: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.95: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.63 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.64: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.96: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.64 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.32: init %empty_tuple.type = call %SelfDestruct.specific_fn.32(%.loc11_3.94)
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc11_3.97: ref %Array.11f = array_index %self.param, %int_10
// CHECK:STDOUT:   %Destroy.facet.loc11_3.65: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.98: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.65 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.66: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.99: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.66 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.33: init %empty_tuple.type = call %SelfDestruct.specific_fn.33(%.loc11_3.97)
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc11_3.100: ref %Array.11f = array_index %self.param, %int_9
// CHECK:STDOUT:   %Destroy.facet.loc11_3.67: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.101: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.67 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.68: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.102: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.68 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.34: init %empty_tuple.type = call %SelfDestruct.specific_fn.34(%.loc11_3.100)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc11_3.103: ref %Array.11f = array_index %self.param, %int_8
// CHECK:STDOUT:   %Destroy.facet.loc11_3.69: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.104: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.69 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.70: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.105: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.70 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.35: init %empty_tuple.type = call %SelfDestruct.specific_fn.35(%.loc11_3.103)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc11_3.106: ref %Array.11f = array_index %self.param, %int_7
// CHECK:STDOUT:   %Destroy.facet.loc11_3.71: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.107: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.71 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.72: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.108: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.72 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.36: init %empty_tuple.type = call %SelfDestruct.specific_fn.36(%.loc11_3.106)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc11_3.109: ref %Array.11f = array_index %self.param, %int_6
// CHECK:STDOUT:   %Destroy.facet.loc11_3.73: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.110: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.73 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.74: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.111: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.74 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.37: init %empty_tuple.type = call %SelfDestruct.specific_fn.37(%.loc11_3.109)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc11_3.112: ref %Array.11f = array_index %self.param, %int_5
// CHECK:STDOUT:   %Destroy.facet.loc11_3.75: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.113: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.75 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.76: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.114: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.76 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.38: init %empty_tuple.type = call %SelfDestruct.specific_fn.38(%.loc11_3.112)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc11_3.115: ref %Array.11f = array_index %self.param, %int_4
// CHECK:STDOUT:   %Destroy.facet.loc11_3.77: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.116: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.77 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.78: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.117: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.78 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.39: init %empty_tuple.type = call %SelfDestruct.specific_fn.39(%.loc11_3.115)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc11_3.118: ref %Array.11f = array_index %self.param, %int_3
// CHECK:STDOUT:   %Destroy.facet.loc11_3.79: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.119: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.79 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.80: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.120: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.80 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.40: init %empty_tuple.type = call %SelfDestruct.specific_fn.40(%.loc11_3.118)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc11_3.121: ref %Array.11f = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.facet.loc11_3.81: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.122: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.81 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.82: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.123: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.82 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.41: init %empty_tuple.type = call %SelfDestruct.specific_fn.41(%.loc11_3.121)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_3.124: ref %Array.11f = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc11_3.83: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.125: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.83 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.84: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.126: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.84 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.42: init %empty_tuple.type = call %SelfDestruct.specific_fn.42(%.loc11_3.124)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_3.127: ref %Array.11f = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc11_3.85: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.128: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.85 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.86: %Destroy.type = facet_value constants.%Array.11f, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   %.loc11_3.129: %Destroy.type = converted constants.%Array.11f, %Destroy.facet.loc11_3.86 [concrete = constants.%Destroy.facet.ea7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.43: init %empty_tuple.type = call %SelfDestruct.specific_fn.43(%.loc11_3.127)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.0e1: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_2) [concrete]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2f: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.490, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.96a: %Destroy.type = facet_value %Adapter, (%DestroyImpl.impl_witness.7cf, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %Adapter = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %Adapter = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Adapter to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %x: ref %Adapter = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%Adapter, (constants.%DestroyImpl.impl_witness.7cf, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%Adapter, (constants.%DestroyImpl.impl_witness.7cf, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.3(%self.param: ref %Array.0e1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc17_3.1: ref %HasNonTrivialDestruction = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc17_3.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc17_3.4: ref %HasNonTrivialDestruction = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc17_3.3: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc17_3.5: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_3.3 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.4: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.490, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   %.loc17_3.6: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc17_3.4 [concrete = constants.%Destroy.facet.a2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc17_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.4(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
