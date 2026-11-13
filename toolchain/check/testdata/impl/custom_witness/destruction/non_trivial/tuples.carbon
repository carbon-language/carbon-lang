// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/tuples.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

// `HasNonTrivialDestruction.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoHasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

// --- todo_nontrivial_all_nontrivial_elements.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (HasNonTrivialDestruction, AlsoHasNonTrivialDestruction, AlsoHasNonTrivialDestruction, HasNonTrivialDestruction);
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_one_trivial_element.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (HasNonTrivialDestruction, HasTrivialDestruction, AlsoHasNonTrivialDestruction, HasNonTrivialDestruction);
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = ((HasNonTrivialDestruction, HasTrivialDestruction), AlsoHasNonTrivialDestruction);
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_elements.carbon = $todo_nontrivial_all_nontrivial_elements
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.c31: type = tuple_type (type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c31 = tuple_value (%HasNonTrivialDestruction, %AlsoHasNonTrivialDestruction, %AlsoHasNonTrivialDestruction, %HasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %tuple.type.66d: type = tuple_type (%HasNonTrivialDestruction, %AlsoHasNonTrivialDestruction, %AlsoHasNonTrivialDestruction, %HasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %pattern_type.244: type = pattern_type %tuple.type.66d [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d3d: %SubobjectDestroy.type = facet_value %tuple.type.66d, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d9b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d3d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a6b: %Destroy.type = facet_value %tuple.type.66d, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.d9b) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.244 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.244 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc9_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc15_28, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Tuple.ref.loc10: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%tuple.type.66d, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.d3d]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Tuple.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.d3d]
// CHECK:STDOUT:   %Tuple.ref.loc11: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value constants.%tuple.type.66d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d9b) [concrete = constants.%Destroy.facet.a6b]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Tuple.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.a6b]
// CHECK:STDOUT:   %x.var: ref %tuple.type.66d = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %tuple.type.66d = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.66d to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = constants.%tuple.type.66d] {
// CHECK:STDOUT:     %Tuple.ref.loc13: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc13_17.2: type = converted %Tuple.ref.loc13, constants.%tuple.type.66d [concrete = constants.%tuple.type.66d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.66d = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.244 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.244 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%tuple.type.66d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d9b) [concrete = constants.%Destroy.facet.a6b]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%tuple.type.66d, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.a6b]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%tuple.type.66d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d9b) [concrete = constants.%Destroy.facet.a6b]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.66d, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.a6b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %tuple.type.66d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %tuple.elem3: ref %HasNonTrivialDestruction = tuple_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.1: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem3)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref %AlsoHasNonTrivialDestruction = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.4: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %AlsoHasNonTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.5: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.5: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.5 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.6: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.6: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.6 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %HasNonTrivialDestruction = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.7: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.7 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.8: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.8 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_one_trivial_element.carbon = $todo_nontrivial_one_trivial_element
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.c31: type = tuple_type (type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c31 = tuple_value (%HasNonTrivialDestruction, %HasTrivialDestruction, %AlsoHasNonTrivialDestruction, %HasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %tuple.type.f92: type = tuple_type (%HasNonTrivialDestruction, %HasTrivialDestruction, %AlsoHasNonTrivialDestruction, %HasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %pattern_type.407: type = pattern_type %tuple.type.f92 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f6b: %SubobjectDestroy.type = facet_value %tuple.type.f92, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.004: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f6b) [concrete]
// CHECK:STDOUT:   %Destroy.facet.86a: %Destroy.type = facet_value %tuple.type.f92, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.004) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.407 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.407 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc9_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc15_28, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Tuple.ref.loc10: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%tuple.type.f92, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.f6b]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Tuple.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.f6b]
// CHECK:STDOUT:   %Tuple.ref.loc11: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value constants.%tuple.type.f92, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.004) [concrete = constants.%Destroy.facet.86a]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Tuple.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.86a]
// CHECK:STDOUT:   %x.var: ref %tuple.type.f92 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %tuple.type.f92 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.f92 to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = constants.%tuple.type.f92] {
// CHECK:STDOUT:     %Tuple.ref.loc13: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc13_17.2: type = converted %Tuple.ref.loc13, constants.%tuple.type.f92 [concrete = constants.%tuple.type.f92]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.f92 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.407 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.407 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%tuple.type.f92, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.004) [concrete = constants.%Destroy.facet.86a]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%tuple.type.f92, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.86a]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%tuple.type.f92, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.004) [concrete = constants.%Destroy.facet.86a]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.f92, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.86a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.5(%self.param: ref %tuple.type.f92) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %tuple.elem3: ref %HasNonTrivialDestruction = tuple_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.1: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem3)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref %AlsoHasNonTrivialDestruction = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.4: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.5: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.5 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.6: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.6 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %HasNonTrivialDestruction = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.7: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.7 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.8: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.8 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nested.carbon = $todo_nontrivial_nested
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.777: %tuple.type.693 = tuple_value (%HasNonTrivialDestruction, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.55c: type = tuple_type (%tuple.type.693, type) [concrete]
// CHECK:STDOUT:   %tuple.5ca: %tuple.type.55c = tuple_value (%tuple.777, %AlsoHasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %tuple.type.aad: type = tuple_type (%HasNonTrivialDestruction, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %tuple.type.1e2: type = tuple_type (%tuple.type.aad, %AlsoHasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %pattern_type.601: type = pattern_type %tuple.type.1e2 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b69: %SubobjectDestroy.type = facet_value %tuple.type.aad, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4a1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b69) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a9a: %Destroy.type = facet_value %tuple.type.aad, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.4a1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2cb: %SubobjectDestroy.type = facet_value %tuple.type.1e2, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.26a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2cb) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3ca: %Destroy.type = facet_value %tuple.type.1e2, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.26a) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.601 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.601 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc9_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc15_28, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Tuple.ref.loc10: %tuple.type.55c = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.5ca]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %.loc10_9.1: type = converted constants.%tuple.777, constants.%tuple.type.aad [concrete = constants.%tuple.type.aad]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%tuple.type.1e2, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.2cb]
// CHECK:STDOUT:   %.loc10_9.2: %SubobjectDestroy.type = converted %Tuple.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.2cb]
// CHECK:STDOUT:   %Tuple.ref.loc11: %tuple.type.55c = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.5ca]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %.loc11_9.1: type = converted constants.%tuple.777, constants.%tuple.type.aad [concrete = constants.%tuple.type.aad]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value constants.%tuple.type.1e2, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.26a) [concrete = constants.%Destroy.facet.3ca]
// CHECK:STDOUT:   %.loc11_9.2: %Destroy.type = converted %Tuple.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.3ca]
// CHECK:STDOUT:   %x.var: ref %tuple.type.1e2 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %tuple.type.1e2 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.1e2 to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.3 [concrete = constants.%tuple.type.1e2] {
// CHECK:STDOUT:     %Tuple.ref.loc13: %tuple.type.55c = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.5ca]
// CHECK:STDOUT:     %.loc13_17.2: type = converted constants.%tuple.777, constants.%tuple.type.aad [concrete = constants.%tuple.type.aad]
// CHECK:STDOUT:     %.loc13_17.3: type = converted %Tuple.ref.loc13, constants.%tuple.type.1e2 [concrete = constants.%tuple.type.1e2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.1e2 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.601 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.601 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%tuple.type.1e2, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.26a) [concrete = constants.%Destroy.facet.3ca]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%tuple.type.1e2, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.3ca]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%tuple.type.1e2, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.26a) [concrete = constants.%Destroy.facet.3ca]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.1e2, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.3ca]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %tuple.type.aad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.1: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %HasNonTrivialDestruction = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_9.4: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.5(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.6(%self.param: ref %tuple.type.1e2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %AlsoHasNonTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.1: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.aad = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%tuple.type.aad, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4a1) [concrete = constants.%Destroy.facet.a9a]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%tuple.type.aad, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.a9a]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%tuple.type.aad, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4a1) [concrete = constants.%Destroy.facet.a9a]
// CHECK:STDOUT:   %.loc10_9.4: %Destroy.type = converted constants.%tuple.type.aad, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.a9a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
