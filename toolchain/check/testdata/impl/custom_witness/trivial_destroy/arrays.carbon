// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/arrays.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/arrays.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_empty_array_with_trivial_element_type.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(i32, 0);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_empty_array_with_nontrivial_element_type.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array =  array(HasNonTrivialDestruction, 0);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_populated.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasTrivialDestruction, 1);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_populated.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasNonTrivialDestruction, 2);

fn Test() {
  Array as Core.TrivialDestroy;
}

// --- todo_trivial_array_of_array.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasTrivialDestruction, 3), 10);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_array_of_array.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasNonTrivialDestruction, 4), 43);

fn Test() {
  Array as Core.TrivialDestroy;
}

// --- todo_trivial_adapter.carbon

// Not in `adapters.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasTrivialDestruction, 2);
}

fn Make() -> Adapter;

fn Test() {
  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasNonTrivialDestruction, 2);
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- todo_trivial_constrained_facet.carbon

// Not in `facets.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

fn MakeElement(generic T: Core.TrivialDestroy) -> T {
  returned var result: T;
  return var;
}

fn MakeArray(generic T: Core.TrivialDestroy) -> array(T, 3) {
  return (MakeElement(T), MakeElement(T), MakeElement(T));
}

fn Test(generic T: Core.TrivialDestroy) {
  array(T, 3) as Core.Destroy;
  array(T, 3) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: array(T, 3) = MakeArray(T);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_trivial_empty_array_with_trivial_element_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.68f: type = class_type @Array, @Array(%i32, %int_0) [concrete]
// CHECK:STDOUT:   %pattern_type.a2e: type = pattern_type %Array.68f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8b7: %SubobjectDestroy.type = facet_value %Array.68f, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.68f, (%DestroyImpl.impl_witness.cca, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a2e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.a2e = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %Array.68f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.68f = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.68f to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc6_7 [concrete = constants.%Array.68f]
// CHECK:STDOUT:   %x: ref %Array.68f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a2e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.a2e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.68f, (constants.%DestroyImpl.impl_witness.cca, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.68f, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.68f, (constants.%DestroyImpl.impl_witness.cca, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.68f, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_empty_array_with_nontrivial_element_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.fbd: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_0) [concrete]
// CHECK:STDOUT:   %pattern_type.16c: type = pattern_type %Array.fbd [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6c7: %SubobjectDestroy.type = facet_value %Array.fbd, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cf7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6c7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.fbd, (%DestroyImpl.impl_witness.cf7, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.16c = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.16c = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %Array.fbd = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.fbd = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.fbd to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc6_7 [concrete = constants.%Array.fbd]
// CHECK:STDOUT:   %x: ref %Array.fbd = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.16c = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.16c = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.fbd, (constants.%DestroyImpl.impl_witness.cf7, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.fbd, (constants.%DestroyImpl.impl_witness.cf7, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_populated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.7b9: type = class_type @Array, @Array(%HasTrivialDestruction, %int_1) [concrete]
// CHECK:STDOUT:   %pattern_type.f24: type = pattern_type %Array.7b9 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.351: %SubobjectDestroy.type = facet_value %Array.7b9, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.137: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.351) [concrete]
// CHECK:STDOUT:   %Destroy.facet.41b: %Destroy.type = facet_value %Array.7b9, (%DestroyImpl.impl_witness.137, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f24 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.f24 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %Array.7b9 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.7b9 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.7b9 to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc6_7 [concrete = constants.%Array.7b9]
// CHECK:STDOUT:   %x: ref %Array.7b9 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f24 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.f24 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.7b9, (constants.%DestroyImpl.impl_witness.137, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.41b]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.7b9, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.41b]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.7b9, (constants.%DestroyImpl.impl_witness.137, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.41b]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.7b9, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.41b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_array_of_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.8ec: type = class_type @Array, @Array(%HasTrivialDestruction, %int_3) [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Array.a4f: type = class_type @Array, @Array(%Array.8ec, %int_10) [concrete]
// CHECK:STDOUT:   %pattern_type.e6e: type = pattern_type %Array.a4f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ef9: %SubobjectDestroy.type = facet_value %Array.a4f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d07: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ef9) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ca6: %Destroy.type = facet_value %Array.a4f, (%DestroyImpl.impl_witness.d07, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e6e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e6e = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %Array.a4f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.a4f = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.a4f to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc6_7 [concrete = constants.%Array.a4f]
// CHECK:STDOUT:   %x: ref %Array.a4f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e6e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e6e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.a4f, (constants.%DestroyImpl.impl_witness.d07, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ca6]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.a4f, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.ca6]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.a4f, (constants.%DestroyImpl.impl_witness.d07, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ca6]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.a4f, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.ca6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.96a: %Destroy.type = facet_value %Adapter, (%DestroyImpl.impl_witness.7cf, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %Adapter = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc20_3.1: ref %Adapter = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Adapter to %.loc20_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Adapter.ref.loc20: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %x: ref %Adapter = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%Adapter, (constants.%DestroyImpl.impl_witness.7cf, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%Adapter, (constants.%DestroyImpl.impl_witness.7cf, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   %.loc20_3.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet.96a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_constrained_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190: type = facet_access_type %T.f45 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.cf1: type = class_type @Array, @Array(%T.as_type.190, %int_3) [symbolic]
// CHECK:STDOUT:   %pattern_type.ed8: type = pattern_type %Array.cf1 [symbolic]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d5e: <witness> = require_complete_type %Array.cf1 [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.8a2: <witness> = lookup_impl_witness %Array.cf1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.518: <witness> = lookup_impl_witness %Array.cf1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.cf1, (%DestroyImpl.lookup_impl_witness.8a2, %SubobjectDestroy.lookup_impl_witness.518) [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.ed8 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.ed8 = var_pattern %x.patt [symbolic]
// CHECK:STDOUT:   %MakeArray.specific_fn: <specific function> = specific_function %MakeArray, @MakeArray(%T.f45) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc16_18.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc17_13.2 [symbolic = %require_complete (constants.%require_complete.d5e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc17_13.2 [symbolic = %pattern_type (constants.%pattern_type.ed8)]
// CHECK:STDOUT:   %x.patt.loc21_15.2: @Test.%pattern_type (%pattern_type.ed8) = ref_binding_pattern x [symbolic = %x.patt.loc21_15.2 (constants.%x.patt)]
// CHECK:STDOUT:   %x.var_patt.loc21_3.2: @Test.%pattern_type (%pattern_type.ed8) = var_pattern %x.patt.loc21_15.2 [symbolic = %x.var_patt.loc21_3.2 (constants.%x.var_patt)]
// CHECK:STDOUT:   %MakeArray.specific_fn.loc21_31.2: <specific function> = specific_function constants.%MakeArray, @MakeArray(%T.loc16_18.1) [symbolic = %MakeArray.specific_fn.loc21_31.2 (constants.%MakeArray.specific_fn)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %x.var: ref @Test.%Array.loc17_13.2 (%Array.cf1) = var_storage %x.var_patt.loc21_3.1
// CHECK:STDOUT:     %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:     %T.ref.loc21_41: %Destroy.type = name_ref T, %T.loc16_18.2 [symbolic = %T.loc16_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %MakeArray.specific_fn.loc21_31.1: <specific function> = specific_function %MakeArray.ref, @MakeArray(constants.%T.f45) [symbolic = %MakeArray.specific_fn.loc21_31.2 (constants.%MakeArray.specific_fn)]
// CHECK:STDOUT:     %.loc21_3.1: ref @Test.%Array.loc17_13.2 (%Array.cf1) = splice_block %x.var {}
// CHECK:STDOUT:     %MakeArray.call: init @Test.%Array.loc17_13.2 (%Array.cf1) to %.loc21_3.1 = call %MakeArray.specific_fn.loc21_31.1()
// CHECK:STDOUT:     assign %x.var, %MakeArray.call
// CHECK:STDOUT:     %.loc21_27: type = splice_block %Array.loc21 [symbolic = %Array.loc17_13.2 (constants.%Array.cf1)] {
// CHECK:STDOUT:       %T.ref.loc21_23: %Destroy.type = name_ref T, %T.loc16_18.2 [symbolic = %T.loc16_18.1 (constants.%T.f45)]
// CHECK:STDOUT:       %int_3.loc21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %T.as_type.loc21: type = facet_access_type %T.ref.loc21_23 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:       %.loc21_23: type = converted %T.ref.loc21_23, %T.as_type.loc21 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:       %Array.loc21: type = class_type @Array, @Array(constants.%T.as_type.190, constants.%int_3) [symbolic = %Array.loc17_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @Test.%Array.loc17_13.2 (%Array.cf1) = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt.loc21_15.1: @Test.%pattern_type (%pattern_type.ed8) = ref_binding_pattern x [symbolic = %x.patt.loc21_15.2 (constants.%x.patt)]
// CHECK:STDOUT:       %x.var_patt.loc21_3.1: @Test.%pattern_type (%pattern_type.ed8) = var_pattern %x.patt.loc21_15.1 [symbolic = %x.var_patt.loc21_3.2 (constants.%x.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.facet.loc21_3.1: %Destroy.type = facet_value constants.%Array.cf1, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc17_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc21_3.2: %Destroy.type = converted constants.%Array.cf1, %Destroy.facet.loc21_3.1 [symbolic = %Destroy.facet.loc17_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %Destroy.facet.loc21_3.2: %Destroy.type = facet_value constants.%Array.cf1, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc17_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc21_3.3: %Destroy.type = converted constants.%Array.cf1, %Destroy.facet.loc21_3.2 [symbolic = %Destroy.facet.loc17_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%x.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc16_18.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc16_18.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
