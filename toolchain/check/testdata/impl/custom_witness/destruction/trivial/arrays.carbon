// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/arrays.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/arrays.carbon

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
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_empty_array_with_nontrivial_element_type.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasNonTrivialDestruction, 0);
fn Make() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_populated.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasTrivialDestruction, 1);
fn Make() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

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
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

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
library "[[@TEST_NAME]]";

// Not in `adapters.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

import library "types";

class Adapter {
  adapt array(HasTrivialDestruction, 2);
}

fn Make() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

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
library "[[@TEST_NAME]]";

// Not in `facets.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

fn MakeElement(generic T: Core.TrivialDestroy) -> T {
  returned var result: T;
  return var;
}

fn MakeArray(generic T: Core.TrivialDestroy) -> array(T, 3) {
  return (MakeElement(T), MakeElement(T), MakeElement(T));
}

fn Test(generic T: Core.TrivialDestroy) {
  //@dump-sem-ir-begin
  array(T, 3) as Core.SubobjectDestroy;
  array(T, 3) as Core.Destroy;
  array(T, 3) as Core.TrivialDestroy;

  var unused x: array(T, 3) = MakeArray(T);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_trivial_empty_array_with_trivial_element_type.carbon = $todo_trivial_empty_array_with_trivial_element_type
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.68f: type = class_type @Array, @Array(%i32, %int_0) [concrete]
// CHECK:STDOUT:   %pattern_type.a2e: type = pattern_type %Array.68f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8b7: %SubobjectDestroy.type = facet_value %Array.68f, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.68f, (%custom_witness.b80, %DestroyImpl.impl_witness.cca) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a2e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.a2e = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.68f]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.8b7]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.8b7]
// CHECK:STDOUT:   %Array.ref.loc11: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.68f]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Array.ref.loc11, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Array.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Array.ref.loc12: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.68f]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Array.ref.loc12, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Array.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %Array.68f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.68f = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.68f to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.68f]
// CHECK:STDOUT:   %x: ref %Array.68f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a2e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.a2e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.68f, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.68f, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.68f, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.68f, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %Array.68f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_empty_array_with_nontrivial_element_type.carbon = $todo_trivial_empty_array_with_nontrivial_element_type
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.fbd: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_0) [concrete]
// CHECK:STDOUT:   %pattern_type.16c: type = pattern_type %Array.fbd [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6c7: %SubobjectDestroy.type = facet_value %Array.fbd, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cf7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6c7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.fbd, (%custom_witness.b80, %DestroyImpl.impl_witness.cf7) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.16c = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.16c = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.fbd]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.6c7]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.6c7]
// CHECK:STDOUT:   %Array.ref.loc11: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.fbd]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Array.ref.loc11, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Array.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Array.ref.loc12: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.fbd]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Array.ref.loc12, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Array.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %Array.fbd = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.fbd = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.fbd to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.fbd]
// CHECK:STDOUT:   %x: ref %Array.fbd = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.16c = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.16c = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.fbd, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.fbd, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %Array.fbd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_populated.carbon = $todo_trivial_populated
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.7b9: type = class_type @Array, @Array(%HasTrivialDestruction, %int_1) [concrete]
// CHECK:STDOUT:   %pattern_type.f24: type = pattern_type %Array.7b9 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.351: %SubobjectDestroy.type = facet_value %Array.7b9, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.137: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.351) [concrete]
// CHECK:STDOUT:   %Destroy.facet.db3: %Destroy.type = facet_value %Array.7b9, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.137) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f24 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.f24 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.7b9]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.351]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.351]
// CHECK:STDOUT:   %Array.ref.loc11: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.7b9]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Array.ref.loc11, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Array.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   %Array.ref.loc12: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.7b9]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Array.ref.loc12, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Array.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   %x.var: ref %Array.7b9 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.7b9 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.7b9 to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.7b9]
// CHECK:STDOUT:   %x: ref %Array.7b9 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f24 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.f24 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.7b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.7b9, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.7b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.7b9, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.db3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %Array.7b9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_9.1: ref %HasTrivialDestruction = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_array_of_array.carbon = $todo_trivial_array_of_array
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.8ec: type = class_type @Array, @Array(%HasTrivialDestruction, %int_3) [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Array.a4f: type = class_type @Array, @Array(%Array.8ec, %int_10) [concrete]
// CHECK:STDOUT:   %pattern_type.e6e: type = pattern_type %Array.a4f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8c9: %SubobjectDestroy.type = facet_value %Array.8ec, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bb8: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8c9) [concrete]
// CHECK:STDOUT:   %Destroy.facet.248: %Destroy.type = facet_value %Array.8ec, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.bb8) [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ef9: %SubobjectDestroy.type = facet_value %Array.a4f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d07: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ef9) [concrete]
// CHECK:STDOUT:   %Destroy.facet.afd: %Destroy.type = facet_value %Array.a4f, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.d07) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e6e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e6e = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.a4f]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.ef9]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.ef9]
// CHECK:STDOUT:   %Array.ref.loc11: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.a4f]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Array.ref.loc11, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d07) [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Array.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   %Array.ref.loc12: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.a4f]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Array.ref.loc12, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d07) [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Array.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   %x.var: ref %Array.a4f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.a4f = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Array.a4f to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Array.ref.loc14: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.a4f]
// CHECK:STDOUT:   %x: ref %Array.a4f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e6e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e6e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.a4f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d07) [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.a4f, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.a4f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d07) [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.a4f, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.afd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %Array.8ec) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_9.1: ref %HasTrivialDestruction = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_9.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_9.4: ref %HasTrivialDestruction = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_9.4)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_9.7: ref %HasTrivialDestruction = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.5: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.8: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.5 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.6: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_9.9: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_9.6 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_9.7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %Array.a4f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc10_9.1: ref %Array.8ec = array_index %self.param, %int_9
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.3: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_9.1)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc10_9.4: ref %Array.8ec = array_index %self.param, %int_8
// CHECK:STDOUT:   %Destroy.facet.loc10_9.3: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.5: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.3 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.4: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.6: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.4 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_9.4)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc10_9.7: ref %Array.8ec = array_index %self.param, %int_7
// CHECK:STDOUT:   %Destroy.facet.loc10_9.5: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.8: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.5 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.6: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.9: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.6 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_9.7)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc10_9.10: ref %Array.8ec = array_index %self.param, %int_6
// CHECK:STDOUT:   %Destroy.facet.loc10_9.7: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.11: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.7 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.8: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.12: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.8 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc10_9.10)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc10_9.13: ref %Array.8ec = array_index %self.param, %int_5
// CHECK:STDOUT:   %Destroy.facet.loc10_9.9: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.14: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.9 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.10: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.15: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.10 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.5: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_9.13)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc10_9.16: ref %Array.8ec = array_index %self.param, %int_4
// CHECK:STDOUT:   %Destroy.facet.loc10_9.11: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.17: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.11 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.12: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.18: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.12 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.6: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc10_9.16)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc10_9.19: ref %Array.8ec = array_index %self.param, %int_3
// CHECK:STDOUT:   %Destroy.facet.loc10_9.13: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.20: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.13 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.14: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.21: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.14 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.7: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc10_9.19)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_9.22: ref %Array.8ec = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.facet.loc10_9.15: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.23: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.15 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.16: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.24: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.16 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.8: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%.loc10_9.22)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_9.25: ref %Array.8ec = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc10_9.17: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.26: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.17 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.18: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.27: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.18 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.9: init %empty_tuple.type = call %SelfDestruct.specific_fn.9(%.loc10_9.25)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_9.28: ref %Array.8ec = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.19: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.29: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.19 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.20: %Destroy.type = facet_value constants.%Array.8ec, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bb8) [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   %.loc10_9.30: %Destroy.type = converted constants.%Array.8ec, %Destroy.facet.loc10_9.20 [concrete = constants.%Destroy.facet.248]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_9.10: init %empty_tuple.type = call %SelfDestruct.specific_fn.10(%.loc10_9.28)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_adapter.carbon = $todo_trivial_adapter
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.d89: type = class_type @Array, @Array(%HasTrivialDestruction, %int_2) [concrete]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.534: %Destroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7cf) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Adapter.ref.loc16: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc16, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc16: %SubobjectDestroy.type = converted %Adapter.ref.loc16, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %Adapter.ref.loc17: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc17: %Destroy.type = facet_value %Adapter.ref.loc17, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc17: %Destroy.type = converted %Adapter.ref.loc17, %Destroy.facet.loc17 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %Adapter.ref.loc18: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc18: %Destroy.type = facet_value %Adapter.ref.loc18, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc18: %Destroy.type = converted %Adapter.ref.loc18, %Destroy.facet.loc18 [concrete = constants.%Destroy.facet.534]
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
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc20_3.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_11.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_11.3(%self.param: ref %Array.d89) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc16_11.1: ref %HasTrivialDestruction = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc16_11.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc16_11.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc16_11.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc16_11.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc16_11.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc16_11.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc16_11.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc16_11.4: ref %HasTrivialDestruction = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc16_11.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc16_11.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc16_11.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc16_11.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc16_11.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc16_11.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc16_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_11.4(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_constrained_facet.carbon = $todo_trivial_constrained_facet
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190: type = facet_access_type %T.f45 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.cf1: type = class_type @Array, @Array(%T.as_type.190, %int_3) [symbolic]
// CHECK:STDOUT:   %pattern_type.ed8: type = pattern_type %Array.cf1 [symbolic]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d5e: <witness> = require_complete_type %Array.cf1 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.518: <witness> = lookup_impl_witness %Array.cf1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bdc: %SubobjectDestroy.type = facet_value %Array.cf1, (%SubobjectDestroy.lookup_impl_witness.518) [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.8a2: <witness> = lookup_impl_witness %Array.cf1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.cf1, (%DestroyImpl.lookup_impl_witness.8a2, %SubobjectDestroy.lookup_impl_witness.518) [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.ed8 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.ed8 = var_pattern %x.patt [symbolic]
// CHECK:STDOUT:   %MakeArray.specific_fn: <specific function> = specific_function %MakeArray, @MakeArray(%T.f45) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc15_18.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc17_9.2: type = facet_access_type %T.loc15_18.1 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:   %Array.loc17_13.2: type = class_type @Array, @Array(%T.as_type.loc17_9.2, constants.%int_3) [symbolic = %Array.loc17_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc17_13.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.518)]
// CHECK:STDOUT:   %SubobjectDestroy.facet.loc17_15.2: %SubobjectDestroy.type = facet_value %Array.loc17_13.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet.loc17_15.2 (constants.%SubobjectDestroy.facet.bdc)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc17_13.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.8a2)]
// CHECK:STDOUT:   %Destroy.facet.loc18_15.2: %Destroy.type = facet_value %Array.loc17_13.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc17_13.2 [symbolic = %require_complete (constants.%require_complete.d5e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc17_13.2 [symbolic = %pattern_type (constants.%pattern_type.ed8)]
// CHECK:STDOUT:   %x.patt.loc21_15.2: @Test.%pattern_type (%pattern_type.ed8) = ref_binding_pattern x [symbolic = %x.patt.loc21_15.2 (constants.%x.patt)]
// CHECK:STDOUT:   %x.var_patt.loc21_3.2: @Test.%pattern_type (%pattern_type.ed8) = var_pattern %x.patt.loc21_15.2 [symbolic = %x.var_patt.loc21_3.2 (constants.%x.var_patt)]
// CHECK:STDOUT:   %MakeArray.specific_fn.loc21_31.2: <specific function> = specific_function constants.%MakeArray, @MakeArray(%T.loc15_18.1) [symbolic = %MakeArray.specific_fn.loc21_31.2 (constants.%MakeArray.specific_fn)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc17: %Destroy.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %int_3.loc17: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %T.as_type.loc17_9.1: type = facet_access_type %T.ref.loc17 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %.loc17_9: type = converted %T.ref.loc17, %T.as_type.loc17_9.1 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %Array.loc17_13.1: type = class_type @Array, @Array(constants.%T.as_type.190, constants.%int_3) [symbolic = %Array.loc17_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:     %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:     %SubobjectDestroy.facet.loc17_15.1: %SubobjectDestroy.type = facet_value %Array.loc17_13.1, (constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %SubobjectDestroy.facet.loc17_15.2 (constants.%SubobjectDestroy.facet.bdc)]
// CHECK:STDOUT:     %.loc17_15: %SubobjectDestroy.type = converted %Array.loc17_13.1, %SubobjectDestroy.facet.loc17_15.1 [symbolic = %SubobjectDestroy.facet.loc17_15.2 (constants.%SubobjectDestroy.facet.bdc)]
// CHECK:STDOUT:     %T.ref.loc18: %Destroy.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %int_3.loc18: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %T.as_type.loc18: type = facet_access_type %T.ref.loc18 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %.loc18_9: type = converted %T.ref.loc18, %T.as_type.loc18 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %Array.loc18: type = class_type @Array, @Array(constants.%T.as_type.190, constants.%int_3) [symbolic = %Array.loc17_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:     %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     %Destroy.facet.loc18_15.1: %Destroy.type = facet_value %Array.loc18, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc18_15: %Destroy.type = converted %Array.loc18, %Destroy.facet.loc18_15.1 [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %T.ref.loc19: %Destroy.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %int_3.loc19: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %.loc19_9: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.as_type.loc17_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %Array.loc19: type = class_type @Array, @Array(constants.%T.as_type.190, constants.%int_3) [symbolic = %Array.loc17_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:     %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %TrivialDestroy.ref.loc19: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     %Destroy.facet.loc19: %Destroy.type = facet_value %Array.loc19, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc19_15: %Destroy.type = converted %Array.loc19, %Destroy.facet.loc19 [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %x.var: ref @Test.%Array.loc17_13.2 (%Array.cf1) = var_storage %x.var_patt.loc21_3.1
// CHECK:STDOUT:     %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:     %T.ref.loc21_41: %Destroy.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %MakeArray.specific_fn.loc21_31.1: <specific function> = specific_function %MakeArray.ref, @MakeArray(constants.%T.f45) [symbolic = %MakeArray.specific_fn.loc21_31.2 (constants.%MakeArray.specific_fn)]
// CHECK:STDOUT:     %.loc21_3.1: ref @Test.%Array.loc17_13.2 (%Array.cf1) = splice_block %x.var {}
// CHECK:STDOUT:     %MakeArray.call: init @Test.%Array.loc17_13.2 (%Array.cf1) to %.loc21_3.1 = call %MakeArray.specific_fn.loc21_31.1()
// CHECK:STDOUT:     assign %x.var, %MakeArray.call
// CHECK:STDOUT:     %.loc21_27: type = splice_block %Array.loc21 [symbolic = %Array.loc17_13.2 (constants.%Array.cf1)] {
// CHECK:STDOUT:       %T.ref.loc21_23: %Destroy.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T.f45)]
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
// CHECK:STDOUT:     %Destroy.facet.loc21_3.1: %Destroy.type = facet_value constants.%Array.cf1, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc21_3.2: %Destroy.type = converted constants.%Array.cf1, %Destroy.facet.loc21_3.1 [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %Destroy.facet.loc21_3.2: %Destroy.type = facet_value constants.%Array.cf1, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc21_3.3: %Destroy.type = converted constants.%Array.cf1, %Destroy.facet.loc21_3.2 [symbolic = %Destroy.facet.loc18_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%x.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc15_18.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc15_18.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
