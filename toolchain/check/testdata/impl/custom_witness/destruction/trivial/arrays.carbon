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

// TODO: builtin array type destruction should be trivial, and defer to
// `Core.Array as Core.Destroy`.

// --- types.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_empty_array_with_trivial_element_type.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasTrivialDestruction, 0);
fn MakeArray() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeArray();
  //@dump-sem-ir-end
}

// --- trivial_empty_array_with_nontrivial_element_type.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasNonTrivialDestruction, 0);
fn MakeArray() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeArray();
  //@dump-sem-ir-end
}

// --- trivial_populated.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasTrivialDestruction, 1);
fn MakeArray() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeArray();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_populated.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasNonTrivialDestruction, 1);

fn Test() {
  Array as Core.TrivialDestroy;
}

// --- trivial_array_of_array.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasTrivialDestruction, 3), 10);
fn MakeArray() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeArray();
  //@dump-sem-ir-end
}

// --- trivial_empty_array_of_nontrivial_populated_array.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasNonTrivialDestruction, 1), 0);
fn MakeArray() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeArray();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_array_of_array_populated.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasNonTrivialDestruction, 1), 1);

fn Test() {
  Array as Core.TrivialDestroy;
}

// --- trivial_array_adapter.carbon
library "[[@TEST_NAME]]";

// Not in `adapters.carbon`, since that file uses a smaller `min_prelude` than arrays.

import library "types";

class Adapter {
  adapt array(HasTrivialDestruction, 1);
}

fn MakeAdapter() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeAdapter();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_array_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasNonTrivialDestruction, 1);
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- trivial_constrained_facet.carbon
library "[[@TEST_NAME]]";

// Not in `facets.carbon`, since that file uses a smaller `min_prelude` than arrays.

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
  //@dump-sem-ir-end

  array(T, 3) as Core.Destroy;
  array(T, 3) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: array(T, 3) = MakeArray(T);
  //@dump-sem-ir-end
}

// --- trivial_const_populated.carbon
library "[[@TEST_NAME]]";

// Not in `const.carbon`, since that file uses a smaller `min_prelude` than arrays.

class HasTrivialDestruction {}
alias Array = const array(HasTrivialDestruction, 1);
fn MakeArray() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeArray();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_const_array.carbon
library "[[@TEST_NAME]]";

// Not in `const.carbon`, since that file uses a smaller `min_prelude` than arrays.

import library "types";

alias Array = const array(HasNonTrivialDestruction, 1);

fn Test() {
  Array as Core.TrivialDestroy;
}

// --- trivial_array_of_const.carbon
library "[[@TEST_NAME]]";

// Not in `const.carbon`, since that file uses a smaller `min_prelude` than arrays.

import library "types";

alias Array = array(const HasTrivialDestruction, 1);
fn MakeArray() -> Array;

fn Test() {
  //@dump-sem-ir-begin
  Array as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeArray();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_array_of_const.carbon
library "[[@TEST_NAME]]";

// Not in `const.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

import library "types";

alias Array = array(const HasNonTrivialDestruction, 1);

fn Test() {
  Array as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- trivial_empty_array_with_trivial_element_type.carbon = $trivial_empty_array_with_trivial_element_type
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.744: type = class_type @Array, @Array(%HasTrivialDestruction, %int_0) [concrete]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c12: %SubobjectDestroy.type = facet_value %Array.744, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c12) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.744, (%custom_witness.b80, %DestroyImpl.impl_witness.fca) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.744]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.c12]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c12]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %.loc17_13.1: ref %Array.744 = temporary_storage
// CHECK:STDOUT:   %MakeArray.call: init %Array.744 to %.loc17_13.1 = call %MakeArray.ref()
// CHECK:STDOUT:   %.loc17_13.2: ref %Array.744 = temporary %.loc17_13.1, %MakeArray.call
// CHECK:STDOUT:   %Destroy.facet.loc17_13.1: %Destroy.type = facet_value constants.%Array.744, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.fca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.3: %Destroy.type = converted constants.%Array.744, %Destroy.facet.loc17_13.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_13.2: %Destroy.type = facet_value constants.%Array.744, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.fca) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.4: %Destroy.type = converted constants.%Array.744, %Destroy.facet.loc17_13.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %Array.744) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_empty_array_with_nontrivial_element_type.carbon = $trivial_empty_array_with_nontrivial_element_type
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.fbd: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_0) [concrete]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
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
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
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
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %.loc17_13.1: ref %Array.fbd = temporary_storage
// CHECK:STDOUT:   %MakeArray.call: init %Array.fbd to %.loc17_13.1 = call %MakeArray.ref()
// CHECK:STDOUT:   %.loc17_13.2: ref %Array.fbd = temporary %.loc17_13.1, %MakeArray.call
// CHECK:STDOUT:   %Destroy.facet.loc17_13.1: %Destroy.type = facet_value constants.%Array.fbd, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.3: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc17_13.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_13.2: %Destroy.type = facet_value constants.%Array.fbd, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.cf7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.4: %Destroy.type = converted constants.%Array.fbd, %Destroy.facet.loc17_13.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %Array.fbd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_populated.carbon = $trivial_populated
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
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.351: %SubobjectDestroy.type = facet_value %Array.7b9, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.137: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.351) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.7b9, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.137) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.7b9]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.351]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.351]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %.loc17_13.1: ref %Array.7b9 = temporary_storage
// CHECK:STDOUT:   %MakeArray.call: init %Array.7b9 to %.loc17_13.1 = call %MakeArray.ref()
// CHECK:STDOUT:   %.loc17_13.2: ref %Array.7b9 = temporary %.loc17_13.1, %MakeArray.call
// CHECK:STDOUT:   %Destroy.facet.loc17_13.1: %Destroy.type = facet_value constants.%Array.7b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.3: %Destroy.type = converted constants.%Array.7b9, %Destroy.facet.loc17_13.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_13.2: %Destroy.type = facet_value constants.%Array.7b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.4: %Destroy.type = converted constants.%Array.7b9, %Destroy.facet.loc17_13.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %Array.7b9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_array_of_array.carbon = $trivial_array_of_array
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
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ef9: %SubobjectDestroy.type = facet_value %Array.a4f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d07: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ef9) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.a4f, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.d07) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.a4f]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.ef9]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.ef9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %.loc17_13.1: ref %Array.a4f = temporary_storage
// CHECK:STDOUT:   %MakeArray.call: init %Array.a4f to %.loc17_13.1 = call %MakeArray.ref()
// CHECK:STDOUT:   %.loc17_13.2: ref %Array.a4f = temporary %.loc17_13.1, %MakeArray.call
// CHECK:STDOUT:   %Destroy.facet.loc17_13.1: %Destroy.type = facet_value constants.%Array.a4f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d07) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.3: %Destroy.type = converted constants.%Array.a4f, %Destroy.facet.loc17_13.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_13.2: %Destroy.type = facet_value constants.%Array.a4f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d07) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.4: %Destroy.type = converted constants.%Array.a4f, %Destroy.facet.loc17_13.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %Array.8ec) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %Array.a4f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_empty_array_of_nontrivial_populated_array.carbon = $trivial_empty_array_of_nontrivial_populated_array
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.1a9: type = class_type @Array, @Array(%HasNonTrivialDestruction, %int_1) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.95d: type = class_type @Array, @Array(%Array.1a9, %int_0) [concrete]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cc3: %SubobjectDestroy.type = facet_value %Array.95d, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8d5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cc3) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.95d, (%custom_witness.b80, %DestroyImpl.impl_witness.8d5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc10: type = name_ref Array, file.%Array.loc5_7 [concrete = constants.%Array.95d]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.cc3]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Array.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.cc3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %.loc17_13.1: ref %Array.95d = temporary_storage
// CHECK:STDOUT:   %MakeArray.call: init %Array.95d to %.loc17_13.1 = call %MakeArray.ref()
// CHECK:STDOUT:   %.loc17_13.2: ref %Array.95d = temporary %.loc17_13.1, %MakeArray.call
// CHECK:STDOUT:   %Destroy.facet.loc17_13.1: %Destroy.type = facet_value constants.%Array.95d, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.8d5) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.3: %Destroy.type = converted constants.%Array.95d, %Destroy.facet.loc17_13.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_13.2: %Destroy.type = facet_value constants.%Array.95d, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.8d5) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_13.4: %Destroy.type = converted constants.%Array.95d, %Destroy.facet.loc17_13.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %Array.95d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_array_adapter.carbon = $trivial_array_adapter
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.7b9: type = class_type @Array, @Array(%HasTrivialDestruction, %int_1) [concrete]
// CHECK:STDOUT:   %MakeAdapter.type: type = fn_type @MakeAdapter [concrete]
// CHECK:STDOUT:   %MakeAdapter: %MakeAdapter.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7cf) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Adapter.ref.loc15: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc15, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %Adapter.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeAdapter.ref: %MakeAdapter.type = name_ref MakeAdapter, file.%MakeAdapter.decl [concrete = constants.%MakeAdapter]
// CHECK:STDOUT:   %.loc22_15.1: ref %Adapter = temporary_storage
// CHECK:STDOUT:   %MakeAdapter.call: init %Adapter to %.loc22_15.1 = call %MakeAdapter.ref()
// CHECK:STDOUT:   %.loc22_15.2: ref %Adapter = temporary %.loc22_15.1, %MakeAdapter.call
// CHECK:STDOUT:   %Destroy.facet.loc22_15.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_15.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc22_15.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc22_15.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_15.4: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc22_15.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc22_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.3(%self.param: ref %Array.7b9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.4(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_constrained_facet.carbon = $trivial_constrained_facet
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
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
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc14_18.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc16_9.2: type = facet_access_type %T.loc14_18.1 [symbolic = %T.as_type.loc16_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:   %Array.loc16_13.2: type = class_type @Array, @Array(%T.as_type.loc16_9.2, constants.%int_3) [symbolic = %Array.loc16_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc16_13.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.518)]
// CHECK:STDOUT:   %SubobjectDestroy.facet.loc16_15.2: %SubobjectDestroy.type = facet_value %Array.loc16_13.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet.loc16_15.2 (constants.%SubobjectDestroy.facet.bdc)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc16_13.2 [symbolic = %require_complete (constants.%require_complete.d5e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc16_13.2 [symbolic = %pattern_type (constants.%pattern_type.ed8)]
// CHECK:STDOUT:   %x.patt.loc23_15.2: @Test.%pattern_type (%pattern_type.ed8) = ref_binding_pattern x [symbolic = %x.patt.loc23_15.2 (constants.%x.patt)]
// CHECK:STDOUT:   %x.var_patt.loc23_3.2: @Test.%pattern_type (%pattern_type.ed8) = var_pattern %x.patt.loc23_15.2 [symbolic = %x.var_patt.loc23_3.2 (constants.%x.var_patt)]
// CHECK:STDOUT:   %MakeArray.specific_fn.loc23_31.2: <specific function> = specific_function constants.%MakeArray, @MakeArray(%T.loc14_18.1) [symbolic = %MakeArray.specific_fn.loc23_31.2 (constants.%MakeArray.specific_fn)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc16: %Destroy.type = name_ref T, %T.loc14_18.2 [symbolic = %T.loc14_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %int_3.loc16: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %T.as_type.loc16_9.1: type = facet_access_type %T.ref.loc16 [symbolic = %T.as_type.loc16_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %.loc16_9: type = converted %T.ref.loc16, %T.as_type.loc16_9.1 [symbolic = %T.as_type.loc16_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:     %Array.loc16_13.1: type = class_type @Array, @Array(constants.%T.as_type.190, constants.%int_3) [symbolic = %Array.loc16_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:     %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:     %SubobjectDestroy.facet.loc16_15.1: %SubobjectDestroy.type = facet_value %Array.loc16_13.1, (constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %SubobjectDestroy.facet.loc16_15.2 (constants.%SubobjectDestroy.facet.bdc)]
// CHECK:STDOUT:     %.loc16_15: %SubobjectDestroy.type = converted %Array.loc16_13.1, %SubobjectDestroy.facet.loc16_15.1 [symbolic = %SubobjectDestroy.facet.loc16_15.2 (constants.%SubobjectDestroy.facet.bdc)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %x.var: ref @Test.%Array.loc16_13.2 (%Array.cf1) = var_storage %x.var_patt.loc23_3.1
// CHECK:STDOUT:     %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:     %T.ref.loc23_41: %Destroy.type = name_ref T, %T.loc14_18.2 [symbolic = %T.loc14_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %MakeArray.specific_fn.loc23_31.1: <specific function> = specific_function %MakeArray.ref, @MakeArray(constants.%T.f45) [symbolic = %MakeArray.specific_fn.loc23_31.2 (constants.%MakeArray.specific_fn)]
// CHECK:STDOUT:     %.loc23_3.1: ref @Test.%Array.loc16_13.2 (%Array.cf1) = splice_block %x.var {}
// CHECK:STDOUT:     %MakeArray.call: init @Test.%Array.loc16_13.2 (%Array.cf1) to %.loc23_3.1 = call %MakeArray.specific_fn.loc23_31.1()
// CHECK:STDOUT:     assign %x.var, %MakeArray.call
// CHECK:STDOUT:     %.loc23_27: type = splice_block %Array.loc23 [symbolic = %Array.loc16_13.2 (constants.%Array.cf1)] {
// CHECK:STDOUT:       %T.ref.loc23_23: %Destroy.type = name_ref T, %T.loc14_18.2 [symbolic = %T.loc14_18.1 (constants.%T.f45)]
// CHECK:STDOUT:       %int_3.loc23: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %T.as_type.loc23: type = facet_access_type %T.ref.loc23_23 [symbolic = %T.as_type.loc16_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:       %.loc23_23: type = converted %T.ref.loc23_23, %T.as_type.loc23 [symbolic = %T.as_type.loc16_9.2 (constants.%T.as_type.190)]
// CHECK:STDOUT:       %Array.loc23: type = class_type @Array, @Array(constants.%T.as_type.190, constants.%int_3) [symbolic = %Array.loc16_13.2 (constants.%Array.cf1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @Test.%Array.loc16_13.2 (%Array.cf1) = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt.loc23_15.1: @Test.%pattern_type (%pattern_type.ed8) = ref_binding_pattern x [symbolic = %x.patt.loc23_15.2 (constants.%x.patt)]
// CHECK:STDOUT:       %x.var_patt.loc23_3.1: @Test.%pattern_type (%pattern_type.ed8) = var_pattern %x.patt.loc23_15.1 [symbolic = %x.var_patt.loc23_3.2 (constants.%x.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.facet.loc23_3.1: %Destroy.type = facet_value constants.%Array.cf1, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc19_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc23_3.2: %Destroy.type = converted constants.%Array.cf1, %Destroy.facet.loc23_3.1 [symbolic = %Destroy.facet.loc19_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %Destroy.facet.loc23_3.2: %Destroy.type = facet_value constants.%Array.cf1, (constants.%DestroyImpl.lookup_impl_witness.8a2, constants.%SubobjectDestroy.lookup_impl_witness.518) [symbolic = %Destroy.facet.loc19_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc23_3.3: %Destroy.type = converted constants.%Array.cf1, %Destroy.facet.loc23_3.2 [symbolic = %Destroy.facet.loc19_15.2 (constants.%Destroy.facet)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%x.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc14_18.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc14_18.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_const_populated.carbon = $trivial_const_populated
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
// CHECK:STDOUT:   %const: type = const_type %Array.7b9 [concrete]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.27a: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ea4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.27a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.ea4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc11: type = name_ref Array, file.%Array.loc6_7 [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc11, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.27a]
// CHECK:STDOUT:   %.loc11: %SubobjectDestroy.type = converted %Array.ref.loc11, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.27a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %.loc18_13.1: ref %const = temporary_storage
// CHECK:STDOUT:   %MakeArray.call: init %const to %.loc18_13.1 = call %MakeArray.ref()
// CHECK:STDOUT:   %.loc18_13.2: ref %const = temporary %.loc18_13.1, %MakeArray.call
// CHECK:STDOUT:   %Destroy.facet.loc18_13.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ea4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc18_13.3: %Destroy.type = converted constants.%const, %Destroy.facet.loc18_13.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc18_13.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ea4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc18_13.4: %Destroy.type = converted constants.%const, %Destroy.facet.loc18_13.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.3(%self.param: ref %Array.7b9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.4(%self.param: ref %const) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_array_of_const.carbon = $trivial_array_of_const
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %const: type = const_type %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.aa4: type = class_type @Array, @Array(%const, %int_1) [concrete]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c7b: %SubobjectDestroy.type = facet_value %Array.aa4, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c16: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c7b) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Array.aa4, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.c16) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Array.ref.loc12: type = name_ref Array, file.%Array.loc7_7 [concrete = constants.%Array.aa4]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.ref.loc12, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.c7b]
// CHECK:STDOUT:   %.loc12: %SubobjectDestroy.type = converted %Array.ref.loc12, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c7b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeArray.ref: %MakeArray.type = name_ref MakeArray, file.%MakeArray.decl [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %.loc19_13.1: ref %Array.aa4 = temporary_storage
// CHECK:STDOUT:   %MakeArray.call: init %Array.aa4 to %.loc19_13.1 = call %MakeArray.ref()
// CHECK:STDOUT:   %.loc19_13.2: ref %Array.aa4 = temporary %.loc19_13.1, %MakeArray.call
// CHECK:STDOUT:   %Destroy.facet.loc19_13.1: %Destroy.type = facet_value constants.%Array.aa4, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.c16) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc19_13.3: %Destroy.type = converted constants.%Array.aa4, %Destroy.facet.loc19_13.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc19_13.2: %Destroy.type = facet_value constants.%Array.aa4, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.c16) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc19_13.4: %Destroy.type = converted constants.%Array.aa4, %Destroy.facet.loc19_13.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_9.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_9.3(%self.param: ref %const) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_9.4(%self.param: ref %Array.aa4) = "no_op";
// CHECK:STDOUT:
