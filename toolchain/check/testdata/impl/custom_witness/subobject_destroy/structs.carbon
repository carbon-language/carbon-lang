// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/subobject_destroy/structs.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/subobject_destroy/structs.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class A {}
class B {}

// Used when mixing with `ImplsDestroy`.
class HasTrivialDestruction {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

// --- todo_nontrivial_all_nontrivial_fields.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: ImplsDestroy, .b: AlsoImplsDestroy, .c: AlsoImplsDestroy, .d: ImplsDestroy};
fn Make() -> Struct;

fn Test() {
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_one_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: ImplsDestroy, .b: AlsoImplsDestroy, .c: HasTrivialDestruction, .d: ImplsDestroy};
fn Make() -> Struct;

fn Test() {
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: ImplsDestroy}};
fn Make() -> Struct;

fn Test() {
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.f50: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %AlsoImplsDestroy, .d: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.ae4: type = pattern_type %struct_type.a.b.c.d.f50 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bb: %SubobjectDestroy.type = facet_value %struct_type.a.b.c.d.f50, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e15: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bb) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a69: %Destroy.type = facet_value %struct_type.a.b.c.d.f50, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.e15) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.ae4 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.ae4 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.c.d.f50 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %struct_type.a.b.c.d.f50 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %struct_type.a.b.c.d.f50 to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref.loc13: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.c.d.f50]
// CHECK:STDOUT:   %x: ref %struct_type.a.b.c.d.f50 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.ae4 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.ae4 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.f50, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e15) [concrete = constants.%Destroy.facet.a69]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%struct_type.a.b.c.d.f50, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.a69]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.f50, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e15) [concrete = constants.%Destroy.facet.a69]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%struct_type.a.b.c.d.f50, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.a69]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_one_trivial_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.c6a: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %HasTrivialDestruction, .d: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.a4d: type = pattern_type %struct_type.a.b.c.d.c6a [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_10.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fb: %SubobjectDestroy.type = facet_value %struct_type.a.b.c.d.c6a, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7e6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fb) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9cc: %Destroy.type = facet_value %struct_type.a.b.c.d.c6a, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.7e6) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a4d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.a4d = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.c.d.c6a = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %struct_type.a.b.c.d.c6a = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %struct_type.a.b.c.d.c6a to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref.loc13: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.c.d.c6a]
// CHECK:STDOUT:   %x: ref %struct_type.a.b.c.d.c6a = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a4d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.a4d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.c6a, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.7e6) [concrete = constants.%Destroy.facet.9cc]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%struct_type.a.b.c.d.c6a, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.9cc]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.c6a, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.7e6) [concrete = constants.%Destroy.facet.9cc]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%struct_type.a.b.c.d.c6a, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.9cc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.b.0cd: type = struct_type {.b: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %struct_type.a.13f: type = struct_type {.a: %struct_type.b.0cd} [concrete]
// CHECK:STDOUT:   %pattern_type.378: type = pattern_type %struct_type.a.13f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.985: %SubobjectDestroy.type = facet_value %struct_type.a.13f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.973: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.985) [concrete]
// CHECK:STDOUT:   %Destroy.facet.49c: %Destroy.type = facet_value %struct_type.a.13f, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.973) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.378 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.378 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %struct_type.a.13f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %struct_type.a.13f = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref.loc13: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.13f]
// CHECK:STDOUT:   %x: ref %struct_type.a.13f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.378 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.378 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.973) [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   %.loc13_3.1: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.973) [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.49c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
