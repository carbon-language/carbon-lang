// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/structs.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/structs.carbon

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


// --- todo_nontrivial_all_nontrivial_fields.carbon

library "[[@TEST_NAME]]";

import library "types";

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

alias Struct = {.a: ImplsDestroy, .b: AlsoImplsDestroy, .c: AlsoImplsDestroy, .d: ImplsDestroy};
fn Make() -> Struct;

fn Test() {
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
  //@dump-sem-ir-begin
  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_struct_nontrivial_field_first.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.02c: type = struct_type {.a: %ImplsDestroy, .b: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %struct_type.a.b.02c [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c60 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c60 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d56: %SubobjectDestroy.type = facet_value %struct_type.a.b.02c, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.65e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d56) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ff6: %Destroy.type = facet_value %struct_type.a.b.02c, (%DestroyImpl.impl_witness.65e, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.02c = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_3.1: ref %struct_type.a.b.02c = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %struct_type.a.b.02c to %.loc11_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.02c]
// CHECK:STDOUT:   %x: ref %struct_type.a.b.02c = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c60 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c60 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.02c, (constants.%DestroyImpl.impl_witness.65e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%struct_type.a.b.02c, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.02c, (constants.%DestroyImpl.impl_witness.65e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%struct_type.a.b.02c, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.4(%self.param: ref %struct_type.a.b.02c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_3.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_3.1)
// CHECK:STDOUT:   %.loc11_3.4: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc11_3.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc11_3.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc11_3.3 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc11_3.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc11_3.4 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc11_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nontrivial_field_second.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.a.b.bb2: type = struct_type {.a: %HasTrivialDestruction, .b: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.fad: type = pattern_type %struct_type.a.b.bb2 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.fad = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.fad = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bae: %SubobjectDestroy.type = facet_value %struct_type.a.b.bb2, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.de5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.bae) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3e4: %Destroy.type = facet_value %struct_type.a.b.bb2, (%DestroyImpl.impl_witness.de5, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.bb2 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_3.1: ref %struct_type.a.b.bb2 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %struct_type.a.b.bb2 to %.loc11_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.bb2]
// CHECK:STDOUT:   %x: ref %struct_type.a.b.bb2 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.fad = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.fad = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.bb2, (constants.%DestroyImpl.impl_witness.de5, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.3e4]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%struct_type.a.b.bb2, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.3e4]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.bb2, (constants.%DestroyImpl.impl_witness.de5, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.3e4]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%struct_type.a.b.bb2, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.3e4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.3(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.4(%self.param: ref %struct_type.a.b.bb2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_3.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_3.1)
// CHECK:STDOUT:   %.loc11_3.4: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc11_3.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc11_3.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc11_3.3 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc11_3.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc11_3.4 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc11_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e7f: <witness> = impl_witness @AlsoImplsDestroy.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.f50: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %AlsoImplsDestroy, .d: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.ae4: type = pattern_type %struct_type.a.b.c.d.f50 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.ae4 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.ae4 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.122: %Destroy.type = facet_value %AlsoImplsDestroy, (%DestroyImpl.impl_witness.e7f, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bb: %SubobjectDestroy.type = facet_value %struct_type.a.b.c.d.f50, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e15: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bb) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ecb: %Destroy.type = facet_value %struct_type.a.b.c.d.f50, (%DestroyImpl.impl_witness.e15, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.c.d.f50 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %struct_type.a.b.c.d.f50 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %struct_type.a.b.c.d.f50 to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.c.d.f50]
// CHECK:STDOUT:   %x: ref %struct_type.a.b.c.d.f50 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.ae4 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.ae4 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.f50, (constants.%DestroyImpl.impl_witness.e15, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ecb]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%struct_type.a.b.c.d.f50, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.ecb]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.f50, (constants.%DestroyImpl.impl_witness.e15, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ecb]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%struct_type.a.b.c.d.f50, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.ecb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.3(%self.param: ref %AlsoImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.4(%self.param: ref %struct_type.a.b.c.d.f50) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_3.1: ref %ImplsDestroy = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc17_3.1)
// CHECK:STDOUT:   %.loc17_3.4: ref %AlsoImplsDestroy = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc17_3.3: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc17_3.5: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_3.3 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.4: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc17_3.6: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_3.4 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc17_3.4)
// CHECK:STDOUT:   %.loc17_3.7: ref %AlsoImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc17_3.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc17_3.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_3.5 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc17_3.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc17_3.6 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc17_3.7)
// CHECK:STDOUT:   %.loc17_3.10: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_3.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc17_3.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_3.7 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc17_3.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc17_3.8 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc17_3.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.0cd: type = struct_type {.b: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %struct_type.a.13f: type = struct_type {.a: %struct_type.b.0cd} [concrete]
// CHECK:STDOUT:   %pattern_type.378: type = pattern_type %struct_type.a.13f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.378 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.378 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.135: %SubobjectDestroy.type = facet_value %struct_type.b.0cd, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.436: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.135) [concrete]
// CHECK:STDOUT:   %Destroy.facet.2df: %Destroy.type = facet_value %struct_type.b.0cd, (%DestroyImpl.impl_witness.436, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.985: %SubobjectDestroy.type = facet_value %struct_type.a.13f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.973: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.985) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b26: %Destroy.type = facet_value %struct_type.a.13f, (%DestroyImpl.impl_witness.973, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %struct_type.a.13f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %struct_type.a.13f = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.13f]
// CHECK:STDOUT:   %x: ref %struct_type.a.13f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.378 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.378 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%DestroyImpl.impl_witness.973, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   %.loc11_3.1: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%DestroyImpl.impl_witness.973, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.3(%self.param: ref %struct_type.b.0cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_3.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc11_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_3.4(%self.param: ref %struct_type.a.13f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_3.1: ref %struct_type.b.0cd = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%struct_type.b.0cd, (constants.%DestroyImpl.impl_witness.436, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%struct_type.b.0cd, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%struct_type.b.0cd, (constants.%DestroyImpl.impl_witness.436, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%struct_type.b.0cd, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc11_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
