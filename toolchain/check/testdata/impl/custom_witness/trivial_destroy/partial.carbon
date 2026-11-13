// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/partial.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_empty_partial_class.carbon
library "[[@TEST_NAME]]";

base class Base {}
fn Make() -> Base;

fn Test() {
  partial Base as Core.Destroy;
  partial Base as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: partial Base = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nonempty_partial_class.carbon
library "[[@TEST_NAME]]";

class A {}
class B {}

base class Base {
  var a: A;
  var b: B;
}

fn Make() -> partial Base;

fn Test() {
  partial Base as Core.SubobjectDestroy;
  partial Base as Core.Destroy;
  partial Base as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: partial Base = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_member.carbon
library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Make() -> Base;

fn Test() {
  partial Base as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_impls_destroy.carbon
library "[[@TEST_NAME]]";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  partial Base as Core.TrivialDestroy;
}

// --- todo_trivial_abstract.carbon
library "[[@TEST_NAME]]";

abstract class Abstract {}
fn Make() -> partial Abstract;

fn Test() {
  partial Abstract as Core.SubobjectDestroy;
  partial Abstract as Core.Destroy;
  partial Abstract as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: partial Abstract = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_abstract.carbon
library "[[@TEST_NAME]]";

import library "types";

abstract class Abstract {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Test() {
  partial Abstract as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- todo_trivial_empty_partial_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_16.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.edf, (%DestroyImpl.impl_witness.1d2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %x.var: ref %.edf = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_3.1: ref %.edf = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Base to %.loc11_3.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc11_3.2: init %.edf = as_compatible %Make.call
// CHECK:STDOUT:   %.loc11_3.3: init %.edf = converted %Make.call, %.loc11_3.2
// CHECK:STDOUT:   assign %x.var, %.loc11_3.3
// CHECK:STDOUT:   %.loc11_17.1: type = splice_block %.loc11_17.2 [concrete = constants.%.edf] {
// CHECK:STDOUT:     %Base.ref.loc11: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %.loc11_17.2: type = partial_type %Base.ref.loc11 [concrete = constants.%.edf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %.edf = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_3.4: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_3.5: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_nonempty_partial_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_16.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.96b: %Destroy.type = facet_value %.edf, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.1d2) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %x.var: ref %.edf = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc19_3.1: ref %.edf = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %.edf to %.loc19_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc19_17.1: type = splice_block %.loc19_17.2 [concrete = constants.%.edf] {
// CHECK:STDOUT:     %Base.ref.loc19: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %.loc19_17.2: type = partial_type %Base.ref.loc19 [concrete = constants.%.edf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %.edf = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc19_3.1: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc19_3.2: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc19_3.1 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %Destroy.facet.loc19_3.2: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc19_3.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc19_3.2 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %.79f: type = partial_type %Abstract [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %.79f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_20.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c03: %SubobjectDestroy.type = facet_value %.79f, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.98a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c03) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.79f, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.98a) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.5f8 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.5f8 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %x.var: ref %.79f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_3.1: ref %.79f = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %.79f to %.loc12_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc12_17.1: type = splice_block %.loc12_17.2 [concrete = constants.%.79f] {
// CHECK:STDOUT:     %Abstract.ref.loc12: type = name_ref Abstract, file.%Abstract.decl [concrete = constants.%Abstract]
// CHECK:STDOUT:     %.loc12_17.2: type = partial_type %Abstract.ref.loc12 [concrete = constants.%.79f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %.79f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.5f8 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.5f8 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.2: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.3: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
