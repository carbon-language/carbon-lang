// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/partial.carbon

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

// --- trivial_empty_partial_class.carbon

library "[[@TEST_NAME]]";

base class Base {}

fn Test() {
  var unused x: partial Base = {};
}

// --- trivial_nonempty_partial_class.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

base class Base {
  var a: A;
  var b: B;
}

fn Test() {
  var unused x: partial Base = {.a = {}, .b = {}};
}

// --- nontrivial_member.carbon

library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: partial Base = {.x = {}, .y = {}};
  //@dump-sem-ir-end
}

// --- nontrivial_impls_destroy.carbon

library "[[@TEST_NAME]]";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: partial Base = {};
  //@dump-sem-ir-end
}

// --- trivial_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {}

fn Test() {
  var unused x: partial Abstract = {};
}

// --- nontrivial_abstract.carbon

library "[[@TEST_NAME]]";

import library "types";

abstract class Abstract {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: partial Abstract = {.x = {}, .y = {}};
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- nontrivial_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.x.y.538: type = struct_type {.x: %ImplsDestroy, .y: %AlsoImplsDestroy} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.c22: type = struct_type {.x: %empty_struct_type, .y: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.f0c: %struct_type.x.y.c22 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.val: %AlsoImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %struct.34f: %.edf = struct_value (%ImplsDestroy.val, %AlsoImplsDestroy.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c98: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.09d [concrete]
// CHECK:STDOUT:   %Destroy.facet.b4c: %Destroy.type = facet_value %AlsoImplsDestroy, (%DestroyImpl.impl_witness.c98, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6f7: %Destroy.type = facet_value %.edf, (%DestroyImpl.impl_witness.1d2, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.0c0: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%AlsoImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.09d = impl_witness_table (%Main.import_ref.0c0), @AlsoImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc6_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %.edf = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc13_39.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_48.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_49.1: %struct_type.x.y.c22 = struct_literal (%.loc13_39.1, %.loc13_48.1) [concrete = constants.%struct.f0c]
// CHECK:STDOUT:   %.loc13_49.2: ref %ImplsDestroy = class_element_access %x.var, element0
// CHECK:STDOUT:   %.loc13_39.2: init %ImplsDestroy to %.loc13_49.2 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_49.3: init %ImplsDestroy = converted %.loc13_39.1, %.loc13_39.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_49.4: ref %AlsoImplsDestroy = class_element_access %x.var, element1
// CHECK:STDOUT:   %.loc13_48.2: init %AlsoImplsDestroy to %.loc13_49.4 = class_init () [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_49.5: init %AlsoImplsDestroy = converted %.loc13_48.1, %.loc13_48.2 [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_49.6: init %.edf to %x.var = class_init (%.loc13_49.3, %.loc13_49.5) [concrete = constants.%struct.34f]
// CHECK:STDOUT:   %.loc13_3.1: init %.edf = converted %.loc13_49.1, %.loc13_49.6 [concrete = constants.%struct.34f]
// CHECK:STDOUT:   assign %x.var, %.loc13_3.1
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = constants.%.edf] {
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %.loc13_17.2: type = partial_type %Base.ref [concrete = constants.%.edf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %.edf = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.6f7]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.6f7]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.6f7]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.6f7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.3(%self.param: ref %AlsoImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.4(%self.param: ref %struct_type.x.y.538) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_3.1: ref %AlsoImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.c98, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.c98, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc13_3.1)
// CHECK:STDOUT:   %.loc13_3.4: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_3.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc13_3.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_3.3 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc13_3.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_3.4 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.5(%self.param: ref %.edf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_impls_destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct.c22: %.edf = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.edf, (%DestroyImpl.impl_witness.1d2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %.edf = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc12_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:   %.loc12_33.2: init %.edf to %x.var = class_init () [concrete = constants.%empty_struct.c22]
// CHECK:STDOUT:   %.loc12_3.1: init %.edf = converted %.loc12_33.1, %.loc12_33.2 [concrete = constants.%empty_struct.c22]
// CHECK:STDOUT:   assign %x.var, %.loc12_3.1
// CHECK:STDOUT:   %.loc12_17.1: type = splice_block %.loc12_17.2 [concrete = constants.%.edf] {
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %.loc12_17.2: type = partial_type %Base.ref [concrete = constants.%.edf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %.edf = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.2: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_3.2(%self.param: ref %.edf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.x.y.538: type = struct_type {.x: %ImplsDestroy, .y: %AlsoImplsDestroy} [concrete]
// CHECK:STDOUT:   %.79f: type = partial_type %Abstract [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %.79f [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.5f8 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.5f8 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.c22: type = struct_type {.x: %empty_struct_type, .y: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.f0c: %struct_type.x.y.c22 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.val: %AlsoImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %struct.37d: %.79f = struct_value (%ImplsDestroy.val, %AlsoImplsDestroy.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c98: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.09d [concrete]
// CHECK:STDOUT:   %Destroy.facet.b4c: %Destroy.type = facet_value %AlsoImplsDestroy, (%DestroyImpl.impl_witness.c98, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c03: %SubobjectDestroy.type = facet_value %.79f, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.98a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c03) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9ad: %Destroy.type = facet_value %.79f, (%DestroyImpl.impl_witness.98a, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.as.DestroyImpl.impl.Op: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.0c0: %AlsoImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%AlsoImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.09d = impl_witness_table (%Main.import_ref.0c0), @AlsoImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc6_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %.79f = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc13_43.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_52.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_53.1: %struct_type.x.y.c22 = struct_literal (%.loc13_43.1, %.loc13_52.1) [concrete = constants.%struct.f0c]
// CHECK:STDOUT:   %.loc13_53.2: ref %ImplsDestroy = class_element_access %x.var, element0
// CHECK:STDOUT:   %.loc13_43.2: init %ImplsDestroy to %.loc13_53.2 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_53.3: init %ImplsDestroy = converted %.loc13_43.1, %.loc13_43.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_53.4: ref %AlsoImplsDestroy = class_element_access %x.var, element1
// CHECK:STDOUT:   %.loc13_52.2: init %AlsoImplsDestroy to %.loc13_53.4 = class_init () [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_53.5: init %AlsoImplsDestroy = converted %.loc13_52.1, %.loc13_52.2 [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc13_53.6: init %.79f to %x.var = class_init (%.loc13_53.3, %.loc13_53.5) [concrete = constants.%struct.37d]
// CHECK:STDOUT:   %.loc13_3.1: init %.79f = converted %.loc13_53.1, %.loc13_53.6 [concrete = constants.%struct.37d]
// CHECK:STDOUT:   assign %x.var, %.loc13_3.1
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = constants.%.79f] {
// CHECK:STDOUT:     %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [concrete = constants.%Abstract]
// CHECK:STDOUT:     %.loc13_17.2: type = partial_type %Abstract.ref [concrete = constants.%.79f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %.79f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.5f8 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.5f8 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%.79f, (constants.%DestroyImpl.impl_witness.98a, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.9ad]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.9ad]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%.79f, (constants.%DestroyImpl.impl_witness.98a, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.9ad]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.9ad]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.3(%self.param: ref %AlsoImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.4(%self.param: ref %struct_type.x.y.538) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_3.1: ref %AlsoImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.c98, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.c98, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.b4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc13_3.1)
// CHECK:STDOUT:   %.loc13_3.4: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_3.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc13_3.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_3.3 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc13_3.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc13_3.4 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.5(%self.param: ref %.79f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
