// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/auto/bindings.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/auto/bindings.carbon

// --- var.carbon
library "[[@TEST_NAME]]";

class A {}
impl A as Core.Copy {
    fn Op(self) -> Self { return self; }
}

fn F() {
    var a: A = {};
//@dump-sem-ir-begin
    var unused x: auto = a;
    var unused y: auto = 20;
//@dump-sem-ir-end
}

// --- let.carbon
library "[[@TEST_NAME]]";

class A {}
fn F() {
    let a: A = {};
//@dump-sem-ir-begin
    let unused x: auto = a;
    let unused y: auto = 20;
//@dump-sem-ir-end
}

// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.23e: <witness> = impl_witness @A.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.as.Copy.impl.Op.type: type = fn_type @A.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.as.Copy.impl.Op: %A.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.2d2: %Copy.type = facet_value %A, (%Copy.impl_witness.23e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.527: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.2d2) [concrete]
// CHECK:STDOUT:   %.e9e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.527, %Copy.facet.2d2 [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f6d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.f6d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.f6d = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.f6d = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0b3: <witness> = impl_witness imports.%Copy.impl_witness_table.ee5 [concrete]
// CHECK:STDOUT:   %Copy.facet.d66: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.0b3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d4d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.d66) [concrete]
// CHECK:STDOUT:   %.1b2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d4d, %Copy.facet.d66 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %int_20, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.659: %SubobjectDestroy.type = facet_value Core.IntLiteral, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9b4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.659) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4c6: %Destroy.type = facet_value Core.IntLiteral, (%DestroyImpl.impl_witness.9b4, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35e: %Destroy.type = facet_value %A, (%DestroyImpl.impl_witness.6d1, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.ee5 = impl_witness_table (%Core.import_ref.9c3), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %A = var_storage %x.var_patt
// CHECK:STDOUT:   %a.ref: ref %A = name_ref a, %a
// CHECK:STDOUT:   %.loc11_26: %A = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc11: %.e9e = impl_witness_access constants.%Copy.impl_witness.23e, element0 [concrete = constants.%A.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.loc11_26, %impl.elem0.loc11
// CHECK:STDOUT:   %.loc11_5.1: ref %A = splice_block %x.var {}
// CHECK:STDOUT:   %A.as.Copy.impl.Op.call: init %A to %.loc11_5.1 = call %bound_method.loc11(%.loc11_26)
// CHECK:STDOUT:   assign %x.var, %A.as.Copy.impl.Op.call
// CHECK:STDOUT:   %x: ref %A = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f6d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.f6d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref Core.IntLiteral = var_storage %y.var_patt
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %impl.elem0.loc12: %.1b2 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %int_20, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call: init Core.IntLiteral = call %bound_method.loc12(%int_20) [concrete = constants.%int_20]
// CHECK:STDOUT:   assign %y.var, %Core.IntLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT:   %y: ref Core.IntLiteral = wrapper_binding y, %y.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.f6d = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.f6d = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_5.1: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc12_5.1: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc12_5.1 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %Destroy.facet.loc12_5.2: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc12_5.2: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc12_5.2 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%y.var)
// CHECK:STDOUT:   %Destroy.facet.loc11_5.1: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc11_5.2: %Destroy.type = converted constants.%A, %Destroy.facet.loc11_5.1 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %Destroy.facet.loc11_5.2: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc11_5.3: %Destroy.type = converted constants.%A, %Destroy.facet.loc11_5.2 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12(%self.param: ref Core.IntLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_5.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f6d = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.f6d = value_binding_pattern y [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %x: %A = wrapper_binding x, %a.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f6d = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %y: Core.IntLiteral = wrapper_binding y, %int_20
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.f6d = value_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
