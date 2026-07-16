// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/lifetime.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/lifetime.carbon

// --- extend.carbon
library "[[@TEST_NAME]]";

class A {
  fn Clone() -> Self;
  impl as Core.Copy {
    fn Op(self) -> Self;
  }
}

fn G();

fn F() {
  //@dump-sem-ir-begin
  // All of these A temporaries get lifetime-extended.
  var unused a: A = ({} as A, {} as A).1;
  var unused b: A = ({} as A).Clone();
  G();
  // TODO: Four A objects should be destroyed here.
  // For now, we create five A objects, because we copy from (...).1.
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.Clone.type: type = fn_type @A.Clone [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.Clone: %A.Clone.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.7c4: <witness> = impl_witness @A.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.as.Copy.impl.Op.type: type = fn_type @A.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Copy.impl.Op: %A.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %A, (%Copy.impl_witness.7c4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.573: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.777: type = tuple_type (%A, %A) [concrete]
// CHECK:STDOUT:   %tuple.cc9: %tuple.type.777 = tuple_value (%A.val, %A.val) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %.367: ref %A = temporary invalid, %A.val [concrete]
// CHECK:STDOUT:   %.b68: type = fn_type_with_self_type %Copy.WithSelf.Op.type.573, %Copy.facet [concrete]
// CHECK:STDOUT:   %A.as.Copy.impl.Op.bound: <bound method> = bound_method %A.val, %A.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.9ef = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.9ef = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc16_25.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.367, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %A = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc15_23.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc15_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc15_23.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc15_23.3: init %A to %.loc15_23.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc15_25.1: init %A = converted %.loc15_23.1, %.loc15_23.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc15_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc15_37: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc15_32.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc15_32.3: init %A to %.loc15_32.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc15_34.1: init %A = converted %.loc15_32.1, %.loc15_32.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc15_38.1: %tuple.type.777 = tuple_literal (%.loc15_25.1, %.loc15_34.1) [concrete = constants.%tuple.cc9]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc15_25.2: ref %A = temporary %.loc15_23.2, %.loc15_25.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc15_25.3: %A = acquire_value %.loc15_25.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc15_34.2: ref %A = temporary %.loc15_32.2, %.loc15_34.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc15_34.3: %A = acquire_value %.loc15_34.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %tuple: %tuple.type.777 = tuple_value (%.loc15_25.3, %.loc15_34.3) [concrete = constants.%tuple.cc9]
// CHECK:STDOUT:   %.loc15_38.2: %tuple.type.777 = converted %.loc15_38.1, %tuple [concrete = constants.%tuple.cc9]
// CHECK:STDOUT:   %tuple.elem1: %A = tuple_access %.loc15_38.2, element1 [concrete = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0: %.b68 = impl_witness_access constants.%Copy.impl_witness.7c4, element0 [concrete = constants.%A.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %tuple.elem1, %impl.elem0 [concrete = constants.%A.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %.loc15_3: ref %A = splice_block %a.var {}
// CHECK:STDOUT:   %A.as.Copy.impl.Op.call: init %A to %.loc15_3 = call %bound_method(%tuple.elem1)
// CHECK:STDOUT:   assign %a.var, %A.as.Copy.impl.Op.call
// CHECK:STDOUT:   %A.ref.loc15_17: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %A = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc16_23.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc16_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc16_23.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc16_23.3: init %A to %.loc16_23.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc16_25.1: init %A = converted %.loc16_23.1, %.loc16_23.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc16_25.2: ref %A = temporary %.loc16_23.2, %.loc16_25.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %Clone.ref: %A.Clone.type = name_ref Clone, @A.%A.Clone.decl [concrete = constants.%A.Clone]
// CHECK:STDOUT:   %.loc16_3: ref %A = splice_block %b.var {}
// CHECK:STDOUT:   %A.Clone.call: init %A to %.loc16_3 = call %Clone.ref()
// CHECK:STDOUT:   assign %b.var, %A.Clone.call
// CHECK:STDOUT:   %A.ref.loc16_17: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %b: ref %A = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.9ef = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.9ef = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   %Destroy.Op.call.loc16_25: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   %Destroy.Op.bound.loc16: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc16_3: init %empty_tuple.type = call %Destroy.Op.bound.loc16(%b.var)
// CHECK:STDOUT:   %Destroy.Op.call.loc15_34: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   %Destroy.Op.call.loc15_25: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   %Destroy.Op.bound.loc15: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc15_3: init %empty_tuple.type = call %Destroy.Op.bound.loc15(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_25.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_25.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
