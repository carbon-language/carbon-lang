// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/bool/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/bool/eq.carbon

// --- builtin_call.carbon

library "[[@TEST_NAME]]";

fn Eq(a: bool, b: bool) -> bool = "bool.eq";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

//@dump-sem-ir-begin
var a: C(Eq(true, true)) = True();
//@dump-sem-ir-end
var b: C(Eq(true, false)) = False();
var c: C(Eq(false, true)) = False();
var d: C(Eq(false, false)) = True();

// --- prelude.carbon

library "[[@TEST_NAME]]";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

//@dump-sem-ir-begin
var a: C(true == true) = True();
//@dump-sem-ir-end
var b: C(true == false) = False();
var c: C(false == true) = False();
var d: C(false == false) = True();

// CHECK:STDOUT: --- builtin_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [concrete]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %C.a23: type = class_type @C, @C(%true) [concrete]
// CHECK:STDOUT:   %pattern_type.a06: type = pattern_type %C.a23 [concrete]
// CHECK:STDOUT:   %True.type: type = fn_type @True [concrete]
// CHECK:STDOUT:   %True: %True.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a06 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a06 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.a23 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc60.1: type = splice_block %C.loc60 [concrete = constants.%C.a23] {
// CHECK:STDOUT:     %C.ref.loc52: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Eq.ref.loc54: %Eq.type = name_ref Eq, %Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:     %true.loc56: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc58: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %Eq.call.loc59: init bool = call %Eq.ref.loc54(%true.loc56, %true.loc58) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc60.2: bool = value_of_initializer %Eq.call.loc59 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc60.3: bool = converted %Eq.call.loc59, %.loc60.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc60: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a23]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.a23 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc62: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc49: ref %C.a23 = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %True.call.loc64: init %C.a23 to %.loc49 = call %True.ref.loc62()
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc64
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %C.a23: type = class_type @C, @C(%true) [concrete]
// CHECK:STDOUT:   %pattern_type.a06: type = pattern_type %C.a23 [concrete]
// CHECK:STDOUT:   %True.type: type = fn_type @True [concrete]
// CHECK:STDOUT:   %True: %True.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.type.863: type = facet_type <@EqWith, @EqWith(bool)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness: <witness> = impl_witness imports.%EqWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.863 = facet_value bool, (%EqWith.impl_witness) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.8c0: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(bool, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %.483: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.8c0, %EqWith.facet [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.Equal.type: type = fn_type @bool.as.EqWith.impl.Equal [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.Equal: %bool.as.EqWith.impl.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.Equal.bound.40b: <bound method> = bound_method %true, %bool.as.EqWith.impl.Equal [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.f73: %bool.as.EqWith.impl.Equal.type = import_ref Core//prelude/operators/comparison, loc145, loaded [concrete = constants.%bool.as.EqWith.impl.Equal]
// CHECK:STDOUT:   %Core.import_ref.115 = import_ref Core//prelude/operators/comparison, loc162, unloaded
// CHECK:STDOUT:   %EqWith.impl_witness_table = impl_witness_table (%Core.import_ref.f73, %Core.import_ref.115), @bool.as.EqWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a06 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a06 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.a23 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc41.1: type = splice_block %C.loc41 [concrete = constants.%C.a23] {
// CHECK:STDOUT:     %C.ref.loc36: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc38: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc40: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem0.loc39: %.483 = impl_witness_access constants.%EqWith.impl_witness, element0 [concrete = constants.%bool.as.EqWith.impl.Equal]
// CHECK:STDOUT:     %bound_method.loc39: <bound method> = bound_method %true.loc38, %impl.elem0.loc39 [concrete = constants.%bool.as.EqWith.impl.Equal.bound.40b]
// CHECK:STDOUT:     %bool.as.EqWith.impl.Equal.call.loc39: init bool = call %bound_method.loc39(%true.loc38, %true.loc40) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc41.2: bool = value_of_initializer %bool.as.EqWith.impl.Equal.call.loc39 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc41.3: bool = converted %bool.as.EqWith.impl.Equal.call.loc39, %.loc41.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc41: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a23]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.a23 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc43: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc33: ref %C.a23 = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %True.call.loc45: init %C.a23 to %.loc33 = call %True.ref.loc43()
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc45
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
