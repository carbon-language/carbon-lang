// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %C.a14: type = class_type @C, @C(%true) [concrete]
// CHECK:STDOUT:   %pattern_type.cc5: type = pattern_type %C.a14 [concrete]
// CHECK:STDOUT:   %True.type: type = fn_type @True [concrete]
// CHECK:STDOUT:   %True: %True.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cc5 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cc5 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.a14 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc12_24.1: type = splice_block %C.loc12 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc12: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Eq.ref.loc12: %Eq.type = name_ref Eq, %Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:     %true.loc12_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc12_19: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %bool.eq.loc12: init bool = call %Eq.ref.loc12(%true.loc12_13, %true.loc12_19) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_24.2: bool = value_of_initializer %bool.eq.loc12 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_24.3: bool = converted %bool.eq.loc12, %.loc12_24.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc12: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.a14 = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc12: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc12: ref %C.a14 = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %True.call.loc12: init %C.a14 = call %True.ref.loc12() to %.loc12
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc12
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %C.a14: type = class_type @C, @C(%true) [concrete]
// CHECK:STDOUT:   %pattern_type.cc5: type = pattern_type %C.a14 [concrete]
// CHECK:STDOUT:   %True.type: type = fn_type @True [concrete]
// CHECK:STDOUT:   %True: %True.type = struct_value () [concrete]
// CHECK:STDOUT:   %Eq.type: type = facet_type <@Eq> [concrete]
// CHECK:STDOUT:   %Equal.type.79c: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Eq.impl_witness: <witness> = impl_witness imports.%Eq.impl_witness_table [concrete]
// CHECK:STDOUT:   %Eq.facet: %Eq.type = facet_value bool, (%Eq.impl_witness) [concrete]
// CHECK:STDOUT:   %.7f7: type = fn_type_with_self_type %Equal.type.79c, %Eq.facet [concrete]
// CHECK:STDOUT:   %Equal.type.aa3: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.6e2: %Equal.type.aa3 = struct_value () [concrete]
// CHECK:STDOUT:   %Equal.bound.fe0: <bound method> = bound_method %true, %Equal.6e2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.998: %Equal.type.aa3 = import_ref Core//prelude/operators/comparison, loc29_45, loaded [concrete = constants.%Equal.6e2]
// CHECK:STDOUT:   %Core.import_ref.fb6 = import_ref Core//prelude/operators/comparison, loc30_48, unloaded
// CHECK:STDOUT:   %Eq.impl_witness_table = impl_witness_table (%Core.import_ref.998, %Core.import_ref.fb6), @impl.bb5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cc5 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cc5 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.a14 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc10_22.1: type = splice_block %C.loc10 [concrete = constants.%C.a14] {
// CHECK:STDOUT:     %C.ref.loc10: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc10_10: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc10_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem0.loc10: %.7f7 = impl_witness_access constants.%Eq.impl_witness, element0 [concrete = constants.%Equal.6e2]
// CHECK:STDOUT:     %bound_method.loc10: <bound method> = bound_method %true.loc10_10, %impl.elem0.loc10 [concrete = constants.%Equal.bound.fe0]
// CHECK:STDOUT:     %bool.eq.loc10: init bool = call %bound_method.loc10(%true.loc10_10, %true.loc10_18) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc10_22.2: bool = value_of_initializer %bool.eq.loc10 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc10_22.3: bool = converted %bool.eq.loc10, %.loc10_22.2 [concrete = constants.%true]
// CHECK:STDOUT:     %C.loc10: type = class_type @C, @C(constants.%true) [concrete = constants.%C.a14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.a14 = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %True.ref.loc10: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc10: ref %C.a14 = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %True.call.loc10: init %C.a14 = call %True.ref.loc10() to %.loc10
// CHECK:STDOUT:   assign file.%a.var, %True.call.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
