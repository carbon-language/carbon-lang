// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

namespace N;
class C {};

// Both N.F1 and N.F2 use N.C and not C.
class N.C {}
fn N.F1(x: C);
fn N.F2(x: C) { N.F1(x); }

// --- poison_without_usage.carbon

library "[[@TEST_NAME]]";

namespace N;
class C {};

// Here we use C and poison N.C.
fn N.F1(x: C);

// TODO: Should fail here since C was poisoned for namespace N when it was used
// in N context without qualification.
class N.C {}

// --- fail_poison_with_usage.carbon

library "[[@TEST_NAME]]";

namespace N;
class C {};

// Here we use C and poison N.C.
fn N.F1(x: C);

// TODO: Should fail here since C was poisoned for namespace N when it was used
// in N context without qualification.
class N.C {}

// TODO: Should not fail here since both N.F2() and N.F1() input is the class C
// and not class N.C.
// CHECK:STDERR: fail_poison_with_usage.carbon:[[@LINE+6]]:22: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn N.F2(x: C) { N.F1(x); }
// CHECK:STDERR:                      ^
// CHECK:STDERR: fail_poison_with_usage.carbon:[[@LINE-11]]:9: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn N.F1(x: C);
// CHECK:STDERR:         ^~~~
fn N.F2(x: C) { N.F1(x); }

// CHECK:STDOUT: --- no_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.1: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.2: type = class_type @C.2 [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [template]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .C = %C.decl.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl.loc8
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc5: type = class_decl @C.1 [template = constants.%C.1] {} {}
// CHECK:STDOUT:   %C.decl.loc8: type = class_decl @C.2 [template = constants.%C.2] {} {}
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.2 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.2 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc8 [template = constants.%C.2]
// CHECK:STDOUT:     %x.param: %C.2 = value_param runtime_param0
// CHECK:STDOUT:     %x: %C.2 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [template = constants.%F2] {
// CHECK:STDOUT:     %x.patt: %C.2 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.2 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc8 [template = constants.%C.2]
// CHECK:STDOUT:     %x.param: %C.2 = value_param runtime_param0
// CHECK:STDOUT:     %x: %C.2 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.2
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.2);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: %C.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %F1.ref: %F1.type = name_ref F1, file.%F1.decl [template = constants.%F1]
// CHECK:STDOUT:   %x.ref: %C.2 = name_ref x, %x
// CHECK:STDOUT:   %F1.call: init %empty_tuple.type = call %F1.ref(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison_without_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.1: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .C = %C.decl.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .C = %C.decl.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc5: type = class_decl @C.1 [template = constants.%C.1] {} {}
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.1 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.1 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc5 [template = constants.%C.1]
// CHECK:STDOUT:     %x.param: %C.1 = value_param runtime_param0
// CHECK:STDOUT:     %x: %C.1 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc12: type = class_decl @C.2 [template = constants.%C.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.2
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.1);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_with_usage.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.1: type = class_type @C.1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C.2 [template]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [template]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .C = %C.decl.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .C = %C.decl.loc12
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc5: type = class_decl @C.1 [template = constants.%C.1] {} {}
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [template = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.1 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.1 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc5 [template = constants.%C.1]
// CHECK:STDOUT:     %x.param: %C.1 = value_param runtime_param0
// CHECK:STDOUT:     %x: %C.1 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc12: type = class_decl @C.2 [template = constants.%C.2] {} {}
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [template = constants.%F2] {
// CHECK:STDOUT:     %x.patt: %C.2 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.2 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc12 [template = constants.%C.2]
// CHECK:STDOUT:     %x.param: %C.2 = value_param runtime_param0
// CHECK:STDOUT:     %x: %C.2 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.2
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.1);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: %C.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, file.%N [template = file.%N]
// CHECK:STDOUT:   %F1.ref: %F1.type = name_ref F1, file.%F1.decl [template = constants.%F1]
// CHECK:STDOUT:   %x.ref: %C.2 = name_ref x, %x
// CHECK:STDOUT:   %.loc22: %C.1 = converted %x.ref, <error> [template = <error>]
// CHECK:STDOUT:   %F1.call: init %empty_tuple.type = call %F1.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
