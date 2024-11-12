// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/array.carbon

// --- type_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: [T; 3]) -> T { return a[0]; }

fn G() -> C {
  var a: [C; 3] = ({}, {}, {});
  return F(a);
}

// --- fail_todo_bound_only.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_todo_bound_only.carbon:[[@LINE+4]]:22: error: semantics TODO: `symbolic array bound` [SemanticsTodo]
// CHECK:STDERR: fn F[N:! i32](a: [C; N]) -> i32 { return N; }
// CHECK:STDERR:                      ^
// CHECK:STDERR:
fn F[N:! i32](a: [C; N]) -> i32 { return N; }

fn G() -> C {
  var a: [C; 3] = ({}, {}, {});
  return F(a);
}

// --- fail_todo_type_and_bound.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_todo_type_and_bound.carbon:[[@LINE+4]]:32: error: semantics TODO: `symbolic array bound` [SemanticsTodo]
// CHECK:STDERR: fn F[T:! type, N:! i32](a: [T; N]) -> T;
// CHECK:STDERR:                                ^
// CHECK:STDERR:
fn F[T:! type, N:! i32](a: [T; N]) -> T;

fn G() -> C {
  var a: [C; 3] = ({}, {}, {});
  return F(a);
}

// --- fail_bound_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: [T; 2]) -> T { return a[0]; }

fn G() -> C {
  // TODO: We succeed at deducing T here but fail to convert. Is this the right behavior?
  var a: [C; 3] = ({}, {}, {});
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert from `[C; 3]` to `[C; 2]` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+7]]:12: note: type `[C; 3]` does not implement interface `ImplicitAs([C; 2])` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE-11]]:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[T:! type](a: [T; 2]) -> T { return a[0]; }
  // CHECK:STDERR:                ^~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:22: error: semantics TODO: `symbolic array bound` [SemanticsTodo]
// CHECK:STDERR: fn F[N:! i32](a: [C; N]) -> i32 { return N; }
// CHECK:STDERR:                      ^
fn F[N:! i32](a: [C; N]) -> i32 { return N; }

fn G() -> C {
  var a: [D; 3] = ({}, {}, {});
  return F(a);
}


// CHECK:STDOUT: --- type_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.3: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [symbolic]
// CHECK:STDOUT:   %.6: i32 = int_value 0 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.8: type = array_type %.3, %C [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.10: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.11: i32 = int_value 2 [template]
// CHECK:STDOUT:   %array: %.8 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT:   %.12: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_6.1, runtime_param<invalid> [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @F.%.loc6_24.2 (%.4) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%.loc6_24.2 (%.4) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_20: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_23: i32 = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_24.1: type = array_type %.loc6_23, %T [symbolic = %.loc6_24.2 (constants.%.4)]
// CHECK:STDOUT:     %T.ref.loc6_30: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%.loc6_24.2 (%.4) = value_param runtime_param0
// CHECK:STDOUT:     %a: @F.%.loc6_24.2 (%.4) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc6_24.2: type = array_type constants.%.3, @F.%T.loc6_6.2 (%T) [symbolic = %.loc6_24.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%a.param_patt: @F.%.loc6_24.2 (%.4)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%.loc6_24.2 (%.4) = name_ref a, %a
// CHECK:STDOUT:     %.loc6_43: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:     %.loc6_44.1: ref @F.%.loc6_24.2 (%.4) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_44.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_44.1, %.loc6_43
// CHECK:STDOUT:     %.loc6_44.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_44.2
// CHECK:STDOUT:     return %.loc6_44.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc9: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc9_14: i32 = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_15: type = array_type %.loc9_14, %C [template = constants.%.8]
// CHECK:STDOUT:   %a.var: ref %.8 = var a
// CHECK:STDOUT:   %a: ref %.8 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %tuple.type = tuple_literal (%.loc9_21.1, %.loc9_25.1, %.loc9_29.1)
// CHECK:STDOUT:   %.loc9_30.2: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_30.3: ref %C = array_index %a.var, %.loc9_30.2
// CHECK:STDOUT:   %.loc9_21.2: init %C = class_init (), %.loc9_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.4: init %C = converted %.loc9_21.1, %.loc9_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.5: i32 = int_value 1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc9_30.6: ref %C = array_index %a.var, %.loc9_30.5
// CHECK:STDOUT:   %.loc9_25.2: init %C = class_init (), %.loc9_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.7: init %C = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.8: i32 = int_value 2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc9_30.9: ref %C = array_index %a.var, %.loc9_30.8
// CHECK:STDOUT:   %.loc9_29.2: init %C = class_init (), %.loc9_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.10: init %C = converted %.loc9_29.1, %.loc9_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.11: init %.8 = array_init (%.loc9_30.4, %.loc9_30.7, %.loc9_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_31: init %.8 = converted %.loc9_30.1, %.loc9_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.8 = name_ref a, %a
// CHECK:STDOUT:   %.loc10_10: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%.12]
// CHECK:STDOUT:   %.loc8_8.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc10_12: %.8 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %C = call %.loc10_10(%.loc10_12) to %.loc8_8.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, %C [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.7: i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_value 2 [template]
// CHECK:STDOUT:   %array: %.5 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc10_6.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc10_6.1, runtime_param<invalid> [symbolic = %N.patt.loc10_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc10_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_10.1: type = value_of_initializer %int.make_type_32.loc10_10 [template = i32]
// CHECK:STDOUT:     %.loc10_10.2: type = converted %int.make_type_32.loc10_10, %.loc10_10.1 [template = i32]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %N.ref.loc10_22: i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:     %.loc10_23: type = array_type %N.ref.loc10_22, %C [template = <error>]
// CHECK:STDOUT:     %int.make_type_32.loc10_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_32.loc10_29 [template = i32]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_32.loc10_29, %.loc10_29.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc10_6.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc10_6.1: i32) {
// CHECK:STDOUT:   %N.loc10_6.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc10_6.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: i32](%a.param_patt: <error>) -> i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc10_42: i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:     return %N.ref.loc10_42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_14: i32 = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_15: type = array_type %.loc13_14, %C [template = constants.%.5]
// CHECK:STDOUT:   %a.var: ref %.5 = var a
// CHECK:STDOUT:   %a: ref %.5 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_30.1: %tuple.type = tuple_literal (%.loc13_21.1, %.loc13_25.1, %.loc13_29.1)
// CHECK:STDOUT:   %.loc13_30.2: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_30.3: ref %C = array_index %a.var, %.loc13_30.2
// CHECK:STDOUT:   %.loc13_21.2: init %C = class_init (), %.loc13_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_21.1, %.loc13_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.5: i32 = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_30.6: ref %C = array_index %a.var, %.loc13_30.5
// CHECK:STDOUT:   %.loc13_25.2: init %C = class_init (), %.loc13_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.7: init %C = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.8: i32 = int_value 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_30.9: ref %C = array_index %a.var, %.loc13_30.8
// CHECK:STDOUT:   %.loc13_29.2: init %C = class_init (), %.loc13_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.10: init %C = converted %.loc13_29.1, %.loc13_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.11: init %.5 = array_init (%.loc13_30.4, %.loc13_30.7, %.loc13_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_31: init %.5 = converted %.loc13_30.1, %.loc13_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.5 = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc10_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc10_6.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, %C [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.7: i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_value 2 [template]
// CHECK:STDOUT:   %array: %.5 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc10_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_6.1, runtime_param<invalid> [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc10_16.1: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc10_16.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc10_16.1, runtime_param<invalid> [symbolic = %N.patt.loc10_16.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc10_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc10_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc10_20.2: type = converted %int.make_type_32, %.loc10_20.1 [template = i32]
// CHECK:STDOUT:     %T.ref.loc10_29: type = name_ref T, %T.loc10_6.1 [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:     %N.ref: i32 = name_ref N, %N.loc10_16.1 [symbolic = %N.loc10_16.2 (constants.%N)]
// CHECK:STDOUT:     %.loc10_33: type = array_type %N.ref, %T [template = <error>]
// CHECK:STDOUT:     %T.ref.loc10_39: type = name_ref T, %T.loc10_6.1 [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc10_16.1: i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc10_16.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc10_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc10_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc10_6.1: type, %N.loc10_16.1: i32) {
// CHECK:STDOUT:   %T.loc10_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc10_16.2: i32 = bind_symbolic_name N, 1 [symbolic = %N.loc10_16.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc10_16.2: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc10_16.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %N.param_patt: i32](%a.param_patt: <error>) -> @F.%T.loc10_6.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_14: i32 = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_15: type = array_type %.loc13_14, %C [template = constants.%.5]
// CHECK:STDOUT:   %a.var: ref %.5 = var a
// CHECK:STDOUT:   %a: ref %.5 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_30.1: %tuple.type = tuple_literal (%.loc13_21.1, %.loc13_25.1, %.loc13_29.1)
// CHECK:STDOUT:   %.loc13_30.2: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_30.3: ref %C = array_index %a.var, %.loc13_30.2
// CHECK:STDOUT:   %.loc13_21.2: init %C = class_init (), %.loc13_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_21.1, %.loc13_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.5: i32 = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_30.6: ref %C = array_index %a.var, %.loc13_30.5
// CHECK:STDOUT:   %.loc13_25.2: init %C = class_init (), %.loc13_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.7: init %C = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.8: i32 = int_value 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_30.9: ref %C = array_index %a.var, %.loc13_30.8
// CHECK:STDOUT:   %.loc13_29.2: init %C = class_init (), %.loc13_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.10: init %C = converted %.loc13_29.1, %.loc13_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.11: init %.5 = array_init (%.loc13_30.4, %.loc13_30.7, %.loc13_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_31: init %.5 = converted %.loc13_30.1, %.loc13_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.5 = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc10_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_6.2 => constants.%T
// CHECK:STDOUT:   %N.loc10_16.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc10_16.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.3: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [symbolic]
// CHECK:STDOUT:   %.6: i32 = int_value 0 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.8: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, %C [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.11: i32 = int_value 1 [template]
// CHECK:STDOUT:   %array: %.9 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT:   %.12: type = array_type %.3, %C [template]
// CHECK:STDOUT:   %.13: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT:   %.14: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%.12)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%.12) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.19: %.15 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+48, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+54, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.1 (%.15) = import_ref Core//prelude/operators/as, inst+76, loaded [symbolic = @ImplicitAs.%.2 (constants.%.19)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+69, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_6.1, runtime_param<invalid> [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @F.%.loc6_24.2 (%.4) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%.loc6_24.2 (%.4) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_20: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_23: i32 = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_24.1: type = array_type %.loc6_23, %T [symbolic = %.loc6_24.2 (constants.%.4)]
// CHECK:STDOUT:     %T.ref.loc6_30: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%.loc6_24.2 (%.4) = value_param runtime_param0
// CHECK:STDOUT:     %a: @F.%.loc6_24.2 (%.4) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.15)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.15) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.2 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc6_24.2: type = array_type constants.%.3, @F.%T.loc6_6.2 (%T) [symbolic = %.loc6_24.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%a.param_patt: @F.%.loc6_24.2 (%.4)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%.loc6_24.2 (%.4) = name_ref a, %a
// CHECK:STDOUT:     %.loc6_43: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:     %.loc6_44.1: ref @F.%.loc6_24.2 (%.4) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_44.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_44.1, %.loc6_43
// CHECK:STDOUT:     %.loc6_44.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_44.2
// CHECK:STDOUT:     return %.loc6_44.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc10_14: i32 = int_value 3 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_15: type = array_type %.loc10_14, %C [template = constants.%.9]
// CHECK:STDOUT:   %a.var: ref %.9 = var a
// CHECK:STDOUT:   %a: ref %.9 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_30.1: %tuple.type = tuple_literal (%.loc10_21.1, %.loc10_25.1, %.loc10_29.1)
// CHECK:STDOUT:   %.loc10_30.2: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_30.3: ref %C = array_index %a.var, %.loc10_30.2
// CHECK:STDOUT:   %.loc10_21.2: init %C = class_init (), %.loc10_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.4: init %C = converted %.loc10_21.1, %.loc10_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.5: i32 = int_value 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc10_30.6: ref %C = array_index %a.var, %.loc10_30.5
// CHECK:STDOUT:   %.loc10_25.2: init %C = class_init (), %.loc10_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.7: init %C = converted %.loc10_25.1, %.loc10_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.8: i32 = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_30.9: ref %C = array_index %a.var, %.loc10_30.8
// CHECK:STDOUT:   %.loc10_29.2: init %C = class_init (), %.loc10_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.10: init %C = converted %.loc10_29.1, %.loc10_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.11: init %.9 = array_init (%.loc10_30.4, %.loc10_30.7, %.loc10_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc10_31: init %.9 = converted %.loc10_30.1, %.loc10_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.9 = name_ref a, %a
// CHECK:STDOUT:   %.loc21_10: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%.13]
// CHECK:STDOUT:   %.loc8_8.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc21_12: %.12 = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %F.call: init %C = call %.loc21_10(<error>) to %.loc8_8.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.12
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.12) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, %D [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.7: i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct: %D = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_value 2 [template]
// CHECK:STDOUT:   %array: %.5 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc10_6.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc10_6.1, runtime_param<invalid> [symbolic = %N.patt.loc10_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc10_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_10.1: type = value_of_initializer %int.make_type_32.loc10_10 [template = i32]
// CHECK:STDOUT:     %.loc10_10.2: type = converted %int.make_type_32.loc10_10, %.loc10_10.1 [template = i32]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %N.ref.loc10_22: i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:     %.loc10_23: type = array_type %N.ref.loc10_22, %C [template = <error>]
// CHECK:STDOUT:     %int.make_type_32.loc10_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_32.loc10_29 [template = i32]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_32.loc10_29, %.loc10_29.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc10_6.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc10_6.1: i32) {
// CHECK:STDOUT:   %N.loc10_6.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc10_6.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: i32](%a.param_patt: <error>) -> i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc10_42: i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N)]
// CHECK:STDOUT:     return %N.ref.loc10_42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc13_14: i32 = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_15: type = array_type %.loc13_14, %D [template = constants.%.5]
// CHECK:STDOUT:   %a.var: ref %.5 = var a
// CHECK:STDOUT:   %a: ref %.5 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_30.1: %tuple.type = tuple_literal (%.loc13_21.1, %.loc13_25.1, %.loc13_29.1)
// CHECK:STDOUT:   %.loc13_30.2: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_30.3: ref %D = array_index %a.var, %.loc13_30.2
// CHECK:STDOUT:   %.loc13_21.2: init %D = class_init (), %.loc13_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.4: init %D = converted %.loc13_21.1, %.loc13_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.5: i32 = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_30.6: ref %D = array_index %a.var, %.loc13_30.5
// CHECK:STDOUT:   %.loc13_25.2: init %D = class_init (), %.loc13_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.7: init %D = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.8: i32 = int_value 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_30.9: ref %D = array_index %a.var, %.loc13_30.8
// CHECK:STDOUT:   %.loc13_29.2: init %D = class_init (), %.loc13_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.10: init %D = converted %.loc13_29.1, %.loc13_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.11: init %.5 = array_init (%.loc13_30.4, %.loc13_30.7, %.loc13_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_31: init %.5 = converted %.loc13_30.1, %.loc13_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.5 = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc10_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc10_6.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
