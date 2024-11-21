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
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.7 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.7) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.7) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.36: type = array_type %.3, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %.36 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
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
// CHECK:STDOUT:     %.loc6_23: Core.IntLiteral = int_value 3 [template = constants.%.3]
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
// CHECK:STDOUT:     %.loc6_43.1: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:     %.loc6_43.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc6_43.3: <bound method> = bound_method %.loc6_43.1, %.loc6_43.2 [template = constants.%.32]
// CHECK:STDOUT:     %.loc6_43.4: <specific function> = specific_function %.loc6_43.3, @Convert.2(constants.%.7) [template = constants.%.33]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc6_43.4(%.loc6_43.1) [template = constants.%.34]
// CHECK:STDOUT:     %.loc6_43.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%.34]
// CHECK:STDOUT:     %.loc6_43.6: %i32 = converted %.loc6_43.1, %.loc6_43.5 [template = constants.%.34]
// CHECK:STDOUT:     %.loc6_44.1: ref @F.%.loc6_24.2 (%.4) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_44.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_44.1, %.loc6_43.6
// CHECK:STDOUT:     %.loc6_44.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_44.2
// CHECK:STDOUT:     return %.loc6_44.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc9: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc9_14: Core.IntLiteral = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_15: type = array_type %.loc9_14, %C [template = constants.%.36]
// CHECK:STDOUT:   %a.var: ref %.36 = var a
// CHECK:STDOUT:   %a: ref %.36 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %tuple.type = tuple_literal (%.loc9_21.1, %.loc9_25.1, %.loc9_29.1)
// CHECK:STDOUT:   %.loc9_30.2: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_30.3: ref %C = array_index %a.var, %.loc9_30.2
// CHECK:STDOUT:   %.loc9_21.2: init %C = class_init (), %.loc9_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.4: init %C = converted %.loc9_21.1, %.loc9_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.5: Core.IntLiteral = int_value 1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc9_30.6: ref %C = array_index %a.var, %.loc9_30.5
// CHECK:STDOUT:   %.loc9_25.2: init %C = class_init (), %.loc9_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.7: init %C = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.8: Core.IntLiteral = int_value 2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc9_30.9: ref %C = array_index %a.var, %.loc9_30.8
// CHECK:STDOUT:   %.loc9_29.2: init %C = class_init (), %.loc9_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.10: init %C = converted %.loc9_29.1, %.loc9_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30.11: init %.36 = array_init (%.loc9_30.4, %.loc9_30.7, %.loc9_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_31: init %.36 = converted %.loc9_30.1, %.loc9_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.36 = name_ref a, %a
// CHECK:STDOUT:   %.loc10_10: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%.40]
// CHECK:STDOUT:   %.loc8_8.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc10_12: %.36 = bind_value %a.ref
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
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.36
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
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.4, @impl.3(%.3) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.13) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %N.1, %Convert.13 [symbolic]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.4(%.3) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.27(%N.1) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.30: type = array_type %.29, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %.30 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:     %N.patt.loc10_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc10_6.1, runtime_param<invalid> [symbolic = %N.patt.loc10_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_10.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc10_10: init type = call constants.%Int(%.loc10_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_10.2: type = value_of_initializer %int.make_type_signed.loc10_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_10.3: type = converted %int.make_type_signed.loc10_10, %.loc10_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %N.ref.loc10_22: %i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %.loc10_22.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.13]
// CHECK:STDOUT:     %.loc10_22.2: <bound method> = bound_method %N.ref.loc10_22, %.loc10_22.1 [symbolic = %.loc10_22.6 (constants.%.26)]
// CHECK:STDOUT:     %.loc10_22.3: <specific function> = specific_function %.loc10_22.2, @Convert.4(constants.%.3) [symbolic = %.loc10_22.7 (constants.%.27)]
// CHECK:STDOUT:     %int.convert_checked.loc10_22.1: init Core.IntLiteral = call %.loc10_22.3(%N.ref.loc10_22) [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_22.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10_22.1 [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_22.5: Core.IntLiteral = converted %N.ref.loc10_22, %.loc10_22.4 [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_23: type = array_type %.loc10_22.5, %C [template = <error>]
// CHECK:STDOUT:     %.loc10_29.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc10_29: init type = call constants.%Int(%.loc10_29.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_29.2: type = value_of_initializer %int.make_type_signed.loc10_29 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_29.3: type = converted %int.make_type_signed.loc10_29, %.loc10_29.2 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc10_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
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
// CHECK:STDOUT: generic fn @F(%N.loc10_6.1: %i32) {
// CHECK:STDOUT:   %N.loc10_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc10_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   %.loc10_22.6: <bound method> = bound_method %N.loc10_6.2, constants.%Convert.13 [symbolic = %.loc10_22.6 (constants.%.26)]
// CHECK:STDOUT:   %.loc10_22.7: <specific function> = specific_function %.loc10_22.6, @Convert.4(constants.%.3) [symbolic = %.loc10_22.7 (constants.%.27)]
// CHECK:STDOUT:   %int.convert_checked.loc10_22.2: init Core.IntLiteral = call %.loc10_22.7(%N.loc10_6.2) [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: %i32](%a.param_patt: <error>) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc10_42: %i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:     return %N.ref.loc10_42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_14: Core.IntLiteral = int_value 3 [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_15: type = array_type %.loc13_14, %C [template = constants.%.30]
// CHECK:STDOUT:   %a.var: ref %.30 = var a
// CHECK:STDOUT:   %a: ref %.30 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_30.1: %tuple.type = tuple_literal (%.loc13_21.1, %.loc13_25.1, %.loc13_29.1)
// CHECK:STDOUT:   %.loc13_30.2: Core.IntLiteral = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_30.3: ref %C = array_index %a.var, %.loc13_30.2
// CHECK:STDOUT:   %.loc13_21.2: init %C = class_init (), %.loc13_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_21.1, %.loc13_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.5: Core.IntLiteral = int_value 1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc13_30.6: ref %C = array_index %a.var, %.loc13_30.5
// CHECK:STDOUT:   %.loc13_25.2: init %C = class_init (), %.loc13_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.7: init %C = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.8: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_30.9: ref %C = array_index %a.var, %.loc13_30.8
// CHECK:STDOUT:   %.loc13_29.2: init %C = class_init (), %.loc13_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.10: init %C = converted %.loc13_29.1, %.loc13_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.11: init %.30 = array_init (%.loc13_30.4, %.loc13_30.7, %.loc13_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_31: init %.30 = converted %.loc13_30.1, %.loc13_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.30 = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.1) {
// CHECK:STDOUT:   %N.loc10_6.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc10_6.2 => constants.%N.1
// CHECK:STDOUT:   %.loc10_22.6 => constants.%.26
// CHECK:STDOUT:   %.loc10_22.7 => constants.%.27
// CHECK:STDOUT:   %int.convert_checked.loc10_22.2 => constants.%int.convert_checked
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
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.4, @impl.3(%.3) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.13) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %N.1, %Convert.13 [symbolic]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.4(%.3) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.27(%N.1) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.30: type = array_type %.29, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %.30 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:     %T.patt.loc10_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_6.1, runtime_param<invalid> [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc10_16.1: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc10_16.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc10_16.1, runtime_param<invalid> [symbolic = %N.patt.loc10_16.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc10_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc10_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_20.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc10_20.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_20.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_20.3: type = converted %int.make_type_signed, %.loc10_20.2 [template = constants.%i32]
// CHECK:STDOUT:     %T.ref.loc10_29: type = name_ref T, %T.loc10_6.1 [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc10_16.1 [symbolic = %N.loc10_16.2 (constants.%N.1)]
// CHECK:STDOUT:     %.loc10_32.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.13]
// CHECK:STDOUT:     %.loc10_32.2: <bound method> = bound_method %N.ref, %.loc10_32.1 [symbolic = %.loc10_32.6 (constants.%.26)]
// CHECK:STDOUT:     %.loc10_32.3: <specific function> = specific_function %.loc10_32.2, @Convert.4(constants.%.3) [symbolic = %.loc10_32.7 (constants.%.27)]
// CHECK:STDOUT:     %int.convert_checked.loc10_32.1: init Core.IntLiteral = call %.loc10_32.3(%N.ref) [symbolic = %int.convert_checked.loc10_32.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_32.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10_32.1 [symbolic = %int.convert_checked.loc10_32.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_32.5: Core.IntLiteral = converted %N.ref, %.loc10_32.4 [symbolic = %int.convert_checked.loc10_32.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_33: type = array_type %.loc10_32.5, %T [template = <error>]
// CHECK:STDOUT:     %T.ref.loc10_39: type = name_ref T, %T.loc10_6.1 [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc10_16.1: %i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc10_16.2 (constants.%N.1)]
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
// CHECK:STDOUT: generic fn @F(%T.loc10_6.1: type, %N.loc10_16.1: %i32) {
// CHECK:STDOUT:   %T.loc10_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc10_16.2: %i32 = bind_symbolic_name N, 1 [symbolic = %N.loc10_16.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc10_16.2: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc10_16.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   %.loc10_32.6: <bound method> = bound_method %N.loc10_16.2, constants.%Convert.13 [symbolic = %.loc10_32.6 (constants.%.26)]
// CHECK:STDOUT:   %.loc10_32.7: <specific function> = specific_function %.loc10_32.6, @Convert.4(constants.%.3) [symbolic = %.loc10_32.7 (constants.%.27)]
// CHECK:STDOUT:   %int.convert_checked.loc10_32.2: init Core.IntLiteral = call %.loc10_32.7(%N.loc10_16.2) [symbolic = %int.convert_checked.loc10_32.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %N.param_patt: %i32](%a.param_patt: <error>) -> @F.%T.loc10_6.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_14: Core.IntLiteral = int_value 3 [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_15: type = array_type %.loc13_14, %C [template = constants.%.30]
// CHECK:STDOUT:   %a.var: ref %.30 = var a
// CHECK:STDOUT:   %a: ref %.30 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_30.1: %tuple.type = tuple_literal (%.loc13_21.1, %.loc13_25.1, %.loc13_29.1)
// CHECK:STDOUT:   %.loc13_30.2: Core.IntLiteral = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_30.3: ref %C = array_index %a.var, %.loc13_30.2
// CHECK:STDOUT:   %.loc13_21.2: init %C = class_init (), %.loc13_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.4: init %C = converted %.loc13_21.1, %.loc13_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.5: Core.IntLiteral = int_value 1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc13_30.6: ref %C = array_index %a.var, %.loc13_30.5
// CHECK:STDOUT:   %.loc13_25.2: init %C = class_init (), %.loc13_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.7: init %C = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.8: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_30.9: ref %C = array_index %a.var, %.loc13_30.8
// CHECK:STDOUT:   %.loc13_29.2: init %C = class_init (), %.loc13_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.10: init %C = converted %.loc13_29.1, %.loc13_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.11: init %.30 = array_init (%.loc13_30.4, %.loc13_30.7, %.loc13_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_31: init %.30 = converted %.loc13_30.1, %.loc13_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.30 = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N.1) {
// CHECK:STDOUT:   %T.loc10_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_6.2 => constants.%T
// CHECK:STDOUT:   %N.loc10_16.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc10_16.2 => constants.%N.1
// CHECK:STDOUT:   %.loc10_32.6 => constants.%.26
// CHECK:STDOUT:   %.loc10_32.7 => constants.%.27
// CHECK:STDOUT:   %int.convert_checked.loc10_32.2 => constants.%int.convert_checked
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
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.7 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.7) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.7) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.37: type = array_type %.36, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array: %.37 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT:   %.40: type = array_type %.3, %C [template]
// CHECK:STDOUT:   %.41: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
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
// CHECK:STDOUT:     %.loc6_23: Core.IntLiteral = int_value 2 [template = constants.%.3]
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
// CHECK:STDOUT:     %.loc6_43.1: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:     %.loc6_43.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc6_43.3: <bound method> = bound_method %.loc6_43.1, %.loc6_43.2 [template = constants.%.32]
// CHECK:STDOUT:     %.loc6_43.4: <specific function> = specific_function %.loc6_43.3, @Convert.2(constants.%.7) [template = constants.%.33]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc6_43.4(%.loc6_43.1) [template = constants.%.34]
// CHECK:STDOUT:     %.loc6_43.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%.34]
// CHECK:STDOUT:     %.loc6_43.6: %i32 = converted %.loc6_43.1, %.loc6_43.5 [template = constants.%.34]
// CHECK:STDOUT:     %.loc6_44.1: ref @F.%.loc6_24.2 (%.4) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_44.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_44.1, %.loc6_43.6
// CHECK:STDOUT:     %.loc6_44.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_44.2
// CHECK:STDOUT:     return %.loc6_44.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc10_14: Core.IntLiteral = int_value 3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc10_15: type = array_type %.loc10_14, %C [template = constants.%.37]
// CHECK:STDOUT:   %a.var: ref %.37 = var a
// CHECK:STDOUT:   %a: ref %.37 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc10_30.1: %tuple.type = tuple_literal (%.loc10_21.1, %.loc10_25.1, %.loc10_29.1)
// CHECK:STDOUT:   %.loc10_30.2: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_30.3: ref %C = array_index %a.var, %.loc10_30.2
// CHECK:STDOUT:   %.loc10_21.2: init %C = class_init (), %.loc10_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.4: init %C = converted %.loc10_21.1, %.loc10_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.5: Core.IntLiteral = int_value 1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc10_30.6: ref %C = array_index %a.var, %.loc10_30.5
// CHECK:STDOUT:   %.loc10_25.2: init %C = class_init (), %.loc10_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.7: init %C = converted %.loc10_25.1, %.loc10_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.8: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_30.9: ref %C = array_index %a.var, %.loc10_30.8
// CHECK:STDOUT:   %.loc10_29.2: init %C = class_init (), %.loc10_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.10: init %C = converted %.loc10_29.1, %.loc10_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30.11: init %.37 = array_init (%.loc10_30.4, %.loc10_30.7, %.loc10_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc10_31: init %.37 = converted %.loc10_30.1, %.loc10_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.37 = name_ref a, %a
// CHECK:STDOUT:   %.loc21_10: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%.41]
// CHECK:STDOUT:   %.loc8_8.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc21_12: %.40 = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %F.call: init %C = call %.loc21_10(<error>) to %.loc8_8.2
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
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.40
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.4, @impl.3(%.3) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.13) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %N.1, %Convert.13 [symbolic]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.4(%.3) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.27(%N.1) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.30: type = array_type %.29, %D [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %struct: %D = struct_value () [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %.30 = tuple_value (%struct, %struct, %struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %N.patt.loc10_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc10_6.1, runtime_param<invalid> [symbolic = %N.patt.loc10_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_10.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc10_10: init type = call constants.%Int(%.loc10_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_10.2: type = value_of_initializer %int.make_type_signed.loc10_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_10.3: type = converted %int.make_type_signed.loc10_10, %.loc10_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %N.ref.loc10_22: %i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %.loc10_22.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.13]
// CHECK:STDOUT:     %.loc10_22.2: <bound method> = bound_method %N.ref.loc10_22, %.loc10_22.1 [symbolic = %.loc10_22.6 (constants.%.26)]
// CHECK:STDOUT:     %.loc10_22.3: <specific function> = specific_function %.loc10_22.2, @Convert.4(constants.%.3) [symbolic = %.loc10_22.7 (constants.%.27)]
// CHECK:STDOUT:     %int.convert_checked.loc10_22.1: init Core.IntLiteral = call %.loc10_22.3(%N.ref.loc10_22) [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_22.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10_22.1 [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_22.5: Core.IntLiteral = converted %N.ref.loc10_22, %.loc10_22.4 [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_23: type = array_type %.loc10_22.5, %C [template = <error>]
// CHECK:STDOUT:     %.loc10_29.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc10_29: init type = call constants.%Int(%.loc10_29.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_29.2: type = value_of_initializer %int.make_type_signed.loc10_29 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_29.3: type = converted %int.make_type_signed.loc10_29, %.loc10_29.2 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc10_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
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
// CHECK:STDOUT: generic fn @F(%N.loc10_6.1: %i32) {
// CHECK:STDOUT:   %N.loc10_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc10_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc10_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   %.loc10_22.6: <bound method> = bound_method %N.loc10_6.2, constants.%Convert.13 [symbolic = %.loc10_22.6 (constants.%.26)]
// CHECK:STDOUT:   %.loc10_22.7: <specific function> = specific_function %.loc10_22.6, @Convert.4(constants.%.3) [symbolic = %.loc10_22.7 (constants.%.27)]
// CHECK:STDOUT:   %int.convert_checked.loc10_22.2: init Core.IntLiteral = call %.loc10_22.7(%N.loc10_6.2) [symbolic = %int.convert_checked.loc10_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: %i32](%a.param_patt: <error>) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc10_42: %i32 = name_ref N, %N.loc10_6.1 [symbolic = %N.loc10_6.2 (constants.%N.1)]
// CHECK:STDOUT:     return %N.ref.loc10_42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc13_14: Core.IntLiteral = int_value 3 [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_15: type = array_type %.loc13_14, %D [template = constants.%.30]
// CHECK:STDOUT:   %a.var: ref %.30 = var a
// CHECK:STDOUT:   %a: ref %.30 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_30.1: %tuple.type = tuple_literal (%.loc13_21.1, %.loc13_25.1, %.loc13_29.1)
// CHECK:STDOUT:   %.loc13_30.2: Core.IntLiteral = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_30.3: ref %D = array_index %a.var, %.loc13_30.2
// CHECK:STDOUT:   %.loc13_21.2: init %D = class_init (), %.loc13_30.3 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.4: init %D = converted %.loc13_21.1, %.loc13_21.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.5: Core.IntLiteral = int_value 1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc13_30.6: ref %D = array_index %a.var, %.loc13_30.5
// CHECK:STDOUT:   %.loc13_25.2: init %D = class_init (), %.loc13_30.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.7: init %D = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.8: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_30.9: ref %D = array_index %a.var, %.loc13_30.8
// CHECK:STDOUT:   %.loc13_29.2: init %D = class_init (), %.loc13_30.9 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.10: init %D = converted %.loc13_29.1, %.loc13_29.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_30.11: init %.30 = array_init (%.loc13_30.4, %.loc13_30.7, %.loc13_30.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_31: init %.30 = converted %.loc13_30.1, %.loc13_30.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %.30 = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.1) {
// CHECK:STDOUT:   %N.loc10_6.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc10_6.2 => constants.%N.1
// CHECK:STDOUT:   %.loc10_22.6 => constants.%.26
// CHECK:STDOUT:   %.loc10_22.7 => constants.%.27
// CHECK:STDOUT:   %int.convert_checked.loc10_22.2 => constants.%int.convert_checked
// CHECK:STDOUT: }
// CHECK:STDOUT:
