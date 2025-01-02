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

// --- bound_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! Core.IntLiteral()](a: [C; N]) -> i32 { return N; }

fn G() -> i32 {
  var a: [C; 3] = ({}, {}, {});
  return F(a);
}

// --- type_and_bound.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type, N:! Core.IntLiteral()](a: [T; N]) {}

fn G() {
  var a: [C; 3] = ({}, {}, {});
  F(a);
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
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+7]]:12: note: type `[C; 3]` does not implement interface `Core.ImplicitAs([C; 2])` [MissingImplInMemberAccessNote]
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

fn F[N:! Core.IntLiteral()](a: [C; N]) -> i32 { return N; }

fn G() -> i32 {
  // TODO: We succeed at deducing N here but fail to convert. Is this the right behavior?
  var a: [D; 3] = ({}, {}, {});
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert from `[D; 3]` to `[C; 3]` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:12: note: type `[D; 3]` does not implement interface `Core.ImplicitAs([C; 3])` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE-11]]:29: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[N:! Core.IntLiteral()](a: [C; N]) -> i32 { return N; }
  // CHECK:STDERR:                             ^~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_bound_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! i32](a: [C; N]) -> i32 { return N; }

fn G() -> i32 {
  var a: [C; 3] = ({}, {}, {});
  // TODO: This fails because the array bound in `F` is effectively
  //   `N.(ImplicitAs(IntLiteral).Convert)()`
  // which we can't deduce through. We should decide if we want to support
  // deductions of that form. If not, it'd be nice to diagnose this situation
  // better.
  // CHECK:STDERR: fail_bound_type_mismatch.carbon:[[@LINE+6]]:10: error: cannot deduce value for generic parameter `N` [DeductionIncomplete]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_bound_type_mismatch.carbon:[[@LINE-12]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[N:! i32](a: [C; N]) -> i32 { return N; }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return F(a);
}

// CHECK:STDOUT: --- type_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_3, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %array_type.1 [symbolic]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %array_type.2 = tuple_value (%C.val, %C.val, %C.val) [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %array_type.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_24.2 (%array_type.1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_24.2 (%array_type.1) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_30: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_24.2 (%array_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_24: type = splice_block %array_type.loc6_24.1 [symbolic = %array_type.loc6_24.2 (constants.%array_type.1)] {
// CHECK:STDOUT:       %T.ref.loc6_20: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:       %array_type.loc6_24.1: type = array_type %int_3, %T [symbolic = %array_type.loc6_24.2 (constants.%array_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_24.2 (%array_type.1) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %array_type.loc6_24.2: type = array_type constants.%int_3, @F.%T.loc6_6.2 (%T) [symbolic = %array_type.loc6_24.2 (constants.%array_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_27: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_27 (constants.%require_complete.1)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%array_type.loc6_24.2 (%array_type.1) [symbolic = %require_complete.loc6_17 (constants.%require_complete.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%a.param_patt: @F.%array_type.loc6_24.2 (%array_type.1)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_24.2 (%array_type.1) = name_ref a, %a
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc6_43.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc6_43.2: %i32 = converted %int_0, %.loc6_43.1 [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc6_44.1: ref @F.%array_type.loc6_24.2 (%array_type.1) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_44.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_44.1, %.loc6_43.2
// CHECK:STDOUT:     %.loc6_44.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_44.2
// CHECK:STDOUT:     return %.loc6_44.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %array_type.2 = var a
// CHECK:STDOUT:   %a: ref %array_type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %tuple.type = tuple_literal (%.loc9_21.1, %.loc9_25.1, %.loc9_29.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc9_30.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_21.2: init %C = class_init (), %.loc9_30.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.3: init %C = converted %.loc9_21.1, %.loc9_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc9_30.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_25.2: init %C = class_init (), %.loc9_30.4 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.5: init %C = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc9_30.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_29.2: init %C = class_init (), %.loc9_30.6 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.7: init %C = converted %.loc9_29.1, %.loc9_29.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.8: init %array_type.2 = array_init (%.loc9_30.3, %.loc9_30.5, %.loc9_30.7) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_31: init %array_type.2 = converted %.loc9_30.1, %.loc9_30.8 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.2 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc8: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc10: %array_type.2 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%.loc10) to %.loc8
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %array_type.loc6_24.2 => constants.%array_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %array_type.loc6_24.2 => constants.%array_type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_27 => constants.%complete_type.1
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.1: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %array_type.1 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %N, %Convert.10 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn.1(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3.1, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %array_type.2 = tuple_value (%C.val, %C.val, %C.val) [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1) [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %array_type.2 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_3.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
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
// CHECK:STDOUT:     %N.patt.loc6_6.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_6.1, runtime_param<invalid> [symbolic = %N.patt.loc6_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_37.2 (%array_type.1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_37.2 (%array_type.1) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_26.1: type = splice_block %.loc6_26.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.3: type = converted %int_literal.make_type, %.loc6_26.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_37.2 (%array_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_37: type = splice_block %array_type.loc6_37.1 [symbolic = %array_type.loc6_37.2 (constants.%array_type.1)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_36: Core.IntLiteral = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_37.1: type = array_type %N.ref.loc6_36, %C [symbolic = %array_type.loc6_37.2 (constants.%array_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_37.2 (%array_type.1) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_6.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc6_37.2: type = array_type %N.loc6_6.2, %C [symbolic = %array_type.loc6_37.2 (constants.%array_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc6_37.2 (%array_type.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Convert.bound.loc6_57.2: <bound method> = bound_method %N.loc6_6.2, constants.%Convert.10 [symbolic = %Convert.bound.loc6_57.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_57.2: <specific function> = specific_function %Convert.bound.loc6_57.2, @Convert.2(constants.%int_32) [symbolic = %Convert.specific_fn.loc6_57.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:   %int.convert_checked.loc6_57.2: init %i32 = call %Convert.specific_fn.loc6_57.2(%N.loc6_6.2) [symbolic = %int.convert_checked.loc6_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: Core.IntLiteral](%a.param_patt: @F.%array_type.loc6_37.2 (%array_type.1)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_56: Core.IntLiteral = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:     %Convert.bound.loc6_57.1: <bound method> = bound_method %N.ref.loc6_56, %impl.elem0 [symbolic = %Convert.bound.loc6_57.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:     %Convert.specific_fn.loc6_57.1: <specific function> = specific_function %Convert.bound.loc6_57.1, @Convert.2(constants.%int_32) [symbolic = %Convert.specific_fn.loc6_57.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:     %int.convert_checked.loc6_57.1: init %i32 = call %Convert.specific_fn.loc6_57.1(%N.ref.loc6_56) [symbolic = %int.convert_checked.loc6_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc6_57.1: %i32 = value_of_initializer %int.convert_checked.loc6_57.1 [symbolic = %int.convert_checked.loc6_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc6_57.2: %i32 = converted %N.ref.loc6_56, %.loc6_57.1 [symbolic = %int.convert_checked.loc6_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     return %.loc6_57.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %array_type.2 = var a
// CHECK:STDOUT:   %a: ref %array_type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %tuple.type = tuple_literal (%.loc9_21.1, %.loc9_25.1, %.loc9_29.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc9_30.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_21.2: init %C = class_init (), %.loc9_30.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.3: init %C = converted %.loc9_21.1, %.loc9_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc9_30.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_25.2: init %C = class_init (), %.loc9_30.4 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.5: init %C = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc9_30.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_29.2: init %C = class_init (), %.loc9_30.6 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.7: init %C = converted %.loc9_29.1, %.loc9_29.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.8: init %array_type.2 = array_init (%.loc9_30.3, %.loc9_30.5, %.loc9_30.7) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_31: init %array_type.2 = converted %.loc9_30.1, %.loc9_30.8 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.2 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10_12: %array_type.2 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc10_12)
// CHECK:STDOUT:   %.loc10_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc10_14.2: %i32 = converted %F.call, %.loc10_14.1
// CHECK:STDOUT:   return %.loc10_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%N
// CHECK:STDOUT:   %array_type.loc6_37.2 => constants.%array_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%int_3.1
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%int_3.1
// CHECK:STDOUT:   %array_type.loc6_37.2 => constants.%array_type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4
// CHECK:STDOUT:   %Convert.bound.loc6_57.2 => constants.%Convert.bound.2
// CHECK:STDOUT:   %Convert.specific_fn.loc6_57.2 => constants.%Convert.specific_fn.2
// CHECK:STDOUT:   %int.convert_checked.loc6_57.2 => constants.%int_3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %array_type.1: type = array_type %N, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %array_type.2 = tuple_value (%C.val, %C.val, %C.val) [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C, %int_3) [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %array_type.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
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
// CHECK:STDOUT:     %N.patt.loc6_16.1: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_16.1, runtime_param<invalid> [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_47.2 (%array_type.1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_47.2 (%array_type.1) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_36.1: type = splice_block %.loc6_36.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.3: type = converted %int_literal.make_type, %.loc6_36.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_16.1: Core.IntLiteral = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_47.2 (%array_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_47: type = splice_block %array_type.loc6_47.1 [symbolic = %array_type.loc6_47.2 (constants.%array_type.1)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc6_16.1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_47.1: type = array_type %N.ref, %T [symbolic = %array_type.loc6_47.2 (constants.%array_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_47.2 (%array_type.1) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type, %N.loc6_16.1: Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc6_16.2: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_16.2: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc6_47.2: type = array_type %N.loc6_16.2, @F.%T.loc6_6.2 (%T) [symbolic = %array_type.loc6_47.2 (constants.%array_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc6_47.2 (%array_type.1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %N.param_patt: Core.IntLiteral](%a.param_patt: @F.%array_type.loc6_47.2 (%array_type.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %array_type.2 = var a
// CHECK:STDOUT:   %a: ref %array_type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %tuple.type = tuple_literal (%.loc9_21.1, %.loc9_25.1, %.loc9_29.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc9_30.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_21.2: init %C = class_init (), %.loc9_30.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.3: init %C = converted %.loc9_21.1, %.loc9_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc9_30.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_25.2: init %C = class_init (), %.loc9_30.4 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.5: init %C = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc9_30.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_29.2: init %C = class_init (), %.loc9_30.6 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.7: init %C = converted %.loc9_29.1, %.loc9_29.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.8: init %array_type.2 = array_init (%.loc9_30.3, %.loc9_30.5, %.loc9_30.7) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_31: init %array_type.2 = converted %.loc9_30.1, %.loc9_30.8 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.2 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%int_3) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %array_type.2 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %N.loc6_16.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%N
// CHECK:STDOUT:   %array_type.loc6_47.2 => constants.%array_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%int_3) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %N.loc6_16.2 => constants.%int_3
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%int_3
// CHECK:STDOUT:   %array_type.loc6_47.2 => constants.%array_type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_2, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %array_type.1 [symbolic]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array: %array_type.2 = tuple_value (%C.val, %C.val, %C.val) [template]
// CHECK:STDOUT:   %array_type.3: type = array_type %int_2, %C [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %array_type.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_24.2 (%array_type.1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_24.2 (%array_type.1) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_30: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_24.2 (%array_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_24: type = splice_block %array_type.loc6_24.1 [symbolic = %array_type.loc6_24.2 (constants.%array_type.1)] {
// CHECK:STDOUT:       %T.ref.loc6_20: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:       %array_type.loc6_24.1: type = array_type %int_2, %T [symbolic = %array_type.loc6_24.2 (constants.%array_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_24.2 (%array_type.1) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %array_type.loc6_24.2: type = array_type constants.%int_2, @F.%T.loc6_6.2 (%T) [symbolic = %array_type.loc6_24.2 (constants.%array_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_27: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_27 (constants.%require_complete.1)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%array_type.loc6_24.2 (%array_type.1) [symbolic = %require_complete.loc6_17 (constants.%require_complete.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%a.param_patt: @F.%array_type.loc6_24.2 (%array_type.1)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_24.2 (%array_type.1) = name_ref a, %a
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc6_43.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc6_43.2: %i32 = converted %int_0, %.loc6_43.1 [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc6_44.1: ref @F.%array_type.loc6_24.2 (%array_type.1) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_44.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_44.1, %.loc6_43.2
// CHECK:STDOUT:     %.loc6_44.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_44.2
// CHECK:STDOUT:     return %.loc6_44.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %array_type.2 = var a
// CHECK:STDOUT:   %a: ref %array_type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_30.1: %tuple.type = tuple_literal (%.loc10_21.1, %.loc10_25.1, %.loc10_29.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc10_30.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc10_21.2: init %C = class_init (), %.loc10_30.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc10_30.3: init %C = converted %.loc10_21.1, %.loc10_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc10_30.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc10_25.2: init %C = class_init (), %.loc10_30.4 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc10_30.5: init %C = converted %.loc10_25.1, %.loc10_25.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc10_30.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc10_29.2: init %C = class_init (), %.loc10_30.6 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc10_30.7: init %C = converted %.loc10_29.1, %.loc10_29.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc10_30.8: init %array_type.2 = array_init (%.loc10_30.3, %.loc10_30.5, %.loc10_30.7) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc10_31: init %array_type.2 = converted %.loc10_30.1, %.loc10_30.8 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.2 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc8: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc21: %array_type.3 = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(<error>) to %.loc8
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %array_type.loc6_24.2 => constants.%array_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %array_type.loc6_24.2 => constants.%array_type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_27 => constants.%complete_type.1
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.1: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %array_type.1 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %N, %Convert.10 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn.1(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3.1, %D [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %D.val: %D = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %array_type.2 = tuple_value (%D.val, %D.val, %D.val) [template]
// CHECK:STDOUT:   %array_type.3: type = array_type %int_3.1, %C [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1) [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %array_type.3 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_3.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
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
// CHECK:STDOUT:     %N.patt.loc7_6.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc7_6.1, runtime_param<invalid> [symbolic = %N.patt.loc7_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc7_37.2 (%array_type.1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc7_37.2 (%array_type.1) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_26.1: type = splice_block %.loc7_26.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.3: type = converted %int_literal.make_type, %.loc7_26.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_6.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc7_37.2 (%array_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc7_37: type = splice_block %array_type.loc7_37.1 [symbolic = %array_type.loc7_37.2 (constants.%array_type.1)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %N.ref.loc7_36: Core.IntLiteral = name_ref N, %N.loc7_6.1 [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc7_37.1: type = array_type %N.ref.loc7_36, %C [symbolic = %array_type.loc7_37.2 (constants.%array_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc7_37.2 (%array_type.1) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc7_6.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc7_6.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_6.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc7_37.2: type = array_type %N.loc7_6.2, %C [symbolic = %array_type.loc7_37.2 (constants.%array_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc7_37.2 (%array_type.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Convert.bound.loc7_57.2: <bound method> = bound_method %N.loc7_6.2, constants.%Convert.10 [symbolic = %Convert.bound.loc7_57.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_57.2: <specific function> = specific_function %Convert.bound.loc7_57.2, @Convert.2(constants.%int_32) [symbolic = %Convert.specific_fn.loc7_57.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:   %int.convert_checked.loc7_57.2: init %i32 = call %Convert.specific_fn.loc7_57.2(%N.loc7_6.2) [symbolic = %int.convert_checked.loc7_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: Core.IntLiteral](%a.param_patt: @F.%array_type.loc7_37.2 (%array_type.1)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc7_56: Core.IntLiteral = name_ref N, %N.loc7_6.1 [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:     %Convert.bound.loc7_57.1: <bound method> = bound_method %N.ref.loc7_56, %impl.elem0 [symbolic = %Convert.bound.loc7_57.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:     %Convert.specific_fn.loc7_57.1: <specific function> = specific_function %Convert.bound.loc7_57.1, @Convert.2(constants.%int_32) [symbolic = %Convert.specific_fn.loc7_57.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:     %int.convert_checked.loc7_57.1: init %i32 = call %Convert.specific_fn.loc7_57.1(%N.ref.loc7_56) [symbolic = %int.convert_checked.loc7_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc7_57.1: %i32 = value_of_initializer %int.convert_checked.loc7_57.1 [symbolic = %int.convert_checked.loc7_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc7_57.2: %i32 = converted %N.ref.loc7_56, %.loc7_57.1 [symbolic = %int.convert_checked.loc7_57.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     return %.loc7_57.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %array_type.2 = var a
// CHECK:STDOUT:   %a: ref %array_type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc11_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_30.1: %tuple.type = tuple_literal (%.loc11_21.1, %.loc11_25.1, %.loc11_29.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_30.2: ref %D = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_21.2: init %D = class_init (), %.loc11_30.2 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc11_30.3: init %D = converted %.loc11_21.1, %.loc11_21.2 [template = constants.%D.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc11_30.4: ref %D = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc11_25.2: init %D = class_init (), %.loc11_30.4 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc11_30.5: init %D = converted %.loc11_25.1, %.loc11_25.2 [template = constants.%D.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc11_30.6: ref %D = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc11_29.2: init %D = class_init (), %.loc11_30.6 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc11_30.7: init %D = converted %.loc11_29.1, %.loc11_29.2 [template = constants.%D.val]
// CHECK:STDOUT:   %.loc11_30.8: init %array_type.2 = array_init (%.loc11_30.3, %.loc11_30.5, %.loc11_30.7) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_31: init %array_type.2 = converted %.loc11_30.1, %.loc11_30.8 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.2 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc22_12: %array_type.3 = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %.loc22_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc22_14.2: %i32 = converted %F.call, %.loc22_14.1
// CHECK:STDOUT:   return %.loc22_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc7_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc7_6.2 => constants.%N
// CHECK:STDOUT:   %array_type.loc7_37.2 => constants.%array_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1) {
// CHECK:STDOUT:   %N.loc7_6.2 => constants.%int_3.1
// CHECK:STDOUT:   %N.patt.loc7_6.2 => constants.%int_3.1
// CHECK:STDOUT:   %array_type.loc7_37.2 => constants.%array_type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4
// CHECK:STDOUT:   %Convert.bound.loc7_57.2 => constants.%Convert.bound.2
// CHECK:STDOUT:   %Convert.specific_fn.loc7_57.2 => constants.%Convert.specific_fn.2
// CHECK:STDOUT:   %int.convert_checked.loc7_57.2 => constants.%int_3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.2, %Convert.9 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.2) [symbolic]
// CHECK:STDOUT:   %array_type.1: type = array_type %int.convert_checked, %C [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %array_type.1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3, %C [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array: %array_type.2 = tuple_value (%C.val, %C.val, %C.val) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:     %N.patt.loc6_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_6.1, runtime_param<invalid> [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_23.2 (%array_type.1) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_23.2 (%array_type.1) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_29: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_6.2 (constants.%N.2)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_23.2 (%array_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_23: type = splice_block %array_type.loc6_23.1 [symbolic = %array_type.loc6_23.2 (constants.%array_type.1)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_22: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.2)]
// CHECK:STDOUT:       %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:       %Convert.bound.loc6_22.1: <bound method> = bound_method %N.ref.loc6_22, %impl.elem0 [symbolic = %Convert.bound.loc6_22.2 (constants.%Convert.bound)]
// CHECK:STDOUT:       %Convert.specific_fn.loc6_22.1: <specific function> = specific_function %Convert.bound.loc6_22.1, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn.loc6_22.2 (constants.%Convert.specific_fn)]
// CHECK:STDOUT:       %int.convert_checked.loc6_22.1: init Core.IntLiteral = call %Convert.specific_fn.loc6_22.1(%N.ref.loc6_22) [symbolic = %int.convert_checked.loc6_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc6_22.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc6_22.1 [symbolic = %int.convert_checked.loc6_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc6_22.2: Core.IntLiteral = converted %N.ref.loc6_22, %.loc6_22.1 [symbolic = %int.convert_checked.loc6_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc6_23.1: type = array_type %.loc6_22.2, %C [symbolic = %array_type.loc6_23.2 (constants.%array_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_23.2 (%array_type.1) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.1: %i32) {
// CHECK:STDOUT:   %N.loc6_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc6_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Convert.bound.loc6_22.2: <bound method> = bound_method %N.loc6_6.2, constants.%Convert.9 [symbolic = %Convert.bound.loc6_22.2 (constants.%Convert.bound)]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_22.2: <specific function> = specific_function %Convert.bound.loc6_22.2, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn.loc6_22.2 (constants.%Convert.specific_fn)]
// CHECK:STDOUT:   %int.convert_checked.loc6_22.2: init Core.IntLiteral = call %Convert.specific_fn.loc6_22.2(%N.loc6_6.2) [symbolic = %int.convert_checked.loc6_22.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc6_23.2: type = array_type %int.convert_checked.loc6_22.2, %C [symbolic = %array_type.loc6_23.2 (constants.%array_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc6_23.2 (%array_type.1) [symbolic = %require_complete (constants.%require_complete.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: %i32](%a.param_patt: @F.%array_type.loc6_23.2 (%array_type.1)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_42: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.2)]
// CHECK:STDOUT:     return %N.ref.loc6_42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %array_type.2 = var a
// CHECK:STDOUT:   %a: ref %array_type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %tuple.type = tuple_literal (%.loc9_21.1, %.loc9_25.1, %.loc9_29.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc9_30.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_21.2: init %C = class_init (), %.loc9_30.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.3: init %C = converted %.loc9_21.1, %.loc9_21.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc9_30.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_25.2: init %C = class_init (), %.loc9_30.4 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.5: init %C = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc9_30.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_29.2: init %C = class_init (), %.loc9_30.6 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.7: init %C = converted %.loc9_29.1, %.loc9_29.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc9_30.8: init %array_type.2 = array_init (%.loc9_30.3, %.loc9_30.5, %.loc9_30.7) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_31: init %array_type.2 = converted %.loc9_30.1, %.loc9_30.8 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_31
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.2 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.2) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%N.2
// CHECK:STDOUT:   %Convert.bound.loc6_22.2 => constants.%Convert.bound
// CHECK:STDOUT:   %Convert.specific_fn.loc6_22.2 => constants.%Convert.specific_fn
// CHECK:STDOUT:   %int.convert_checked.loc6_22.2 => constants.%int.convert_checked
// CHECK:STDOUT:   %array_type.loc6_23.2 => constants.%array_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
