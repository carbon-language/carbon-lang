// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/array.carbon

// --- type_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: array(T, 3)) -> T { return F(a); }

fn G() -> C {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- bound_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! Core.IntLiteral()](unused a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- type_and_bound.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type, N:! Core.IntLiteral()](unused a: array(T, N)) {}

fn G() {
  var a: array(C, 3) = ({}, {}, {});
  F(a);
}

// --- fail_bound_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: array(T, 2)) -> T { return F(a); }

fn G() -> C {
  // TODO: We succeed at deducing T here but fail to convert. Is this the right behavior?
  var a: array(C, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(C, 3)` to `array(C, 2)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+7]]:12: note: type `array(C, 3)` does not implement interface `Core.ImplicitAs(array(C, 2))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE-11]]:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[T:! type](a: array(T, 2)) -> T { return F(a); }
  // CHECK:STDERR:                ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[N:! Core.IntLiteral()](unused a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  // TODO: We succeed at deducing N here but fail to convert. Is this the right behavior?
  var a: array(D, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(D, 3)` to `array(C, 3)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:12: note: type `array(D, 3)` does not implement interface `Core.ImplicitAs(array(C, 3))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE-11]]:36: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[N:! Core.IntLiteral()](unused a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR:                                    ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_bound_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! i32](unused a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  // TODO: This fails because the array bound in `F` is effectively
  //   `N.(ImplicitAs(IntLiteral).Convert)()`
  // which we can't deduce through. We should decide if we want to support
  // deductions of that form. If not, it'd be nice to diagnose this situation
  // better.
  // CHECK:STDERR: fail_bound_type_mismatch.carbon:[[@LINE+7]]:10: error: cannot deduce value for generic parameter `N` [DeductionIncomplete]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_bound_type_mismatch.carbon:[[@LINE-12]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[N:! i32](unused a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_todo_array_length_from_tuple.carbon
library "[[@TEST_NAME]]";

class C {}

fn F[N:! i32](unused a: array(C, N)) {}

fn G() {
  // TODO: Deduce N as 3 from the tuple's size.
  //
  // CHECK:STDERR: fail_todo_array_length_from_tuple.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `N` [DeductionIncomplete]
  // CHECK:STDERR:   F(({}, {}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_array_length_from_tuple.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[N:! i32](unused a: array(C, N)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(({}, {}, {}));
}

// CHECK:STDOUT: --- type_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.3ec: type = array_type %int_3, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3f: type = pattern_type %array_type.3ec [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.3ec [symbolic]
// CHECK:STDOUT:   %F.specific_fn.643: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.a69: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.931: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.f21: type = pattern_type %array_type.931 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.931 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn.540: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.c7a: <witness> = complete_type_witness %array_type.931 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc17 (%pattern_type.b3f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc17 (%pattern_type.b3f) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc26 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc26 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc26: type = name_ref T, %T.loc11.2 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:     %.loc26.3: Core.Form = init_form %T.ref.loc26 [symbolic = %.loc26.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11.2: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc23.1 (%array_type.3ec) = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %array_type.loc23.2 [symbolic = %array_type.loc23.1 (constants.%array_type.3ec)] {
// CHECK:STDOUT:       %T.ref.loc20: type = name_ref T, %T.loc11.2 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type.loc23.2: type = array_type %int_3, %T.ref.loc20 [symbolic = %array_type.loc23.1 (constants.%array_type.3ec)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc23.1 (%array_type.3ec) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc11.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc11.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern %return.param_patt, %C.ref.loc40 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc40: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc40.2: Core.Form = init_form %C.ref.loc40 [concrete = constants.%.a69]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11.2: type) {
// CHECK:STDOUT:   %T.loc11.1: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:   %array_type.loc23.1: type = array_type constants.%int_3, %T.loc11.1 [symbolic = %array_type.loc23.1 (constants.%array_type.3ec)]
// CHECK:STDOUT:   %pattern_type.loc17: type = pattern_type %array_type.loc23.1 [symbolic = %pattern_type.loc17 (constants.%pattern_type.b3f)]
// CHECK:STDOUT:   %.loc26.2: Core.Form = init_form %T.loc11.1 [symbolic = %.loc26.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc26: type = pattern_type %T.loc11.1 [symbolic = %pattern_type.loc26 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc23.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc29.2: <specific function> = specific_function constants.%F, @F(%T.loc11.1) [symbolic = %F.specific_fn.loc29.2 (constants.%F.specific_fn.643)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc23.1 (%array_type.3ec)) -> out %return.param: @F.%T.loc11.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc23.1 (%array_type.3ec) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc29.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc29.2 (constants.%F.specific_fn.643)]
// CHECK:STDOUT:     %.loc26.1: ref @F.%T.loc11.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc11.1 (%T) to %.loc26.1 = call %F.specific_fn.loc29.1(%a.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f21 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f21 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.931 = var %a.var_patt
// CHECK:STDOUT:   %.loc54.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc57.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc60.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc61.1: %tuple.type.8d4 = tuple_literal (%.loc54.1, %.loc57.1, %.loc60.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc61.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc54.2: init %C to %.loc61.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.3: init %C = converted %.loc54.1, %.loc54.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc61.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc57.2: init %C to %.loc61.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.5: init %C = converted %.loc57.1, %.loc57.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc61.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc60.2: init %C to %.loc61.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.7: init %C = converted %.loc60.1, %.loc60.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.8: init %array_type.931 to %a.var = array_init (%.loc61.3, %.loc61.5, %.loc61.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc42: init %array_type.931 = converted %.loc61.1, %.loc61.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc42
// CHECK:STDOUT:   %.loc50: type = splice_block %array_type [concrete = constants.%array_type.931] {
// CHECK:STDOUT:     %C.ref.loc47: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc47 [concrete = constants.%array_type.931]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.931 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.931 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.540]
// CHECK:STDOUT:   %.loc40.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %.loc66: %array_type.931 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %C to %.loc40.1 = call %F.specific_fn(%.loc66)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.931) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%T
// CHECK:STDOUT:   %array_type.loc23.1 => constants.%array_type.3ec
// CHECK:STDOUT:   %pattern_type.loc17 => constants.%pattern_type.b3f
// CHECK:STDOUT:   %.loc26.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc26 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc29.2 => constants.%F.specific_fn.643
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%C
// CHECK:STDOUT:   %array_type.loc23.1 => constants.%array_type.931
// CHECK:STDOUT:   %pattern_type.loc17 => constants.%pattern_type.f21
// CHECK:STDOUT:   %.loc26.2 => constants.%.a69
// CHECK:STDOUT:   %pattern_type.loc26 => constants.%pattern_type.7c7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c7a
// CHECK:STDOUT:   %F.specific_fn.loc29.2 => constants.%F.specific_fn.540
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.c79: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.cc9: type = pattern_type %array_type.c79 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d0d: <witness> = require_complete_type %array_type.c79 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d6e: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.7fa: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.7fa(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.931: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %pattern_type.f21: type = pattern_type %array_type.931 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.931 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.c7a: <witness> = complete_type_witness %array_type.931 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.cc9) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.cc9) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc31: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %.loc17.1: type = splice_block %.loc17.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc17.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc17.3: type = converted %IntLiteral.call, %.loc17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc11.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc11.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc28.1 (%array_type.c79) = value_param call_param0
// CHECK:STDOUT:     %.loc28: type = splice_block %array_type.loc28.2 [symbolic = %array_type.loc28.1 (constants.%array_type.c79)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc27: Core.IntLiteral = name_ref N, %N.loc11.2 [symbolic = %N.loc11.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc28.2: type = array_type %N.ref.loc27, %C.ref [symbolic = %array_type.loc28.1 (constants.%array_type.c79)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc28.1 (%array_type.c79) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc42: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc11.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc11.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc11.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc28.1: type = array_type %N.loc11.1, constants.%C [symbolic = %array_type.loc28.1 (constants.%array_type.c79)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc28.1 [symbolic = %pattern_type (constants.%pattern_type.cc9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc28.1 [symbolic = %require_complete (constants.%require_complete.d0d)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc11.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d6e)]
// CHECK:STDOUT:   %bound_method.loc35.3: <bound method> = bound_method %N.loc11.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc35.3 (constants.%bound_method.7fa)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35.2: init %i32 = call %bound_method.loc35.3(%N.loc11.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc28.1 (%array_type.c79)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc34: Core.IntLiteral = name_ref N, %N.loc11.2 [symbolic = %N.loc11.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc35.1: <bound method> = bound_method %N.ref.loc34, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d6e)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc35.2: <bound method> = bound_method %N.ref.loc34, %specific_fn [symbolic = %bound_method.loc35.3 (constants.%bound_method.7fa)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35.1: init %i32 = call %bound_method.loc35.2(%N.ref.loc34) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc35: init %i32 = converted %N.ref.loc34, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f21 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f21 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.931 = var %a.var_patt
// CHECK:STDOUT:   %.loc56.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc59.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc62.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc63.1: %tuple.type.8d4 = tuple_literal (%.loc56.1, %.loc59.1, %.loc62.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc63.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc56.2: init %C to %.loc63.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc63.3: init %C = converted %.loc56.1, %.loc56.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc63.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc59.2: init %C to %.loc63.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc63.5: init %C = converted %.loc59.1, %.loc59.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc63.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc62.2: init %C to %.loc63.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc63.7: init %C = converted %.loc62.1, %.loc62.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc63.8: init %array_type.931 to %a.var = array_init (%.loc63.3, %.loc63.5, %.loc63.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc44: init %array_type.931 = converted %.loc63.1, %.loc63.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc44
// CHECK:STDOUT:   %.loc52: type = splice_block %array_type [concrete = constants.%array_type.931] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.931]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.931 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.931 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc68: %array_type.931 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc68)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.931) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc11.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc28.1 => constants.%array_type.c79
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc11.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc28.1 => constants.%array_type.931
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c7a
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061
// CHECK:STDOUT:   %bound_method.loc35.3 => constants.%bound_method.fa7
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35.2 => constants.%int_3.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %array_type.6d2: type = array_type %N, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4d8: type = pattern_type %array_type.6d2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.6d2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.931: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.f21: type = pattern_type %array_type.931 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.931 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C, %int_3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.c7a: <witness> = complete_type_witness %array_type.931 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.4d8) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.4d8) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11.2: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21.1: type = splice_block %.loc21.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc21.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc21.3: type = converted %IntLiteral.call, %.loc21.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc15.2: Core.IntLiteral = symbolic_binding N, 1 [symbolic = %N.loc15.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc32.1 (%array_type.6d2) = value_param call_param0
// CHECK:STDOUT:     %.loc32: type = splice_block %array_type.loc32.2 [symbolic = %array_type.loc32.1 (constants.%array_type.6d2)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc11.2 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc15.2 [symbolic = %N.loc15.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc32.2: type = array_type %N.ref, %T.ref [symbolic = %array_type.loc32.1 (constants.%array_type.6d2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc32.1 (%array_type.6d2) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11.2: type, %N.loc15.2: Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc11.1: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc15.1: Core.IntLiteral = symbolic_binding N, 1 [symbolic = %N.loc15.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc32.1: type = array_type %N.loc15.1, %T.loc11.1 [symbolic = %array_type.loc32.1 (constants.%array_type.6d2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc32.1 [symbolic = %pattern_type (constants.%pattern_type.4d8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc32.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc32.1 (%array_type.6d2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f21 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f21 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.931 = var %a.var_patt
// CHECK:STDOUT:   %.loc53.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc56.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc59.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc60.1: %tuple.type.8d4 = tuple_literal (%.loc53.1, %.loc56.1, %.loc59.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc60.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc53.2: init %C to %.loc60.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc60.3: init %C = converted %.loc53.1, %.loc53.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc60.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc56.2: init %C to %.loc60.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc60.5: init %C = converted %.loc56.1, %.loc56.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc60.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc59.2: init %C to %.loc60.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc60.7: init %C = converted %.loc59.1, %.loc59.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc60.8: init %array_type.931 to %a.var = array_init (%.loc60.3, %.loc60.5, %.loc60.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc41: init %array_type.931 = converted %.loc60.1, %.loc60.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc41
// CHECK:STDOUT:   %.loc49: type = splice_block %array_type [concrete = constants.%array_type.931] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.931]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.931 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.931 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%int_3) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc64: %array_type.931 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc64)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.931) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%T
// CHECK:STDOUT:   %N.loc15.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc32.1 => constants.%array_type.6d2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4d8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%int_3) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%C
// CHECK:STDOUT:   %N.loc15.1 => constants.%int_3
// CHECK:STDOUT:   %array_type.loc32.1 => constants.%array_type.931
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c7a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type.a0b: type = array_type %int_2, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %array_type.a0b [symbolic]
// CHECK:STDOUT:   %.184347.1: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.a0b [symbolic]
// CHECK:STDOUT:   %F.specific_fn.643: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.a69: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.931: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.f21: type = pattern_type %array_type.931 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type.931 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %array_type.158: type = array_type %int_2, %C [concrete]
// CHECK:STDOUT:   %pattern_type.6d3: type = pattern_type %array_type.158 [concrete]
// CHECK:STDOUT:   %F.specific_fn.540: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.b8b: <witness> = complete_type_witness %array_type.158 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc17 (%pattern_type.b42) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc17 (%pattern_type.b42) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc26 (%pattern_type.51d1c4.1) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc26 (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt, %T.ref.loc26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc26: type = name_ref T, %T.loc11.2 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:     %.loc26.3: Core.Form = init_form %T.ref.loc26 [symbolic = %.loc26.2 (constants.%.184347.1)]
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11.2: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc23.1 (%array_type.a0b) = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %array_type.loc23.2 [symbolic = %array_type.loc23.1 (constants.%array_type.a0b)] {
// CHECK:STDOUT:       %T.ref.loc20: type = name_ref T, %T.loc11.2 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %array_type.loc23.2: type = array_type %int_2, %T.ref.loc20 [symbolic = %array_type.loc23.1 (constants.%array_type.a0b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc23.1 (%array_type.a0b) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc11.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc11.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern %return.param_patt, %C.ref.loc40 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc40: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc40.2: Core.Form = init_form %C.ref.loc40 [concrete = constants.%.a69]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11.2: type) {
// CHECK:STDOUT:   %T.loc11.1: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:   %array_type.loc23.1: type = array_type constants.%int_2, %T.loc11.1 [symbolic = %array_type.loc23.1 (constants.%array_type.a0b)]
// CHECK:STDOUT:   %pattern_type.loc17: type = pattern_type %array_type.loc23.1 [symbolic = %pattern_type.loc17 (constants.%pattern_type.b42)]
// CHECK:STDOUT:   %.loc26.2: Core.Form = init_form %T.loc11.1 [symbolic = %.loc26.2 (constants.%.184347.1)]
// CHECK:STDOUT:   %pattern_type.loc26: type = pattern_type %T.loc11.1 [symbolic = %pattern_type.loc26 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc23.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc29.2: <specific function> = specific_function constants.%F, @F(%T.loc11.1) [symbolic = %F.specific_fn.loc29.2 (constants.%F.specific_fn.643)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc23.1 (%array_type.a0b)) -> out %return.param: @F.%T.loc11.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc23.1 (%array_type.a0b) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc29.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc29.2 (constants.%F.specific_fn.643)]
// CHECK:STDOUT:     %.loc26.1: ref @F.%T.loc11.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc11.1 (%T) to %.loc26.1 = call %F.specific_fn.loc29.1(%a.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f21 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f21 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.931 = var %a.var_patt
// CHECK:STDOUT:   %.loc54.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc57.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc60.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc61.1: %tuple.type.8d4 = tuple_literal (%.loc54.1, %.loc57.1, %.loc60.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc61.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc54.2: init %C to %.loc61.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.3: init %C = converted %.loc54.1, %.loc54.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc61.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc57.2: init %C to %.loc61.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.5: init %C = converted %.loc57.1, %.loc57.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc61.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc60.2: init %C to %.loc61.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.7: init %C = converted %.loc60.1, %.loc60.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc61.8: init %array_type.931 to %a.var = array_init (%.loc61.3, %.loc61.5, %.loc61.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc42: init %array_type.931 = converted %.loc61.1, %.loc61.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc42
// CHECK:STDOUT:   %.loc50: type = splice_block %array_type [concrete = constants.%array_type.931] {
// CHECK:STDOUT:     %C.ref.loc47: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc47 [concrete = constants.%array_type.931]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.931 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.931 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.540]
// CHECK:STDOUT:   %.loc40.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %.loc66: %array_type.158 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %C to %.loc40.1 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.931) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%T
// CHECK:STDOUT:   %array_type.loc23.1 => constants.%array_type.a0b
// CHECK:STDOUT:   %pattern_type.loc17 => constants.%pattern_type.b42
// CHECK:STDOUT:   %.loc26.2 => constants.%.184347.1
// CHECK:STDOUT:   %pattern_type.loc26 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc29.2 => constants.%F.specific_fn.643
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%C
// CHECK:STDOUT:   %array_type.loc23.1 => constants.%array_type.158
// CHECK:STDOUT:   %pattern_type.loc17 => constants.%pattern_type.6d3
// CHECK:STDOUT:   %.loc26.2 => constants.%.a69
// CHECK:STDOUT:   %pattern_type.loc26 => constants.%pattern_type.7c7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.b8b
// CHECK:STDOUT:   %F.specific_fn.loc29.2 => constants.%F.specific_fn.540
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.c79: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.cc9: type = pattern_type %array_type.c79 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d0d: <witness> = require_complete_type %array_type.c79 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d6e: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.7fa: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.7fa(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.b6d: type = array_type %int_3.1ba, %D [concrete]
// CHECK:STDOUT:   %pattern_type.454: type = pattern_type %array_type.b6d [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.b6d = tuple_value (%D.val, %D.val, %D.val) [concrete]
// CHECK:STDOUT:   %array_type.931: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %pattern_type.f21: type = pattern_type %array_type.931 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.c7a: <witness> = complete_type_witness %array_type.931 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.cc9) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.cc9) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc35: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %.loc21.1: type = splice_block %.loc21.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc21.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc21.3: type = converted %IntLiteral.call, %.loc21.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc15.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc15.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc32.1 (%array_type.c79) = value_param call_param0
// CHECK:STDOUT:     %.loc32: type = splice_block %array_type.loc32.2 [symbolic = %array_type.loc32.1 (constants.%array_type.c79)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc31: Core.IntLiteral = name_ref N, %N.loc15.2 [symbolic = %N.loc15.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc32.2: type = array_type %N.ref.loc31, %C.ref [symbolic = %array_type.loc32.1 (constants.%array_type.c79)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc32.1 (%array_type.c79) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc46: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc15.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc15.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc15.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc32.1: type = array_type %N.loc15.1, constants.%C [symbolic = %array_type.loc32.1 (constants.%array_type.c79)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc32.1 [symbolic = %pattern_type (constants.%pattern_type.cc9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc32.1 [symbolic = %require_complete (constants.%require_complete.d0d)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc15.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d6e)]
// CHECK:STDOUT:   %bound_method.loc39.3: <bound method> = bound_method %N.loc15.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc39.3 (constants.%bound_method.7fa)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39.2: init %i32 = call %bound_method.loc39.3(%N.loc15.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc32.1 (%array_type.c79)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc38: Core.IntLiteral = name_ref N, %N.loc15.2 [symbolic = %N.loc15.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc39.1: <bound method> = bound_method %N.ref.loc38, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d6e)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc39.2: <bound method> = bound_method %N.ref.loc38, %specific_fn [symbolic = %bound_method.loc39.3 (constants.%bound_method.7fa)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39.1: init %i32 = call %bound_method.loc39.2(%N.ref.loc38) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc39: init %i32 = converted %N.ref.loc38, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc39
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.454 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.454 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.b6d = var %a.var_patt
// CHECK:STDOUT:   %.loc60.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc63.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc66.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc67.1: %tuple.type.8d4 = tuple_literal (%.loc60.1, %.loc63.1, %.loc66.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc67.2: ref %D = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc60.2: init %D to %.loc67.2 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc67.3: init %D = converted %.loc60.1, %.loc60.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc67.4: ref %D = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc63.2: init %D to %.loc67.4 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc67.5: init %D = converted %.loc63.1, %.loc63.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc67.6: ref %D = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc66.2: init %D to %.loc67.6 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc67.7: init %D = converted %.loc66.1, %.loc66.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc67.8: init %array_type.b6d to %a.var = array_init (%.loc67.3, %.loc67.5, %.loc67.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc48: init %array_type.b6d = converted %.loc67.1, %.loc67.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc48
// CHECK:STDOUT:   %.loc56: type = splice_block %array_type [concrete = constants.%array_type.b6d] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %D.ref [concrete = constants.%array_type.b6d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.b6d = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.b6d = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc72: %array_type.931 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.b6d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc15.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc32.1 => constants.%array_type.c79
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc15.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc32.1 => constants.%array_type.931
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c7a
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061
// CHECK:STDOUT:   %bound_method.loc39.3 => constants.%bound_method.fa7
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39.2 => constants.%int_3.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.5de: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.640: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.240: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.dd4: %Int.as.ImplicitAs.impl.Convert.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.640) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.462: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.462, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.5de, %Int.as.ImplicitAs.impl.Convert.dd4 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.dd4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.d76: <bound method> = bound_method %N.5de, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.d76(%N.5de) [symbolic]
// CHECK:STDOUT:   %array_type.8c3: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.0fb: type = pattern_type %array_type.8c3 [symbolic]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a24: <witness> = require_complete_type %array_type.8c3 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.5de, %Int.as.Copy.impl.Op.664 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.207: <bound method> = bound_method %N.5de, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.931: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.f21: type = pattern_type %array_type.931 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.5c1: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.931 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/parts/int, loc158, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.0fb) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.0fb) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc27: Core.Form = init_form %i32.loc27 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %.loc13: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc11.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc24.1 (%array_type.8c3) = value_param call_param0
// CHECK:STDOUT:     %.loc24: type = splice_block %array_type.loc24.2 [symbolic = %array_type.loc24.1 (constants.%array_type.8c3)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc23: %i32 = name_ref N, %N.loc11.2 [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:       %impl.elem0.loc23: %.0a7 = impl_witness_access constants.%ImplicitAs.impl_witness.640, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.dd4]
// CHECK:STDOUT:       %bound_method.loc23.2: <bound method> = bound_method %N.ref.loc23, %impl.elem0.loc23 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc23.3: <bound method> = bound_method %N.ref.loc23, %specific_fn.loc23 [symbolic = %bound_method.loc23.1 (constants.%bound_method.d76)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc23.2: init Core.IntLiteral = call %bound_method.loc23.3(%N.ref.loc23) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc23.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc23.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc23.2: Core.IntLiteral = converted %N.ref.loc23, %.loc23.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc24.2: type = array_type %.loc23.2, %C.ref [symbolic = %array_type.loc24.1 (constants.%array_type.8c3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc24.1 (%array_type.8c3) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc38: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc11.2: %i32) {
// CHECK:STDOUT:   %N.loc11.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc11.1, constants.%Int.as.ImplicitAs.impl.Convert.dd4 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc23.1: <bound method> = bound_method %N.loc11.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc23.1 (constants.%bound_method.d76)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc23.1: init Core.IntLiteral = call %bound_method.loc23.1(%N.loc11.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc24.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc23.1, constants.%C [symbolic = %array_type.loc24.1 (constants.%array_type.8c3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc24.1 [symbolic = %pattern_type (constants.%pattern_type.0fb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc24.1 [symbolic = %require_complete (constants.%require_complete.a24)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc11.1, constants.%Int.as.Copy.impl.Op.664 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc30.3: <bound method> = bound_method %N.loc11.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc30.3 (constants.%bound_method.207)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc24.1 (%array_type.8c3)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc30: %i32 = name_ref N, %N.loc11.2 [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:     %impl.elem0.loc30: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:     %bound_method.loc30.1: <bound method> = bound_method %N.ref.loc30, %impl.elem0.loc30 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc30.2: <bound method> = bound_method %N.ref.loc30, %specific_fn.loc30 [symbolic = %bound_method.loc30.3 (constants.%bound_method.207)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc30.2(%N.ref.loc30) [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f21 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f21 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.931 = var %a.var_patt
// CHECK:STDOUT:   %.loc52.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc55.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc59.1: %tuple.type.8d4 = tuple_literal (%.loc52.1, %.loc55.1, %.loc58.1) [concrete = constants.%tuple.5c1]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc59.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc52.2: init %C to %.loc59.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc59.3: init %C = converted %.loc52.1, %.loc52.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc59.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc55.2: init %C to %.loc59.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc59.5: init %C = converted %.loc55.1, %.loc55.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc59.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc58.2: init %C to %.loc59.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc59.7: init %C = converted %.loc58.1, %.loc58.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc59.8: init %array_type.931 to %a.var = array_init (%.loc59.3, %.loc59.5, %.loc59.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc40: init %array_type.931 = converted %.loc59.1, %.loc59.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc40
// CHECK:STDOUT:   %.loc48: type = splice_block %array_type [concrete = constants.%array_type.931] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.931]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.931 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.931 = name_ref a, %a
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.931) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.5de) {
// CHECK:STDOUT:   %N.loc11.1 => constants.%N.5de
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc23.1 => constants.%bound_method.d76
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc23.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc24.1 => constants.%array_type.8c3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_array_length_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.5de: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.640: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.240: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.dd4: %Int.as.ImplicitAs.impl.Convert.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.640) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.462: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.462, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.5de, %Int.as.ImplicitAs.impl.Convert.dd4 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.dd4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.5de, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.5de) [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.0fb: type = pattern_type %array_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a24: <witness> = require_complete_type %array_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/parts/int, loc158, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.0fb) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.0fb) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc11.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc24.1 (%array_type) = value_param call_param0
// CHECK:STDOUT:     %.loc24: type = splice_block %array_type.loc24.2 [symbolic = %array_type.loc24.1 (constants.%array_type)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc11.2 [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:       %impl.elem0: %.0a7 = impl_witness_access constants.%ImplicitAs.impl_witness.640, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.dd4]
// CHECK:STDOUT:       %bound_method.loc23.2: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc23.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc23.1 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc23.2: init Core.IntLiteral = call %bound_method.loc23.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc23.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc23.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc23.2: Core.IntLiteral = converted %N.ref, %.loc23.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc24.2: type = array_type %.loc23.2, %C.ref [symbolic = %array_type.loc24.1 (constants.%array_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc24.1 (%array_type) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc11.2: %i32) {
// CHECK:STDOUT:   %N.loc11.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc11.1 (constants.%N.5de)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc11.1, constants.%Int.as.ImplicitAs.impl.Convert.dd4 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc23.1: <bound method> = bound_method %N.loc11.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc23.1 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc23.1: init Core.IntLiteral = call %bound_method.loc23.1(%N.loc11.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc23.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc24.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc23.1, constants.%C [symbolic = %array_type.loc24.1 (constants.%array_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc24.1 [symbolic = %pattern_type (constants.%pattern_type.0fb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc24.1 [symbolic = %require_complete (constants.%require_complete.a24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc24.1 (%array_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc37: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc40: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc43: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc44: %tuple.type = tuple_literal (%.loc37, %.loc40, %.loc43) [concrete = constants.%tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.5de) {
// CHECK:STDOUT:   %N.loc11.1 => constants.%N.5de
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc23.1 => constants.%bound_method
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc23.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc24.1 => constants.%array_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
