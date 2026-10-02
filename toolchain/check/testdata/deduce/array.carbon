// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
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

fn F[T: type](a: array(T, 3)) -> T { return F(a); }

fn G() -> C {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- bound_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N: Core.IntLiteral](unused a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- type_and_bound.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T: type, N: Core.IntLiteral](unused a: array(T, N)) {}

fn G() {
  var a: array(C, 3) = ({}, {}, {});
  F(a);
}

// --- fail_bound_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T: type](a: array(T, 2)) -> T { return F(a); }

fn G() -> C {
  // TODO: We succeed at deducing T here but fail to convert. Is this the right behavior?
  var a: array(C, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(C, 3)` to `array(C, 2)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+7]]:12: note: type `array(C, 3)` does not implement interface `Core.ImplicitAs(array(C, 2))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE-11]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[T: type](a: array(T, 2)) -> T { return F(a); }
  // CHECK:STDERR:               ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[N: Core.IntLiteral](unused a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  // TODO: We succeed at deducing N here but fail to convert. Is this the right behavior?
  var a: array(D, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(D, 3)` to `array(C, 3)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:12: note: type `array(D, 3)` does not implement interface `Core.ImplicitAs(array(C, 3))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE-11]]:33: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[N: Core.IntLiteral](unused a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR:                                 ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_bound_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N: i32](unused a: array(C, N)) -> i32 { return N; }

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
  // CHECK:STDERR: fn F[N: i32](unused a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_todo_array_length_from_tuple.carbon
library "[[@TEST_NAME]]";

class C {}

fn F[N: i32](unused a: array(C, N)) {}

fn G() {
  // TODO: Deduce N as 3 from the tuple's size.
  //
  // CHECK:STDERR: fail_todo_array_length_from_tuple.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `N` [DeductionIncomplete]
  // CHECK:STDERR:   F(({}, {}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_array_length_from_tuple.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[N: i32](unused a: array(C, N)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(({}, {}, {}));
}

// CHECK:STDOUT: --- type_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.319: type = class_type @Array, @Array(%T, %int_3) [symbolic]
// CHECK:STDOUT:   %pattern_type.2b6: type = pattern_type %Array.319 [symbolic]
// CHECK:STDOUT:   %a.param_patt.b6c: %pattern_type.2b6 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.32c: %pattern_type.2b6 = wrapper_binding_pattern a, %a.param_patt.b6c [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.af9: <witness> = require_complete_type %Array.319 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.c05: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.c39: type = class_type @Array, @Array(%C, %int_3) [concrete]
// CHECK:STDOUT:   %array_type.9a3: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %complete_type.1a4: <witness> = complete_type_witness %array_type.9a3 [concrete]
// CHECK:STDOUT:   %pattern_type.1a7: type = pattern_type %Array.c39 [concrete]
// CHECK:STDOUT:   %a.patt.496: %pattern_type.1a7 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt.496 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.c39 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %a.param_patt.80a: %pattern_type.1a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.dac: %pattern_type.1a7 = wrapper_binding_pattern a, %a.param_patt.80a [concrete]
// CHECK:STDOUT:   %F.specific_fn.c73: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2d8: %pattern_type.1a7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b3: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt.2d8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.param_patt.loc6_16.1: @F.%pattern_type.loc6_16 (%pattern_type.2b6) = value_param_pattern [symbolic = %a.param_patt.loc6_16.2 (constants.%a.param_patt.b6c)]
// CHECK:STDOUT:     %a.patt.loc6_16.1: @F.%pattern_type.loc6_16 (%pattern_type.2b6) = wrapper_binding_pattern a, %a.param_patt.loc6_16.1 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt.32c)]
// CHECK:STDOUT:     %return.param_patt.loc6_34.1: @F.%pattern_type.loc6_34 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_34.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:     %return.patt.loc6_31.1: @F.%pattern_type.loc6_34 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_34.1, %T.ref.loc6_34 [symbolic = %return.patt.loc6_31.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_34: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_34.3: Core.Form = init_form %T.ref.loc6_34 [symbolic = %.loc6_34.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%Array.loc6_28.1 (%Array.319) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28: type = splice_block %Array.loc6_28.2 [symbolic = %Array.loc6_28.1 (constants.%Array.319)] {
// CHECK:STDOUT:       %T.ref.loc6_24: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %Array.loc6_28.2: type = class_type @Array, @Array(constants.%T, constants.%int_3) [symbolic = %Array.loc6_28.1 (constants.%Array.319)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%Array.loc6_28.1 (%Array.319) = wrapper_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc8 [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_11.2: Core.Form = init_form %C.ref.loc8 [concrete = constants.%.a44]
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
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %Array.loc6_28.1: type = class_type @Array, @Array(%T.loc6_7.1, constants.%int_3) [symbolic = %Array.loc6_28.1 (constants.%Array.319)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %Array.loc6_28.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.2b6)]
// CHECK:STDOUT:   %a.param_patt.loc6_16.2: @F.%pattern_type.loc6_16 (%pattern_type.2b6) = value_param_pattern [symbolic = %a.param_patt.loc6_16.2 (constants.%a.param_patt.b6c)]
// CHECK:STDOUT:   %a.patt.loc6_16.2: @F.%pattern_type.loc6_16 (%pattern_type.2b6) = wrapper_binding_pattern a, %a.param_patt.loc6_16.2 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt.32c)]
// CHECK:STDOUT:   %.loc6_34.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_34.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_34: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_34 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc6_34.2: @F.%pattern_type.loc6_34 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_34.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc6_31.2: @F.%pattern_type.loc6_34 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_34.2, %T.loc6_7.1 [symbolic = %return.patt.loc6_31.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc6_28.1 [symbolic = %require_complete (constants.%require_complete.af9)]
// CHECK:STDOUT:   %F.specific_fn.loc6_45.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_45.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%Array.loc6_28.1 (%Array.319)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%Array.loc6_28.1 (%Array.319) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc6_45.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_45.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:     %.loc6_34.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_34.1 = call %F.specific_fn.loc6_45.1(%a.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Array.c39 = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %Array.c39 to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %Array.c39 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %Array [concrete = constants.%Array.c39] {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%C, constants.%int_3) [concrete = constants.%Array.c39]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.c39 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1a7 = ref_binding_pattern a [concrete = constants.%a.patt.496]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %Array.c39 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.c73]
// CHECK:STDOUT:   %.loc8_11.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10: %Array.c39 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %C to %.loc8_11.1 = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_3.1: ref %C = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_3.2: ref %C = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_3.3: ref %C = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3: <bound method> = bound_method %.loc9_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3(%.loc9_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %Array.c39) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %Array.loc6_28.1 => constants.%Array.319
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.2b6
// CHECK:STDOUT:   %a.param_patt.loc6_16.2 => constants.%a.param_patt.b6c
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt.32c
// CHECK:STDOUT:   %.loc6_34.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_34 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_34.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.b39
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.af9
// CHECK:STDOUT:   %F.specific_fn.loc6_45.2 => constants.%F.specific_fn.c05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %Array.loc6_28.1 => constants.%Array.c39
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.1a7
// CHECK:STDOUT:   %a.param_patt.loc6_16.2 => constants.%a.param_patt.80a
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt.dac
// CHECK:STDOUT:   %.loc6_34.2 => constants.%.a44
// CHECK:STDOUT:   %pattern_type.loc6_34 => constants.%pattern_type.98b
// CHECK:STDOUT:   %return.param_patt.loc6_34.2 => constants.%return.param_patt.89a
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.e4a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.1a4
// CHECK:STDOUT:   %F.specific_fn.loc6_45.2 => constants.%F.specific_fn.c73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt.aa5: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.f60: type = class_type @Array, @Array(%C, %N.fe9) [symbolic]
// CHECK:STDOUT:   %pattern_type.d24: type = pattern_type %Array.f60 [symbolic]
// CHECK:STDOUT:   %a.param_patt.9fa: %pattern_type.d24 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.439: %pattern_type.d24 = wrapper_binding_pattern a, %a.param_patt.9fa [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.11c: <witness> = require_complete_type %Array.f60 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb: <bound method> = bound_method %N.fe9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f47: <bound method> = bound_method %N.fe9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.f47(%N.fe9) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.c39: type = class_type @Array, @Array(%C, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %array_type.9a3: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %complete_type.1a4: <witness> = complete_type_witness %array_type.9a3 [concrete]
// CHECK:STDOUT:   %pattern_type.1a7: type = pattern_type %Array.c39 [concrete]
// CHECK:STDOUT:   %a.patt.496: %pattern_type.1a7 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt.496 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.c39 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %a.param_patt.80a: %pattern_type.1a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.dac: %pattern_type.1a7 = wrapper_binding_pattern a, %a.param_patt.80a [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2d8: %pattern_type.1a7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b3: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt.2d8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.d29: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.af5: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %N.patt.loc6_7.1: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.aa5)]
// CHECK:STDOUT:     %a.param_patt.loc6_34.1: @F.%pattern_type (%pattern_type.d24) = value_param_pattern [symbolic = %a.param_patt.loc6_34.2 (constants.%a.param_patt.9fa)]
// CHECK:STDOUT:     %a.patt.loc6_34.1: @F.%pattern_type (%pattern_type.d24) = wrapper_binding_pattern a, %a.param_patt.loc6_34.1 [symbolic = %a.patt.loc6_34.2 (constants.%a.patt.439)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_52: Core.Form = init_form %i32 [concrete = constants.%.55c]
// CHECK:STDOUT:     %.loc6_13: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_7.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:     %a.param: @F.%Array.loc6_46.1 (%Array.f60) = value_param call_param0
// CHECK:STDOUT:     %.loc6_46: type = splice_block %Array.loc6_46.2 [symbolic = %Array.loc6_46.1 (constants.%Array.f60)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_45: Core.IntLiteral = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:       %Array.loc6_46.2: type = class_type @Array, @Array(constants.%C, constants.%N.fe9) [symbolic = %Array.loc6_46.1 (constants.%Array.f60)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%Array.loc6_46.1 (%Array.f60) = wrapper_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32 [concrete = constants.%.55c]
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
// CHECK:STDOUT: generic fn @F(%N.loc6_7.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.patt.loc6_7.2: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.aa5)]
// CHECK:STDOUT:   %N.loc6_7.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:   %Array.loc6_46.1: type = class_type @Array, @Array(constants.%C, %N.loc6_7.1) [symbolic = %Array.loc6_46.1 (constants.%Array.f60)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc6_46.1 [symbolic = %pattern_type (constants.%pattern_type.d24)]
// CHECK:STDOUT:   %a.param_patt.loc6_34.2: @F.%pattern_type (%pattern_type.d24) = value_param_pattern [symbolic = %a.param_patt.loc6_34.2 (constants.%a.param_patt.9fa)]
// CHECK:STDOUT:   %a.patt.loc6_34.2: @F.%pattern_type (%pattern_type.d24) = wrapper_binding_pattern a, %a.param_patt.loc6_34.2 [symbolic = %a.patt.loc6_34.2 (constants.%a.patt.439)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc6_46.1 [symbolic = %require_complete (constants.%require_complete.11c)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb)]
// CHECK:STDOUT:   %bound_method.loc6_66.3: <bound method> = bound_method %N.loc6_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc6_66.3 (constants.%bound_method.f47)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_66.2: init %i32 = call %bound_method.loc6_66.3(%N.loc6_7.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_66.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%Array.loc6_46.1 (%Array.f60)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_65: Core.IntLiteral = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc6_66.1: <bound method> = bound_method %N.ref.loc6_65, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_66.2: <bound method> = bound_method %N.ref.loc6_65, %specific_fn [symbolic = %bound_method.loc6_66.3 (constants.%bound_method.f47)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_66.1: init %i32 = call %bound_method.loc6_66.2(%N.ref.loc6_65) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_66.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc6_66: init %i32 = converted %N.ref.loc6_65, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_66.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_66.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc6_66
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Array.c39 = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %Array.c39 to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %Array.c39 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %Array [concrete = constants.%Array.c39] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%C, constants.%int_3.1ba) [concrete = constants.%Array.c39]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.c39 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1a7 = ref_binding_pattern a [concrete = constants.%a.patt.496]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %Array.c39 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %Array.c39 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_3.1: ref %C = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_3.2: ref %C = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_3.3: ref %C = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3: <bound method> = bound_method %.loc9_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3(%.loc9_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %Array.c39) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.fe9) {
// CHECK:STDOUT:   %N.patt.loc6_7.2 => constants.%N.patt.aa5
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%N.fe9
// CHECK:STDOUT:   %Array.loc6_46.1 => constants.%Array.f60
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d24
// CHECK:STDOUT:   %a.param_patt.loc6_34.2 => constants.%a.param_patt.9fa
// CHECK:STDOUT:   %a.patt.loc6_34.2 => constants.%a.patt.439
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.patt.loc6_7.2 => constants.%N.patt.aa5
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %Array.loc6_46.1 => constants.%Array.c39
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a7
// CHECK:STDOUT:   %a.param_patt.loc6_34.2 => constants.%a.param_patt.80a
// CHECK:STDOUT:   %a.patt.loc6_34.2 => constants.%a.patt.dac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.1a4
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8
// CHECK:STDOUT:   %bound_method.loc6_66.3 => constants.%bound_method.d29
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_66.2 => constants.%int_3.af5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.bb7: type = class_type @Array, @Array(%T, %N) [symbolic]
// CHECK:STDOUT:   %pattern_type.e8b: type = pattern_type %Array.bb7 [symbolic]
// CHECK:STDOUT:   %a.param_patt.3a3: %pattern_type.e8b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.168: %pattern_type.e8b = wrapper_binding_pattern a, %a.param_patt.3a3 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.341: <witness> = require_complete_type %Array.bb7 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.c39: type = class_type @Array, @Array(%C, %int_3) [concrete]
// CHECK:STDOUT:   %array_type.9a3: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %complete_type.1a4: <witness> = complete_type_witness %array_type.9a3 [concrete]
// CHECK:STDOUT:   %pattern_type.1a7: type = pattern_type %Array.c39 [concrete]
// CHECK:STDOUT:   %a.patt.496: %pattern_type.1a7 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt.496 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.c39 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %a.param_patt.80a: %pattern_type.1a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.dac: %pattern_type.1a7 = wrapper_binding_pattern a, %a.param_patt.80a [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C, %int_3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2d8: %pattern_type.1a7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b3: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt.2d8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc6_16.1: %pattern_type.dc0 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.param_patt.loc6_43.1: @F.%pattern_type (%pattern_type.e8b) = value_param_pattern [symbolic = %a.param_patt.loc6_43.2 (constants.%a.param_patt.3a3)]
// CHECK:STDOUT:     %a.patt.loc6_43.1: @F.%pattern_type (%pattern_type.e8b) = wrapper_binding_pattern a, %a.param_patt.loc6_43.1 [symbolic = %a.patt.loc6_43.2 (constants.%a.patt.168)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc6_7: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_22: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.frozen.loc6_16: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_16.2: Core.IntLiteral = symbolic_binding N, 1 [symbolic = %N.loc6_16.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%Array.loc6_55.1 (%Array.bb7) = value_param call_param0
// CHECK:STDOUT:     %.loc6_55: type = splice_block %Array.loc6_55.2 [symbolic = %Array.loc6_55.1 (constants.%Array.bb7)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc6_16.2 [symbolic = %N.loc6_16.1 (constants.%N)]
// CHECK:STDOUT:       %Array.loc6_55.2: type = class_type @Array, @Array(constants.%T, constants.%N) [symbolic = %Array.loc6_55.1 (constants.%Array.bb7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%Array.loc6_55.1 (%Array.bb7) = wrapper_binding a, %a.param
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
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type, %N.loc6_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %N.patt.loc6_16.2: %pattern_type.dc0 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:   %N.loc6_16.1: Core.IntLiteral = symbolic_binding N, 1 [symbolic = %N.loc6_16.1 (constants.%N)]
// CHECK:STDOUT:   %Array.loc6_55.1: type = class_type @Array, @Array(%T.loc6_7.1, %N.loc6_16.1) [symbolic = %Array.loc6_55.1 (constants.%Array.bb7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc6_55.1 [symbolic = %pattern_type (constants.%pattern_type.e8b)]
// CHECK:STDOUT:   %a.param_patt.loc6_43.2: @F.%pattern_type (%pattern_type.e8b) = value_param_pattern [symbolic = %a.param_patt.loc6_43.2 (constants.%a.param_patt.3a3)]
// CHECK:STDOUT:   %a.patt.loc6_43.2: @F.%pattern_type (%pattern_type.e8b) = wrapper_binding_pattern a, %a.param_patt.loc6_43.2 [symbolic = %a.patt.loc6_43.2 (constants.%a.patt.168)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc6_55.1 [symbolic = %require_complete (constants.%require_complete.341)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%Array.loc6_55.1 (%Array.bb7)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Array.c39 = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %Array.c39 to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %Array.c39 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %Array [concrete = constants.%Array.c39] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%C, constants.%int_3) [concrete = constants.%Array.c39]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.c39 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1a7 = ref_binding_pattern a [concrete = constants.%a.patt.496]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %Array.c39 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%int_3) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %Array.c39 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_3.1: ref %C = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_3.2: ref %C = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_3.3: ref %C = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3: <bound method> = bound_method %.loc9_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3(%.loc9_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %Array.c39) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc6_16.1 => constants.%N
// CHECK:STDOUT:   %Array.loc6_55.1 => constants.%Array.bb7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e8b
// CHECK:STDOUT:   %a.param_patt.loc6_43.2 => constants.%a.param_patt.3a3
// CHECK:STDOUT:   %a.patt.loc6_43.2 => constants.%a.patt.168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%int_3) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc6_16.1 => constants.%int_3
// CHECK:STDOUT:   %Array.loc6_55.1 => constants.%Array.c39
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a7
// CHECK:STDOUT:   %a.param_patt.loc6_43.2 => constants.%a.param_patt.80a
// CHECK:STDOUT:   %a.patt.loc6_43.2 => constants.%a.patt.dac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.1a4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.740: type = class_type @Array, @Array(%T, %int_2) [symbolic]
// CHECK:STDOUT:   %pattern_type.e62: type = pattern_type %Array.740 [symbolic]
// CHECK:STDOUT:   %a.param_patt.dd1: %pattern_type.e62 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.10c: %pattern_type.e62 = wrapper_binding_pattern a, %a.param_patt.dd1 [symbolic]
// CHECK:STDOUT:   %.184347.1: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d2d9.1: %pattern_type.51d1c4.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d1c4.1 = return_slot_pattern %return.param_patt.41d2d9.1, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fa2: <witness> = require_complete_type %Array.740 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.c05: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.c39: type = class_type @Array, @Array(%C, %int_3) [concrete]
// CHECK:STDOUT:   %pattern_type.1a7: type = pattern_type %Array.c39 [concrete]
// CHECK:STDOUT:   %a.patt.496: %pattern_type.1a7 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt.496 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.c39 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %Array.9c6: type = class_type @Array, @Array(%C, %int_2) [concrete]
// CHECK:STDOUT:   %pattern_type.460: type = pattern_type %Array.9c6 [concrete]
// CHECK:STDOUT:   %a.param_patt.37a: %pattern_type.460 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.93d: %pattern_type.460 = wrapper_binding_pattern a, %a.param_patt.37a [concrete]
// CHECK:STDOUT:   %F.specific_fn.c73: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %array_type.cc9: type = array_type %int_2, %C [concrete]
// CHECK:STDOUT:   %complete_type.d6a: <witness> = complete_type_witness %array_type.cc9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc10_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2d8: %pattern_type.1a7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b3: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt.2d8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc10_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc10_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc10_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc10_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.param_patt.loc6_16.1: @F.%pattern_type.loc6_16 (%pattern_type.e62) = value_param_pattern [symbolic = %a.param_patt.loc6_16.2 (constants.%a.param_patt.dd1)]
// CHECK:STDOUT:     %a.patt.loc6_16.1: @F.%pattern_type.loc6_16 (%pattern_type.e62) = wrapper_binding_pattern a, %a.param_patt.loc6_16.1 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt.10c)]
// CHECK:STDOUT:     %return.param_patt.loc6_34.1: @F.%pattern_type.loc6_34 (%pattern_type.51d1c4.1) = out_param_pattern [symbolic = %return.param_patt.loc6_34.2 (constants.%return.param_patt.41d2d9.1)]
// CHECK:STDOUT:     %return.patt.loc6_31.1: @F.%pattern_type.loc6_34 (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt.loc6_34.1, %T.ref.loc6_34 [symbolic = %return.patt.loc6_31.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_34: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_34.3: Core.Form = init_form %T.ref.loc6_34 [symbolic = %.loc6_34.2 (constants.%.184347.1)]
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%Array.loc6_28.1 (%Array.740) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28: type = splice_block %Array.loc6_28.2 [symbolic = %Array.loc6_28.1 (constants.%Array.740)] {
// CHECK:STDOUT:       %T.ref.loc6_24: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %Array.loc6_28.2: type = class_type @Array, @Array(constants.%T, constants.%int_2) [symbolic = %Array.loc6_28.1 (constants.%Array.740)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%Array.loc6_28.1 (%Array.740) = wrapper_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc8 [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_11.2: Core.Form = init_form %C.ref.loc8 [concrete = constants.%.a44]
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
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %Array.loc6_28.1: type = class_type @Array, @Array(%T.loc6_7.1, constants.%int_2) [symbolic = %Array.loc6_28.1 (constants.%Array.740)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %Array.loc6_28.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.e62)]
// CHECK:STDOUT:   %a.param_patt.loc6_16.2: @F.%pattern_type.loc6_16 (%pattern_type.e62) = value_param_pattern [symbolic = %a.param_patt.loc6_16.2 (constants.%a.param_patt.dd1)]
// CHECK:STDOUT:   %a.patt.loc6_16.2: @F.%pattern_type.loc6_16 (%pattern_type.e62) = wrapper_binding_pattern a, %a.param_patt.loc6_16.2 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt.10c)]
// CHECK:STDOUT:   %.loc6_34.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_34.2 (constants.%.184347.1)]
// CHECK:STDOUT:   %pattern_type.loc6_34: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_34 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %return.param_patt.loc6_34.2: @F.%pattern_type.loc6_34 (%pattern_type.51d1c4.1) = out_param_pattern [symbolic = %return.param_patt.loc6_34.2 (constants.%return.param_patt.41d2d9.1)]
// CHECK:STDOUT:   %return.patt.loc6_31.2: @F.%pattern_type.loc6_34 (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt.loc6_34.2, %T.loc6_7.1 [symbolic = %return.patt.loc6_31.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc6_28.1 [symbolic = %require_complete (constants.%require_complete.fa2)]
// CHECK:STDOUT:   %F.specific_fn.loc6_45.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_45.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%Array.loc6_28.1 (%Array.740)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%Array.loc6_28.1 (%Array.740) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc6_45.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_45.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:     %.loc6_34.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_34.1 = call %F.specific_fn.loc6_45.1(%a.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Array.c39 = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc10_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_35.1: %tuple.type.8d4 = tuple_literal (%.loc10_26.1, %.loc10_30.1, %.loc10_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc10_26.2: init %C to %.loc10_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.3: init %C = converted %.loc10_26.1, %.loc10_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc10_30.2: init %C to %.loc10_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.5: init %C = converted %.loc10_30.1, %.loc10_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc10_34.2: init %C to %.loc10_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.7: init %C = converted %.loc10_34.1, %.loc10_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.8: init %Array.c39 to %a.var = array_init (%.loc10_35.3, %.loc10_35.5, %.loc10_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_3: init %Array.c39 = converted %.loc10_35.1, %.loc10_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_20: type = splice_block %Array [concrete = constants.%Array.c39] {
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%C, constants.%int_3) [concrete = constants.%Array.c39]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.c39 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1a7 = ref_binding_pattern a [concrete = constants.%a.patt.496]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %Array.c39 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.c73]
// CHECK:STDOUT:   %.loc8_11.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %.loc21: %Array.9c6 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %C to %.loc8_11.1 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_3.1: ref %C = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_3.1: <bound method> = bound_method %.loc10_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_3.1(%.loc10_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_3.2: ref %C = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_3.2: <bound method> = bound_method %.loc10_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_3.2(%.loc10_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_3.3: ref %C = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_3.3: <bound method> = bound_method %.loc10_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_3.3(%.loc10_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_3.3(%self.param: ref %Array.c39) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %Array.loc6_28.1 => constants.%Array.740
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.e62
// CHECK:STDOUT:   %a.param_patt.loc6_16.2 => constants.%a.param_patt.dd1
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt.10c
// CHECK:STDOUT:   %.loc6_34.2 => constants.%.184347.1
// CHECK:STDOUT:   %pattern_type.loc6_34 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %return.param_patt.loc6_34.2 => constants.%return.param_patt.41d2d9.1
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.b39
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.fa2
// CHECK:STDOUT:   %F.specific_fn.loc6_45.2 => constants.%F.specific_fn.c05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %Array.loc6_28.1 => constants.%Array.9c6
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.460
// CHECK:STDOUT:   %a.param_patt.loc6_16.2 => constants.%a.param_patt.37a
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt.93d
// CHECK:STDOUT:   %.loc6_34.2 => constants.%.a44
// CHECK:STDOUT:   %pattern_type.loc6_34 => constants.%pattern_type.98b
// CHECK:STDOUT:   %return.param_patt.loc6_34.2 => constants.%return.param_patt.89a
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.e4a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d6a
// CHECK:STDOUT:   %F.specific_fn.loc6_45.2 => constants.%F.specific_fn.c73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt.aa5: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.f60: type = class_type @Array, @Array(%C, %N.fe9) [symbolic]
// CHECK:STDOUT:   %pattern_type.d24: type = pattern_type %Array.f60 [symbolic]
// CHECK:STDOUT:   %a.param_patt.9fa: %pattern_type.d24 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.439: %pattern_type.d24 = wrapper_binding_pattern a, %a.param_patt.9fa [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.11c: <witness> = require_complete_type %Array.f60 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb: <bound method> = bound_method %N.fe9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f47: <bound method> = bound_method %N.fe9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.f47(%N.fe9) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.f45: type = class_type @Array, @Array(%D, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %pattern_type.467: type = pattern_type %Array.f45 [concrete]
// CHECK:STDOUT:   %a.patt.349: %pattern_type.467 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.467 = var_pattern %a.patt.349 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.f45 = tuple_value (%D.val, %D.val, %D.val) [concrete]
// CHECK:STDOUT:   %Array.c39: type = class_type @Array, @Array(%C, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %pattern_type.1a7: type = pattern_type %Array.c39 [concrete]
// CHECK:STDOUT:   %a.param_patt.80a: %pattern_type.1a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.dac: %pattern_type.1a7 = wrapper_binding_pattern a, %a.param_patt.80a [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %array_type.9a3: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %complete_type.1a4: <witness> = complete_type_witness %array_type.9a3 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc11_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %self.param_patt.b8e: %pattern_type.d8d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.86b: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt.b8e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.e5e: %pattern_type.467 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1d3: %pattern_type.467 = wrapper_binding_pattern self, %self.param_patt.e5e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.d29: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.af5: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc11_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = ref_param_pattern [concrete = constants.%self.param_patt.b8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.86b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %D = ref_param call_param0
// CHECK:STDOUT:     %self: ref %D = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc11_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = ref_param_pattern [concrete = constants.%self.param_patt.b8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.86b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %D = ref_param call_param0
// CHECK:STDOUT:     %self: ref %D = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.467 = ref_param_pattern [concrete = constants.%self.param_patt.e5e]
// CHECK:STDOUT:     %self.patt: %pattern_type.467 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1d3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.f45 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.f45 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc11_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.467 = ref_param_pattern [concrete = constants.%self.param_patt.e5e]
// CHECK:STDOUT:     %self.patt: %pattern_type.467 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1d3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.f45 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.f45 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %N.patt.loc7_7.1: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_7.2 (constants.%N.patt.aa5)]
// CHECK:STDOUT:     %a.param_patt.loc7_34.1: @F.%pattern_type (%pattern_type.d24) = value_param_pattern [symbolic = %a.param_patt.loc7_34.2 (constants.%a.param_patt.9fa)]
// CHECK:STDOUT:     %a.patt.loc7_34.1: @F.%pattern_type (%pattern_type.d24) = wrapper_binding_pattern a, %a.param_patt.loc7_34.1 [symbolic = %a.patt.loc7_34.2 (constants.%a.patt.439)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_52: Core.Form = init_form %i32 [concrete = constants.%.55c]
// CHECK:STDOUT:     %.loc7_13: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_7.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc7_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:     %a.param: @F.%Array.loc7_46.1 (%Array.f60) = value_param call_param0
// CHECK:STDOUT:     %.loc7_46: type = splice_block %Array.loc7_46.2 [symbolic = %Array.loc7_46.1 (constants.%Array.f60)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc7_45: Core.IntLiteral = name_ref N, %N.loc7_7.2 [symbolic = %N.loc7_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:       %Array.loc7_46.2: type = class_type @Array, @Array(constants.%C, constants.%N.fe9) [symbolic = %Array.loc7_46.1 (constants.%Array.f60)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%Array.loc7_46.1 (%Array.f60) = wrapper_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9: Core.Form = init_form %i32 [concrete = constants.%.55c]
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
// CHECK:STDOUT: generic fn @F(%N.loc7_7.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.patt.loc7_7.2: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_7.2 (constants.%N.patt.aa5)]
// CHECK:STDOUT:   %N.loc7_7.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc7_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:   %Array.loc7_46.1: type = class_type @Array, @Array(constants.%C, %N.loc7_7.1) [symbolic = %Array.loc7_46.1 (constants.%Array.f60)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc7_46.1 [symbolic = %pattern_type (constants.%pattern_type.d24)]
// CHECK:STDOUT:   %a.param_patt.loc7_34.2: @F.%pattern_type (%pattern_type.d24) = value_param_pattern [symbolic = %a.param_patt.loc7_34.2 (constants.%a.param_patt.9fa)]
// CHECK:STDOUT:   %a.patt.loc7_34.2: @F.%pattern_type (%pattern_type.d24) = wrapper_binding_pattern a, %a.param_patt.loc7_34.2 [symbolic = %a.patt.loc7_34.2 (constants.%a.patt.439)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc7_46.1 [symbolic = %require_complete (constants.%require_complete.11c)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc7_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb)]
// CHECK:STDOUT:   %bound_method.loc7_66.3: <bound method> = bound_method %N.loc7_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc7_66.3 (constants.%bound_method.f47)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_66.2: init %i32 = call %bound_method.loc7_66.3(%N.loc7_7.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_66.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%Array.loc7_46.1 (%Array.f60)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc7_65: Core.IntLiteral = name_ref N, %N.loc7_7.2 [symbolic = %N.loc7_7.1 (constants.%N.fe9)]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc7_66.1: <bound method> = bound_method %N.ref.loc7_65, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_66.2: <bound method> = bound_method %N.ref.loc7_65, %specific_fn [symbolic = %bound_method.loc7_66.3 (constants.%bound_method.f47)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_66.1: init %i32 = call %bound_method.loc7_66.2(%N.ref.loc7_65) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_66.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc7_66: init %i32 = converted %N.ref.loc7_65, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_66.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_66.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc7_66
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Array.f45 = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc11_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_35.1: %tuple.type.8d4 = tuple_literal (%.loc11_26.1, %.loc11_30.1, %.loc11_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_35.2: ref %D = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_26.2: init %D to %.loc11_35.2 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.3: init %D = converted %.loc11_26.1, %.loc11_26.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_35.4: ref %D = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc11_30.2: init %D to %.loc11_35.4 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.5: init %D = converted %.loc11_30.1, %.loc11_30.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc11_35.6: ref %D = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc11_34.2: init %D to %.loc11_35.6 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.7: init %D = converted %.loc11_34.1, %.loc11_34.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.8: init %Array.f45 to %a.var = array_init (%.loc11_35.3, %.loc11_35.5, %.loc11_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %Array.f45 = converted %.loc11_35.1, %.loc11_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_20: type = splice_block %Array [concrete = constants.%Array.f45] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%D, constants.%int_3.1ba) [concrete = constants.%Array.f45]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.f45 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.467 = ref_binding_pattern a [concrete = constants.%a.patt.349]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.467 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %Array.f45 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc22: %Array.c39 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.2(%self.param: ref %D) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.2(%self.param: ref %D) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.2(%self.param: ref %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11_3.3(%self.param: ref %Array.f45) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc11_3.1: ref %D = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11_3.1: <bound method> = bound_method %.loc11_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc11_3.1(%.loc11_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_3.2: ref %D = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11_3.2: <bound method> = bound_method %.loc11_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc11_3.2(%.loc11_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_3.3: ref %D = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11_3.3: <bound method> = bound_method %.loc11_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc11_3.3(%.loc11_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11_3.3(%self.param: ref %Array.f45) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11_3.3(%self.param: ref %Array.f45) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.fe9) {
// CHECK:STDOUT:   %N.patt.loc7_7.2 => constants.%N.patt.aa5
// CHECK:STDOUT:   %N.loc7_7.1 => constants.%N.fe9
// CHECK:STDOUT:   %Array.loc7_46.1 => constants.%Array.f60
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d24
// CHECK:STDOUT:   %a.param_patt.loc7_34.2 => constants.%a.param_patt.9fa
// CHECK:STDOUT:   %a.patt.loc7_34.2 => constants.%a.patt.439
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.patt.loc7_7.2 => constants.%N.patt.aa5
// CHECK:STDOUT:   %N.loc7_7.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %Array.loc7_46.1 => constants.%Array.c39
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a7
// CHECK:STDOUT:   %a.param_patt.loc7_34.2 => constants.%a.param_patt.80a
// CHECK:STDOUT:   %a.patt.loc7_34.2 => constants.%a.patt.dac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.1a4
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8
// CHECK:STDOUT:   %bound_method.loc7_66.3 => constants.%bound_method.d29
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_66.2 => constants.%int_3.af5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b2d: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.5d3: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.500: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bfc: %Int.as.ImplicitAs.impl.Convert.type.500 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.803: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ce, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.de1: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.76f: %Int.as.ImplicitAs.impl.Convert.type.de1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.803) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.797: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.917: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.797, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.5d3, %Int.as.ImplicitAs.impl.Convert.76f [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.76f, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.210: <bound method> = bound_method %N.5d3, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.210(%N.5d3) [symbolic]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.0a6: type = class_type @Array, @Array(%C, %Int.as.ImplicitAs.impl.Convert.call) [symbolic]
// CHECK:STDOUT:   %pattern_type.fc5: type = pattern_type %Array.0a6 [symbolic]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.fc5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.542: %pattern_type.fc5 = wrapper_binding_pattern a, %a.param_patt [symbolic]
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.97c: <witness> = require_complete_type %Array.0a6 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.5d3, %Int.as.Copy.impl.Op.617 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.0b0: <bound method> = bound_method %N.5d3, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.c39: type = class_type @Array, @Array(%C, %int_3) [concrete]
// CHECK:STDOUT:   %pattern_type.1a7: type = pattern_type %Array.c39 [concrete]
// CHECK:STDOUT:   %a.patt.496: %pattern_type.1a7 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt.496 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.5c1: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.c39 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2d8: %pattern_type.1a7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0b3: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt.2d8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5b7: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.500) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.bfc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ce = impl_witness_table (%Core.import_ref.5b7), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a7 = ref_param_pattern [concrete = constants.%self.param_patt.2d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0b3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.c39 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.c39 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %N.patt.loc6_7.1: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:     %a.param_patt.loc6_22.1: @F.%pattern_type (%pattern_type.fc5) = value_param_pattern [symbolic = %a.param_patt.loc6_22.2 (constants.%a.param_patt)]
// CHECK:STDOUT:     %a.patt.loc6_22.1: @F.%pattern_type (%pattern_type.fc5) = wrapper_binding_pattern a, %a.param_patt.loc6_22.1 [symbolic = %a.patt.loc6_22.2 (constants.%a.patt.542)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32.loc6_40 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_40: Core.Form = init_form %i32.loc6_40 [concrete = constants.%.55c]
// CHECK:STDOUT:     %.loc6_9: type = splice_block %i32.loc6_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32.loc6_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:     %a.param: @F.%Array.loc6_34.1 (%Array.0a6) = value_param call_param0
// CHECK:STDOUT:     %.loc6_34: type = splice_block %Array.loc6_34.2 [symbolic = %Array.loc6_34.1 (constants.%Array.0a6)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_33: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:       %impl.elem0.loc6_33: %.917 = impl_witness_access constants.%ImplicitAs.impl_witness.803, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.76f]
// CHECK:STDOUT:       %bound_method.loc6_33.2: <bound method> = bound_method %N.ref.loc6_33, %impl.elem0.loc6_33 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn.loc6_33: <specific function> = specific_function %impl.elem0.loc6_33, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc6_33.3: <bound method> = bound_method %N.ref.loc6_33, %specific_fn.loc6_33 [symbolic = %bound_method.loc6_33.1 (constants.%bound_method.210)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc6_33.2: init Core.IntLiteral = call %bound_method.loc6_33.3(%N.ref.loc6_33) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc6_33.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc6_33.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc6_33.2: Core.IntLiteral = converted %N.ref.loc6_33, %.loc6_33.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %Array.loc6_34.2: type = class_type @Array, @Array(constants.%C, constants.%Int.as.ImplicitAs.impl.Convert.call) [symbolic = %Array.loc6_34.1 (constants.%Array.0a6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%Array.loc6_34.1 (%Array.0a6) = wrapper_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32 [concrete = constants.%.55c]
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
// CHECK:STDOUT: generic fn @F(%N.loc6_7.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc6_7.2: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:   %N.loc6_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.ImplicitAs.impl.Convert.76f [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc6_33.1: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc6_33.1 (constants.%bound_method.210)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6_33.1: init Core.IntLiteral = call %bound_method.loc6_33.1(%N.loc6_7.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %Array.loc6_34.1: type = class_type @Array, @Array(constants.%C, %Int.as.ImplicitAs.impl.Convert.call.loc6_33.1) [symbolic = %Array.loc6_34.1 (constants.%Array.0a6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc6_34.1 [symbolic = %pattern_type (constants.%pattern_type.fc5)]
// CHECK:STDOUT:   %a.param_patt.loc6_22.2: @F.%pattern_type (%pattern_type.fc5) = value_param_pattern [symbolic = %a.param_patt.loc6_22.2 (constants.%a.param_patt)]
// CHECK:STDOUT:   %a.patt.loc6_22.2: @F.%pattern_type (%pattern_type.fc5) = wrapper_binding_pattern a, %a.param_patt.loc6_22.2 [symbolic = %a.patt.loc6_22.2 (constants.%a.patt.542)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc6_34.1 [symbolic = %require_complete (constants.%require_complete.97c)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.617 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc6_53.3: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_53.3 (constants.%bound_method.0b0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%Array.loc6_34.1 (%Array.0a6)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_53: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:     %impl.elem0.loc6_53: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:     %bound_method.loc6_53.1: <bound method> = bound_method %N.ref.loc6_53, %impl.elem0.loc6_53 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn.loc6_53: <specific function> = specific_function %impl.elem0.loc6_53, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_53.2: <bound method> = bound_method %N.ref.loc6_53, %specific_fn.loc6_53 [symbolic = %bound_method.loc6_53.3 (constants.%bound_method.0b0)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_53.2(%N.ref.loc6_53) [symbolic = %N.loc6_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Array.c39 = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple.5c1]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %Array.c39 to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %Array.c39 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %Array [concrete = constants.%Array.c39] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%C, constants.%int_3) [concrete = constants.%Array.c39]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.c39 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1a7 = ref_binding_pattern a [concrete = constants.%a.patt.496]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1a7 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %Array.c39 = name_ref a, %a
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%a.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_3.1: ref %C = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_3.2: ref %C = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_3.3: ref %C = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3: <bound method> = bound_method %.loc9_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3(%.loc9_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %Array.c39) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %Array.c39) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.5d3) {
// CHECK:STDOUT:   %N.patt.loc6_7.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%N.5d3
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc6_33.1 => constants.%bound_method.210
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6_33.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %Array.loc6_34.1 => constants.%Array.0a6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fc5
// CHECK:STDOUT:   %a.param_patt.loc6_22.2 => constants.%a.param_patt
// CHECK:STDOUT:   %a.patt.loc6_22.2 => constants.%a.patt.542
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_array_length_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b2d: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.5d3: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.500: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bfc: %Int.as.ImplicitAs.impl.Convert.type.500 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.803: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ce, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.de1: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.76f: %Int.as.ImplicitAs.impl.Convert.type.de1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.803) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.797: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.917: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.797, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.5d3, %Int.as.ImplicitAs.impl.Convert.76f [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.76f, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.5d3, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.5d3) [symbolic]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.0a6: type = class_type @Array, @Array(%C, %Int.as.ImplicitAs.impl.Convert.call) [symbolic]
// CHECK:STDOUT:   %pattern_type.fc5: type = pattern_type %Array.0a6 [symbolic]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.fc5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.fc5 = wrapper_binding_pattern a, %a.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.97c: <witness> = require_complete_type %Array.0a6 [symbolic]
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
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5b7: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.500) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.bfc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ce = impl_witness_table (%Core.import_ref.5b7), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
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
// CHECK:STDOUT:     %N.patt.loc5_7.1: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_7.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:     %a.param_patt.loc5_22.1: @F.%pattern_type (%pattern_type.fc5) = value_param_pattern [symbolic = %a.param_patt.loc5_22.2 (constants.%a.param_patt)]
// CHECK:STDOUT:     %a.patt.loc5_22.1: @F.%pattern_type (%pattern_type.fc5) = wrapper_binding_pattern a, %a.param_patt.loc5_22.1 [symbolic = %a.patt.loc5_22.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_9: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc5_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc5_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:     %a.param: @F.%Array.loc5_34.1 (%Array.0a6) = value_param call_param0
// CHECK:STDOUT:     %.loc5_34: type = splice_block %Array.loc5_34.2 [symbolic = %Array.loc5_34.1 (constants.%Array.0a6)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc5_7.2 [symbolic = %N.loc5_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:       %impl.elem0: %.917 = impl_witness_access constants.%ImplicitAs.impl_witness.803, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.76f]
// CHECK:STDOUT:       %bound_method.loc5_33.2: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc5_33.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc5_33.1 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc5_33.2: init Core.IntLiteral = call %bound_method.loc5_33.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc5_33.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc5_33.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc5_33.2: Core.IntLiteral = converted %N.ref, %.loc5_33.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %Array.loc5_34.2: type = class_type @Array, @Array(constants.%C, constants.%Int.as.ImplicitAs.impl.Convert.call) [symbolic = %Array.loc5_34.1 (constants.%Array.0a6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%Array.loc5_34.1 (%Array.0a6) = wrapper_binding a, %a.param
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
// CHECK:STDOUT: generic fn @F(%N.loc5_7.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc5_7.2: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_7.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:   %N.loc5_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc5_7.1 (constants.%N.5d3)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc5_7.1, constants.%Int.as.ImplicitAs.impl.Convert.76f [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc5_33.1: <bound method> = bound_method %N.loc5_7.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_33.1 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_33.1: init Core.IntLiteral = call %bound_method.loc5_33.1(%N.loc5_7.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_33.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %Array.loc5_34.1: type = class_type @Array, @Array(constants.%C, %Int.as.ImplicitAs.impl.Convert.call.loc5_33.1) [symbolic = %Array.loc5_34.1 (constants.%Array.0a6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc5_34.1 [symbolic = %pattern_type (constants.%pattern_type.fc5)]
// CHECK:STDOUT:   %a.param_patt.loc5_22.2: @F.%pattern_type (%pattern_type.fc5) = value_param_pattern [symbolic = %a.param_patt.loc5_22.2 (constants.%a.param_patt)]
// CHECK:STDOUT:   %a.patt.loc5_22.2: @F.%pattern_type (%pattern_type.fc5) = wrapper_binding_pattern a, %a.param_patt.loc5_22.2 [symbolic = %a.patt.loc5_22.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc5_34.1 [symbolic = %require_complete (constants.%require_complete.97c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%Array.loc5_34.1 (%Array.0a6)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc17_7: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_11: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_16: %tuple.type = tuple_literal (%.loc17_7, %.loc17_11, %.loc17_15) [concrete = constants.%tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.5d3) {
// CHECK:STDOUT:   %N.patt.loc5_7.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc5_7.1 => constants.%N.5d3
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc5_33.1 => constants.%bound_method
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_33.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %Array.loc5_34.1 => constants.%Array.0a6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fc5
// CHECK:STDOUT:   %a.param_patt.loc5_22.2 => constants.%a.param_patt
// CHECK:STDOUT:   %a.patt.loc5_22.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
