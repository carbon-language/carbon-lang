// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/derived_to_base.carbon

// --- derived_to_base.carbon

library "[[@TEST_NAME]]";

base class A {
  var a: ();
}

base class B {
  extend base: A;
  var b: ();
}

class C {
  extend base: B;
  var c: ();
}

//@dump-sem-ir-begin
fn ConvertCToB(p: C*) -> B* { return p; }
fn ConvertBToA(p: B*) -> A* { return p; }
fn ConvertCToA(p: C*) -> A* { return p; }

fn ConvertValue(c: C) {
  let a: A = c;
}

fn ConvertRef(c: C*) -> A* {
  return &(*c as A);
}

fn ConvertInit() {
  let a: A = {.base = {.base = {.a = ()}, .b = ()}, .c = ()} as C;
}
//@dump-sem-ir-end

// --- fail_bad_derived_to_base.carbon

library "[[@TEST_NAME]]";

base class A1 {
  var a: ();
}

base class A2 {
  var a: ();
}

class B2 {
  extend base: A2;
  var b: ();
}

// CHECK:STDERR: fail_bad_derived_to_base.carbon:[[@LINE+7]]:38: error: cannot implicitly convert expression of type `B2*` to `A1*` [ConversionFailure]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR: fail_bad_derived_to_base.carbon:[[@LINE+4]]:38: note: type `B2*` does not implement interface `Core.ImplicitAs(A1*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR: fail_bad_derived_to_base.carbon:[[@LINE+7]]:47: error: cannot implicitly convert expression of type `Incomplete*` to `A2*` [ConversionFailure]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR: fail_bad_derived_to_base.carbon:[[@LINE+4]]:47: note: type `Incomplete*` does not implement interface `Core.ImplicitAs(A2*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR:
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }

// CHECK:STDOUT: --- derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [concrete]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [concrete]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [concrete]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [concrete]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ConvertRef.type: type = fn_type @ConvertRef [concrete]
// CHECK:STDOUT:   %ConvertRef: %ConvertRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertInit.type: type = fn_type @ConvertInit [concrete]
// CHECK:STDOUT:   %ConvertInit: %ConvertInit.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.b.8ca: type = struct_type {.base: %struct_type.a, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct_type.base.c.7dc: type = struct_type {.base: %struct_type.base.b.8ca, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%A.val, %empty_tuple) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B.val, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %ConvertCToB.decl: %ConvertCToB.type = fn_decl @ConvertCToB [concrete = constants.%ConvertCToB] {
// CHECK:STDOUT:     %p.patt: %pattern_type.44a = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.44a = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.960 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.960 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc19_27: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc19_20: type = splice_block %ptr.loc19_20 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc19_20: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.e79 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.e79 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [concrete = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.patt: %pattern_type.960 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.960 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5f8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5f8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc20_27: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %p.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %.loc20_20: type = splice_block %ptr.loc20_20 [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc20_20: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e79 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [concrete = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.patt: %pattern_type.44a = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.44a = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5f8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5f8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc21_27: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc21_20: type = splice_block %ptr.loc21_20 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc21_20: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [concrete = constants.%ConvertValue] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [concrete = constants.%ConvertRef] {
// CHECK:STDOUT:     %c.patt: %pattern_type.44a = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.44a = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5f8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5f8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc27: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc27_26: type = ptr_type %A.ref.loc27 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %c.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc27: type = splice_block %ptr.loc27_19 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc27_19: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.019 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [concrete = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param: %ptr.019) -> %ptr.e79 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %.loc19_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc19_39.2: ref %B = class_element_access %.loc19_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc19_39.2
// CHECK:STDOUT:   %.loc19_39.3: %ptr.e79 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc19_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param: %ptr.e79) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e79 = name_ref p, %p
// CHECK:STDOUT:   %.loc20_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc20_39.2: ref %A = class_element_access %.loc20_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc20_39.2
// CHECK:STDOUT:   %.loc20_39.3: %ptr.6db = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc20_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param: %ptr.019) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc21_39.2: ref %B = class_element_access %.loc21_39.1, element0
// CHECK:STDOUT:   %.loc21_39.3: ref %A = class_element_access %.loc21_39.2, element0
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc21_39.3
// CHECK:STDOUT:   %.loc21_39.4: %ptr.6db = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc21_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc24_14.1: ref %B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc24_14.2: ref %A = class_element_access %.loc24_14.1, element0
// CHECK:STDOUT:   %.loc24_14.3: ref %A = converted %c.ref, %.loc24_14.2
// CHECK:STDOUT:   %a: ref %A = bind_name a, %.loc24_14.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c.param: %ptr.019) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ptr.019 = name_ref c, %c
// CHECK:STDOUT:   %.loc28_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc28_15.1: ref %B = class_element_access %.loc28_12, element0
// CHECK:STDOUT:   %.loc28_15.2: ref %A = class_element_access %.loc28_15.1, element0
// CHECK:STDOUT:   %.loc28_15.3: ref %A = converted %.loc28_12, %.loc28_15.2
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc28_15.3
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc32_39.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc32_40.1: %struct_type.a = struct_literal (%.loc32_39.1)
// CHECK:STDOUT:   %.loc32_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc32_50.1: %struct_type.base.b.8ca = struct_literal (%.loc32_40.1, %.loc32_49.1)
// CHECK:STDOUT:   %.loc32_59.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc32_60.1: %struct_type.base.c.7dc = struct_literal (%.loc32_50.1, %.loc32_59.1)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc32_60.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc32_60.3: ref %B = class_element_access %.loc32_60.2, element0
// CHECK:STDOUT:   %.loc32_50.2: ref %A = class_element_access %.loc32_60.3, element0
// CHECK:STDOUT:   %.loc32_40.2: ref %empty_tuple.type = class_element_access %.loc32_50.2, element0
// CHECK:STDOUT:   %.loc32_39.2: init %empty_tuple.type = tuple_init () to %.loc32_40.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc32_40.3: init %empty_tuple.type = converted %.loc32_39.1, %.loc32_39.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc32_40.4: init %A = class_init (%.loc32_40.3), %.loc32_50.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc32_50.3: init %A = converted %.loc32_40.1, %.loc32_40.4 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc32_50.4: ref %empty_tuple.type = class_element_access %.loc32_60.3, element1
// CHECK:STDOUT:   %.loc32_49.2: init %empty_tuple.type = tuple_init () to %.loc32_50.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc32_50.5: init %empty_tuple.type = converted %.loc32_49.1, %.loc32_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc32_50.6: init %B = class_init (%.loc32_50.3, %.loc32_50.5), %.loc32_60.3 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc32_60.4: init %B = converted %.loc32_50.1, %.loc32_50.6 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc32_60.5: ref %empty_tuple.type = class_element_access %.loc32_60.2, element1
// CHECK:STDOUT:   %.loc32_59.2: init %empty_tuple.type = tuple_init () to %.loc32_60.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc32_60.6: init %empty_tuple.type = converted %.loc32_59.1, %.loc32_59.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc32_60.7: init %C = class_init (%.loc32_60.4, %.loc32_60.6), %.loc32_60.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc32_60.8: ref %C = temporary %.loc32_60.2, %.loc32_60.7
// CHECK:STDOUT:   %.loc32_62.1: ref %C = converted %.loc32_60.1, %.loc32_60.8
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc32_62.2: ref %B = class_element_access %.loc32_62.1, element0
// CHECK:STDOUT:   %.loc32_62.3: ref %A = class_element_access %.loc32_62.2, element0
// CHECK:STDOUT:   %.loc32_62.4: ref %A = converted %.loc32_62.1, %.loc32_62.3
// CHECK:STDOUT:   %a: ref %A = bind_name a, %.loc32_62.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
