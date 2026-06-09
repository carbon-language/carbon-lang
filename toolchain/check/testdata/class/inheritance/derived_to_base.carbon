// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/derived_to_base.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

base class A {
  var a: i32;
}

base class B {
  extend base: A;
  var b: i32;
}

class C {
  extend base: B;
  var c: i32;
}

//@dump-sem-ir-begin
fn ConvertCToB(p: C*) -> B* { return p; }
fn ConvertBToA(p: B*) -> A* { return p; }
fn ConvertCToA(p: C*) -> A* { return p; }

fn ConvertValue(c: C) {
  let unused a: A = c;
}

fn ConvertRef(c: C*) -> A* {
  return &(*c as A);
}

fn ConvertInit() {
  let unused a: A = {.base = {.base = {.a = 1}, .b = 2}, .c = 3} as C;
}
//@dump-sem-ir-end

// --- qualified.carbon

library "[[@TEST_NAME]]";

base class A {
}

class B {
  extend base: A;
}

fn TakeConstAPtr(p: const A*);

fn PassNonConstBPtr(p: B*) {
  //@dump-sem-ir-begin
  TakeConstAPtr(p);
  //@dump-sem-ir-end
}

fn PassConstBPtr(p: const B*) {
  //@dump-sem-ir-begin
  TakeConstAPtr(p);
  //@dump-sem-ir-end
}

// --- fail_todo_qualified_non_ptr.carbon

library "[[@TEST_NAME]]";

base class A {
}

class B {
  extend base: A;
}

fn TakeConstA(p: const A);

fn PassNonConstB(p: B) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE+10]]:14: error: cannot implicitly convert expression of type `B` to `const A` [ConversionFailure]
  // CHECK:STDERR:   TakeConstA(p);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE+7]]:14: note: type `B` does not implement interface `Core.ImplicitAs(const A)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   TakeConstA(p);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE-10]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeConstA(p: const A);
  // CHECK:STDERR:               ^~~~~~~~~~
  // CHECK:STDERR:
  TakeConstA(p);
  //@dump-sem-ir-end
}

fn PassConstB(p: const B) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE+10]]:14: error: cannot implicitly convert expression of type `const B` to `const A` [ConversionFailure]
  // CHECK:STDERR:   TakeConstA(p);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE+7]]:14: note: type `const B` does not implement interface `Core.ImplicitAs(const A)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   TakeConstA(p);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE-26]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeConstA(p: const A);
  // CHECK:STDERR:               ^~~~~~~~~~
  // CHECK:STDERR:
  TakeConstA(p);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %struct_type.a.a92: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %struct_type.base.b.6c4: type = struct_type {.base: %A, .b: %i32} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %struct_type.base.c.faf: type = struct_type {.base: %B, .c: %i32} [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %.d69: Core.Form = init_form %ptr.432 [concrete]
// CHECK:STDOUT:   %pattern_type.3c8: type = pattern_type %ptr.432 [concrete]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [concrete]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.09a: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d4b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.70e: %ptr.as.Copy.impl.Op.type.d4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.5e3: %Copy.type = facet_value %ptr.432, (%Copy.impl_witness.09a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.78a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.5e3) [concrete]
// CHECK:STDOUT:   %.fa1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.78a, %Copy.facet.5e3 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.bbe: <specific function> = specific_function %ptr.as.Copy.impl.Op.70e, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT:   %ptr.d43: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %.1a4: Core.Form = init_form %ptr.d43 [concrete]
// CHECK:STDOUT:   %pattern_type.8a5: type = pattern_type %ptr.d43 [concrete]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [concrete]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d19: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%A) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa3: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%A) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a8: %ptr.as.Copy.impl.Op.type.aa3 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.bd2: %Copy.type = facet_value %ptr.d43, (%Copy.impl_witness.d19) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.df0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.bd2) [concrete]
// CHECK:STDOUT:   %.e91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.df0, %Copy.facet.bd2 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.d1a: <specific function> = specific_function %ptr.as.Copy.impl.Op.5a8, @ptr.as.Copy.impl.Op(%A) [concrete]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [concrete]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [concrete]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ConvertRef.type: type = fn_type @ConvertRef [concrete]
// CHECK:STDOUT:   %ConvertRef: %ConvertRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertInit.type: type = fn_type @ConvertInit [concrete]
// CHECK:STDOUT:   %ConvertInit: %ConvertInit.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.48c: %struct_type.a.a6c = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.base.b.bf0: type = struct_type {.base: %struct_type.a.a6c, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.ff9: %struct_type.base.b.bf0 = struct_value (%struct.48c, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.base.c.136: type = struct_type {.base: %struct_type.base.b.bf0, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.2aa: %struct_type.base.c.136 = struct_value (%struct.ff9, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %.640: type = partial_type %B [concrete]
// CHECK:STDOUT:   %.3d6: type = partial_type %A [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.7e2: %.3d6 = struct_value (%int_1.0c6) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.0c6) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.a8a: %.640 = struct_value (%A.val, %int_2.295) [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%A.val, %int_2.295) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B.val, %int_3.410) [concrete]
// CHECK:STDOUT:   %.084: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %.107: ref %B = class_element_access %.084, element0 [concrete]
// CHECK:STDOUT:   %.5ce: ref %A = class_element_access %.107, element0 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.loc32_66.8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.084, %Destroy.Op.1a2547.8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %ConvertCToB.decl: %ConvertCToB.type = fn_decl @ConvertCToB [concrete = constants.%ConvertCToB] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.fcb = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.fcb = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.3c8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.3c8 = return_slot_pattern %return.param_patt, %ptr.loc19_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc19_27: type = ptr_type %B.ref [concrete = constants.%ptr.432]
// CHECK:STDOUT:     %.loc19_27: Core.Form = init_form %ptr.loc19_27 [concrete = constants.%.d69]
// CHECK:STDOUT:     %p.param: %ptr.6b6 = value_param call_param0
// CHECK:STDOUT:     %.loc19_20: type = splice_block %ptr.loc19_20 [concrete = constants.%ptr.6b6] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc19_20: type = ptr_type %C.ref [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.6b6 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.432 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.432 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [concrete = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.3c8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.3c8 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8a5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8a5 = return_slot_pattern %return.param_patt, %ptr.loc20_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc20_27: type = ptr_type %A.ref [concrete = constants.%ptr.d43]
// CHECK:STDOUT:     %.loc20_27: Core.Form = init_form %ptr.loc20_27 [concrete = constants.%.1a4]
// CHECK:STDOUT:     %p.param: %ptr.432 = value_param call_param0
// CHECK:STDOUT:     %.loc20_20: type = splice_block %ptr.loc20_20 [concrete = constants.%ptr.432] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc20_20: type = ptr_type %B.ref [concrete = constants.%ptr.432]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.432 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.d43 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.d43 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [concrete = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.fcb = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.fcb = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8a5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8a5 = return_slot_pattern %return.param_patt, %ptr.loc21_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc21_27: type = ptr_type %A.ref [concrete = constants.%ptr.d43]
// CHECK:STDOUT:     %.loc21_27: Core.Form = init_form %ptr.loc21_27 [concrete = constants.%.1a4]
// CHECK:STDOUT:     %p.param: %ptr.6b6 = value_param call_param0
// CHECK:STDOUT:     %.loc21_20: type = splice_block %ptr.loc21_20 [concrete = constants.%ptr.6b6] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc21_20: type = ptr_type %C.ref [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.6b6 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.d43 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.d43 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [concrete = constants.%ConvertValue] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [concrete = constants.%ConvertRef] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.fcb = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.fcb = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8a5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8a5 = return_slot_pattern %return.param_patt, %ptr.loc27_26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc27: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc27_26: type = ptr_type %A.ref.loc27 [concrete = constants.%ptr.d43]
// CHECK:STDOUT:     %.loc27_26: Core.Form = init_form %ptr.loc27_26 [concrete = constants.%.1a4]
// CHECK:STDOUT:     %c.param: %ptr.6b6 = value_param call_param0
// CHECK:STDOUT:     %.loc27_19: type = splice_block %ptr.loc27_19 [concrete = constants.%ptr.6b6] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc27_19: type = ptr_type %C.ref [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.6b6 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %ptr.d43 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.d43 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [concrete = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param: %ptr.6b6) -> out %return.param: %ptr.432 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.6b6 = name_ref p, %p
// CHECK:STDOUT:   %.loc19_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc19_39.2: ref %B = class_element_access %.loc19_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.432 = addr_of %.loc19_39.2
// CHECK:STDOUT:   %.loc19_39.3: %ptr.432 = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.fa1 = impl_witness_access constants.%Copy.impl_witness.09a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.70e]
// CHECK:STDOUT:   %bound_method.loc19_39.1: <bound method> = bound_method %.loc19_39.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.bbe]
// CHECK:STDOUT:   %bound_method.loc19_39.2: <bound method> = bound_method %.loc19_39.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.432 = call %bound_method.loc19_39.2(%.loc19_39.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param: %ptr.432) -> out %return.param: %ptr.d43 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.432 = name_ref p, %p
// CHECK:STDOUT:   %.loc20_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc20_39.2: ref %A = class_element_access %.loc20_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.d43 = addr_of %.loc20_39.2
// CHECK:STDOUT:   %.loc20_39.3: %ptr.d43 = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.e91 = impl_witness_access constants.%Copy.impl_witness.d19, element0 [concrete = constants.%ptr.as.Copy.impl.Op.5a8]
// CHECK:STDOUT:   %bound_method.loc20_39.1: <bound method> = bound_method %.loc20_39.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.d1a]
// CHECK:STDOUT:   %bound_method.loc20_39.2: <bound method> = bound_method %.loc20_39.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.d43 = call %bound_method.loc20_39.2(%.loc20_39.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param: %ptr.6b6) -> out %return.param: %ptr.d43 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.6b6 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc21_39.2: ref %B = class_element_access %.loc21_39.1, element0
// CHECK:STDOUT:   %.loc21_39.3: ref %A = class_element_access %.loc21_39.2, element0
// CHECK:STDOUT:   %addr: %ptr.d43 = addr_of %.loc21_39.3
// CHECK:STDOUT:   %.loc21_39.4: %ptr.d43 = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.e91 = impl_witness_access constants.%Copy.impl_witness.d19, element0 [concrete = constants.%ptr.as.Copy.impl.Op.5a8]
// CHECK:STDOUT:   %bound_method.loc21_39.1: <bound method> = bound_method %.loc21_39.4, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.d1a]
// CHECK:STDOUT:   %bound_method.loc21_39.2: <bound method> = bound_method %.loc21_39.4, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.d43 = call %bound_method.loc21_39.2(%.loc21_39.4)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc24_21.1: ref %B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc24_21.2: ref %A = class_element_access %.loc24_21.1, element0
// CHECK:STDOUT:   %.loc24_21.3: ref %A = converted %c.ref, %.loc24_21.2
// CHECK:STDOUT:   %.loc24_21.4: %A = acquire_value %.loc24_21.3
// CHECK:STDOUT:   %a: %A = value_binding a, %.loc24_21.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c.param: %ptr.6b6) -> out %return.param: %ptr.d43 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ptr.6b6 = name_ref c, %c
// CHECK:STDOUT:   %.loc28_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc28_15.1: ref %B = class_element_access %.loc28_12, element0
// CHECK:STDOUT:   %.loc28_15.2: ref %A = class_element_access %.loc28_15.1, element0
// CHECK:STDOUT:   %.loc28_15.3: ref %A = converted %.loc28_12, %.loc28_15.2
// CHECK:STDOUT:   %addr: %ptr.d43 = addr_of %.loc28_15.3
// CHECK:STDOUT:   %impl.elem0: %.e91 = impl_witness_access constants.%Copy.impl_witness.d19, element0 [concrete = constants.%ptr.as.Copy.impl.Op.5a8]
// CHECK:STDOUT:   %bound_method.loc28_10.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.d1a]
// CHECK:STDOUT:   %bound_method.loc28_10.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.d43 = call %bound_method.loc28_10.2(%addr)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc32_46.1: %struct_type.a.a6c = struct_literal (%int_1) [concrete = constants.%struct.48c]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc32_55.1: %struct_type.base.b.bf0 = struct_literal (%.loc32_46.1, %int_2) [concrete = constants.%struct.ff9]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc32_64.1: %struct_type.base.c.136 = struct_literal (%.loc32_55.1, %int_3) [concrete = constants.%struct.2aa]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc32_46: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc32_46.1: <bound method> = bound_method %int_1, %impl.elem0.loc32_46 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc32_46: <specific function> = specific_function %impl.elem0.loc32_46, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_46.2: <bound method> = bound_method %int_1, %specific_fn.loc32_46 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_46: init %i32 = call %bound_method.loc32_46.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc32_46.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_46 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc32_64.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc32_64.3: ref %.640 = class_element_access %.loc32_64.2, element0
// CHECK:STDOUT:   %.loc32_55.2: ref %.3d6 = class_element_access %.loc32_64.3, element0
// CHECK:STDOUT:   %.loc32_46.3: ref %i32 = class_element_access %.loc32_55.2, element0
// CHECK:STDOUT:   %.loc32_46.4: init %i32 to %.loc32_46.3 = in_place_init %.loc32_46.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc32_46.5: init %.3d6 to %.loc32_55.2 = class_init (%.loc32_46.4) [concrete = constants.%struct.7e2]
// CHECK:STDOUT:   %.loc32_55.3: init %.3d6 = converted %.loc32_46.1, %.loc32_46.5 [concrete = constants.%struct.7e2]
// CHECK:STDOUT:   %.loc32_55.4: init %A = as_compatible %.loc32_55.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0.loc32_55: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc32_55.1: <bound method> = bound_method %int_2, %impl.elem0.loc32_55 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc32_55: <specific function> = specific_function %impl.elem0.loc32_55, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_55.2: <bound method> = bound_method %int_2, %specific_fn.loc32_55 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_55: init %i32 = call %bound_method.loc32_55.2(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc32_55.5: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_55 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc32_55.6: ref %i32 = class_element_access %.loc32_64.3, element1
// CHECK:STDOUT:   %.loc32_55.7: init %i32 to %.loc32_55.6 = in_place_init %.loc32_55.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc32_55.8: init %.640 to %.loc32_64.3 = class_init (%.loc32_55.4, %.loc32_55.7) [concrete = constants.%struct.a8a]
// CHECK:STDOUT:   %.loc32_64.4: init %.640 = converted %.loc32_55.1, %.loc32_55.8 [concrete = constants.%struct.a8a]
// CHECK:STDOUT:   %.loc32_64.5: init %B = as_compatible %.loc32_64.4 [concrete = constants.%B.val]
// CHECK:STDOUT:   %impl.elem0.loc32_64: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc32_64.1: <bound method> = bound_method %int_3, %impl.elem0.loc32_64 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc32_64: <specific function> = specific_function %impl.elem0.loc32_64, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_64.2: <bound method> = bound_method %int_3, %specific_fn.loc32_64 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_64: init %i32 = call %bound_method.loc32_64.2(%int_3) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc32_64.6: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_64 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc32_64.7: ref %i32 = class_element_access %.loc32_64.2, element1
// CHECK:STDOUT:   %.loc32_64.8: init %i32 to %.loc32_64.7 = in_place_init %.loc32_64.6 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc32_64.9: init %C to %.loc32_64.2 = class_init (%.loc32_64.5, %.loc32_64.8) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc32_66.1: init %C = converted %.loc32_64.1, %.loc32_64.9 [concrete = constants.%C.val]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc32_66.2: ref %C = temporary %.loc32_64.2, %.loc32_66.1 [concrete = constants.%.084]
// CHECK:STDOUT:   %.loc32_66.3: ref %B = class_element_access %.loc32_66.2, element0 [concrete = constants.%.107]
// CHECK:STDOUT:   %.loc32_66.4: ref %A = class_element_access %.loc32_66.3, element0 [concrete = constants.%.5ce]
// CHECK:STDOUT:   %.loc32_66.5: ref %A = converted %.loc32_66.1, %.loc32_66.4 [concrete = constants.%.5ce]
// CHECK:STDOUT:   %.loc32_66.6: %A = acquire_value %.loc32_66.5
// CHECK:STDOUT:   %a: %A = value_binding a, %.loc32_66.6
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.084)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.3(%self.param: ref %struct_type.a.a92) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.4(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.5(%self.param: ref %struct_type.base.b.6c4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.6(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.7(%self.param: ref %struct_type.base.c.faf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.8(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const.b97: type = const_type %A [concrete]
// CHECK:STDOUT:   %ptr.903: type = ptr_type %const.b97 [concrete]
// CHECK:STDOUT:   %TakeConstAPtr.type: type = fn_type @TakeConstAPtr [concrete]
// CHECK:STDOUT:   %TakeConstAPtr: %TakeConstAPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %const.825: type = const_type %B [concrete]
// CHECK:STDOUT:   %ptr.c09: type = ptr_type %const.825 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNonConstBPtr(%p.param: %ptr.432) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstAPtr.ref: %TakeConstAPtr.type = name_ref TakeConstAPtr, file.%TakeConstAPtr.decl [concrete = constants.%TakeConstAPtr]
// CHECK:STDOUT:   %p.ref: %ptr.432 = name_ref p, %p
// CHECK:STDOUT:   %.loc15_17.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc15_17.2: ref %A = class_element_access %.loc15_17.1, element0
// CHECK:STDOUT:   %addr: %ptr.903 = addr_of %.loc15_17.2
// CHECK:STDOUT:   %.loc15_17.3: %ptr.903 = as_compatible %addr
// CHECK:STDOUT:   %.loc15_17.4: %ptr.903 = converted %p.ref, %.loc15_17.3
// CHECK:STDOUT:   %TakeConstAPtr.call: init %empty_tuple.type = call %TakeConstAPtr.ref(%.loc15_17.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassConstBPtr(%p.param: %ptr.c09) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstAPtr.ref: %TakeConstAPtr.type = name_ref TakeConstAPtr, file.%TakeConstAPtr.decl [concrete = constants.%TakeConstAPtr]
// CHECK:STDOUT:   %p.ref: %ptr.c09 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_17.1: ref %const.825 = deref %p.ref
// CHECK:STDOUT:   %.loc21_17.2: ref %const.b97 = class_element_access %.loc21_17.1, element0
// CHECK:STDOUT:   %addr: %ptr.903 = addr_of %.loc21_17.2
// CHECK:STDOUT:   %.loc21_17.3: %ptr.903 = converted %p.ref, %addr
// CHECK:STDOUT:   %TakeConstAPtr.call: init %empty_tuple.type = call %TakeConstAPtr.ref(%.loc21_17.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_qualified_non_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const.b97: type = const_type %A [concrete]
// CHECK:STDOUT:   %TakeConstA.type: type = fn_type @TakeConstA [concrete]
// CHECK:STDOUT:   %TakeConstA: %TakeConstA.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.825: type = const_type %B [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNonConstB(%p.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstA.ref: %TakeConstA.type = name_ref TakeConstA, file.%TakeConstA.decl [concrete = constants.%TakeConstA]
// CHECK:STDOUT:   %p.ref: %B = name_ref p, %p
// CHECK:STDOUT:   %.loc25: %const.b97 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstA.call: init %empty_tuple.type = call %TakeConstA.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassConstB(%p.param: %const.825) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstA.ref: %TakeConstA.type = name_ref TakeConstA, file.%TakeConstA.decl [concrete = constants.%TakeConstA]
// CHECK:STDOUT:   %p.ref: %const.825 = name_ref p, %p
// CHECK:STDOUT:   %.loc41: %const.b97 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstA.call: init %empty_tuple.type = call %TakeConstA.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
