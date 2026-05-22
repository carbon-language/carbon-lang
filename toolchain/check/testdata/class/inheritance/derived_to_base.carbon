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
// CHECK:STDOUT:   %struct_type.a.72d: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %struct_type.base.b.0f2: type = struct_type {.base: %A, .b: %i32} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %struct_type.base.c.657: type = struct_type {.base: %B, .c: %i32} [concrete]
// CHECK:STDOUT:   %ptr.4af: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.109: type = pattern_type %ptr.4af [concrete]
// CHECK:STDOUT:   %ptr.c86: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %.636: Core.Form = init_form %ptr.c86 [concrete]
// CHECK:STDOUT:   %pattern_type.155: type = pattern_type %ptr.c86 [concrete]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [concrete]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.3a7: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.e5e: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.4fd: %ptr.as.Copy.impl.Op.type.e5e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.d2e: %Copy.type = facet_value %ptr.c86, (%Copy.impl_witness.3a7) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.554: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.d2e) [concrete]
// CHECK:STDOUT:   %.28a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.554, %Copy.facet.d2e [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.894: <specific function> = specific_function %ptr.as.Copy.impl.Op.4fd, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT:   %ptr.f5d: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %.67c4: Core.Form = init_form %ptr.f5d [concrete]
// CHECK:STDOUT:   %pattern_type.929: type = pattern_type %ptr.f5d [concrete]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [concrete]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.167: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%A) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.74a: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%A) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.c78: %ptr.as.Copy.impl.Op.type.74a = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.e7c: %Copy.type = facet_value %ptr.f5d, (%Copy.impl_witness.167) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f33: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.e7c) [concrete]
// CHECK:STDOUT:   %.df5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f33, %Copy.facet.e7c [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.abf: <specific function> = specific_function %ptr.as.Copy.impl.Op.c78, @ptr.as.Copy.impl.Op(%A) [concrete]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [concrete]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9a0: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [concrete]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.780: type = pattern_type %A [concrete]
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
// CHECK:STDOUT:   %.686: type = partial_type %B [concrete]
// CHECK:STDOUT:   %.9e1: type = partial_type %A [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.e77: %.9e1 = struct_value (%int_1.d5e) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.d5e) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.a66: %.686 = struct_value (%A.val, %int_2.4ff) [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%A.val, %int_2.4ff) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.353: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B.val, %int_3.49a) [concrete]
// CHECK:STDOUT:   %.a5c: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %.b43: ref %B = class_element_access %.a5c, element0 [concrete]
// CHECK:STDOUT:   %.ad1: ref %A = class_element_access %.b43, element0 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.8: type = fn_type @Destroy.Op.loc32_66.8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.8: %Destroy.Op.type.af7ec0.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.a5c, %Destroy.Op.1dc86d.8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %ConvertCToB.decl: %ConvertCToB.type = fn_decl @ConvertCToB [concrete = constants.%ConvertCToB] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.109 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.109 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.155 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.155 = return_slot_pattern %return.param_patt, %ptr.loc19_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc19_27: type = ptr_type %B.ref [concrete = constants.%ptr.c86]
// CHECK:STDOUT:     %.loc19_27: Core.Form = init_form %ptr.loc19_27 [concrete = constants.%.636]
// CHECK:STDOUT:     %p.param: %ptr.4af = value_param call_param0
// CHECK:STDOUT:     %.loc19_20: type = splice_block %ptr.loc19_20 [concrete = constants.%ptr.4af] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc19_20: type = ptr_type %C.ref [concrete = constants.%ptr.4af]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.4af = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.c86 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.c86 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [concrete = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.155 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.155 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.929 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.929 = return_slot_pattern %return.param_patt, %ptr.loc20_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc20_27: type = ptr_type %A.ref [concrete = constants.%ptr.f5d]
// CHECK:STDOUT:     %.loc20_27: Core.Form = init_form %ptr.loc20_27 [concrete = constants.%.67c4]
// CHECK:STDOUT:     %p.param: %ptr.c86 = value_param call_param0
// CHECK:STDOUT:     %.loc20_20: type = splice_block %ptr.loc20_20 [concrete = constants.%ptr.c86] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc20_20: type = ptr_type %B.ref [concrete = constants.%ptr.c86]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.c86 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.f5d = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.f5d = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [concrete = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.109 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.109 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.929 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.929 = return_slot_pattern %return.param_patt, %ptr.loc21_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc21_27: type = ptr_type %A.ref [concrete = constants.%ptr.f5d]
// CHECK:STDOUT:     %.loc21_27: Core.Form = init_form %ptr.loc21_27 [concrete = constants.%.67c4]
// CHECK:STDOUT:     %p.param: %ptr.4af = value_param call_param0
// CHECK:STDOUT:     %.loc21_20: type = splice_block %ptr.loc21_20 [concrete = constants.%ptr.4af] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc21_20: type = ptr_type %C.ref [concrete = constants.%ptr.4af]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.4af = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.f5d = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.f5d = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [concrete = constants.%ConvertValue] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.9a0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.9a0 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [concrete = constants.%ConvertRef] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.109 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.109 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.929 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.929 = return_slot_pattern %return.param_patt, %ptr.loc27_26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc27: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc27_26: type = ptr_type %A.ref.loc27 [concrete = constants.%ptr.f5d]
// CHECK:STDOUT:     %.loc27_26: Core.Form = init_form %ptr.loc27_26 [concrete = constants.%.67c4]
// CHECK:STDOUT:     %c.param: %ptr.4af = value_param call_param0
// CHECK:STDOUT:     %.loc27_19: type = splice_block %ptr.loc27_19 [concrete = constants.%ptr.4af] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc27_19: type = ptr_type %C.ref [concrete = constants.%ptr.4af]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.4af = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %ptr.f5d = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.f5d = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [concrete = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param: %ptr.4af) -> out %return.param: %ptr.c86 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.4af = name_ref p, %p
// CHECK:STDOUT:   %.loc19_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc19_39.2: ref %B = class_element_access %.loc19_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.c86 = addr_of %.loc19_39.2
// CHECK:STDOUT:   %.loc19_39.3: %ptr.c86 = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.28a = impl_witness_access constants.%Copy.impl_witness.3a7, element0 [concrete = constants.%ptr.as.Copy.impl.Op.4fd]
// CHECK:STDOUT:   %bound_method.loc19_39.1: <bound method> = bound_method %.loc19_39.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.894]
// CHECK:STDOUT:   %bound_method.loc19_39.2: <bound method> = bound_method %.loc19_39.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.c86 = call %bound_method.loc19_39.2(%.loc19_39.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param: %ptr.c86) -> out %return.param: %ptr.f5d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.c86 = name_ref p, %p
// CHECK:STDOUT:   %.loc20_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc20_39.2: ref %A = class_element_access %.loc20_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.f5d = addr_of %.loc20_39.2
// CHECK:STDOUT:   %.loc20_39.3: %ptr.f5d = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.df5 = impl_witness_access constants.%Copy.impl_witness.167, element0 [concrete = constants.%ptr.as.Copy.impl.Op.c78]
// CHECK:STDOUT:   %bound_method.loc20_39.1: <bound method> = bound_method %.loc20_39.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.abf]
// CHECK:STDOUT:   %bound_method.loc20_39.2: <bound method> = bound_method %.loc20_39.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.f5d = call %bound_method.loc20_39.2(%.loc20_39.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param: %ptr.4af) -> out %return.param: %ptr.f5d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.4af = name_ref p, %p
// CHECK:STDOUT:   %.loc21_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc21_39.2: ref %B = class_element_access %.loc21_39.1, element0
// CHECK:STDOUT:   %.loc21_39.3: ref %A = class_element_access %.loc21_39.2, element0
// CHECK:STDOUT:   %addr: %ptr.f5d = addr_of %.loc21_39.3
// CHECK:STDOUT:   %.loc21_39.4: %ptr.f5d = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.df5 = impl_witness_access constants.%Copy.impl_witness.167, element0 [concrete = constants.%ptr.as.Copy.impl.Op.c78]
// CHECK:STDOUT:   %bound_method.loc21_39.1: <bound method> = bound_method %.loc21_39.4, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.abf]
// CHECK:STDOUT:   %bound_method.loc21_39.2: <bound method> = bound_method %.loc21_39.4, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.f5d = call %bound_method.loc21_39.2(%.loc21_39.4)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.780 = value_binding_pattern a [concrete]
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
// CHECK:STDOUT: fn @ConvertRef(%c.param: %ptr.4af) -> out %return.param: %ptr.f5d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ptr.4af = name_ref c, %c
// CHECK:STDOUT:   %.loc28_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc28_15.1: ref %B = class_element_access %.loc28_12, element0
// CHECK:STDOUT:   %.loc28_15.2: ref %A = class_element_access %.loc28_15.1, element0
// CHECK:STDOUT:   %.loc28_15.3: ref %A = converted %.loc28_12, %.loc28_15.2
// CHECK:STDOUT:   %addr: %ptr.f5d = addr_of %.loc28_15.3
// CHECK:STDOUT:   %impl.elem0: %.df5 = impl_witness_access constants.%Copy.impl_witness.167, element0 [concrete = constants.%ptr.as.Copy.impl.Op.c78]
// CHECK:STDOUT:   %bound_method.loc28_10.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.abf]
// CHECK:STDOUT:   %bound_method.loc28_10.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.f5d = call %bound_method.loc28_10.2(%addr)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.780 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc32_46.1: %struct_type.a.a6c = struct_literal (%int_1) [concrete = constants.%struct.48c]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc32_55.1: %struct_type.base.b.bf0 = struct_literal (%.loc32_46.1, %int_2) [concrete = constants.%struct.ff9]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc32_64.1: %struct_type.base.c.136 = struct_literal (%.loc32_55.1, %int_3) [concrete = constants.%struct.2aa]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc32_46: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc32_46.1: <bound method> = bound_method %int_1, %impl.elem0.loc32_46 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc32_46: <specific function> = specific_function %impl.elem0.loc32_46, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_46.2: <bound method> = bound_method %int_1, %specific_fn.loc32_46 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_46: init %i32 = call %bound_method.loc32_46.2(%int_1) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc32_46.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_46 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc32_64.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc32_64.3: ref %.686 = class_element_access %.loc32_64.2, element0
// CHECK:STDOUT:   %.loc32_55.2: ref %.9e1 = class_element_access %.loc32_64.3, element0
// CHECK:STDOUT:   %.loc32_46.3: ref %i32 = class_element_access %.loc32_55.2, element0
// CHECK:STDOUT:   %.loc32_46.4: init %i32 to %.loc32_46.3 = in_place_init %.loc32_46.2 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc32_46.5: init %.9e1 to %.loc32_55.2 = class_init (%.loc32_46.4) [concrete = constants.%struct.e77]
// CHECK:STDOUT:   %.loc32_55.3: init %.9e1 = converted %.loc32_46.1, %.loc32_46.5 [concrete = constants.%struct.e77]
// CHECK:STDOUT:   %.loc32_55.4: init %A = as_compatible %.loc32_55.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0.loc32_55: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc32_55.1: <bound method> = bound_method %int_2, %impl.elem0.loc32_55 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc32_55: <specific function> = specific_function %impl.elem0.loc32_55, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_55.2: <bound method> = bound_method %int_2, %specific_fn.loc32_55 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_55: init %i32 = call %bound_method.loc32_55.2(%int_2) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc32_55.5: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_55 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc32_55.6: ref %i32 = class_element_access %.loc32_64.3, element1
// CHECK:STDOUT:   %.loc32_55.7: init %i32 to %.loc32_55.6 = in_place_init %.loc32_55.5 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc32_55.8: init %.686 to %.loc32_64.3 = class_init (%.loc32_55.4, %.loc32_55.7) [concrete = constants.%struct.a66]
// CHECK:STDOUT:   %.loc32_64.4: init %.686 = converted %.loc32_55.1, %.loc32_55.8 [concrete = constants.%struct.a66]
// CHECK:STDOUT:   %.loc32_64.5: init %B = as_compatible %.loc32_64.4 [concrete = constants.%B.val]
// CHECK:STDOUT:   %impl.elem0.loc32_64: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc32_64.1: <bound method> = bound_method %int_3, %impl.elem0.loc32_64 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a]
// CHECK:STDOUT:   %specific_fn.loc32_64: <specific function> = specific_function %impl.elem0.loc32_64, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_64.2: <bound method> = bound_method %int_3, %specific_fn.loc32_64 [concrete = constants.%bound_method.353]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_64: init %i32 = call %bound_method.loc32_64.2(%int_3) [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc32_64.6: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_64 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc32_64.7: ref %i32 = class_element_access %.loc32_64.2, element1
// CHECK:STDOUT:   %.loc32_64.8: init %i32 to %.loc32_64.7 = in_place_init %.loc32_64.6 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc32_64.9: init %C to %.loc32_64.2 = class_init (%.loc32_64.5, %.loc32_64.8) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc32_66.1: init %C = converted %.loc32_64.1, %.loc32_64.9 [concrete = constants.%C.val]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc32_66.2: ref %C = temporary %.loc32_64.2, %.loc32_66.1 [concrete = constants.%.a5c]
// CHECK:STDOUT:   %.loc32_66.3: ref %B = class_element_access %.loc32_66.2, element0 [concrete = constants.%.b43]
// CHECK:STDOUT:   %.loc32_66.4: ref %A = class_element_access %.loc32_66.3, element0 [concrete = constants.%.ad1]
// CHECK:STDOUT:   %.loc32_66.5: ref %A = converted %.loc32_66.1, %.loc32_66.4 [concrete = constants.%.ad1]
// CHECK:STDOUT:   %.loc32_66.6: %A = acquire_value %.loc32_66.5
// CHECK:STDOUT:   %a: %A = value_binding a, %.loc32_66.6
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.a5c)
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
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.3(%self.param: ref %struct_type.a.72d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.4(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.5(%self.param: ref %struct_type.base.b.0f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.6(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_66.7(%self.param: ref %struct_type.base.c.657) {
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
// CHECK:STDOUT:   %const.20a: type = const_type %A [concrete]
// CHECK:STDOUT:   %ptr.e39: type = ptr_type %const.20a [concrete]
// CHECK:STDOUT:   %TakeConstAPtr.type: type = fn_type @TakeConstAPtr [concrete]
// CHECK:STDOUT:   %TakeConstAPtr: %TakeConstAPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ffa: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %const.e9e: type = const_type %B [concrete]
// CHECK:STDOUT:   %ptr.127: type = ptr_type %const.e9e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNonConstBPtr(%p.param: %ptr.ffa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstAPtr.ref: %TakeConstAPtr.type = name_ref TakeConstAPtr, file.%TakeConstAPtr.decl [concrete = constants.%TakeConstAPtr]
// CHECK:STDOUT:   %p.ref: %ptr.ffa = name_ref p, %p
// CHECK:STDOUT:   %.loc15_17.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc15_17.2: ref %A = class_element_access %.loc15_17.1, element0
// CHECK:STDOUT:   %addr: %ptr.e39 = addr_of %.loc15_17.2
// CHECK:STDOUT:   %.loc15_17.3: %ptr.e39 = as_compatible %addr
// CHECK:STDOUT:   %.loc15_17.4: %ptr.e39 = converted %p.ref, %.loc15_17.3
// CHECK:STDOUT:   %TakeConstAPtr.call: init %empty_tuple.type = call %TakeConstAPtr.ref(%.loc15_17.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassConstBPtr(%p.param: %ptr.127) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstAPtr.ref: %TakeConstAPtr.type = name_ref TakeConstAPtr, file.%TakeConstAPtr.decl [concrete = constants.%TakeConstAPtr]
// CHECK:STDOUT:   %p.ref: %ptr.127 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_17.1: ref %const.e9e = deref %p.ref
// CHECK:STDOUT:   %.loc21_17.2: ref %const.20a = class_element_access %.loc21_17.1, element0
// CHECK:STDOUT:   %addr: %ptr.e39 = addr_of %.loc21_17.2
// CHECK:STDOUT:   %.loc21_17.3: %ptr.e39 = converted %p.ref, %addr
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
// CHECK:STDOUT:   %const.89f: type = const_type %A [concrete]
// CHECK:STDOUT:   %TakeConstA.type: type = fn_type @TakeConstA [concrete]
// CHECK:STDOUT:   %TakeConstA: %TakeConstA.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.f50: type = const_type %B [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNonConstB(%p.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstA.ref: %TakeConstA.type = name_ref TakeConstA, file.%TakeConstA.decl [concrete = constants.%TakeConstA]
// CHECK:STDOUT:   %p.ref: %B = name_ref p, %p
// CHECK:STDOUT:   %.loc25: %const.89f = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstA.call: init %empty_tuple.type = call %TakeConstA.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassConstB(%p.param: %const.f50) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstA.ref: %TakeConstA.type = name_ref TakeConstA, file.%TakeConstA.decl [concrete = constants.%TakeConstA]
// CHECK:STDOUT:   %p.ref: %const.f50 = name_ref p, %p
// CHECK:STDOUT:   %.loc41: %const.89f = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstA.call: init %empty_tuple.type = call %TakeConstA.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
