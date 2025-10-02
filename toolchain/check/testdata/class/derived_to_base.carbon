// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/derived_to_base.carbon

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
  let a: A = c;
}

fn ConvertRef(c: C*) -> A* {
  return &(*c as A);
}

fn ConvertInit() {
  let a: A = {.base = {.base = {.a = 1}, .b = 2}, .c = 3} as C;
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
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE+7]]:14: note: type `B` does not implement interface `Core.ImplicitAs(const A)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_todo_qualified_non_ptr.carbon:[[@LINE+7]]:14: note: type `const B` does not implement interface `Core.ImplicitAs(const A)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [concrete]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e33: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.662: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.057: %ptr.as.Copy.impl.Op.type.662 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.654: %Copy.type = facet_value %ptr.e79, (%Copy.impl_witness.e33) [concrete]
// CHECK:STDOUT:   %.56e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.654 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.244: <specific function> = specific_function %ptr.as.Copy.impl.Op.057, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [concrete]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6d3: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%A) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.af5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%A) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.2f8: %ptr.as.Copy.impl.Op.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.2fb: %Copy.type = facet_value %ptr.6db, (%Copy.impl_witness.6d3) [concrete]
// CHECK:STDOUT:   %.7dd: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.2fb [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.833: <specific function> = specific_function %ptr.as.Copy.impl.Op.2f8, @ptr.as.Copy.impl.Op(%A) [concrete]
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
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.base.b.bf0: type = struct_type {.base: %struct_type.a.a6c, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.base.c.136: type = struct_type {.base: %struct_type.base.b.bf0, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.8bd: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%A.val, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B.val, %int_3.822) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.075: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.b80: %DestroyT.as_type.as.Destroy.impl.Op.type.075 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0: %.56e = impl_witness_access constants.%Copy.impl_witness.e33, element0 [concrete = constants.%ptr.as.Copy.impl.Op.057]
// CHECK:STDOUT:   %bound_method.loc19_39.1: <bound method> = bound_method %.loc19_39.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.244]
// CHECK:STDOUT:   %bound_method.loc19_39.2: <bound method> = bound_method %.loc19_39.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e79 = call %bound_method.loc19_39.2(%.loc19_39.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param: %ptr.e79) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e79 = name_ref p, %p
// CHECK:STDOUT:   %.loc20_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc20_39.2: ref %A = class_element_access %.loc20_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc20_39.2
// CHECK:STDOUT:   %.loc20_39.3: %ptr.6db = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.7dd = impl_witness_access constants.%Copy.impl_witness.6d3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.2f8]
// CHECK:STDOUT:   %bound_method.loc20_39.1: <bound method> = bound_method %.loc20_39.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.833]
// CHECK:STDOUT:   %bound_method.loc20_39.2: <bound method> = bound_method %.loc20_39.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.6db = call %bound_method.loc20_39.2(%.loc20_39.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
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
// CHECK:STDOUT:   %impl.elem0: %.7dd = impl_witness_access constants.%Copy.impl_witness.6d3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.2f8]
// CHECK:STDOUT:   %bound_method.loc21_39.1: <bound method> = bound_method %.loc21_39.4, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.833]
// CHECK:STDOUT:   %bound_method.loc21_39.2: <bound method> = bound_method %.loc21_39.4, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.6db = call %bound_method.loc21_39.2(%.loc21_39.4)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
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
// CHECK:STDOUT:   %.loc24_14.4: %A = bind_value %.loc24_14.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc24_14.4
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
// CHECK:STDOUT:   %impl.elem0: %.7dd = impl_witness_access constants.%Copy.impl_witness.6d3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.2f8]
// CHECK:STDOUT:   %bound_method.loc28_10.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%A) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.833]
// CHECK:STDOUT:   %bound_method.loc28_10.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.6db = call %bound_method.loc28_10.2(%addr)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc32_39.1: %struct_type.a.a6c = struct_literal (%int_1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc32_48.1: %struct_type.base.b.bf0 = struct_literal (%.loc32_39.1, %int_2)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc32_57.1: %struct_type.base.c.136 = struct_literal (%.loc32_48.1, %int_3)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc32_39: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc32_39.1: <bound method> = bound_method %int_1, %impl.elem0.loc32_39 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc32_39: <specific function> = specific_function %impl.elem0.loc32_39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_39.2: <bound method> = bound_method %int_1, %specific_fn.loc32_39 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_39: init %i32 = call %bound_method.loc32_39.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32_39.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_39 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32_57.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc32_57.3: ref %B = class_element_access %.loc32_57.2, element0
// CHECK:STDOUT:   %.loc32_48.2: ref %A = class_element_access %.loc32_57.3, element0
// CHECK:STDOUT:   %.loc32_39.3: ref %i32 = class_element_access %.loc32_48.2, element0
// CHECK:STDOUT:   %.loc32_39.4: init %i32 = initialize_from %.loc32_39.2 to %.loc32_39.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32_39.5: init %A = class_init (%.loc32_39.4), %.loc32_48.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc32_48.3: init %A = converted %.loc32_39.1, %.loc32_39.5 [concrete = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0.loc32_48: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc32_48.1: <bound method> = bound_method %int_2, %impl.elem0.loc32_48 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc32_48: <specific function> = specific_function %impl.elem0.loc32_48, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_48.2: <bound method> = bound_method %int_2, %specific_fn.loc32_48 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_48: init %i32 = call %bound_method.loc32_48.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc32_48.4: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_48 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc32_48.5: ref %i32 = class_element_access %.loc32_57.3, element1
// CHECK:STDOUT:   %.loc32_48.6: init %i32 = initialize_from %.loc32_48.4 to %.loc32_48.5 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc32_48.7: init %B = class_init (%.loc32_48.3, %.loc32_48.6), %.loc32_57.3 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc32_57.4: init %B = converted %.loc32_48.1, %.loc32_48.7 [concrete = constants.%B.val]
// CHECK:STDOUT:   %impl.elem0.loc32_57: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc32_57.1: <bound method> = bound_method %int_3, %impl.elem0.loc32_57 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc32_57: <specific function> = specific_function %impl.elem0.loc32_57, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_57.2: <bound method> = bound_method %int_3, %specific_fn.loc32_57 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_57: init %i32 = call %bound_method.loc32_57.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc32_57.5: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32_57 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc32_57.6: ref %i32 = class_element_access %.loc32_57.2, element1
// CHECK:STDOUT:   %.loc32_57.7: init %i32 = initialize_from %.loc32_57.5 to %.loc32_57.6 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc32_57.8: init %C = class_init (%.loc32_57.4, %.loc32_57.7), %.loc32_57.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc32_57.9: ref %C = temporary %.loc32_57.2, %.loc32_57.8
// CHECK:STDOUT:   %.loc32_59.1: ref %C = converted %.loc32_57.1, %.loc32_57.9
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc32_59.2: ref %B = class_element_access %.loc32_59.1, element0
// CHECK:STDOUT:   %.loc32_59.3: ref %A = class_element_access %.loc32_59.2, element0
// CHECK:STDOUT:   %.loc32_59.4: ref %A = converted %.loc32_59.1, %.loc32_59.3
// CHECK:STDOUT:   %.loc32_59.5: %A = bind_value %.loc32_59.4
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc32_59.5
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc32_57.10: %type_where = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc32_57.9, constants.%DestroyT.as_type.as.Destroy.impl.Op.b80
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc32_57.3: <bound method> = bound_method %.loc32_57.9, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc32_57.9
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc32_57.3(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const.d98: type = const_type %A [concrete]
// CHECK:STDOUT:   %ptr.985: type = ptr_type %const.d98 [concrete]
// CHECK:STDOUT:   %TakeConstAPtr.type: type = fn_type @TakeConstAPtr [concrete]
// CHECK:STDOUT:   %TakeConstAPtr: %TakeConstAPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %const.44f: type = const_type %B [concrete]
// CHECK:STDOUT:   %ptr.eb4: type = ptr_type %const.44f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNonConstBPtr(%p.param: %ptr.e79) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstAPtr.ref: %TakeConstAPtr.type = name_ref TakeConstAPtr, file.%TakeConstAPtr.decl [concrete = constants.%TakeConstAPtr]
// CHECK:STDOUT:   %p.ref: %ptr.e79 = name_ref p, %p
// CHECK:STDOUT:   %.loc15_17.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc15_17.2: ref %A = class_element_access %.loc15_17.1, element0
// CHECK:STDOUT:   %addr: %ptr.985 = addr_of %.loc15_17.2
// CHECK:STDOUT:   %.loc15_17.3: %ptr.985 = as_compatible %addr
// CHECK:STDOUT:   %.loc15_17.4: %ptr.985 = converted %p.ref, %.loc15_17.3
// CHECK:STDOUT:   %TakeConstAPtr.call: init %empty_tuple.type = call %TakeConstAPtr.ref(%.loc15_17.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassConstBPtr(%p.param: %ptr.eb4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstAPtr.ref: %TakeConstAPtr.type = name_ref TakeConstAPtr, file.%TakeConstAPtr.decl [concrete = constants.%TakeConstAPtr]
// CHECK:STDOUT:   %p.ref: %ptr.eb4 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_17.1: ref %const.44f = deref %p.ref
// CHECK:STDOUT:   %.loc21_17.2: ref %const.d98 = class_element_access %.loc21_17.1, element0
// CHECK:STDOUT:   %addr: %ptr.985 = addr_of %.loc21_17.2
// CHECK:STDOUT:   %.loc21_17.3: %ptr.985 = converted %p.ref, %addr
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
// CHECK:STDOUT:   %const.d98: type = const_type %A [concrete]
// CHECK:STDOUT:   %TakeConstA.type: type = fn_type @TakeConstA [concrete]
// CHECK:STDOUT:   %TakeConstA: %TakeConstA.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.44f: type = const_type %B [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNonConstB(%p.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstA.ref: %TakeConstA.type = name_ref TakeConstA, file.%TakeConstA.decl [concrete = constants.%TakeConstA]
// CHECK:STDOUT:   %p.ref: %B = name_ref p, %p
// CHECK:STDOUT:   %.loc25: %const.d98 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstA.call: init %empty_tuple.type = call %TakeConstA.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassConstB(%p.param: %const.44f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeConstA.ref: %TakeConstA.type = name_ref TakeConstA, file.%TakeConstA.decl [concrete = constants.%TakeConstA]
// CHECK:STDOUT:   %p.ref: %const.44f = name_ref p, %p
// CHECK:STDOUT:   %.loc41: %const.d98 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeConstA.call: init %empty_tuple.type = call %TakeConstA.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
