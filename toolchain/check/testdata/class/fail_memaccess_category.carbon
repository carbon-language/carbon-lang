// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_memaccess_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_memaccess_category.carbon

class A {
  fn F[addr self: A*]();
}

class B {
  var a: A;
}

fn F(s: {.a: A}, b: B) {
  // `s` has only a value representation, so this must be invalid.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   s.a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~
  // CHECK:STDERR:
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-23]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~
  // CHECK:STDERR:
  b.a.F();
}

// CHECK:STDOUT: --- fail_memaccess_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %F.type.649: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.485: %F.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %struct_type.a.72c: type = struct_type {.a: %A} [concrete]
// CHECK:STDOUT:   %complete_type.2b9: <witness> = complete_type_witness %struct_type.a.72c [concrete]
// CHECK:STDOUT:   %pattern_type.c19: type = pattern_type %struct_type.a.72c [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.b6e: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(%A) [concrete]
// CHECK:STDOUT:   %Op.type.b96: type = fn_type @Op.2, @impl(%A) [concrete]
// CHECK:STDOUT:   %Op.885: %Op.type.b96 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %A, (%Destroy.impl_witness.b6e) [concrete]
// CHECK:STDOUT:   %.af1: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.885, @Op.2(%A) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/destroy, loc15_23, loaded [symbolic = @impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %s.patt: %pattern_type.c19 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.c19 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.049 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.049 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %s.param: %struct_type.a.72c = value_param call_param0
// CHECK:STDOUT:     %.loc22: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.72c] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %struct_type.a: type = struct_type {.a: %A} [concrete = constants.%struct_type.a.72c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %struct_type.a.72c = bind_name s, %s.param
// CHECK:STDOUT:     %b.param: %B = value_param call_param1
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %b: %B = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.649 = fn_decl @F.1 [concrete = constants.%F.485] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc15_20: type = splice_block %ptr [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc19: %B.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %A} [concrete = constants.%struct_type.a.72c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.2b9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .a = %.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(%self.param: %ptr.6db);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%s.param: %struct_type.a.72c, %b.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %struct_type.a.72c = name_ref s, %s
// CHECK:STDOUT:   %.loc31_4.1: %A = struct_access %s.ref, element0
// CHECK:STDOUT:   %F.ref.loc31: %F.type.649 = name_ref F, @A.%F.decl [concrete = constants.%F.485]
// CHECK:STDOUT:   %F.bound.loc31: <bound method> = bound_method %.loc31_4.1, %F.ref.loc31
// CHECK:STDOUT:   %.loc31_4.2: ref %A = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc31: %.af1 = impl_witness_access constants.%Destroy.impl_witness.b6e, element0 [concrete = constants.%Op.885]
// CHECK:STDOUT:   %bound_method.loc31: <bound method> = bound_method %.loc31_4.2, %impl.elem0.loc31
// CHECK:STDOUT:   %addr.loc31: %ptr.6db = addr_of %.loc31_4.2
// CHECK:STDOUT:   %F.call.loc31: init %empty_tuple.type = call %F.bound.loc31(%addr.loc31)
// CHECK:STDOUT:   %b.ref: %B = name_ref b, %b
// CHECK:STDOUT:   %a.ref: %B.elem = name_ref a, @B.%.loc19 [concrete = @B.%.loc19]
// CHECK:STDOUT:   %.loc42_4.1: ref %A = class_element_access %b.ref, element0
// CHECK:STDOUT:   %.loc42_4.2: %A = bind_value %.loc42_4.1
// CHECK:STDOUT:   %F.ref.loc42: %F.type.649 = name_ref F, @A.%F.decl [concrete = constants.%F.485]
// CHECK:STDOUT:   %F.bound.loc42: <bound method> = bound_method %.loc42_4.2, %F.ref.loc42
// CHECK:STDOUT:   %.loc42_4.3: ref %A = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc42: %.af1 = impl_witness_access constants.%Destroy.impl_witness.b6e, element0 [concrete = constants.%Op.885]
// CHECK:STDOUT:   %bound_method.loc42: <bound method> = bound_method %.loc42_4.3, %impl.elem0.loc42
// CHECK:STDOUT:   %addr.loc42: %ptr.6db = addr_of %.loc42_4.3
// CHECK:STDOUT:   %F.call.loc42: init %empty_tuple.type = call %F.bound.loc42(%addr.loc42)
// CHECK:STDOUT:   %specific_fn.loc42: <specific function> = specific_function %impl.elem0.loc42, @Op.2(constants.%A) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc42_4.3, %specific_fn.loc42
// CHECK:STDOUT:   %.loc42_4.4: %A = bind_value %.loc42_4.3
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc42_4.4)
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Op.2(constants.%A) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.loc31_4.2, %specific_fn.loc31
// CHECK:STDOUT:   %.loc31_4.3: %A = bind_value %.loc31_4.2
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc31_4.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
