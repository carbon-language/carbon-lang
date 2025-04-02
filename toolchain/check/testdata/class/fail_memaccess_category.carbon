// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:4: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   s.a.F();
  // CHECK:STDERR:    ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~
  // CHECK:STDERR:
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:4: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:    ^
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
// CHECK:STDOUT:   %F.type.649: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.485: %F.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %struct_type.a.72c: type = struct_type {.a: %A} [concrete]
// CHECK:STDOUT:   %complete_type.2b9: <witness> = complete_type_witness %struct_type.a.72c [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %s.patt: %struct_type.a.72c = binding_pattern s
// CHECK:STDOUT:     %s.param_patt: %struct_type.a.72c = value_param_pattern %s.patt, call_param0
// CHECK:STDOUT:     %b.patt: %B = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %B = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %s.param: %struct_type.a.72c = value_param call_param0
// CHECK:STDOUT:     %.loc19: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.72c] {
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
// CHECK:STDOUT:     %self.patt: %ptr.6db = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.6db = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc12_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc12_20: type = splice_block %ptr [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.loc16_8: %B.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc16_3: %B.elem = var_pattern %.loc16_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %B.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.72c [concrete = constants.%complete_type.2b9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .a = %.loc16_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %ptr.6db]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%s.param_patt: %struct_type.a.72c, %b.param_patt: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %struct_type.a.72c = name_ref s, %s
// CHECK:STDOUT:   %.loc28_4.1: %A = struct_access %s.ref, element0
// CHECK:STDOUT:   %F.ref.loc28: %F.type.649 = name_ref F, @A.%F.decl [concrete = constants.%F.485]
// CHECK:STDOUT:   %F.bound.loc28: <bound method> = bound_method %.loc28_4.1, %F.ref.loc28
// CHECK:STDOUT:   %.loc28_4.2: ref %A = temporary_storage
// CHECK:STDOUT:   %addr.loc28: %ptr.6db = addr_of %.loc28_4.2
// CHECK:STDOUT:   %F.call.loc28: init %empty_tuple.type = call %F.bound.loc28(%addr.loc28)
// CHECK:STDOUT:   %b.ref: %B = name_ref b, %b
// CHECK:STDOUT:   %a.ref: %B.elem = name_ref a, @B.%.loc16_8 [concrete = @B.%.loc16_8]
// CHECK:STDOUT:   %.loc39_4.1: ref %A = class_element_access %b.ref, element0
// CHECK:STDOUT:   %.loc39_4.2: %A = bind_value %.loc39_4.1
// CHECK:STDOUT:   %F.ref.loc39: %F.type.649 = name_ref F, @A.%F.decl [concrete = constants.%F.485]
// CHECK:STDOUT:   %F.bound.loc39: <bound method> = bound_method %.loc39_4.2, %F.ref.loc39
// CHECK:STDOUT:   %.loc39_4.3: ref %A = temporary_storage
// CHECK:STDOUT:   %addr.loc39: %ptr.6db = addr_of %.loc39_4.3
// CHECK:STDOUT:   %F.call.loc39: init %empty_tuple.type = call %F.bound.loc39(%addr.loc39)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
