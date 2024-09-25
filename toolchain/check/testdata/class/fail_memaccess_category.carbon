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
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:8: error: `addr self` method cannot be invoked on a value
  // CHECK:STDERR:   s.a.F();
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-12]]:13: note: initializing `addr self` parameter of method declared here
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+6]]:8: error: `addr self` method cannot be invoked on a value
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-23]]:13: note: initializing `addr self` parameter of method declared here
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:             ^~~~
  b.a.F();
}

// CHECK:STDOUT: --- fail_memaccess_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = ptr_type %A [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: %A} [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %.7 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.9: type = struct_type {.a: %.5} [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.7 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc19: type = struct_type {.a: %A} [template = constants.%.7]
// CHECK:STDOUT:     %s.loc19_6.1: %.7 = param s, runtime_param0
// CHECK:STDOUT:     @F.2.%s: %.7 = bind_name s, %s.loc19_6.1
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %b.loc19_18.1: %B = param b, runtime_param1
// CHECK:STDOUT:     @F.2.%b: %B = bind_name b, %b.loc19_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc12_20: type = ptr_type %A [template = constants.%.1]
// CHECK:STDOUT:     %self.loc12_13.1: %.1 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc12_13.3: %.1 = bind_name self, %self.loc12_13.1
// CHECK:STDOUT:     %.loc12_8: %.1 = addr_pattern %self.loc12_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %.6 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.7 [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr @A.%self.loc12_13.3: %.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%s: %.7, %b: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %.7 = name_ref s, %s
// CHECK:STDOUT:   %.loc28_4: %A = struct_access %s.ref, element0
// CHECK:STDOUT:   %F.ref.loc28: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc28_6: <bound method> = bound_method %.loc28_4, %F.ref.loc28
// CHECK:STDOUT:   %F.call.loc28: init %.2 = call %.loc28_6(<invalid>) [template = <error>]
// CHECK:STDOUT:   %b.ref: %B = name_ref b, %b
// CHECK:STDOUT:   %a.ref: %.6 = name_ref a, @B.%.loc16 [template = @B.%.loc16]
// CHECK:STDOUT:   %.loc38_4.1: ref %A = class_element_access %b.ref, element0
// CHECK:STDOUT:   %.loc38_4.2: %A = bind_value %.loc38_4.1
// CHECK:STDOUT:   %F.ref.loc38: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc38_6: <bound method> = bound_method %.loc38_4.2, %F.ref.loc38
// CHECK:STDOUT:   %F.call.loc38: init %.2 = call %.loc38_6(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
