// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon

package User;

class C {};
class D {};

impl C as Core.Add {
  fn Op[self: C](other: C) -> C;
}
impl C as Core.AddAssign {
  fn Op[addr self: C*](other: C);
}

fn Test(a: C, b: D) -> C {
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+7]]:10: ERROR: Cannot implicitly convert from `D` to `C`.
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE-10]]:3: Initializing parameter 1 of function declared here.
  // CHECK:STDERR:   fn Op[self: C](other: C) -> C;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestAssign(b: D) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+6]]:3: ERROR: Cannot implicitly convert from `D` to `C`.
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE-19]]:3: Initializing parameter 1 of function declared here.
  // CHECK:STDERR:   fn Op[addr self: C*](other: C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  a += b;
}

// CHECK:STDOUT: --- fail_no_impl_for_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.2: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %.5: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %Self.2: %.5 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %C [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.4: %Op.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.2, %Op.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.9 [template]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [template]
// CHECK:STDOUT:   %TestAssign: %TestAssign.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %.5, %Op.type.4 [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Add = %import_ref.1
// CHECK:STDOUT:     .AddAssign = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/arithmetic, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: %.10 = import_ref Core//prelude/operators/arithmetic, inst+25, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.4: %Op.type.2 = import_ref Core//prelude/operators/arithmetic, inst+19, loaded [template = constants.%Op.2]
// CHECK:STDOUT:   %import_ref.5: type = import_ref Core//prelude/operators/arithmetic, inst+27, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/arithmetic, inst+29, unloaded
// CHECK:STDOUT:   %import_ref.7: %.12 = import_ref Core//prelude/operators/arithmetic, inst+50, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.8: %Op.type.4 = import_ref Core//prelude/operators/arithmetic, inst+44, loaded [template = constants.%Op.4]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/arithmetic, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/arithmetic, inst+44, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .TestAssign = %TestAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %AddAssign.ref: type = name_ref AddAssign, imports.%import_ref.5 [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %C.ref.loc23_12: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc23_9.1: %C = param a
// CHECK:STDOUT:     @Test.%a: %C = bind_name a, %a.loc23_9.1
// CHECK:STDOUT:     %D.ref.loc23: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc23_15.1: %D = param b
// CHECK:STDOUT:     @Test.%b: %D = bind_name b, %b.loc23_15.1
// CHECK:STDOUT:     %C.ref.loc23_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @Test.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign.decl: %TestAssign.type = fn_decl @TestAssign [template = constants.%TestAssign] {
// CHECK:STDOUT:     %D.ref.loc34: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc34_15.1: %D = param b
// CHECK:STDOUT:     @TestAssign.%b: %D = bind_name b, %b.loc34_15.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Op = imports.%import_ref.3
// CHECK:STDOUT:   witness = (imports.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   .Op = imports.%import_ref.7
// CHECK:STDOUT:   witness = (imports.%import_ref.8)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C as %.2 {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %C.ref.loc17_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc17_9.1: %C = param self
// CHECK:STDOUT:     %self.loc17_9.2: %C = bind_name self, %self.loc17_9.1
// CHECK:STDOUT:     %C.ref.loc17_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc17_18.1: %C = param other
// CHECK:STDOUT:     %other.loc17_18.2: %C = bind_name other, %other.loc17_18.1
// CHECK:STDOUT:     %C.ref.loc17_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %.5 {
// CHECK:STDOUT:   %Op.decl: %Op.type.3 = fn_decl @Op.3 [template = constants.%Op.3] {
// CHECK:STDOUT:     %C.ref.loc20_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc20_21: type = ptr_type %C [template = constants.%.6]
// CHECK:STDOUT:     %self.loc20_14.1: %.6 = param self
// CHECK:STDOUT:     %self.loc20_14.3: %.6 = bind_name self, %self.loc20_14.1
// CHECK:STDOUT:     %.loc20_9: %.6 = addr_pattern %self.loc20_14.3
// CHECK:STDOUT:     %C.ref.loc20_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc20_24.1: %C = param other
// CHECK:STDOUT:     %other.loc20_24.2: %C = bind_name other, %other.loc20_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@impl.1.%self.loc17_9.2: %C](@impl.1.%other.loc17_18.2: %C) -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self.1: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.2.%Self (%Self.1)](%other: @Op.2.%Self (%Self.1)) -> @Op.2.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr @impl.2.%self.loc20_14.3: %.6](@impl.2.%other.loc20_24.2: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.4(constants.%Self.2: %.5) {
// CHECK:STDOUT:   %Self: %.5 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.4.%Self (%Self.2) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: @Op.4.%.2 (%.7)](%other: @Op.4.%Self (%Self.2));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%a: %C, %b: %D) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %.1: %Op.type.2 = interface_witness_access @impl.1.%.1, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc31_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc31_12.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.call: init %C = call %.loc31_12.1(<invalid>) [template = <error>]
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%b: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref %C = var a
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %.loc35_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc35_15.2: init %C = class_init (), %a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc35_16: init %C = converted %.loc35_15.1, %.loc35_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign %a.var, %.loc35_16
// CHECK:STDOUT:   %a.ref: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %.1: %Op.type.4 = interface_witness_access @impl.2.%.1, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %.loc42_5: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc42_3: %.6 = addr_of %a.ref
// CHECK:STDOUT:   %Op.call: init %.3 = call %.loc42_5(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
