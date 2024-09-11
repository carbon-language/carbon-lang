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
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+10]]:10: ERROR: Cannot implicitly convert from `D` to `C`.
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+7]]:10: Type `D` does not implement interface `ImplicitAs`.
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE-13]]:3: Initializing parameter 1 of function declared here.
  // CHECK:STDERR:   fn Op[self: C](other: C) -> C;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestAssign(b: D) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+9]]:3: ERROR: Cannot implicitly convert from `D` to `C`.
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+6]]:3: Type `D` does not implement interface `ImplicitAs`.
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE-25]]:3: Initializing parameter 1 of function declared here.
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
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.12: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.3: @ImplicitAs.%.1 (%.12) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.4: %.12 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %.12, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %.15: type = interface_type @ImplicitAs, @ImplicitAs(%C) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%C) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.16: type = assoc_entity_type %.15, %Convert.type.2 [template]
// CHECK:STDOUT:   %.17: %.16 = assoc_entity element0, imports.%import_ref.14 [template]
// CHECK:STDOUT:   %.18: %.13 = assoc_entity element0, imports.%import_ref.15 [symbolic]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [template]
// CHECK:STDOUT:   %TestAssign: %TestAssign.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.19: type = assoc_entity_type %.5, %Op.type.4 [template]
// CHECK:STDOUT:   %.20: %.19 = assoc_entity element0, imports.%import_ref.16 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Add = %import_ref.1
// CHECK:STDOUT:     .AddAssign = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.10
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
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
// CHECK:STDOUT:   %import_ref.7: %.19 = import_ref Core//prelude/operators/arithmetic, inst+50, loaded [template = constants.%.20]
// CHECK:STDOUT:   %import_ref.8: %Op.type.4 = import_ref Core//prelude/operators/arithmetic, inst+44, loaded [template = constants.%Op.4]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/arithmetic, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.10: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+37, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.12: @ImplicitAs.%.2 (%.13) = import_ref Core//prelude/operators/as, inst+59, loaded [symbolic = @ImplicitAs.%.3 (constants.%.18)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/arithmetic, inst+44, unloaded
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
// CHECK:STDOUT:     %a.loc23_9.1: %C = param a, runtime_param0
// CHECK:STDOUT:     @Test.%a: %C = bind_name a, %a.loc23_9.1
// CHECK:STDOUT:     %D.ref.loc23: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc23_15.1: %D = param b, runtime_param1
// CHECK:STDOUT:     @Test.%b: %D = bind_name b, %b.loc23_15.1
// CHECK:STDOUT:     %C.ref.loc23_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @Test.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign.decl: %TestAssign.type = fn_decl @TestAssign [template = constants.%TestAssign] {
// CHECK:STDOUT:     %D.ref.loc37: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc37_15.1: %D = param b, runtime_param0
// CHECK:STDOUT:     @TestAssign.%b: %D = bind_name b, %b.loc37_15.1
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
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %Self: %.12 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.12), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.13) = assoc_entity element0, imports.%import_ref.14 [symbolic = %.3 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.11
// CHECK:STDOUT:     .Convert = imports.%import_ref.12
// CHECK:STDOUT:     witness = (imports.%import_ref.13)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C as %.2 {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %C.ref.loc17_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc17_9.1: %C = param self, runtime_param0
// CHECK:STDOUT:     %self.loc17_9.2: %C = bind_name self, %self.loc17_9.1
// CHECK:STDOUT:     %C.ref.loc17_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc17_18.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc17_18.2: %C = bind_name other, %other.loc17_18.1
// CHECK:STDOUT:     %C.ref.loc17_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <witness> = interface_witness (%Op.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %.5 {
// CHECK:STDOUT:   %Op.decl: %Op.type.3 = fn_decl @Op.3 [template = constants.%Op.3] {
// CHECK:STDOUT:     %C.ref.loc20_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc20_21: type = ptr_type %C [template = constants.%.6]
// CHECK:STDOUT:     %self.loc20_14.1: %.6 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc20_14.3: %.6 = bind_name self, %self.loc20_14.1
// CHECK:STDOUT:     %.loc20_9: %.6 = addr_pattern %self.loc20_14.3
// CHECK:STDOUT:     %C.ref.loc20_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc20_24.1: %C = param other, runtime_param1
// CHECK:STDOUT:     %other.loc20_24.2: %C = bind_name other, %other.loc20_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = interface_witness (%Op.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc19
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
// CHECK:STDOUT:   %Op.ref: %.10 = name_ref Op, imports.%import_ref.3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc34_12.1: %Op.type.2 = interface_witness_access @impl.1.%.loc16, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc34_12.2: <bound method> = bound_method %a.ref, %.loc34_12.1
// CHECK:STDOUT:   %.loc34_12.3: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc34_12.4: type = interface_type @ImplicitAs, @ImplicitAs(constants.%C) [template = constants.%.15]
// CHECK:STDOUT:   %.loc34_12.5: %.16 = specific_constant imports.%import_ref.12, @ImplicitAs(constants.%C) [template = constants.%.17]
// CHECK:STDOUT:   %Convert.ref: %.16 = name_ref Convert, %.loc34_12.5 [template = constants.%.17]
// CHECK:STDOUT:   %.loc34_12.6: %C = converted %b.ref, <error> [template = <error>]
// CHECK:STDOUT:   %Op.call: init %C = call %.loc34_12.2(<invalid>) [template = <error>]
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.3: @ImplicitAs.%.1 (%.12)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %Self: %.12 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.4)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%b: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref %C = var a
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %.loc38_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc38_15.2: init %C = class_init (), %a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc38_16: init %C = converted %.loc38_15.1, %.loc38_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign %a.var, %.loc38_16
// CHECK:STDOUT:   %a.ref: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %Op.ref: %.19 = name_ref Op, imports.%import_ref.7 [template = constants.%.20]
// CHECK:STDOUT:   %.loc48_5.1: %Op.type.4 = interface_witness_access @impl.2.%.loc19, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %.loc48_5.2: <bound method> = bound_method %a.ref, %.loc48_5.1
// CHECK:STDOUT:   %.loc48_3: %.6 = addr_of %a.ref
// CHECK:STDOUT:   %.loc48_5.3: type = interface_type @ImplicitAs, @ImplicitAs(constants.%C) [template = constants.%.15]
// CHECK:STDOUT:   %.loc48_5.4: %.16 = specific_constant imports.%import_ref.12, @ImplicitAs(constants.%C) [template = constants.%.17]
// CHECK:STDOUT:   %Convert.ref: %.16 = name_ref Convert, %.loc48_5.4 [template = constants.%.17]
// CHECK:STDOUT:   %.loc48_5.5: %C = converted %b.ref, <error> [template = <error>]
// CHECK:STDOUT:   %Op.call: init %.3 = call %.loc48_5.2(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.4(constants.%Self.2) {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.4(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT:   %.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%C) {
// CHECK:STDOUT:   %Dest => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.15
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.16
// CHECK:STDOUT:   %.3 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
