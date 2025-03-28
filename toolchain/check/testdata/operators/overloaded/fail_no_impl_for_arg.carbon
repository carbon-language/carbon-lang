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
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+10]]:14: error: cannot implicitly convert value of type `D` to `C` [ConversionFailure]
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+7]]:14: note: type `D` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE-13]]:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Op[self: C](other: C) -> C;
  // CHECK:STDERR:                  ^~~~~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestAssign(b: D) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+10]]:8: error: cannot implicitly convert value of type `D` to `C` [ConversionFailure]
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+7]]:8: note: type `D` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE-25]]:24: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Op[addr self: C*](other: C);
  // CHECK:STDERR:                        ^~~~~~~~
  // CHECK:STDERR:
  a += b;
}

// CHECK:STDOUT: --- fail_no_impl_for_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness.796: <witness> = impl_witness (@impl.b32.%Op.decl) [concrete]
// CHECK:STDOUT:   %Op.type.7a3: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.c84: %Op.type.7a3 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %C, (%impl_witness.796) [concrete]
// CHECK:STDOUT:   %AddAssign.type: type = facet_type <@AddAssign> [concrete]
// CHECK:STDOUT:   %Op.type.421: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %impl_witness.95d: <witness> = impl_witness (@impl.14c.%Op.decl) [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.0b8: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.d8e: %Op.type.0b8 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssign.facet: %AddAssign.type = facet_value %C, (%impl_witness.95d) [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c2e: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [concrete]
// CHECK:STDOUT:   %TestAssign: %TestAssign.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.22d: type = fn_type_with_self_type %Op.type.421, %AddAssign.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .AddAssign = %Core.AddAssign
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Add: type = import_ref Core//prelude/operators/arithmetic, Add, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Core.AddAssign: type = import_ref Core//prelude/operators/arithmetic, AddAssign, loaded [concrete = constants.%AddAssign.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .TestAssign = %TestAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.b32 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%Core.Add [concrete = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness (@impl.b32.%Op.decl) [concrete = constants.%impl_witness.796]
// CHECK:STDOUT:   impl_decl @impl.14c [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %AddAssign.ref: type = name_ref AddAssign, imports.%Core.AddAssign [concrete = constants.%AddAssign.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (@impl.14c.%Op.decl) [concrete = constants.%impl_witness.95d]
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc23_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc23_12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign.decl: %TestAssign.type = fn_decl @TestAssign [concrete = constants.%TestAssign] {
// CHECK:STDOUT:     %b.patt: %D = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %D = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b32: %C.ref as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.7a3 = fn_decl @Op.2 [concrete = constants.%Op.c84] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc17_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc17_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc17_25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.14c: %C.ref as %AddAssign.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.0b8 = fn_decl @Op.4 [concrete = constants.%Op.d8e] {
// CHECK:STDOUT:     %self.patt: %ptr.019 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.019 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc20_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc20_21: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc20_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc20_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %C](%other.param_patt: %C) -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr %self.param_patt: %ptr.019](%other.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%a.param_patt: %C, %b.param_patt: %D) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.c2e = impl_witness_access constants.%impl_witness.796, element0 [concrete = constants.%Op.c84]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %.loc23: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc34: %C = converted %b.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %Op.call: init %C = call %bound_method(%a.ref, <error>) to %.loc23
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%b.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %.loc38_3.1: %C = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C = var a
// CHECK:STDOUT:   %.loc38_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc38_15.2: init %C = class_init (), %a.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc38_3.2: init %C = converted %.loc38_15.1, %.loc38_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %a.var, %.loc38_3.2
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.22d = impl_witness_access constants.%impl_witness.95d, element0 [concrete = constants.%Op.d8e]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %a.ref
// CHECK:STDOUT:   %.loc49: %C = converted %b.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr, <error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
