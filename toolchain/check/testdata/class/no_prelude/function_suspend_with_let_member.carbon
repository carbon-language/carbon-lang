
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/function_suspend_with_let_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/function_suspend_with_let_member.carbon

// --- let_after.carbon

library "[[@TEST_NAME]]";

class C {
  fn F() -> () { return x; }
  let x:! () = ();
}

// --- let_before.carbon

library "[[@TEST_NAME]]";

class C {
  let x:! () = ();
  fn F() -> () { return x; }
}

// --- multiple_lets.carbon

library "[[@TEST_NAME]]";

class C {
  let a:! () = ();
  fn F(b:! ((),)) {
    let c:! ((), ()) = ((), ());

    var a1: () = a;
    var b1: ((),) = b;
    var c1: ((), ()) = c;
    var d1: ((), (), ()) = d;
  }
  let d:! ((), (), ()) = ((), (), ());
}

// --- fail_invalid_let_after.carbon

class C {
  fn F() -> () { return x; }
  // CHECK:STDERR: fail_invalid_let_after.carbon:[[@LINE+3]]:13: error: expected `=`; `let` declaration must have an initializer
  // CHECK:STDERR:   let x:! ();
  // CHECK:STDERR:             ^
  let x:! ();
}

// CHECK:STDOUT: --- let_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0 [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc5_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc6_12.2: type = converted %.loc6_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_18: %.1 = converted %.loc6_17, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0, %.loc6_18 [symbolic = constants.%x]
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.1 = name_ref x, @C.%x [symbolic = constants.%x]
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- let_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc5_12.2: type = converted %.loc5_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_18: %.1 = converted %.loc5_17, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0, %.loc5_18 [symbolic = constants.%x]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc6_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_14.2: type = converted %.loc6_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@C.%x: %.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0 [symbolic = %x (constants.%x)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %.1 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: %.1 = name_ref x, @C.%x [symbolic = %x (constants.%x)]
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%x) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_lets.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple.1: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type (%.1) [template]
// CHECK:STDOUT:   %b: %.2 = bind_symbolic_name b 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %tuple.2: %.3 = tuple_value (%tuple.1, %tuple.1, %tuple.1) [template]
// CHECK:STDOUT:   %d: %.3 = bind_symbolic_name d 1 [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.1, %.1) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %tuple.3: %.7 = tuple_value (%tuple.1, %tuple.1) [template]
// CHECK:STDOUT:   %c: %.7 = bind_symbolic_name c 3 [symbolic]
// CHECK:STDOUT:   %tuple.4: %.2 = tuple_value (%tuple.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc5_12.2: type = converted %.loc5_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple.loc5: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_18: %.1 = converted %.loc5_17, %tuple.loc5 [template = constants.%tuple.1]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0, %.loc5_18 [symbolic = constants.%a]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc6_14: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_16.1: %.2 = tuple_literal (%.loc6_14)
// CHECK:STDOUT:     %.loc6_16.2: type = converted %.loc6_14, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_16.3: type = converted %.loc6_16.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %b.loc6_8.1: %.2 = param b, runtime_param<invalid>
// CHECK:STDOUT:     %b.loc6_8.2: %.2 = bind_symbolic_name b 1, %b.loc6_8.1 [symbolic = @F.%b.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_13: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_21: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_22.1: %.3 = tuple_literal (%.loc14_13, %.loc14_17, %.loc14_21)
// CHECK:STDOUT:   %.loc14_22.2: type = converted %.loc14_13, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_22.3: type = converted %.loc14_17, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_22.4: type = converted %.loc14_21, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_22.5: type = converted %.loc14_22.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_28: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_32: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_36: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_37.1: %.3 = tuple_literal (%.loc14_28, %.loc14_32, %.loc14_36)
// CHECK:STDOUT:   %tuple.loc14_28: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc14_37.2: %.1 = converted %.loc14_28, %tuple.loc14_28 [template = constants.%tuple.1]
// CHECK:STDOUT:   %tuple.loc14_32: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc14_37.3: %.1 = converted %.loc14_32, %tuple.loc14_32 [template = constants.%tuple.1]
// CHECK:STDOUT:   %tuple.loc14_36: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc14_37.4: %.1 = converted %.loc14_36, %tuple.loc14_36 [template = constants.%tuple.1]
// CHECK:STDOUT:   %tuple.loc14_37: %.3 = tuple_value (%.loc14_37.2, %.loc14_37.3, %.loc14_37.4) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc14_38: %.3 = converted %.loc14_37.1, %tuple.loc14_37 [template = constants.%tuple.2]
// CHECK:STDOUT:   %d: %.3 = bind_symbolic_name d 1, %.loc14_38 [symbolic = constants.%d]
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .d = %d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@C.%a: %.1, @C.%b.loc6_8.2: %.2) {
// CHECK:STDOUT:   %b.1: %.2 = bind_symbolic_name b 1 [symbolic = %b.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %c.1: %.7 = bind_symbolic_name c 3 [symbolic = %c.1 (constants.%c)]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %d: %.3 = bind_symbolic_name d 1 [symbolic = %d (constants.%d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@C.%b.loc6_8.2: %.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc7_19: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc7_20.1: %.7 = tuple_literal (%.loc7_15, %.loc7_19)
// CHECK:STDOUT:     %.loc7_20.2: type = converted %.loc7_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_20.3: type = converted %.loc7_19, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_20.4: type = converted %.loc7_20.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:     %.loc7_26: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc7_30: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc7_31.1: %.7 = tuple_literal (%.loc7_26, %.loc7_30)
// CHECK:STDOUT:     %tuple.loc7_26: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc7_31.2: %.1 = converted %.loc7_26, %tuple.loc7_26 [template = constants.%tuple.1]
// CHECK:STDOUT:     %tuple.loc7_30: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc7_31.3: %.1 = converted %.loc7_30, %tuple.loc7_30 [template = constants.%tuple.1]
// CHECK:STDOUT:     %tuple.loc7_31: %.7 = tuple_value (%.loc7_31.2, %.loc7_31.3) [template = constants.%tuple.3]
// CHECK:STDOUT:     %.loc7_32: %.7 = converted %.loc7_31.1, %tuple.loc7_31 [template = constants.%tuple.3]
// CHECK:STDOUT:     %c.loc7: %.7 = bind_symbolic_name c 3, %.loc7_32 [symbolic = %c.1 (constants.%c)]
// CHECK:STDOUT:     %.loc9_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %a1.var: ref %.1 = var a1
// CHECK:STDOUT:     %a1: ref %.1 = bind_name a1, %a1.var
// CHECK:STDOUT:     %a.ref: %.1 = name_ref a, @C.%a [symbolic = %a (constants.%a)]
// CHECK:STDOUT:     %.loc9_18: init %.1 = tuple_init () to %a1.var [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc9_19: init %.1 = converted %a.ref, %.loc9_18 [template = constants.%tuple.1]
// CHECK:STDOUT:     assign %a1.var, %.loc9_19
// CHECK:STDOUT:     %.loc10_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_17.1: %.2 = tuple_literal (%.loc10_15)
// CHECK:STDOUT:     %.loc10_17.2: type = converted %.loc10_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc10_17.3: type = converted %.loc10_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %b1.var: ref %.2 = var b1
// CHECK:STDOUT:     %b1: ref %.2 = bind_name b1, %b1.var
// CHECK:STDOUT:     %b.ref: %.2 = name_ref b, @C.%b.loc6_8.2 [symbolic = %b.1 (constants.%b)]
// CHECK:STDOUT:     %.loc10_21.1: %.1 = tuple_access %b.ref, element0
// CHECK:STDOUT:     %.loc10_21.2: ref %.1 = tuple_access %b1.var, element0
// CHECK:STDOUT:     %.loc10_21.3: init %.1 = tuple_init () to %.loc10_21.2 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc10_21.4: init %.1 = converted %.loc10_21.1, %.loc10_21.3 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc10_21.5: init %.2 = tuple_init (%.loc10_21.4) to %b1.var [template = constants.%tuple.4]
// CHECK:STDOUT:     %.loc10_22: init %.2 = converted %b.ref, %.loc10_21.5 [template = constants.%tuple.4]
// CHECK:STDOUT:     assign %b1.var, %.loc10_22
// CHECK:STDOUT:     %.loc11_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_19: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_20.1: %.7 = tuple_literal (%.loc11_15, %.loc11_19)
// CHECK:STDOUT:     %.loc11_20.2: type = converted %.loc11_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_20.3: type = converted %.loc11_19, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_20.4: type = converted %.loc11_20.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:     %c1.var: ref %.7 = var c1
// CHECK:STDOUT:     %c1: ref %.7 = bind_name c1, %c1.var
// CHECK:STDOUT:     %c.ref: %.7 = name_ref c, %c.loc7 [symbolic = %c.1 (constants.%c)]
// CHECK:STDOUT:     %.loc11_24.1: %.1 = tuple_access %c.ref, element0
// CHECK:STDOUT:     %.loc11_24.2: ref %.1 = tuple_access %c1.var, element0
// CHECK:STDOUT:     %.loc11_24.3: init %.1 = tuple_init () to %.loc11_24.2 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc11_24.4: init %.1 = converted %.loc11_24.1, %.loc11_24.3 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc11_24.5: %.1 = tuple_access %c.ref, element1
// CHECK:STDOUT:     %.loc11_24.6: ref %.1 = tuple_access %c1.var, element1
// CHECK:STDOUT:     %.loc11_24.7: init %.1 = tuple_init () to %.loc11_24.6 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc11_24.8: init %.1 = converted %.loc11_24.5, %.loc11_24.7 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc11_24.9: init %.7 = tuple_init (%.loc11_24.4, %.loc11_24.8) to %c1.var [template = constants.%tuple.3]
// CHECK:STDOUT:     %.loc11_25: init %.7 = converted %c.ref, %.loc11_24.9 [template = constants.%tuple.3]
// CHECK:STDOUT:     assign %c1.var, %.loc11_25
// CHECK:STDOUT:     %.loc12_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc12_19: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc12_23: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc12_24.1: %.3 = tuple_literal (%.loc12_15, %.loc12_19, %.loc12_23)
// CHECK:STDOUT:     %.loc12_24.2: type = converted %.loc12_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc12_24.3: type = converted %.loc12_19, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc12_24.4: type = converted %.loc12_23, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc12_24.5: type = converted %.loc12_24.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %d1.var: ref %.3 = var d1
// CHECK:STDOUT:     %d1: ref %.3 = bind_name d1, %d1.var
// CHECK:STDOUT:     %d.ref: %.3 = name_ref d, @C.%d [symbolic = %d (constants.%d)]
// CHECK:STDOUT:     %.loc12_28.1: %.1 = tuple_access %d.ref, element0
// CHECK:STDOUT:     %.loc12_28.2: ref %.1 = tuple_access %d1.var, element0
// CHECK:STDOUT:     %.loc12_28.3: init %.1 = tuple_init () to %.loc12_28.2 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc12_28.4: init %.1 = converted %.loc12_28.1, %.loc12_28.3 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc12_28.5: %.1 = tuple_access %d.ref, element1
// CHECK:STDOUT:     %.loc12_28.6: ref %.1 = tuple_access %d1.var, element1
// CHECK:STDOUT:     %.loc12_28.7: init %.1 = tuple_init () to %.loc12_28.6 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc12_28.8: init %.1 = converted %.loc12_28.5, %.loc12_28.7 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc12_28.9: %.1 = tuple_access %d.ref, element2
// CHECK:STDOUT:     %.loc12_28.10: ref %.1 = tuple_access %d1.var, element2
// CHECK:STDOUT:     %.loc12_28.11: init %.1 = tuple_init () to %.loc12_28.10 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc12_28.12: init %.1 = converted %.loc12_28.9, %.loc12_28.11 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc12_28.13: init %.3 = tuple_init (%.loc12_28.4, %.loc12_28.8, %.loc12_28.12) to %d1.var [template = constants.%tuple.2]
// CHECK:STDOUT:     %.loc12_29: init %.3 = converted %d.ref, %.loc12_28.13 [template = constants.%tuple.2]
// CHECK:STDOUT:     assign %d1.var, %.loc12_29
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%a, constants.%b) {
// CHECK:STDOUT:   %b.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_let_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0 [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc3_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc3_14.2: type = converted %.loc3_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc7_12.2: type = converted %.loc7_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x 0, <error> [symbolic = constants.%x]
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.1 = name_ref x, @C.%x [symbolic = constants.%x]
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
