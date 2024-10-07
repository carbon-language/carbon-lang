
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is mostly checking against crashes for compile time bindings in
// difficult contexts.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/compile_time_bindings.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/compile_time_bindings.carbon

// --- fail_let_after.carbon

library "[[@TEST_NAME]]";

class C {
  fn F() -> () { return x; }
  // CHECK:STDERR: fail_let_after.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface`
  // CHECK:STDERR:   let x:! () = ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  let x:! () = ();
}

// --- fail_let_before.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_let_before.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface`
  // CHECK:STDERR:   let x:! () = ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  let x:! () = ();
  fn F() -> () { return x; }
}

// --- fail_multiple_lets.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_multiple_lets.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface`
  // CHECK:STDERR:   let a:! () = ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  let a:! () = ();
  fn F(b:! ((),)) {
    let c:! ((), ()) = ((), ());

    var a1: () = a;
    var b1: ((),) = b;
    var c1: ((), ()) = c;
    var d1: ((), (), ()) = d;
  }
  // CHECK:STDERR: fail_multiple_lets.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface`
  // CHECK:STDERR:   let d:! ((), (), ()) = ((), (), ());
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let d:! ((), (), ()) = ((), (), ());
}

// --- fail_invalid_let_after.carbon

library "[[@TEST_NAME]]";

class C {
  fn F() -> () { return x; }
  // CHECK:STDERR: fail_invalid_let_after.carbon:[[@LINE+8]]:7: error: semantics TODO: ``let` compile time binding outside function or interface`
  // CHECK:STDERR:   let x:! ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_invalid_let_after.carbon:[[@LINE+4]]:13: error: expected `=`; `let` declaration must have an initializer
  // CHECK:STDERR:   let x:! ();
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  let x:! ();
}

// --- use_in_function.carbon

library "[[@TEST_NAME]]";

fn F() -> i32 {
  let Zero:! i32 = 0;
  return Zero;
}

// --- fail_return_in_interface.carbon

library "[[@TEST_NAME]]";

interface I {
  let T:! type = i32;
  // CHECK:STDERR: fail_return_in_interface.carbon:[[@LINE+7]]:13: error: cannot implicitly convert from `<associated type in I>` to `type`
  // CHECK:STDERR:   fn F() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR: fail_return_in_interface.carbon:[[@LINE+4]]:13: note: type `<associated type in I>` does not implement interface `ImplicitAs`
  // CHECK:STDERR:   fn F() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> T;
}

// --- fail_return_in_class.carbon

library "[[@TEST_NAME]]";

class I {
  // CHECK:STDERR: fail_return_in_class.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface`
  // CHECK:STDERR:   let T:! type = i32;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let T:! type = i32;
  // CHECK:STDERR: fail_return_in_class.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression
  // CHECK:STDERR:   fn F() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> T;
}

// --- fail_return_in_package_scope.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_return_in_package_scope.carbon:[[@LINE+4]]:5: error: semantics TODO: ``let` compile time binding outside function or interface`
// CHECK:STDERR: let T:! type = i32;
// CHECK:STDERR:     ^~~~~~~~
// CHECK:STDERR:
let T:! type = i32;
// CHECK:STDERR: fail_return_in_package_scope.carbon:[[@LINE+3]]:11: error: cannot evaluate type expression
// CHECK:STDERR: fn F() -> T;
// CHECK:STDERR:           ^
fn F() -> T;

// CHECK:STDOUT: --- fail_let_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %.loc5_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_12.2: type = converted %.loc10_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_18: %.1 = converted %.loc10_17, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %x: %.1 = bind_name x, %.loc10_18
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.1 = name_ref x, @C.%x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_let_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc9_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc9_12.2: type = converted %.loc9_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_18: %.1 = converted %.loc9_17, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %x: %.1 = bind_name x, %.loc9_18
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %.loc10_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_14.2: type = converted %.loc10_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.1 = name_ref x, @C.%x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_multiple_lets.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple.1: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (%.1) [template]
// CHECK:STDOUT:   %b: %.2 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = tuple_type (%.1, %.1, %.1) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %tuple.2: %.3 = tuple_value (%tuple.1, %tuple.1, %tuple.1) [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.1, %.1) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %tuple.3: %.7 = tuple_value (%tuple.1, %tuple.1) [template]
// CHECK:STDOUT:   %c: %.7 = bind_symbolic_name c, 1 [symbolic]
// CHECK:STDOUT:   %tuple.4: %.2 = tuple_value (%tuple.1) [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc9_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc9_12.2: type = converted %.loc9_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple.loc9: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc9_18: %.1 = converted %.loc9_17, %tuple.loc9 [template = constants.%tuple.1]
// CHECK:STDOUT:   %a: %.1 = bind_name a, %.loc9_18
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %b.patt: %.2 = symbolic_binding_pattern b, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_14: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_16.1: %.2 = tuple_literal (%.loc10_14)
// CHECK:STDOUT:     %.loc10_16.2: type = converted %.loc10_14, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc10_16.3: type = converted %.loc10_16.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %b.param: %.2 = param b, runtime_param<invalid>
// CHECK:STDOUT:     %b.loc10: %.2 = bind_symbolic_name b, 0, %b.param [symbolic = %b.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_13: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc22_17: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc22_21: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc22_22.1: %.3 = tuple_literal (%.loc22_13, %.loc22_17, %.loc22_21)
// CHECK:STDOUT:   %.loc22_22.2: type = converted %.loc22_13, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_22.3: type = converted %.loc22_17, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_22.4: type = converted %.loc22_21, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_22.5: type = converted %.loc22_22.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc22_28: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc22_32: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc22_36: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc22_37.1: %.3 = tuple_literal (%.loc22_28, %.loc22_32, %.loc22_36)
// CHECK:STDOUT:   %tuple.loc22_28: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc22_37.2: %.1 = converted %.loc22_28, %tuple.loc22_28 [template = constants.%tuple.1]
// CHECK:STDOUT:   %tuple.loc22_32: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc22_37.3: %.1 = converted %.loc22_32, %tuple.loc22_32 [template = constants.%tuple.1]
// CHECK:STDOUT:   %tuple.loc22_36: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc22_37.4: %.1 = converted %.loc22_36, %tuple.loc22_36 [template = constants.%tuple.1]
// CHECK:STDOUT:   %tuple.loc22_37: %.3 = tuple_value (%.loc22_37.2, %.loc22_37.3, %.loc22_37.4) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc22_38: %.3 = converted %.loc22_37.1, %tuple.loc22_37 [template = constants.%tuple.2]
// CHECK:STDOUT:   %d: %.3 = bind_name d, %.loc22_38
// CHECK:STDOUT:   %.loc23: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .d = %d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%b.loc10: %.2) {
// CHECK:STDOUT:   %b.1: %.2 = bind_symbolic_name b, 0 [symbolic = %b.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %c.1: %.7 = bind_symbolic_name c, 1 [symbolic = %c.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%b.loc10: %.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_19: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_20.1: %.7 = tuple_literal (%.loc11_15, %.loc11_19)
// CHECK:STDOUT:     %.loc11_20.2: type = converted %.loc11_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_20.3: type = converted %.loc11_19, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_20.4: type = converted %.loc11_20.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:     %.loc11_26: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_30: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_31.1: %.7 = tuple_literal (%.loc11_26, %.loc11_30)
// CHECK:STDOUT:     %tuple.loc11_26: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc11_31.2: %.1 = converted %.loc11_26, %tuple.loc11_26 [template = constants.%tuple.1]
// CHECK:STDOUT:     %tuple.loc11_30: %.1 = tuple_value () [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc11_31.3: %.1 = converted %.loc11_30, %tuple.loc11_30 [template = constants.%tuple.1]
// CHECK:STDOUT:     %tuple.loc11_31: %.7 = tuple_value (%.loc11_31.2, %.loc11_31.3) [template = constants.%tuple.3]
// CHECK:STDOUT:     %.loc11_32: %.7 = converted %.loc11_31.1, %tuple.loc11_31 [template = constants.%tuple.3]
// CHECK:STDOUT:     %c.loc11: %.7 = bind_symbolic_name c, 1, %.loc11_32 [symbolic = %c.1 (constants.%c)]
// CHECK:STDOUT:     %.loc13_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc13_14.2: type = converted %.loc13_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %a1.var: ref %.1 = var a1
// CHECK:STDOUT:     %a1: ref %.1 = bind_name a1, %a1.var
// CHECK:STDOUT:     %a.ref: %.1 = name_ref a, @C.%a
// CHECK:STDOUT:     %.loc13_18: init %.1 = tuple_init () to %a1.var [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc13_19: init %.1 = converted %a.ref, %.loc13_18 [template = constants.%tuple.1]
// CHECK:STDOUT:     assign %a1.var, %.loc13_19
// CHECK:STDOUT:     %.loc14_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc14_17.1: %.2 = tuple_literal (%.loc14_15)
// CHECK:STDOUT:     %.loc14_17.2: type = converted %.loc14_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc14_17.3: type = converted %.loc14_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %b1.var: ref %.2 = var b1
// CHECK:STDOUT:     %b1: ref %.2 = bind_name b1, %b1.var
// CHECK:STDOUT:     %b.ref: %.2 = name_ref b, %b.loc10 [symbolic = %b.1 (constants.%b)]
// CHECK:STDOUT:     %.loc14_21.1: %.1 = tuple_access %b.ref, element0
// CHECK:STDOUT:     %.loc14_21.2: ref %.1 = tuple_access %b1.var, element0
// CHECK:STDOUT:     %.loc14_21.3: init %.1 = tuple_init () to %.loc14_21.2 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc14_21.4: init %.1 = converted %.loc14_21.1, %.loc14_21.3 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc14_21.5: init %.2 = tuple_init (%.loc14_21.4) to %b1.var [template = constants.%tuple.4]
// CHECK:STDOUT:     %.loc14_22: init %.2 = converted %b.ref, %.loc14_21.5 [template = constants.%tuple.4]
// CHECK:STDOUT:     assign %b1.var, %.loc14_22
// CHECK:STDOUT:     %.loc15_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc15_19: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc15_20.1: %.7 = tuple_literal (%.loc15_15, %.loc15_19)
// CHECK:STDOUT:     %.loc15_20.2: type = converted %.loc15_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc15_20.3: type = converted %.loc15_19, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc15_20.4: type = converted %.loc15_20.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:     %c1.var: ref %.7 = var c1
// CHECK:STDOUT:     %c1: ref %.7 = bind_name c1, %c1.var
// CHECK:STDOUT:     %c.ref: %.7 = name_ref c, %c.loc11 [symbolic = %c.1 (constants.%c)]
// CHECK:STDOUT:     %.loc15_24.1: %.1 = tuple_access %c.ref, element0
// CHECK:STDOUT:     %.loc15_24.2: ref %.1 = tuple_access %c1.var, element0
// CHECK:STDOUT:     %.loc15_24.3: init %.1 = tuple_init () to %.loc15_24.2 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc15_24.4: init %.1 = converted %.loc15_24.1, %.loc15_24.3 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc15_24.5: %.1 = tuple_access %c.ref, element1
// CHECK:STDOUT:     %.loc15_24.6: ref %.1 = tuple_access %c1.var, element1
// CHECK:STDOUT:     %.loc15_24.7: init %.1 = tuple_init () to %.loc15_24.6 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc15_24.8: init %.1 = converted %.loc15_24.5, %.loc15_24.7 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc15_24.9: init %.7 = tuple_init (%.loc15_24.4, %.loc15_24.8) to %c1.var [template = constants.%tuple.3]
// CHECK:STDOUT:     %.loc15_25: init %.7 = converted %c.ref, %.loc15_24.9 [template = constants.%tuple.3]
// CHECK:STDOUT:     assign %c1.var, %.loc15_25
// CHECK:STDOUT:     %.loc16_15: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc16_19: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc16_23: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc16_24.1: %.3 = tuple_literal (%.loc16_15, %.loc16_19, %.loc16_23)
// CHECK:STDOUT:     %.loc16_24.2: type = converted %.loc16_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc16_24.3: type = converted %.loc16_19, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc16_24.4: type = converted %.loc16_23, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc16_24.5: type = converted %.loc16_24.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %d1.var: ref %.3 = var d1
// CHECK:STDOUT:     %d1: ref %.3 = bind_name d1, %d1.var
// CHECK:STDOUT:     %d.ref: %.3 = name_ref d, @C.%d
// CHECK:STDOUT:     %.loc16_28.1: %.1 = tuple_access %d.ref, element0
// CHECK:STDOUT:     %.loc16_28.2: ref %.1 = tuple_access %d1.var, element0
// CHECK:STDOUT:     %.loc16_28.3: init %.1 = tuple_init () to %.loc16_28.2 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc16_28.4: init %.1 = converted %.loc16_28.1, %.loc16_28.3 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc16_28.5: %.1 = tuple_access %d.ref, element1
// CHECK:STDOUT:     %.loc16_28.6: ref %.1 = tuple_access %d1.var, element1
// CHECK:STDOUT:     %.loc16_28.7: init %.1 = tuple_init () to %.loc16_28.6 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc16_28.8: init %.1 = converted %.loc16_28.5, %.loc16_28.7 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc16_28.9: %.1 = tuple_access %d.ref, element2
// CHECK:STDOUT:     %.loc16_28.10: ref %.1 = tuple_access %d1.var, element2
// CHECK:STDOUT:     %.loc16_28.11: init %.1 = tuple_init () to %.loc16_28.10 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc16_28.12: init %.1 = converted %.loc16_28.9, %.loc16_28.11 [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc16_28.13: init %.3 = tuple_init (%.loc16_28.4, %.loc16_28.8, %.loc16_28.12) to %d1.var [template = constants.%tuple.2]
// CHECK:STDOUT:     %.loc16_29: init %.3 = converted %d.ref, %.loc16_28.13 [template = constants.%tuple.2]
// CHECK:STDOUT:     assign %d1.var, %.loc16_29
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%b) {
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
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %.loc5_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_12.2: type = converted %.loc14_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x: %.1 = bind_name x, <error>
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.1 = name_ref x, @C.%x
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_26: %.1 = converted %x.ref, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc5_26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %Zero: i32 = bind_symbolic_name Zero, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_14.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_14.2: type = converted %int.make_type_32.loc5, %.loc5_14.1 [template = i32]
// CHECK:STDOUT:   %.loc5_20: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %Zero: i32 = bind_symbolic_name Zero, 0, %.loc5_20 [symbolic = constants.%Zero]
// CHECK:STDOUT:   %Zero.ref: i32 = name_ref Zero, %Zero [symbolic = constants.%Zero]
// CHECK:STDOUT:   return %Zero.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.3: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.8: %.4 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element1, @I.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.4) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.8)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_21.2: type = converted %int.make_type_32, %.loc5_21.1 [template = i32]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc5_21.3: %.2 = assoc_entity element0, %T [template = constants.%.3]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %T.ref: %.2 = name_ref T, @I.%.loc5_21.3 [template = constants.%.3]
// CHECK:STDOUT:     %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(type) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:     %.loc13_13.1: %.6 = specific_constant imports.%import_ref.4, @ImplicitAs(type) [template = constants.%.7]
// CHECK:STDOUT:     %Convert.ref: %.6 = name_ref Convert, %.loc13_13.1 [template = constants.%.7]
// CHECK:STDOUT:     %.loc13_13.2: type = converted %T.ref, <error> [template = <error>]
// CHECK:STDOUT:     %return: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: %.9 = assoc_entity element1, %F.decl [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc5_21.3
// CHECK:STDOUT:   .F = %.loc13
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.4) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.3)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
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
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %Dest => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I: type = class_type @I [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = class_decl @I [template = constants.%I] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @I {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc9_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc9_21.2: type = converted %int.make_type_32, %.loc9_21.1 [template = i32]
// CHECK:STDOUT:   %T: type = bind_name T, %.loc9_21.2
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @I.%T
// CHECK:STDOUT:     %return: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%I
// CHECK:STDOUT:   .T = %T
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_package_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .T = @__global_init.%T
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @__global_init.%T
// CHECK:STDOUT:     %return: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc8_19.2: type = converted %int.make_type_32, %.loc8_19.1 [template = i32]
// CHECK:STDOUT:   %T: type = bind_name T, %.loc8_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
