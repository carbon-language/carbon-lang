
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
  // CHECK:STDERR: fail_let_after.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let x:! () = ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  let x:! () = ();
}

// --- fail_let_before.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_let_before.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let x:! () = ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  let x:! () = ();
  fn F() -> () { return x; }
}

// --- fail_multiple_lets.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_multiple_lets.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
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
  // CHECK:STDERR: fail_multiple_lets.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let d:! ((), (), ()) = ((), (), ());
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let d:! ((), (), ()) = ((), (), ());
}

// --- fail_invalid_let_after.carbon

library "[[@TEST_NAME]]";

class C {
  fn F() -> () { return x; }
  // CHECK:STDERR: fail_invalid_let_after.carbon:[[@LINE+8]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let x:! ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_invalid_let_after.carbon:[[@LINE+4]]:13: error: expected `=`; `let` declaration must have an initializer [ExpectedInitializerAfterLet]
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

// --- use_in_block.carbon

library "[[@TEST_NAME]]";

fn F() -> i32 {
  if (true) {
    let Zero:! i32 = 0;
    return Zero;
  }
  return 1;
}

// --- fail_return_in_interface.carbon

library "[[@TEST_NAME]]";

interface I {
  let T:! type = i32;
  // CHECK:STDERR: fail_return_in_interface.carbon:[[@LINE+7]]:13: error: cannot implicitly convert from `<associated type in I>` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR: fail_return_in_interface.carbon:[[@LINE+4]]:13: note: type `<associated type in I>` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> T;
}

// --- fail_return_in_class.carbon

library "[[@TEST_NAME]]";

class I {
  // CHECK:STDERR: fail_return_in_class.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let T:! type = i32;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let T:! type = i32;
  // CHECK:STDERR: fail_return_in_class.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   fn F() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> T;
}

// --- fail_return_in_package_scope.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_return_in_package_scope.carbon:[[@LINE+4]]:5: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
// CHECK:STDERR: let T:! type = i32;
// CHECK:STDERR:     ^~~~~~~~
// CHECK:STDERR:
let T:! type = i32;
// CHECK:STDERR: fail_return_in_package_scope.carbon:[[@LINE+4]]:11: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F() -> T;
// CHECK:STDERR:           ^
// CHECK:STDERR:
fn F() -> T;

// --- fail_use_in_impl.carbon

library "[[@TEST_NAME]]";

interface Empty {}

impl i32 as Empty {
  // CHECK:STDERR: fail_use_in_impl.carbon:[[@LINE+3]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let Zero:! i32 = 0;
  // CHECK:STDERR:       ^~~~~~~~~~
  let Zero:! i32 = 0;
}

// CHECK:STDOUT: --- fail_let_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_12.2: type = converted %.loc10_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc10_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_18: %empty_tuple.type = converted %.loc10_17, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, %.loc10_18
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, @C.%x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_let_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
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
// CHECK:STDOUT:   %.loc9_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc9_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_18: %empty_tuple.type = converted %.loc9_17, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, %.loc9_18
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_14.2: type = converted %.loc10_14.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, @C.%x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_multiple_lets.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%empty_tuple.type) [template]
// CHECK:STDOUT:   %b: %tuple.type.1 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %tuple.type.1 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.2 = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.3 = tuple_value (%empty_tuple, %empty_tuple) [template]
// CHECK:STDOUT:   %c: %tuple.type.3 = bind_symbolic_name c, 1 [symbolic]
// CHECK:STDOUT:   %tuple.elem0.1: %empty_tuple.type = tuple_access %b, element0 [symbolic]
// CHECK:STDOUT:   %tuple.3: %tuple.type.1 = tuple_value (%empty_tuple) [template]
// CHECK:STDOUT:   %tuple.elem0.2: %empty_tuple.type = tuple_access %c, element0 [symbolic]
// CHECK:STDOUT:   %tuple.elem1: %empty_tuple.type = tuple_access %c, element1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
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
// CHECK:STDOUT:   %.loc9_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_12.2: type = converted %.loc9_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc9_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc9: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_18: %empty_tuple.type = converted %.loc9_17, %empty_tuple.loc9 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %a: %empty_tuple.type = bind_name a, %.loc9_18
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %b.patt.loc10_8.2: %tuple.type.1 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc10_8.1 (constants.%b.patt)]
// CHECK:STDOUT:     %b.param_patt: %tuple.type.1 = value_param_pattern %b.patt.loc10_8.2, runtime_param<invalid> [symbolic = %b.patt.loc10_8.1 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_16.1: %tuple.type.1 = tuple_literal (%.loc10_14)
// CHECK:STDOUT:     %.loc10_16.2: type = converted %.loc10_14, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_16.3: type = converted %.loc10_16.1, constants.%tuple.type.1 [template = constants.%tuple.type.1]
// CHECK:STDOUT:     %b.param: %tuple.type.1 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc10_8.2: %tuple.type.1 = bind_symbolic_name b, 0, %b.param [symbolic = %b.loc10_8.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_13: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_21: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_22.1: %tuple.type.2 = tuple_literal (%.loc22_13, %.loc22_17, %.loc22_21)
// CHECK:STDOUT:   %.loc22_22.2: type = converted %.loc22_13, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc22_22.3: type = converted %.loc22_17, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc22_22.4: type = converted %.loc22_21, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc22_22.5: type = converted %.loc22_22.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc22_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_32: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_36: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_37.1: %tuple.type.2 = tuple_literal (%.loc22_28, %.loc22_32, %.loc22_36)
// CHECK:STDOUT:   %empty_tuple.loc22_28: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_37.2: %empty_tuple.type = converted %.loc22_28, %empty_tuple.loc22_28 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc22_32: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_37.3: %empty_tuple.type = converted %.loc22_32, %empty_tuple.loc22_32 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc22_36: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_37.4: %empty_tuple.type = converted %.loc22_36, %empty_tuple.loc22_36 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc22_37.2, %.loc22_37.3, %.loc22_37.4) [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc22_38: %tuple.type.2 = converted %.loc22_37.1, %tuple [template = constants.%tuple.1]
// CHECK:STDOUT:   %d: %tuple.type.2 = bind_name d, %.loc22_38
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .d = %d
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%b.loc10_8.2: %tuple.type.1) {
// CHECK:STDOUT:   %b.loc10_8.1: %tuple.type.1 = bind_symbolic_name b, 0 [symbolic = %b.loc10_8.1 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc10_8.1: %tuple.type.1 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc10_8.1 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %c.loc11_9.2: %tuple.type.3 = bind_symbolic_name c, 1 [symbolic = %c.loc11_9.2 (constants.%c)]
// CHECK:STDOUT:   %tuple.elem0.loc14_21.3: %empty_tuple.type = tuple_access %b.loc10_8.1, element0 [symbolic = %tuple.elem0.loc14_21.3 (constants.%tuple.elem0.1)]
// CHECK:STDOUT:   %tuple.elem0.loc15_24.3: %empty_tuple.type = tuple_access %c.loc11_9.2, element0 [symbolic = %tuple.elem0.loc15_24.3 (constants.%tuple.elem0.2)]
// CHECK:STDOUT:   %tuple.elem1.loc15_24.3: %empty_tuple.type = tuple_access %c.loc11_9.2, element1 [symbolic = %tuple.elem1.loc15_24.3 (constants.%tuple.elem1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%b.param_patt: %tuple.type.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_19: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_20.1: %tuple.type.3 = tuple_literal (%.loc11_15, %.loc11_19)
// CHECK:STDOUT:     %.loc11_20.2: type = converted %.loc11_15, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_20.3: type = converted %.loc11_19, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_20.4: type = converted %.loc11_20.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %.loc11_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_31.1: %tuple.type.3 = tuple_literal (%.loc11_26, %.loc11_30)
// CHECK:STDOUT:     %empty_tuple.loc11_26: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_31.2: %empty_tuple.type = converted %.loc11_26, %empty_tuple.loc11_26 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %empty_tuple.loc11_30: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_31.3: %empty_tuple.type = converted %.loc11_30, %empty_tuple.loc11_30 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple: %tuple.type.3 = tuple_value (%.loc11_31.2, %.loc11_31.3) [template = constants.%tuple.2]
// CHECK:STDOUT:     %.loc11_32: %tuple.type.3 = converted %.loc11_31.1, %tuple [template = constants.%tuple.2]
// CHECK:STDOUT:     %c.loc11_9.1: %tuple.type.3 = bind_symbolic_name c, 1, %.loc11_32 [symbolic = %c.loc11_9.2 (constants.%c)]
// CHECK:STDOUT:     %.loc13_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_14.2: type = converted %.loc13_14.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %a1.var: ref %empty_tuple.type = var a1
// CHECK:STDOUT:     %a1: ref %empty_tuple.type = bind_name a1, %a1.var
// CHECK:STDOUT:     %a.ref: %empty_tuple.type = name_ref a, @C.%a
// CHECK:STDOUT:     %.loc13_18: init %empty_tuple.type = tuple_init () to %a1.var [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_19: init %empty_tuple.type = converted %a.ref, %.loc13_18 [template = constants.%empty_tuple]
// CHECK:STDOUT:     assign %a1.var, %.loc13_19
// CHECK:STDOUT:     %.loc14_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_17.1: %tuple.type.1 = tuple_literal (%.loc14_15)
// CHECK:STDOUT:     %.loc14_17.2: type = converted %.loc14_15, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc14_17.3: type = converted %.loc14_17.1, constants.%tuple.type.1 [template = constants.%tuple.type.1]
// CHECK:STDOUT:     %b1.var: ref %tuple.type.1 = var b1
// CHECK:STDOUT:     %b1: ref %tuple.type.1 = bind_name b1, %b1.var
// CHECK:STDOUT:     %b.ref: %tuple.type.1 = name_ref b, %b.loc10_8.2 [symbolic = %b.loc10_8.1 (constants.%b)]
// CHECK:STDOUT:     %tuple.elem0.loc14_21.1: %empty_tuple.type = tuple_access %b.ref, element0 [symbolic = %tuple.elem0.loc14_21.3 (constants.%tuple.elem0.1)]
// CHECK:STDOUT:     %tuple.elem0.loc14_21.2: ref %empty_tuple.type = tuple_access %b1.var, element0
// CHECK:STDOUT:     %.loc14_21.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc14_21.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc14_21.2: init %empty_tuple.type = converted %tuple.elem0.loc14_21.1, %.loc14_21.1 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc14_21.3: init %tuple.type.1 = tuple_init (%.loc14_21.2) to %b1.var [template = constants.%tuple.3]
// CHECK:STDOUT:     %.loc14_22: init %tuple.type.1 = converted %b.ref, %.loc14_21.3 [template = constants.%tuple.3]
// CHECK:STDOUT:     assign %b1.var, %.loc14_22
// CHECK:STDOUT:     %.loc15_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_19: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_20.1: %tuple.type.3 = tuple_literal (%.loc15_15, %.loc15_19)
// CHECK:STDOUT:     %.loc15_20.2: type = converted %.loc15_15, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_20.3: type = converted %.loc15_19, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_20.4: type = converted %.loc15_20.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %c1.var: ref %tuple.type.3 = var c1
// CHECK:STDOUT:     %c1: ref %tuple.type.3 = bind_name c1, %c1.var
// CHECK:STDOUT:     %c.ref: %tuple.type.3 = name_ref c, %c.loc11_9.1 [symbolic = %c.loc11_9.2 (constants.%c)]
// CHECK:STDOUT:     %tuple.elem0.loc15_24.1: %empty_tuple.type = tuple_access %c.ref, element0 [symbolic = %tuple.elem0.loc15_24.3 (constants.%tuple.elem0.2)]
// CHECK:STDOUT:     %tuple.elem0.loc15_24.2: ref %empty_tuple.type = tuple_access %c1.var, element0
// CHECK:STDOUT:     %.loc15_24.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc15_24.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_24.2: init %empty_tuple.type = converted %tuple.elem0.loc15_24.1, %.loc15_24.1 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple.elem1.loc15_24.1: %empty_tuple.type = tuple_access %c.ref, element1 [symbolic = %tuple.elem1.loc15_24.3 (constants.%tuple.elem1)]
// CHECK:STDOUT:     %tuple.elem1.loc15_24.2: ref %empty_tuple.type = tuple_access %c1.var, element1
// CHECK:STDOUT:     %.loc15_24.3: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc15_24.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_24.4: init %empty_tuple.type = converted %tuple.elem1.loc15_24.1, %.loc15_24.3 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_24.5: init %tuple.type.3 = tuple_init (%.loc15_24.2, %.loc15_24.4) to %c1.var [template = constants.%tuple.2]
// CHECK:STDOUT:     %.loc15_25: init %tuple.type.3 = converted %c.ref, %.loc15_24.5 [template = constants.%tuple.2]
// CHECK:STDOUT:     assign %c1.var, %.loc15_25
// CHECK:STDOUT:     %.loc16_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_19: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_23: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_24.1: %tuple.type.2 = tuple_literal (%.loc16_15, %.loc16_19, %.loc16_23)
// CHECK:STDOUT:     %.loc16_24.2: type = converted %.loc16_15, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_24.3: type = converted %.loc16_19, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_24.4: type = converted %.loc16_23, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_24.5: type = converted %.loc16_24.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %d1.var: ref %tuple.type.2 = var d1
// CHECK:STDOUT:     %d1: ref %tuple.type.2 = bind_name d1, %d1.var
// CHECK:STDOUT:     %d.ref: %tuple.type.2 = name_ref d, @C.%d
// CHECK:STDOUT:     %tuple.elem0.loc16_28.1: %empty_tuple.type = tuple_access %d.ref, element0
// CHECK:STDOUT:     %tuple.elem0.loc16_28.2: ref %empty_tuple.type = tuple_access %d1.var, element0
// CHECK:STDOUT:     %.loc16_28.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc16_28.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.2: init %empty_tuple.type = converted %tuple.elem0.loc16_28.1, %.loc16_28.1 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple.elem1.loc16_28.1: %empty_tuple.type = tuple_access %d.ref, element1
// CHECK:STDOUT:     %tuple.elem1.loc16_28.2: ref %empty_tuple.type = tuple_access %d1.var, element1
// CHECK:STDOUT:     %.loc16_28.3: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc16_28.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.4: init %empty_tuple.type = converted %tuple.elem1.loc16_28.1, %.loc16_28.3 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple.elem2.loc16_28.1: %empty_tuple.type = tuple_access %d.ref, element2
// CHECK:STDOUT:     %tuple.elem2.loc16_28.2: ref %empty_tuple.type = tuple_access %d1.var, element2
// CHECK:STDOUT:     %.loc16_28.5: init %empty_tuple.type = tuple_init () to %tuple.elem2.loc16_28.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.6: init %empty_tuple.type = converted %tuple.elem2.loc16_28.1, %.loc16_28.5 [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.7: init %tuple.type.2 = tuple_init (%.loc16_28.2, %.loc16_28.4, %.loc16_28.6) to %d1.var [template = constants.%tuple.1]
// CHECK:STDOUT:     %.loc16_29: init %tuple.type.2 = converted %d.ref, %.loc16_28.7 [template = constants.%tuple.1]
// CHECK:STDOUT:     assign %d1.var, %.loc16_29
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%b) {
// CHECK:STDOUT:   %b.loc10_8.1 => constants.%b
// CHECK:STDOUT:   %b.patt.loc10_8.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_let_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc14_12.2: type = converted %.loc14_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, <error>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, @C.%x
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_26: %empty_tuple.type = converted %x.ref, %tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc5_26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4: init type = call constants.%Int(%int_32.loc4) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_signed.loc4, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_14.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_14.2: type = converted %int.make_type_signed.loc5, %.loc5_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc5_21.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc5_21.2: %i32 = converted %int_0, %.loc5_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0, %.loc5_21.2 [symbolic = constants.%Zero]
// CHECK:STDOUT:   %Zero.ref: %i32 = name_ref Zero, %Zero [symbolic = constants.%Zero]
// CHECK:STDOUT:   return %Zero.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_in_block.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0 [symbolic]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4: init type = call constants.%Int(%int_32.loc4) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_signed.loc4, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_16.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_16.2: type = converted %int.make_type_signed.loc6, %.loc6_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_0, %impl.elem0.loc6 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %i32 = call %Convert.specific_fn.loc6(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc6_23.1: %i32 = value_of_initializer %int.convert_checked.loc6 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc6_23.2: %i32 = converted %int_0, %.loc6_23.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0, %.loc6_23.2 [symbolic = constants.%Zero]
// CHECK:STDOUT:   %Zero.ref: %i32 = name_ref Zero, %Zero [symbolic = constants.%Zero]
// CHECK:STDOUT:   return %Zero.ref
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc9: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc9: <bound method> = bound_method %int_1, %impl.elem0.loc9 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9: <specific function> = specific_function %Convert.bound.loc9, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32 = call %Convert.specific_fn.loc9(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_11.1: %i32 = value_of_initializer %int.convert_checked.loc9 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_11.2: %i32 = converted %int_1, %.loc9_11.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   return %.loc9_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, @I.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21.2: type = converted %int.make_type_signed, %.loc5_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %T [template = constants.%assoc0.1]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, @I.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.loc13: type = converted %T.ref, <error> [template = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, %F.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I: type = class_type @I [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_21.2: type = converted %int.make_type_signed, %.loc9_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %T: type = bind_name T, %.loc9_21.2
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @I.%T
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%I
// CHECK:STDOUT:   .T = %T
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_package_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .T = @__global_init.%T
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @__global_init.%T
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_19.2: type = converted %int.make_type_signed, %.loc8_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %T: type = bind_name T, %.loc8_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [template]
// CHECK:STDOUT:   %Self.1: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [template = constants.%Empty.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_6.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_6.2: type = converted %int.make_type_signed.loc6, %.loc6_6.1 [template = constants.%i32]
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [template = constants.%Empty.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc6_6.2 as %Empty.ref {
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc10: init type = call constants.%Int(%int_32.loc10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_14.1: type = value_of_initializer %int.make_type_signed.loc10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_14.2: type = converted %int.make_type_signed.loc10, %.loc10_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_21.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_21.2: %i32 = converted %int_0, %.loc10_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Zero: %i32 = bind_name Zero, %.loc10_21.2
// CHECK:STDOUT:   %interface: <witness> = interface_witness () [template = constants.%interface.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Zero
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
