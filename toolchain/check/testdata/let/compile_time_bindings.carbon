
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

// --- return_in_interface.carbon

library "[[@TEST_NAME]]";

interface I {
  let T:! type = i32;
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
  // CHECK:STDERR: fail_use_in_impl.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let Zero:! i32 = 0;
  // CHECK:STDERR:       ^~~~~~~~~~
  // CHECK:STDERR:
  let Zero:! i32 = 0;
}

// CHECK:STDOUT: --- fail_let_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_12.1: type = splice_block %.loc10_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_12.3: type = converted %.loc10_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_17.2: %empty_tuple.type = converted %.loc10_17.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, %.loc10_17.2
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, @C.%x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_let_before.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_12.1: type = splice_block %.loc9_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc9_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_12.3: type = converted %.loc9_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_17.2: %empty_tuple.type = converted %.loc9_17.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, %.loc9_17.2
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_14.2: type = converted %.loc10_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, @C.%x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_multiple_lets.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type.9fb: type = tuple_type (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %b: %tuple.type.9fb = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.559: type = pattern_type %tuple.type.9fb [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.2d5: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %ptr.7fe: type = ptr_type %tuple.type.2d5 [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %tuple.type.2d5 [concrete]
// CHECK:STDOUT:   %tuple.7e4: %tuple.type.2d5 = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.bcd: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %ptr.709: type = ptr_type %tuple.type.bcd [concrete]
// CHECK:STDOUT:   %c: %tuple.type.bcd = bind_symbolic_name c, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type.bcd [concrete]
// CHECK:STDOUT:   %tuple.d8f: %tuple.type.bcd = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.elem0.af8: %empty_tuple.type = tuple_access %b, element0 [symbolic]
// CHECK:STDOUT:   %tuple.f41: %tuple.type.9fb = tuple_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.elem0.eaf: %empty_tuple.type = tuple_access %c, element0 [symbolic]
// CHECK:STDOUT:   %tuple.elem1: %empty_tuple.type = tuple_access %c, element1 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.073: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.2d5) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.dce: %T.as.Destroy.impl.Op.type.073 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.234: <specific function> = specific_function %T.as.Destroy.impl.Op.dce, @T.as.Destroy.impl.Op(%tuple.type.2d5) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.4e8: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.bcd) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.166: %T.as.Destroy.impl.Op.type.4e8 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.48a: <specific function> = specific_function %T.as.Destroy.impl.Op.166, @T.as.Destroy.impl.Op(%tuple.type.bcd) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.23e: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.9fb) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.f19: %T.as.Destroy.impl.Op.type.23e = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.652: type = ptr_type %tuple.type.9fb [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.d2b: <specific function> = specific_function %T.as.Destroy.impl.Op.f19, @T.as.Destroy.impl.Op(%tuple.type.9fb) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.a63: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.ea3: %T.as.Destroy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.393: <specific function> = specific_function %T.as.Destroy.impl.Op.ea3, @T.as.Destroy.impl.Op(%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_12.1: type = splice_block %.loc9_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc9_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_12.3: type = converted %.loc9_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.loc9: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_17.2: %empty_tuple.type = converted %.loc9_17.1, %empty_tuple.loc9 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %a: %empty_tuple.type = bind_name a, %.loc9_17.2
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %b.patt: %pattern_type.559 = symbolic_binding_pattern b, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_16.1: type = splice_block %.loc10_16.4 [concrete = constants.%tuple.type.9fb] {
// CHECK:STDOUT:       %.loc10_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc10_16.2: %tuple.type.9fb = tuple_literal (%.loc10_14)
// CHECK:STDOUT:       %.loc10_16.3: type = converted %.loc10_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc10_16.4: type = converted %.loc10_16.2, constants.%tuple.type.9fb [concrete = constants.%tuple.type.9fb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc10_8.2: %tuple.type.9fb = bind_symbolic_name b, 0 [symbolic = %b.loc10_8.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.8c1 = binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_32: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_36: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_37.1: %tuple.type.2d5 = tuple_literal (%.loc22_28, %.loc22_32, %.loc22_36)
// CHECK:STDOUT:   %.loc22_22.1: type = splice_block %.loc22_22.6 [concrete = constants.%tuple.type.2d5] {
// CHECK:STDOUT:     %.loc22_13: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_21: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_22.2: %tuple.type.2d5 = tuple_literal (%.loc22_13, %.loc22_17, %.loc22_21)
// CHECK:STDOUT:     %.loc22_22.3: type = converted %.loc22_13, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc22_22.4: type = converted %.loc22_17, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc22_22.5: type = converted %.loc22_21, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc22_22.6: type = converted %.loc22_22.2, constants.%tuple.type.2d5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.loc22_28: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_37.2: %empty_tuple.type = converted %.loc22_28, %empty_tuple.loc22_28 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc22_32: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_37.3: %empty_tuple.type = converted %.loc22_32, %empty_tuple.loc22_32 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc22_36: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_37.4: %empty_tuple.type = converted %.loc22_36, %empty_tuple.loc22_36 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple: %tuple.type.2d5 = tuple_value (%.loc22_37.2, %.loc22_37.3, %.loc22_37.4) [concrete = constants.%tuple.7e4]
// CHECK:STDOUT:   %.loc22_37.5: %tuple.type.2d5 = converted %.loc22_37.1, %tuple [concrete = constants.%tuple.7e4]
// CHECK:STDOUT:   %d: %tuple.type.2d5 = bind_name d, %.loc22_37.5
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT:   .d = %d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(%b.loc10_8.2: %tuple.type.9fb) {
// CHECK:STDOUT:   %b.loc10_8.1: %tuple.type.9fb = bind_symbolic_name b, 0 [symbolic = %b.loc10_8.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %c.loc11_9.2: %tuple.type.bcd = bind_symbolic_name c, 1 [symbolic = %c.loc11_9.2 (constants.%c)]
// CHECK:STDOUT:   %tuple.elem0.loc14_21.3: %empty_tuple.type = tuple_access %b.loc10_8.1, element0 [symbolic = %tuple.elem0.loc14_21.3 (constants.%tuple.elem0.af8)]
// CHECK:STDOUT:   %tuple.elem0.loc15_24.3: %empty_tuple.type = tuple_access %c.loc11_9.2, element0 [symbolic = %tuple.elem0.loc15_24.3 (constants.%tuple.elem0.eaf)]
// CHECK:STDOUT:   %tuple.elem1.loc15_24.3: %empty_tuple.type = tuple_access %c.loc11_9.2, element1 [symbolic = %tuple.elem1.loc15_24.3 (constants.%tuple.elem1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: %pattern_type.5b8 = symbolic_binding_pattern c, 1 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc11_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_31.1: %tuple.type.bcd = tuple_literal (%.loc11_26, %.loc11_30)
// CHECK:STDOUT:     %.loc11_20.1: type = splice_block %.loc11_20.5 [concrete = constants.%tuple.type.bcd] {
// CHECK:STDOUT:       %.loc11_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_19: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_20.2: %tuple.type.bcd = tuple_literal (%.loc11_15, %.loc11_19)
// CHECK:STDOUT:       %.loc11_20.3: type = converted %.loc11_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc11_20.4: type = converted %.loc11_19, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc11_20.5: type = converted %.loc11_20.2, constants.%tuple.type.bcd [concrete = constants.%tuple.type.bcd]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_tuple.loc11_26: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_31.2: %empty_tuple.type = converted %.loc11_26, %empty_tuple.loc11_26 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %empty_tuple.loc11_30: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_31.3: %empty_tuple.type = converted %.loc11_30, %empty_tuple.loc11_30 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple: %tuple.type.bcd = tuple_value (%.loc11_31.2, %.loc11_31.3) [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:     %.loc11_31.4: %tuple.type.bcd = converted %.loc11_31.1, %tuple [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:     %c.loc11_9.1: %tuple.type.bcd = bind_symbolic_name c, 1, %.loc11_31.4 [symbolic = %c.loc11_9.2 (constants.%c)]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a1.patt: %pattern_type.cb1 = binding_pattern a1 [concrete]
// CHECK:STDOUT:       %a1.var_patt: %pattern_type.cb1 = var_pattern %a1.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a1.var: ref %empty_tuple.type = var %a1.var_patt
// CHECK:STDOUT:     %a.ref: %empty_tuple.type = name_ref a, @C.%a
// CHECK:STDOUT:     %.loc13_18: init %empty_tuple.type = tuple_init () to %a1.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_5: init %empty_tuple.type = converted %a.ref, %.loc13_18 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     assign %a1.var, %.loc13_5
// CHECK:STDOUT:     %.loc13_14.1: type = splice_block %.loc13_14.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc13_14.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc13_14.3: type = converted %.loc13_14.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a1: ref %empty_tuple.type = bind_name a1, %a1.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %b1.patt: %pattern_type.559 = binding_pattern b1 [concrete]
// CHECK:STDOUT:       %b1.var_patt: %pattern_type.559 = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b1.var: ref %tuple.type.9fb = var %b1.var_patt
// CHECK:STDOUT:     %b.ref: %tuple.type.9fb = name_ref b, %b.loc10_8.2 [symbolic = %b.loc10_8.1 (constants.%b)]
// CHECK:STDOUT:     %tuple.elem0.loc14_21.1: %empty_tuple.type = tuple_access %b.ref, element0 [symbolic = %tuple.elem0.loc14_21.3 (constants.%tuple.elem0.af8)]
// CHECK:STDOUT:     %tuple.elem0.loc14_21.2: ref %empty_tuple.type = tuple_access %b1.var, element0
// CHECK:STDOUT:     %.loc14_21.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc14_21.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc14_21.2: init %empty_tuple.type = converted %tuple.elem0.loc14_21.1, %.loc14_21.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc14_21.3: init %tuple.type.9fb = tuple_init (%.loc14_21.2) to %b1.var [concrete = constants.%tuple.f41]
// CHECK:STDOUT:     %.loc14_5: init %tuple.type.9fb = converted %b.ref, %.loc14_21.3 [concrete = constants.%tuple.f41]
// CHECK:STDOUT:     assign %b1.var, %.loc14_5
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.4 [concrete = constants.%tuple.type.9fb] {
// CHECK:STDOUT:       %.loc14_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc14_17.2: %tuple.type.9fb = tuple_literal (%.loc14_15)
// CHECK:STDOUT:       %.loc14_17.3: type = converted %.loc14_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc14_17.4: type = converted %.loc14_17.2, constants.%tuple.type.9fb [concrete = constants.%tuple.type.9fb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b1: ref %tuple.type.9fb = bind_name b1, %b1.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c1.patt: %pattern_type.5b8 = binding_pattern c1 [concrete]
// CHECK:STDOUT:       %c1.var_patt: %pattern_type.5b8 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c1.var: ref %tuple.type.bcd = var %c1.var_patt
// CHECK:STDOUT:     %c.ref: %tuple.type.bcd = name_ref c, %c.loc11_9.1 [symbolic = %c.loc11_9.2 (constants.%c)]
// CHECK:STDOUT:     %tuple.elem0.loc15_24.1: %empty_tuple.type = tuple_access %c.ref, element0 [symbolic = %tuple.elem0.loc15_24.3 (constants.%tuple.elem0.eaf)]
// CHECK:STDOUT:     %tuple.elem0.loc15_24.2: ref %empty_tuple.type = tuple_access %c1.var, element0
// CHECK:STDOUT:     %.loc15_24.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc15_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_24.2: init %empty_tuple.type = converted %tuple.elem0.loc15_24.1, %.loc15_24.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple.elem1.loc15_24.1: %empty_tuple.type = tuple_access %c.ref, element1 [symbolic = %tuple.elem1.loc15_24.3 (constants.%tuple.elem1)]
// CHECK:STDOUT:     %tuple.elem1.loc15_24.2: ref %empty_tuple.type = tuple_access %c1.var, element1
// CHECK:STDOUT:     %.loc15_24.3: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc15_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_24.4: init %empty_tuple.type = converted %tuple.elem1.loc15_24.1, %.loc15_24.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_24.5: init %tuple.type.bcd = tuple_init (%.loc15_24.2, %.loc15_24.4) to %c1.var [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:     %.loc15_5: init %tuple.type.bcd = converted %c.ref, %.loc15_24.5 [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:     assign %c1.var, %.loc15_5
// CHECK:STDOUT:     %.loc15_20.1: type = splice_block %.loc15_20.5 [concrete = constants.%tuple.type.bcd] {
// CHECK:STDOUT:       %.loc15_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_19: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_20.2: %tuple.type.bcd = tuple_literal (%.loc15_15, %.loc15_19)
// CHECK:STDOUT:       %.loc15_20.3: type = converted %.loc15_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc15_20.4: type = converted %.loc15_19, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc15_20.5: type = converted %.loc15_20.2, constants.%tuple.type.bcd [concrete = constants.%tuple.type.bcd]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c1: ref %tuple.type.bcd = bind_name c1, %c1.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %d1.patt: %pattern_type.8c1 = binding_pattern d1 [concrete]
// CHECK:STDOUT:       %d1.var_patt: %pattern_type.8c1 = var_pattern %d1.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d1.var: ref %tuple.type.2d5 = var %d1.var_patt
// CHECK:STDOUT:     %d.ref: %tuple.type.2d5 = name_ref d, @C.%d
// CHECK:STDOUT:     %tuple.elem0.loc16_28.1: %empty_tuple.type = tuple_access %d.ref, element0
// CHECK:STDOUT:     %tuple.elem0.loc16_28.2: ref %empty_tuple.type = tuple_access %d1.var, element0
// CHECK:STDOUT:     %.loc16_28.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc16_28.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.2: init %empty_tuple.type = converted %tuple.elem0.loc16_28.1, %.loc16_28.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple.elem1.loc16_28.1: %empty_tuple.type = tuple_access %d.ref, element1
// CHECK:STDOUT:     %tuple.elem1.loc16_28.2: ref %empty_tuple.type = tuple_access %d1.var, element1
// CHECK:STDOUT:     %.loc16_28.3: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc16_28.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.4: init %empty_tuple.type = converted %tuple.elem1.loc16_28.1, %.loc16_28.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %tuple.elem2.loc16_28.1: %empty_tuple.type = tuple_access %d.ref, element2
// CHECK:STDOUT:     %tuple.elem2.loc16_28.2: ref %empty_tuple.type = tuple_access %d1.var, element2
// CHECK:STDOUT:     %.loc16_28.5: init %empty_tuple.type = tuple_init () to %tuple.elem2.loc16_28.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.6: init %empty_tuple.type = converted %tuple.elem2.loc16_28.1, %.loc16_28.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_28.7: init %tuple.type.2d5 = tuple_init (%.loc16_28.2, %.loc16_28.4, %.loc16_28.6) to %d1.var [concrete = constants.%tuple.7e4]
// CHECK:STDOUT:     %.loc16_5: init %tuple.type.2d5 = converted %d.ref, %.loc16_28.7 [concrete = constants.%tuple.7e4]
// CHECK:STDOUT:     assign %d1.var, %.loc16_5
// CHECK:STDOUT:     %.loc16_24.1: type = splice_block %.loc16_24.6 [concrete = constants.%tuple.type.2d5] {
// CHECK:STDOUT:       %.loc16_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_19: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_23: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_24.2: %tuple.type.2d5 = tuple_literal (%.loc16_15, %.loc16_19, %.loc16_23)
// CHECK:STDOUT:       %.loc16_24.3: type = converted %.loc16_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc16_24.4: type = converted %.loc16_19, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc16_24.5: type = converted %.loc16_23, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc16_24.6: type = converted %.loc16_24.2, constants.%tuple.type.2d5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d1: ref %tuple.type.2d5 = bind_name d1, %d1.var
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.bound.loc16: <bound method> = bound_method %d1.var, constants.%T.as.Destroy.impl.Op.dce
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.dce, @T.as.Destroy.impl.Op(constants.%tuple.type.2d5) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.234]
// CHECK:STDOUT:     %bound_method.loc16: <bound method> = bound_method %d1.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:     %addr.loc16: %ptr.7fe = addr_of %d1.var
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.call.loc16: init %empty_tuple.type = call %bound_method.loc16(%addr.loc16)
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.bound.loc15: <bound method> = bound_method %c1.var, constants.%T.as.Destroy.impl.Op.166
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.166, @T.as.Destroy.impl.Op(constants.%tuple.type.bcd) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.48a]
// CHECK:STDOUT:     %bound_method.loc15: <bound method> = bound_method %c1.var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:     %addr.loc15: %ptr.709 = addr_of %c1.var
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.call.loc15: init %empty_tuple.type = call %bound_method.loc15(%addr.loc15)
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.bound.loc14: <bound method> = bound_method %b1.var, constants.%T.as.Destroy.impl.Op.f19
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.specific_fn.3: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.f19, @T.as.Destroy.impl.Op(constants.%tuple.type.9fb) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.d2b]
// CHECK:STDOUT:     %bound_method.loc14: <bound method> = bound_method %b1.var, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:     %addr.loc14: %ptr.652 = addr_of %b1.var
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.call.loc14: init %empty_tuple.type = call %bound_method.loc14(%addr.loc14)
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.bound.loc13: <bound method> = bound_method %a1.var, constants.%T.as.Destroy.impl.Op.ea3
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.specific_fn.4: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.ea3, @T.as.Destroy.impl.Op(constants.%empty_tuple.type) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.393]
// CHECK:STDOUT:     %bound_method.loc13: <bound method> = bound_method %a1.var, %T.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:     %addr.loc13: %ptr.843 = addr_of %a1.var
// CHECK:STDOUT:     %T.as.Destroy.impl.Op.call.loc13: init %empty_tuple.type = call %bound_method.loc13(%addr.loc13)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%b) {
// CHECK:STDOUT:   %b.loc10_8.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_let_after.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT:   .x = <unexpected>.inst32.loc14_7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, <unexpected>.inst32.loc14_7
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Zero.patt: %pattern_type.7ce = symbolic_binding_pattern Zero, 0 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc5_14: type = splice_block %i32.loc5 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc5_20.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_20.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_20.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc5_20.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc5_20.2: %i32 = converted %int_0, %.loc5_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0, %.loc5_20.2 [symbolic = constants.%Zero]
// CHECK:STDOUT:   %Zero.ref: %i32 = name_ref Zero, %Zero [symbolic = constants.%Zero]
// CHECK:STDOUT:   return %Zero.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_in_block.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Zero.patt: %pattern_type.7ce = symbolic_binding_pattern Zero, 0 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_16: type = splice_block %i32.loc6 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc6: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_22.1: <bound method> = bound_method %int_0, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_22.2: <bound method> = bound_method %int_0, %specific_fn.loc6 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_22.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_22.2: %i32 = converted %int_0, %.loc6_22.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Zero: %i32 = bind_symbolic_name Zero, 0, %.loc6_22.2 [symbolic = constants.%Zero]
// CHECK:STDOUT:   %Zero.ref: %i32 = name_ref Zero, %Zero [symbolic = constants.%Zero]
// CHECK:STDOUT:   return %Zero.ref
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_11.1: <bound method> = bound_method %int_1, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_11.2: <bound method> = bound_method %int_1, %specific_fn.loc9 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_11.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_11.2: %i32 = converted %int_1, %.loc9_11.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   return %.loc9_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_in_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %Self.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ca5: type = pattern_type %impl.elem0 [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%I.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %return.patt: @I.F.%pattern_type (%pattern_type.ca5) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @I.F.%pattern_type (%pattern_type.ca5) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc6_13.2: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %impl.elem0.loc6_13.2 [symbolic = %impl.elem0.loc6_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %return.param: ref @I.F.%impl.elem0.loc6_13.1 (%impl.elem0) = out_param call_param0
// CHECK:STDOUT:     %return: ref @I.F.%impl.elem0.loc6_13.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %I.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %I.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const T:! type = %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc6_13.1: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc6_13.1 [symbolic = %pattern_type (constants.%pattern_type.ca5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @I.F.%impl.elem0.loc6_13.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem0.loc6_13.1 => constants.%impl.elem0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ca5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I: type = class_type @I [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = class_decl @I [concrete = constants.%I] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @I {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = binding_pattern T [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %T: type = bind_name T, %i32
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @I.%T
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%I
// CHECK:STDOUT:   .T = %T
// CHECK:STDOUT:   .F = %I.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_in_package_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .T = %T
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = binding_pattern T [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T: type = bind_name T, @__global_init.%i32
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.193: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness file.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   impl_decl @i32.as.Empty.impl [concrete] {} {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.impl_witness_table = impl_witness_table (), @i32.as.Empty.impl [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness %Empty.impl_witness_table [concrete = constants.%Empty.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.193]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.Empty.impl: %i32.loc6 as %Empty.ref {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Zero.patt: %pattern_type.7ce = binding_pattern Zero [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc11_14: type = splice_block %i32.loc11 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_20.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_20.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_20.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_20.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_20.2: %i32 = converted %int_0, %.loc11_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Zero: %i32 = bind_name Zero, %.loc11_20.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Zero
// CHECK:STDOUT:   witness = file.%Empty.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
