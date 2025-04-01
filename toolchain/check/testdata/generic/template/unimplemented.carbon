// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon

// --- fail_todo_unimplemented_operator.carbon

library "[[@TEST_NAME]]";

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent expression.
fn F[template T:! type](x: T) -> i32 {
  // CHECK:STDERR: fail_todo_unimplemented_operator.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Mul` in type `<dependent type>` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return x.n * 3;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return x.n * 3;
}

// --- fail_todo_unimplemented_value.carbon

library "[[@TEST_NAME]]";

class C {
  var n: i32;
}

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent value where the type is concrete but determined through
// the template dependent value.
fn F(template c:! C) -> i32 {
  // CHECK:STDERR: fail_todo_unimplemented_value.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.Mul` in type `<dependent type>` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return c.n * 3;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return c.n * 3;
}

// --- fail_todo_unimplemented_convert.carbon

library "[[@TEST_NAME]]";

fn F[template T:! type](x: T) {
  // TODO: These diagnostics aren't very good, and we should only produce one error here.
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+8]]:3: error: member name of type `<dependent type>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   var v: T = 0;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+4]]:3: error: value of type `<dependent type>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   var v: T = 0;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  var v: T = 0;
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `T` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var w: i32 = x;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_unimplemented_convert.carbon:[[@LINE+4]]:3: note: type `T` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var w: i32 = x;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  var w: i32 = x;
}

// CHECK:STDOUT: --- fail_todo_unimplemented_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Mul = %Core.Mul
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.%T.loc6_15.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%T.loc6_15.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %T.loc6_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc6_15.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_15.1 [template = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc6_15.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_15.1: type) {
// CHECK:STDOUT:   %T.loc6_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.loc6_15.2 (%T) [template = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %.loc11_11.3: <instruction> = refine_type_action %x.ref, @F.%T.loc6_15.2 (%T) [template]
// CHECK:STDOUT:   %.loc11_11.4: <instruction> = access_member_action %.loc11_11.1, n [template]
// CHECK:STDOUT:   %.loc11_11.5: type = type_of_inst %.loc11_11.4 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_15.1: type](%x.param_patt: @F.%T.loc6_15.2 (%T)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc6_15.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc11_11.1: @F.%T.loc6_15.2 (%T) = splice_inst %.loc11_11.3
// CHECK:STDOUT:     %.loc11_11.2: @F.%.loc11_11.5 (@F.%.loc11_11.5) = splice_inst %.loc11_11.4
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %c: %C = bind_symbolic_name c, 0, template [template]
// CHECK:STDOUT:   %c.patt: %C = symbolic_binding_pattern c, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Mul = %Core.Mul
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt.loc11_15.1: %C = symbolic_binding_pattern c, 0, template [template = %c.patt.loc11_15.2 (constants.%c.patt)]
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c.loc11_15.1: %C = bind_symbolic_name c, 0, template [template = %c.loc11_15.2 (constants.%c)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_8: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %C.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%c.loc11_15.1: %C) {
// CHECK:STDOUT:   %c.loc11_15.2: %C = bind_symbolic_name c, 0, template [template = %c.loc11_15.2 (constants.%c)]
// CHECK:STDOUT:   %c.patt.loc11_15.2: %C = symbolic_binding_pattern c, 0, template [template = %c.patt.loc11_15.2 (constants.%c.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc16_11.2: <instruction> = access_member_action %c.ref, n [template]
// CHECK:STDOUT:   %.loc16_11.3: type = type_of_inst %.loc16_11.2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.patt.loc11_15.1: %C) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: %C = name_ref c, %c.loc11_15.1 [template = %c.loc11_15.2 (constants.%c)]
// CHECK:STDOUT:     %.loc16_11.1: @F.%.loc16_11.3 (@F.%.loc16_11.3) = splice_inst %.loc16_11.2
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%c) {
// CHECK:STDOUT:   %c.loc11_15.2 => constants.%c
// CHECK:STDOUT:   %c.patt.loc11_15.2 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d628ce.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.%T.loc4_15.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%T.loc4_15.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_15.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc4_15.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.loc4_15.2 (%T) [template = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc4_15.2)> [template = %ImplicitAs.type (constants.%ImplicitAs.type.d628ce.2)]
// CHECK:STDOUT:   %.loc14_3.4: <instruction> = access_member_action %ImplicitAs.type, Convert [template]
// CHECK:STDOUT:   %.loc14_3.5: type = type_of_inst %.loc14_3.4 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_15.1: type](%x.param_patt: @F.%T.loc4_15.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%T.loc4_15.2 (%T) = binding_pattern v
// CHECK:STDOUT:       %.loc14_3.1: @F.%T.loc4_15.2 (%T) = var_pattern %v.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%T.loc4_15.2 (%T) = var v
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc14_3.2: @F.%.loc14_3.5 (@F.%.loc14_3.5) = splice_inst %.loc14_3.4
// CHECK:STDOUT:     %.loc14_3.3: @F.%T.loc4_15.2 (%T) = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %v.var, <error>
// CHECK:STDOUT:     %T.ref.loc14: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %v: ref @F.%T.loc4_15.2 (%T) = bind_name v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %i32 = binding_pattern w
// CHECK:STDOUT:       %.loc22_3.1: %i32 = var_pattern %w.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref %i32 = var w
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_15.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc22_3.2: %i32 = converted %x.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %w.var, <error>
// CHECK:STDOUT:     %.loc22_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
