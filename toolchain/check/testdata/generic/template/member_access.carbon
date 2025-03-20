// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/member_access.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

fn F[template T:! type](x: T) -> i32 {
  return x.n;
}

class C {
  var n: i32;
}

fn Test1(c: C) {
  F(c);
}

fn Test2(x: {.m: i32, .n: i32}) {
  F(x);
}

// --- fail_no_such_member.carbon

library "[[@TEST_NAME]]";

fn F[template T:! type](x: T) -> i32 {
  // CHECK:STDERR: fail_no_such_member.carbon:[[@LINE+3]]:10: error: member name `n` not found in `D` [MemberNameNotFoundInScope]
  // CHECK:STDERR:   return x.n;
  // CHECK:STDERR:          ^~~
  return x.n;
}

class D {
  var m: i32;
}

fn Test(d: D) {
  // CHECK:STDERR: fail_no_such_member.carbon:[[@LINE+4]]:3: note: in `F(D)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   F(d);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  F(d);
}

// --- fail_member_wrong_type.carbon

library "[[@TEST_NAME]]";

fn F[template T:! type](x: T) -> i32 {
  // CHECK:STDERR: fail_member_wrong_type.carbon:[[@LINE+6]]:3: error: cannot implicitly convert value of type `F` to `i32` [ConversionFailure]
  // CHECK:STDERR:   return x.n;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_member_wrong_type.carbon:[[@LINE+3]]:3: note: type `F` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return x.n;
  // CHECK:STDERR:   ^~~~~~~~~~~
  return x.n;
}

class E {
  class F {}
  var n: F;
}

fn Test(e: E) {
  // CHECK:STDERR: fail_member_wrong_type.carbon:[[@LINE+4]]:3: note: in `F(E)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   F(e);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  F(e);
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Test1.type: type = fn_type @Test1 [concrete]
// CHECK:STDOUT:   %Test1: %Test1.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %struct_type.m.n: type = struct_type {.m: %i32, .n: %i32} [concrete]
// CHECK:STDOUT:   %Test2.type: type = fn_type @Test2 [concrete]
// CHECK:STDOUT:   %Test2: %Test2.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.92d: <specific function> = specific_function %F, @F(%struct_type.m.n) [concrete]
// CHECK:STDOUT:   %inst.as_compatible.c0a: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fbe: %C = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.d20: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.39c: %i32 = splice_block %.e83 {
// CHECK:STDOUT:       %n.ref: %C.elem = name_ref n, @C.%.loc9_8 [concrete = @C.%.loc9_8]
// CHECK:STDOUT:       %.d20: ref %i32 = class_element_access %.fbe, element0
// CHECK:STDOUT:       %.e83: %i32 = bind_value %.d20
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.435: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.460: %i32 = splice_block %.39c {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.622: <witness> = complete_type_witness %struct_type.m.n [concrete]
// CHECK:STDOUT:   %inst.as_compatible.8f8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.61c: %struct_type.m.n = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.struct_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fd5: %i32 = struct_access %.61c, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.57b: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fbf: %i32 = splice_block %.fd5 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Test1 = %Test1.decl
// CHECK:STDOUT:     .Test2 = %Test2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.%T.loc4_15.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%T.loc4_15.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_15.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc4_15.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Test1.decl: %Test1.type = fn_decl @Test1 [concrete = constants.%Test1] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test2.decl: %Test2.type = fn_decl @Test2 [concrete = constants.%Test2] {
// CHECK:STDOUT:     %x.patt: %struct_type.m.n = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %struct_type.m.n = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %struct_type.m.n = value_param call_param0
// CHECK:STDOUT:     %.loc16: type = splice_block %struct_type.m.n [concrete = constants.%struct_type.m.n] {
// CHECK:STDOUT:       %int_32.loc16_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc16_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.m.n: type = struct_type {.m: %i32, .n: %i32} [concrete = constants.%struct_type.m.n]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %struct_type.m.n = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc9_8: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc9_3: %C.elem = var_pattern %.loc9_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.loc4_15.2 (%T) [template = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %.loc5_11.3: <instruction> = refine_type_action %x.ref, @F.%T.loc4_15.2 (%T) [template]
// CHECK:STDOUT:   %.loc5_11.4: <instruction> = access_member_action %.loc5_11.1, n [template]
// CHECK:STDOUT:   %.loc5_11.5: type = type_of_inst %.loc5_11.4 [template]
// CHECK:STDOUT:   %.loc5_13.2: <instruction> = convert_to_value_action %.loc5_11.2, %i32 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_15.1: type](%x.param_patt: @F.%T.loc4_15.2 (%T)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_15.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc5_11.1: @F.%T.loc4_15.2 (%T) = splice_inst %.loc5_11.3
// CHECK:STDOUT:     %.loc5_11.2: @F.%.loc5_11.5 (@F.%.loc5_11.5) = splice_inst %.loc5_11.4
// CHECK:STDOUT:     %.loc5_13.1: %i32 = splice_inst %.loc5_13.2
// CHECK:STDOUT:     return %.loc5_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test1(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test2(%x.param_patt: %struct_type.m.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %x.ref: %struct_type.m.n = name_ref x, %x
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct_type.m.n) [concrete = constants.%F.specific_fn.92d]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.54b
// CHECK:STDOUT:   %.loc5_11.3 => constants.%inst.as_compatible.c0a
// CHECK:STDOUT:   %.loc5_11.4 => constants.%inst.splice_block.d20
// CHECK:STDOUT:   %.loc5_11.5 => constants.%i32
// CHECK:STDOUT:   %.loc5_13.2 => constants.%inst.splice_block.435
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct_type.m.n) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%struct_type.m.n
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%struct_type.m.n
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.622
// CHECK:STDOUT:   %.loc5_11.3 => constants.%inst.as_compatible.8f8
// CHECK:STDOUT:   %.loc5_11.4 => constants.%inst.struct_access
// CHECK:STDOUT:   %.loc5_11.5 => constants.%i32
// CHECK:STDOUT:   %.loc5_13.2 => constants.%inst.splice_block.57b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_such_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.m: type = struct_type {.m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.218: <witness> = complete_type_witness %struct_type.m [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%D) [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.63b: %D = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.432: <error> = splice_block <error> [concrete = <error>] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.%T.loc4_15.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%T.loc4_15.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_15.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc4_15.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc12_8: %D.elem = field_decl m, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc12_3: %D.elem = var_pattern %.loc12_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %D.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.m [concrete = constants.%complete_type.218]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .m = %.loc12_8
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.loc4_15.2 (%T) [template = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %.loc8_11.3: <instruction> = refine_type_action %x.ref, @F.%T.loc4_15.2 (%T) [template]
// CHECK:STDOUT:   %.loc8_11.4: <instruction> = access_member_action %.loc8_11.1, n [template]
// CHECK:STDOUT:   %.loc8_11.5: type = type_of_inst %.loc8_11.4 [template]
// CHECK:STDOUT:   %.loc8_13.2: <instruction> = convert_to_value_action %.loc8_11.2, %i32 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_15.1: type](%x.param_patt: @F.%T.loc4_15.2 (%T)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_15.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc8_11.1: @F.%T.loc4_15.2 (%T) = splice_inst %.loc8_11.3
// CHECK:STDOUT:     %.loc8_11.2: @F.%.loc8_11.5 (@F.%.loc8_11.5) = splice_inst %.loc8_11.4
// CHECK:STDOUT:     %.loc8_13.1: %i32 = splice_inst %.loc8_13.2
// CHECK:STDOUT:     return %.loc8_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%D) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.218
// CHECK:STDOUT:   %.loc8_11.3 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.loc8_11.4 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc8_11.5 => <error>
// CHECK:STDOUT:   %.loc8_13.2 => constants.%inst.splice_block
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_wrong_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %F.a8a: type = class_type @F.2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %E.elem: type = unbound_element_type %E, %F.a8a [concrete]
// CHECK:STDOUT:   %struct_type.n.e4c: type = struct_type {.n: %F.a8a} [concrete]
// CHECK:STDOUT:   %complete_type.5ba: <witness> = complete_type_witness %struct_type.n.e4c [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.c41, @F.1(%E) [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a66: %E = as_compatible @F.1.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.2a2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.6c9: %F.a8a = splice_block %.e29 {
// CHECK:STDOUT:       %n.ref: %E.elem = name_ref n, @E.%.loc16_8 [concrete = @E.%.loc16_8]
// CHECK:STDOUT:       %.999: ref %F.a8a = class_element_access %.a66, element0
// CHECK:STDOUT:       %.e29: %F.a8a = bind_value %.999
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.a04: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.31f: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %.fd6: %i32 = converted %.6c9, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F.1 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.1.%T.loc4_15.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.1.%T.loc4_15.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.1.%T.loc4_15.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x: @F.1.%T.loc4_15.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %e.patt: %E = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %E = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.2 [concrete = constants.%F.a8a] {} {}
// CHECK:STDOUT:   %.loc16_8: %E.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc16_3: %E.elem = var_pattern %.loc16_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %E.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.e4c [concrete = constants.%complete_type.5ba]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .n = %.loc16_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%F.a8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.1.%T.loc4_15.2 (%T) [template = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %.loc11_11.3: <instruction> = refine_type_action %x.ref, @F.1.%T.loc4_15.2 (%T) [template]
// CHECK:STDOUT:   %.loc11_11.4: <instruction> = access_member_action %.loc11_11.1, n [template]
// CHECK:STDOUT:   %.loc11_11.5: type = type_of_inst %.loc11_11.4 [template]
// CHECK:STDOUT:   %.loc11_13.2: <instruction> = convert_to_value_action %.loc11_11.2, %i32 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_15.1: type](%x.param_patt: @F.1.%T.loc4_15.2 (%T)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.1.%T.loc4_15.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc11_11.1: @F.1.%T.loc4_15.2 (%T) = splice_inst %.loc11_11.3
// CHECK:STDOUT:     %.loc11_11.2: @F.1.%.loc11_11.5 (@F.1.%.loc11_11.5) = splice_inst %.loc11_11.4
// CHECK:STDOUT:     %.loc11_13.1: %i32 = splice_inst %.loc11_13.2
// CHECK:STDOUT:     return %.loc11_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%e.param_patt: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F.c41]
// CHECK:STDOUT:   %e.ref: %E = name_ref e, %e
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F.1(constants.%E) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%e.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%E) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%E
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%E
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5ba
// CHECK:STDOUT:   %.loc11_11.3 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.loc11_11.4 => constants.%inst.splice_block.2a2
// CHECK:STDOUT:   %.loc11_11.5 => constants.%F.a8a
// CHECK:STDOUT:   %.loc11_13.2 => constants.%inst.splice_block.a04
// CHECK:STDOUT: }
// CHECK:STDOUT:
