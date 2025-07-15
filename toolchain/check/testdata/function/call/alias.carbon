// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/alias.carbon

fn A() -> () { return (); }

alias B = A;

fn Main() {
  var b: () = B();
}

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.1, @Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.ea3, @Op.1(%empty_tuple.type) [concrete]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %B: %A.type = bind_alias B, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_24: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_25: %empty_tuple.type = converted %.loc15_24, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc15_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var %b.var_patt
// CHECK:STDOUT:   %B.ref: %A.type = name_ref B, file.%B [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %B.ref()
// CHECK:STDOUT:   assign %b.var, %A.call
// CHECK:STDOUT:   %.loc20_11.1: type = splice_block %.loc20_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc20_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc20_11.3: type = converted %.loc20_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %b.var, constants.%Op.ea3
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.ea3, @Op.1(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %b.var, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.843 = addr_of %b.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
