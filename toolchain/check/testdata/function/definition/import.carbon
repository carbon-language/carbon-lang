// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/import.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- fns.carbon

library "[[@TEST_NAME]]";

fn A() {}
fn B(b: i32) -> i32 { return b; }
fn C(c: (i32,)) -> {.c: i32} { return {.c = c.0}; }
fn D();

// --- extern.carbon

library "[[@TEST_NAME]]";

extern fn A();

// ============================================================================
// Test files
// ============================================================================

// --- basics.carbon

library "[[@TEST_NAME]]";

import library "fns";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));

// --- fail_def_ownership.carbon

library "[[@TEST_NAME]]";

import library "fns";

// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE+8]]:1: error: redeclaration of `fn A` is redundant [RedeclRedundant]
// CHECK:STDERR: fn A() {};
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: fns.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn A() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn A() {};
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE+8]]:1: error: redeclaration of `fn B` is redundant [RedeclRedundant]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE-14]]:1: in import [InImport]
// CHECK:STDERR: fns.carbon:5:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn B(b: i32) -> i32 { return b; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(b: i32) -> i32;

// --- fail_redecl_then_def.carbon

library "[[@TEST_NAME]]";

import library "extern";

// CHECK:STDERR: fail_redecl_then_def.carbon:[[@LINE+8]]:1: error: redeclarations of `fn A` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl_then_def.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: extern.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn A();

// CHECK:STDERR: fail_redecl_then_def.carbon:[[@LINE+8]]:1: error: redeclarations of `fn A` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: fn A() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redecl_then_def.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: extern.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn A() {}

// --- fail_mix_extern_decl.carbon

library "[[@TEST_NAME]]";

import library "fns";

// CHECK:STDERR: fail_mix_extern_decl.carbon:[[@LINE+7]]:1: error: redeclarations of `fn D` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_mix_extern_decl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: fns.carbon:7:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
extern fn D();

fn D() {}

// CHECK:STDOUT: --- fns.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_9: init type = call constants.%Int(%int_32.loc5_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_9.1: type = value_of_initializer %int.make_type_signed.loc5_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_9.2: type = converted %int.make_type_signed.loc5_9, %.loc5_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_17: init type = call constants.%Int(%int_32.loc5_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_17.1: type = value_of_initializer %int.make_type_signed.loc5_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_17.2: type = converted %int.make_type_signed.loc5_17, %.loc5_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %c.patt: %tuple.type.2 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.2 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %struct_type.c = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.c = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_10: init type = call constants.%Int(%int_32.loc6_10) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc6_10)
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_signed.loc6_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_signed.loc6_10, %.loc6_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.4: type = converted %.loc6_14.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %int_32.loc6_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_25: init type = call constants.%Int(%int_32.loc6_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_25.1: type = value_of_initializer %int.make_type_signed.loc6_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_25.2: type = converted %int.make_type_signed.loc6_25, %.loc6_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %c: %tuple.type.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   return %b.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.2) -> %struct_type.c {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %tuple.type.2 = name_ref c, %c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc6_48: %struct_type.c = struct_literal (%tuple.elem0)
// CHECK:STDOUT:   %struct: %struct_type.c = struct_value (%tuple.elem0)
// CHECK:STDOUT:   %.loc6_49: %struct_type.c = converted %.loc6_48, %struct
// CHECK:STDOUT:   return %.loc6_49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%int_1.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//fns, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//fns, inst+34, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//fns, inst+61, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//fns, inst+72, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:     .B = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc6_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_signed.loc7, %.loc7_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_signed.loc8, %.loc8_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.1) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc7: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc7: <specific function> = specific_function %Convert.bound.loc7, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc7: init %i32 = call %Convert.specific_fn.loc7(%int_1.loc7) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_16.1: %i32 = value_of_initializer %int.convert_checked.loc7 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_16.2: %i32 = converted %int_1.loc7, %.loc7_16.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc7_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc8_25.1: %tuple.type.2 = tuple_literal (%int_1.loc8)
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %Convert.specific_fn.loc8(%int_1.loc8) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_25.3: %i32 = converted %int_1.loc8, %.loc8_25.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.loc8_25.3) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_25.4: %tuple.type.1 = converted %.loc8_25.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc8_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_def_ownership.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//fns, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//fns, inst+72, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {} {
// CHECK:STDOUT:     %int_32.loc23_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc23_9: init type = call constants.%Int(%int_32.loc23_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_9.1: type = value_of_initializer %int.make_type_signed.loc23_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_9.2: type = converted %int.make_type_signed.loc23_9, %.loc23_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc23_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc23_17: init type = call constants.%Int(%int_32.loc23_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %int.make_type_signed.loc23_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %int.make_type_signed.loc23_17, %.loc23_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_then_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl.loc14
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %A.decl.loc14: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %A.decl.loc24: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mix_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//fns, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//fns, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//fns, inst+61, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:     .B = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = %D.decl.loc13
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %D.decl.loc13: %D.type = fn_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %D.decl.loc15: %D.type = fn_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
