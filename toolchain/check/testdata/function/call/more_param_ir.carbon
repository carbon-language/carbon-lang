// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/more_param_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/more_param_ir.carbon

fn Foo(a: i32, b: i32) {}

fn Main() {
  var x: (i32,) = (1,);
  // Generates multiple IR instructions for the first parameter.
  Foo(x.0, 6);
}

// CHECK:STDOUT: --- more_param_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.0e4: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.0e4 = tuple_value (%int_1.c60) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.b1c: <bound method> = bound_method %int_6.462, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b26: <specific function> = specific_function %Convert.bound.b1c, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.b03: %i32 = int_value 6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_11: type = splice_block %i32.loc11_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc11_19: type = splice_block %i32.loc11_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a.param_patt: %i32, %b.param_patt: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %tuple.type.0e4 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.0e4 = bind_name x, %x.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_22.1: %tuple.type.985 = tuple_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0.loc14: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc14: <bound method> = bound_method %int_1, %impl.elem0.loc14 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc14: <specific function> = specific_function %Convert.bound.loc14, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %Convert.specific_fn.loc14(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc14_22.2: init %i32 = converted %int_1, %int.convert_checked.loc14 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc14_22.3: init %tuple.type.0e4 = tuple_init (%.loc14_22.2) to %x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_23: init %tuple.type.0e4 = converted %.loc14_22.1, %.loc14_22.3 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var, %.loc14_23
// CHECK:STDOUT:   %Foo.ref: %Foo.type = name_ref Foo, file.%Foo.decl [template = constants.%Foo]
// CHECK:STDOUT:   %x.ref: ref %tuple.type.0e4 = name_ref x, %x
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.462]
// CHECK:STDOUT:   %.loc16_8: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0.loc16: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc16: <bound method> = bound_method %int_6, %impl.elem0.loc16 [template = constants.%Convert.bound.b1c]
// CHECK:STDOUT:   %Convert.specific_fn.loc16: <specific function> = specific_function %Convert.bound.loc16, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b26]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %Convert.specific_fn.loc16(%int_6) [template = constants.%int_6.b03]
// CHECK:STDOUT:   %.loc16_12.1: %i32 = value_of_initializer %int.convert_checked.loc16 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %.loc16_12.2: %i32 = converted %int_6, %.loc16_12.1 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %Foo.call: init %empty_tuple.type = call %Foo.ref(%.loc16_8, %.loc16_12.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
