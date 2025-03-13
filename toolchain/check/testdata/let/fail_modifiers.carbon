// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected let b: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected let b: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default let c: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default let c: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: final let d: i32 = 1;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
final let d: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `virtual` not allowed on `let` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual let e: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:9: error: `final` not allowed on declaration with `default` [ModifierNotAllowedWith]
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default final let f: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:9: error: `default` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default default let g: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:11: error: `private` not allowed on declaration with `protected` [ModifierNotAllowedWith]
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected private let h: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:11: error: `protected` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected protected let i: i32 = 1;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .b [protected] = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .g = %g
// CHECK:STDOUT:     .h [protected] = %h
// CHECK:STDOUT:     .i [protected] = %i
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_18: type = splice_block %i32.loc15 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_24.1: <bound method> = bound_method @__global_init.%int_1.loc15, %impl.elem0.loc15 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_24.2: <bound method> = bound_method @__global_init.%int_1.loc15, %specific_fn.loc15 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %bound_method.loc15_24.2(@__global_init.%int_1.loc15) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_24.1: %i32 = value_of_initializer %int.convert_checked.loc15 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_24.2: %i32 = converted @__global_init.%int_1.loc15, %.loc15_24.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc15_24.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %i32 = binding_pattern c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_16: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc21: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_22.1: <bound method> = bound_method @__global_init.%int_1.loc21, %impl.elem0.loc21 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_22.2: <bound method> = bound_method @__global_init.%int_1.loc21, %specific_fn.loc21 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %bound_method.loc21_22.2(@__global_init.%int_1.loc21) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc21_22.1: %i32 = value_of_initializer %int.convert_checked.loc21 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc21_22.2: %i32 = converted @__global_init.%int_1.loc21, %.loc21_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc21_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %i32 = binding_pattern d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_14: type = splice_block %i32.loc27 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc27: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_20.1: <bound method> = bound_method @__global_init.%int_1.loc27, %impl.elem0.loc27 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_20.2: <bound method> = bound_method @__global_init.%int_1.loc27, %specific_fn.loc27 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %bound_method.loc27_20.2(@__global_init.%int_1.loc27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc27_20.1: %i32 = value_of_initializer %int.convert_checked.loc27 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc27_20.2: %i32 = converted @__global_init.%int_1.loc27, %.loc27_20.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %d: %i32 = bind_name d, %.loc27_20.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %i32 = binding_pattern e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc33_16: type = splice_block %i32.loc33 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc33: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc33_22.1: <bound method> = bound_method @__global_init.%int_1.loc33, %impl.elem0.loc33 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_22.2: <bound method> = bound_method @__global_init.%int_1.loc33, %specific_fn.loc33 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc33: init %i32 = call %bound_method.loc33_22.2(@__global_init.%int_1.loc33) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33_22.1: %i32 = value_of_initializer %int.convert_checked.loc33 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33_22.2: %i32 = converted @__global_init.%int_1.loc33, %.loc33_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %e: %i32 = bind_name e, %.loc33_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %i32 = binding_pattern f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc46_22: type = splice_block %i32.loc46 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc46: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc46: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc46: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc46_28.1: <bound method> = bound_method @__global_init.%int_1.loc46, %impl.elem0.loc46 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc46: <specific function> = specific_function %impl.elem0.loc46, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc46_28.2: <bound method> = bound_method @__global_init.%int_1.loc46, %specific_fn.loc46 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc46: init %i32 = call %bound_method.loc46_28.2(@__global_init.%int_1.loc46) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc46_28.1: %i32 = value_of_initializer %int.convert_checked.loc46 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc46_28.2: %i32 = converted @__global_init.%int_1.loc46, %.loc46_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %f: %i32 = bind_name f, %.loc46_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %i32 = binding_pattern g
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc59_24: type = splice_block %i32.loc59 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc59: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc59: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc59: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc59_30.1: <bound method> = bound_method @__global_init.%int_1.loc59, %impl.elem0.loc59 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc59: <specific function> = specific_function %impl.elem0.loc59, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc59_30.2: <bound method> = bound_method @__global_init.%int_1.loc59, %specific_fn.loc59 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc59: init %i32 = call %bound_method.loc59_30.2(@__global_init.%int_1.loc59) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc59_30.1: %i32 = value_of_initializer %int.convert_checked.loc59 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc59_30.2: %i32 = converted @__global_init.%int_1.loc59, %.loc59_30.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %g: %i32 = bind_name g, %.loc59_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %i32 = binding_pattern h
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc72_26: type = splice_block %i32.loc72 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc72: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc72: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc72: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc72_32.1: <bound method> = bound_method @__global_init.%int_1.loc72, %impl.elem0.loc72 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc72: <specific function> = specific_function %impl.elem0.loc72, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc72_32.2: <bound method> = bound_method @__global_init.%int_1.loc72, %specific_fn.loc72 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc72: init %i32 = call %bound_method.loc72_32.2(@__global_init.%int_1.loc72) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc72_32.1: %i32 = value_of_initializer %int.convert_checked.loc72 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc72_32.2: %i32 = converted @__global_init.%int_1.loc72, %.loc72_32.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %h: %i32 = bind_name h, %.loc72_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %i32 = binding_pattern i
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc85_28: type = splice_block %i32.loc85 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc85: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc85: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc85: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc85_34.1: <bound method> = bound_method @__global_init.%int_1.loc85, %impl.elem0.loc85 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc85: <specific function> = specific_function %impl.elem0.loc85, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc85_34.2: <bound method> = bound_method @__global_init.%int_1.loc85, %specific_fn.loc85 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc85: init %i32 = call %bound_method.loc85_34.2(@__global_init.%int_1.loc85) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc85_34.1: %i32 = value_of_initializer %int.convert_checked.loc85 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc85_34.2: %i32 = converted @__global_init.%int_1.loc85, %.loc85_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %i: %i32 = bind_name i, %.loc85_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc33: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc46: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc59: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc72: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc85: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
