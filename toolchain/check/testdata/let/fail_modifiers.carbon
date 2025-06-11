// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_18: type = splice_block %i32.loc18 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc18: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_24.1: <bound method> = bound_method @__global_init.%int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_24.2: <bound method> = bound_method @__global_init.%int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %bound_method.loc18_24.2(@__global_init.%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_24.1: %i32 = value_of_initializer %int.convert_checked.loc18 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_24.2: %i32 = converted @__global_init.%int_1.loc18, %.loc18_24.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc18_24.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_16: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc24: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_22.1: <bound method> = bound_method @__global_init.%int_1.loc24, %impl.elem0.loc24 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_22.2: <bound method> = bound_method @__global_init.%int_1.loc24, %specific_fn.loc24 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc24: init %i32 = call %bound_method.loc24_22.2(@__global_init.%int_1.loc24) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc24_22.1: %i32 = value_of_initializer %int.convert_checked.loc24 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc24_22.2: %i32 = converted @__global_init.%int_1.loc24, %.loc24_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc24_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc30_14: type = splice_block %i32.loc30 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc30: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc30_20.1: <bound method> = bound_method @__global_init.%int_1.loc30, %impl.elem0.loc30 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_20.2: <bound method> = bound_method @__global_init.%int_1.loc30, %specific_fn.loc30 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc30: init %i32 = call %bound_method.loc30_20.2(@__global_init.%int_1.loc30) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc30_20.1: %i32 = value_of_initializer %int.convert_checked.loc30 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc30_20.2: %i32 = converted @__global_init.%int_1.loc30, %.loc30_20.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %d: %i32 = bind_name d, %.loc30_20.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.7ce = binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36_16: type = splice_block %i32.loc36 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc36: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc36: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc36: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc36_22.1: <bound method> = bound_method @__global_init.%int_1.loc36, %impl.elem0.loc36 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc36: <specific function> = specific_function %impl.elem0.loc36, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_22.2: <bound method> = bound_method @__global_init.%int_1.loc36, %specific_fn.loc36 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc36: init %i32 = call %bound_method.loc36_22.2(@__global_init.%int_1.loc36) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc36_22.1: %i32 = value_of_initializer %int.convert_checked.loc36 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc36_22.2: %i32 = converted @__global_init.%int_1.loc36, %.loc36_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %e: %i32 = bind_name e, %.loc36_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.7ce = binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc49_22: type = splice_block %i32.loc49 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc49: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc49_28.1: <bound method> = bound_method @__global_init.%int_1.loc49, %impl.elem0.loc49 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc49: <specific function> = specific_function %impl.elem0.loc49, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_28.2: <bound method> = bound_method @__global_init.%int_1.loc49, %specific_fn.loc49 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc49: init %i32 = call %bound_method.loc49_28.2(@__global_init.%int_1.loc49) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc49_28.1: %i32 = value_of_initializer %int.convert_checked.loc49 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc49_28.2: %i32 = converted @__global_init.%int_1.loc49, %.loc49_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %f: %i32 = bind_name f, %.loc49_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.7ce = binding_pattern g [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc62_24: type = splice_block %i32.loc62 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc62: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc62: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc62: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc62_30.1: <bound method> = bound_method @__global_init.%int_1.loc62, %impl.elem0.loc62 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc62: <specific function> = specific_function %impl.elem0.loc62, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_30.2: <bound method> = bound_method @__global_init.%int_1.loc62, %specific_fn.loc62 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc62: init %i32 = call %bound_method.loc62_30.2(@__global_init.%int_1.loc62) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc62_30.1: %i32 = value_of_initializer %int.convert_checked.loc62 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc62_30.2: %i32 = converted @__global_init.%int_1.loc62, %.loc62_30.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %g: %i32 = bind_name g, %.loc62_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.7ce = binding_pattern h [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc75_26: type = splice_block %i32.loc75 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc75: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc75: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc75: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc75_32.1: <bound method> = bound_method @__global_init.%int_1.loc75, %impl.elem0.loc75 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc75: <specific function> = specific_function %impl.elem0.loc75, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc75_32.2: <bound method> = bound_method @__global_init.%int_1.loc75, %specific_fn.loc75 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc75: init %i32 = call %bound_method.loc75_32.2(@__global_init.%int_1.loc75) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc75_32.1: %i32 = value_of_initializer %int.convert_checked.loc75 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc75_32.2: %i32 = converted @__global_init.%int_1.loc75, %.loc75_32.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %h: %i32 = bind_name h, %.loc75_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc88_28: type = splice_block %i32.loc88 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc88: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc88: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc88: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc88_34.1: <bound method> = bound_method @__global_init.%int_1.loc88, %impl.elem0.loc88 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc88: <specific function> = specific_function %impl.elem0.loc88, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc88_34.2: <bound method> = bound_method @__global_init.%int_1.loc88, %specific_fn.loc88 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc88: init %i32 = call %bound_method.loc88_34.2(@__global_init.%int_1.loc88) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc88_34.1: %i32 = value_of_initializer %int.convert_checked.loc88 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc88_34.2: %i32 = converted @__global_init.%int_1.loc88, %.loc88_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %i: %i32 = bind_name i, %.loc88_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc30: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc36: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc62: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc75: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc88: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
