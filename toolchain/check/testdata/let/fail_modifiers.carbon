// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_modifiers.carbon

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ protected let b: i32 = 1;
// CHECK:STDERR:        · ━━━━┳━━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `protected` not allowed; requires class scope
// CHECK:STDERR:        ·        ┃        [ModifierProtectedNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
protected let b: i32 = 1;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ default let c: i32 = 1;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `default` not allowed; requires interface scope
// CHECK:STDERR:        ·       ┃        [ModifierRequiresInterface]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
default let c: i32 = 1;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ final let d: i32 = 1;
// CHECK:STDERR:        · ━━┳━━
// CHECK:STDERR:        ·   ┗━━┫ error: `final` not allowed; requires interface scope
// CHECK:STDERR:        ·      ┃        [ModifierRequiresInterface]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
final let d: i32 = 1;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ virtual let e: i32 = 1;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `virtual` not allowed on `let` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
virtual let e: i32 = 1;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+15]]:9
// CHECK:STDERR: ->     │ default final let f: i32 = 1;
// CHECK:STDERR:        · ───┬─── ━━┳━━
// CHECK:STDERR:        ·    │      ┗━━┫ error: `final` not allowed on declaration with `default`
// CHECK:STDERR:        ·    │         ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·    ╰───────────┤ `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ default final let f: i32 = 1;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `default` not allowed; requires interface scope
// CHECK:STDERR:        ·       ┃        [ModifierRequiresInterface]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
default final let f: i32 = 1;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+14]]:9
// CHECK:STDERR: ->     │ default default let g: i32 = 1;
// CHECK:STDERR:        · ───┬─── ━━━┳━━━
// CHECK:STDERR:        ·    │       ┗━━┫ error: `default` repeated on declaration [ModifierRepeated]
// CHECK:STDERR:        ·    ╰────────────┤ `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ default default let g: i32 = 1;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `default` not allowed; requires interface scope
// CHECK:STDERR:        ·       ┃        [ModifierRequiresInterface]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
default default let g: i32 = 1;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+15]]:11
// CHECK:STDERR: ->     │ protected private let h: i32 = 1;
// CHECK:STDERR:        · ────┬──── ━━━┳━━━
// CHECK:STDERR:        ·     │        ┗━━┫ error: `private` not allowed on declaration with `protected`
// CHECK:STDERR:        ·     │           ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·     ╰─────────────┤ `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ protected private let h: i32 = 1;
// CHECK:STDERR:        · ━━━━┳━━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `protected` not allowed; requires class scope
// CHECK:STDERR:        ·        ┃        [ModifierProtectedNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
protected private let h: i32 = 1;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+14]]:11
// CHECK:STDERR: ->     │ protected protected let i: i32 = 1;
// CHECK:STDERR:        · ────┬──── ━━━━┳━━━━
// CHECK:STDERR:        ·     │         ┗━━┫ error: `protected` repeated on declaration [ModifierRepeated]
// CHECK:STDERR:        ·     ╰──────────────┤ `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ protected protected let i: i32 = 1;
// CHECK:STDERR:        · ━━━━┳━━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `protected` not allowed; requires class scope
// CHECK:STDERR:        ·        ┃        [ModifierProtectedNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
protected protected let i: i32 = 1;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.d94 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.d94 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %g.patt: %pattern_type.d94 = value_binding_pattern g [concrete]
// CHECK:STDOUT:   %h.patt: %pattern_type.d94 = value_binding_pattern h [concrete]
// CHECK:STDOUT:   %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc22: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc22_24.1: <bound method> = bound_method @__global_init.%int_1.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_24.2: <bound method> = bound_method @__global_init.%int_1.loc22, %specific_fn.loc22 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22_24.2(@__global_init.%int_1.loc22) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc22_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc22_24.2: %i32 = converted @__global_init.%int_1.loc22, %.loc22_24.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc22_24.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc31: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc31_22.1: <bound method> = bound_method @__global_init.%int_1.loc31, %impl.elem0.loc31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_22.2: <bound method> = bound_method @__global_init.%int_1.loc31, %specific_fn.loc31 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31: init %i32 = call %bound_method.loc31_22.2(@__global_init.%int_1.loc31) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc31_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc31_22.2: %i32 = converted @__global_init.%int_1.loc31, %.loc31_22.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: %i32 = wrapper_binding c, %.loc31_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc40: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc40_20.1: <bound method> = bound_method @__global_init.%int_1.loc40, %impl.elem0.loc40 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc40: <specific function> = specific_function %impl.elem0.loc40, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40_20.2: <bound method> = bound_method @__global_init.%int_1.loc40, %specific_fn.loc40 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc40: init %i32 = call %bound_method.loc40_20.2(@__global_init.%int_1.loc40) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc40_20.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc40 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc40_20.2: %i32 = converted @__global_init.%int_1.loc40, %.loc40_20.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %d: %i32 = wrapper_binding d, %.loc40_20.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc49: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc49_22.1: <bound method> = bound_method @__global_init.%int_1.loc49, %impl.elem0.loc49 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc49: <specific function> = specific_function %impl.elem0.loc49, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_22.2: <bound method> = bound_method @__global_init.%int_1.loc49, %specific_fn.loc49 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49: init %i32 = call %bound_method.loc49_22.2(@__global_init.%int_1.loc49) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc49_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc49_22.2: %i32 = converted @__global_init.%int_1.loc49, %.loc49_22.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %e: %i32 = wrapper_binding e, %.loc49_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.d94 = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc66: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc66_28.1: <bound method> = bound_method @__global_init.%int_1.loc66, %impl.elem0.loc66 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc66: <specific function> = specific_function %impl.elem0.loc66, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc66_28.2: <bound method> = bound_method @__global_init.%int_1.loc66, %specific_fn.loc66 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc66: init %i32 = call %bound_method.loc66_28.2(@__global_init.%int_1.loc66) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc66_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc66 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc66_28.2: %i32 = converted @__global_init.%int_1.loc66, %.loc66_28.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc66: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %f: %i32 = wrapper_binding f, %.loc66_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.d94 = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc82: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc82_30.1: <bound method> = bound_method @__global_init.%int_1.loc82, %impl.elem0.loc82 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc82: <specific function> = specific_function %impl.elem0.loc82, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc82_30.2: <bound method> = bound_method @__global_init.%int_1.loc82, %specific_fn.loc82 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc82: init %i32 = call %bound_method.loc82_30.2(@__global_init.%int_1.loc82) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc82_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc82 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc82_30.2: %i32 = converted @__global_init.%int_1.loc82, %.loc82_30.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc82: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %g: %i32 = wrapper_binding g, %.loc82_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.d94 = value_binding_pattern g [concrete = constants.%g.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc99: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc99_32.1: <bound method> = bound_method @__global_init.%int_1.loc99, %impl.elem0.loc99 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc99: <specific function> = specific_function %impl.elem0.loc99, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc99_32.2: <bound method> = bound_method @__global_init.%int_1.loc99, %specific_fn.loc99 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc99: init %i32 = call %bound_method.loc99_32.2(@__global_init.%int_1.loc99) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc99_32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc99 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc99_32.2: %i32 = converted @__global_init.%int_1.loc99, %.loc99_32.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc99: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %h: %i32 = wrapper_binding h, %.loc99_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.d94 = value_binding_pattern h [concrete = constants.%h.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc115: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc115_34.1: <bound method> = bound_method @__global_init.%int_1.loc115, %impl.elem0.loc115 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc115: <specific function> = specific_function %impl.elem0.loc115, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc115_34.2: <bound method> = bound_method @__global_init.%int_1.loc115, %specific_fn.loc115 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc115: init %i32 = call %bound_method.loc115_34.2(@__global_init.%int_1.loc115) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc115_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc115 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc115_34.2: %i32 = converted @__global_init.%int_1.loc115, %.loc115_34.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc115: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i: %i32 = wrapper_binding i, %.loc115_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc31: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc40: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc66: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc82: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc99: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_1.loc115: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
