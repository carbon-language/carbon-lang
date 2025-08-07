// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char/basics.carbon

// --- basics.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let c: Core.Char = 'a';
let nul: Core.Char = '\0';
let lit: Core.CharLiteral() = '\u{1E15}';
//@dump-sem-ir-end

let lit_as_char: Core.Char = c;

// --- fail_size_small.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_size_small.carbon:[[@LINE+4]]:20: error: character value U+0080 too large for type `Core.Char` [CharTooLargeForType]
// CHECK:STDERR: let c: Core.Char = '\u{80}';
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR:
let c: Core.Char = '\u{80}';

// --- fail_size_multi_byte.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_size_multi_byte.carbon:[[@LINE+4]]:20: error: character value U+1E15 too large for type `Core.Char` [CharTooLargeForType]
// CHECK:STDERR: let c: Core.Char = '\u{1E15}';
// CHECK:STDERR:                    ^~~~~~~~~~
// CHECK:STDERR:
let c: Core.Char = '\u{1E15}';

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %Char [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.715: type = facet_type <@ImplicitAs, @ImplicitAs(%Char)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.f57: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Char) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b70: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.aed [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.715 = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness.b70) [concrete]
// CHECK:STDOUT:   %.a57: type = fn_type_with_self_type %ImplicitAs.Convert.type.f57, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8c6: type = pattern_type Core.CharLiteral [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.142: <bound method> = bound_method %.54f, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_97: %Char = int_value 97 [concrete]
// CHECK:STDOUT:   %.dc9: Core.CharLiteral = char_value U+0000 [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.f0f: <bound method> = bound_method %.dc9, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_0: %Char = int_value 0 [concrete]
// CHECK:STDOUT:   %CharLiteral.type: type = fn_type @CharLiteral [concrete]
// CHECK:STDOUT:   %CharLiteral: %CharLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a81: Core.CharLiteral = char_value U+1E15 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Char = %Core.Char
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .CharLiteral = %Core.CharLiteral
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Char: type = import_ref Core//prelude/types/char, Char, loaded [concrete = constants.%Char]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.0f8: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/char, loc18_36, loaded [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.aed = impl_witness_table (%Core.import_ref.0f8), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.CharLiteral: %CharLiteral.type = import_ref Core//prelude/types/char, CharLiteral, loaded [concrete = constants.%CharLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_12: type = splice_block %Char.ref.loc5 [concrete = constants.%Char] {
// CHECK:STDOUT:     %Core.ref.loc5: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Char.ref.loc5: type = name_ref Char, imports.%Core.Char [concrete = constants.%Char]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc5: %.a57 = impl_witness_access constants.%ImplicitAs.impl_witness.b70, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc5: <bound method> = bound_method @__global_init.%.loc5, %impl.elem0.loc5 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.142]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc5: init %Char = call %bound_method.loc5(@__global_init.%.loc5) [concrete = constants.%int_97]
// CHECK:STDOUT:   %.loc5_20.1: %Char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc5 [concrete = constants.%int_97]
// CHECK:STDOUT:   %.loc5_20.2: %Char = converted @__global_init.%.loc5, %.loc5_20.1 [concrete = constants.%int_97]
// CHECK:STDOUT:   %c: %Char = bind_name c, %.loc5_20.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nul.patt: %pattern_type.b09 = binding_pattern nul [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_14: type = splice_block %Char.ref.loc6 [concrete = constants.%Char] {
// CHECK:STDOUT:     %Core.ref.loc6: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Char.ref.loc6: type = name_ref Char, imports.%Core.Char [concrete = constants.%Char]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc6: %.a57 = impl_witness_access constants.%ImplicitAs.impl_witness.b70, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method @__global_init.%.loc6, %impl.elem0.loc6 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.f0f]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %Char = call %bound_method.loc6(@__global_init.%.loc6) [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc6_22.1: %Char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc6_22.2: %Char = converted @__global_init.%.loc6, %.loc6_22.1 [concrete = constants.%int_0]
// CHECK:STDOUT:   %nul: %Char = bind_name nul, %.loc6_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %lit.patt: %pattern_type.8c6 = binding_pattern lit [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_27.1: type = splice_block %.loc7_27.3 [concrete = Core.CharLiteral] {
// CHECK:STDOUT:     %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %CharLiteral.ref: %CharLiteral.type = name_ref CharLiteral, imports.%Core.CharLiteral [concrete = constants.%CharLiteral]
// CHECK:STDOUT:     %CharLiteral.call: init type = call %CharLiteral.ref() [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc7_27.2: type = value_of_initializer %CharLiteral.call [concrete = Core.CharLiteral]
// CHECK:STDOUT:     %.loc7_27.3: type = converted %CharLiteral.call, %.loc7_27.2 [concrete = Core.CharLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %lit: Core.CharLiteral = bind_name lit, @__global_init.%.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc6: Core.CharLiteral = char_value U+0000 [concrete = constants.%.dc9]
// CHECK:STDOUT:   %.loc7: Core.CharLiteral = char_value U+1E15 [concrete = constants.%.a81]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
