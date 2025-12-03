// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/print/char.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/print/char.carbon

import Core library "io";

fn PrintChar(a: char) -> i32 = "print.char";

fn Main() {
  //@dump-sem-ir-begin
  PrintChar('1');
  Core.PrintChar('2');
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %PrintChar.type.c95: type = fn_type @PrintChar.loc15 [concrete]
// CHECK:STDOUT:   %PrintChar.843: %PrintChar.type.c95 = struct_value () [concrete]
// CHECK:STDOUT:   %.75c: Core.CharLiteral = char_value U+0031 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3be: type = facet_type <@ImplicitAs, @ImplicitAs(%char)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.f57: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%char) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.892: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1f2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.3be = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness.892) [concrete]
// CHECK:STDOUT:   %.8d6: type = fn_type_with_self_type %ImplicitAs.Convert.type.f57, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.fd2: <bound method> = bound_method %.75c, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_49: %char = int_value 49 [concrete]
// CHECK:STDOUT:   %PrintChar.type.089: type = fn_type @PrintChar.1 [concrete]
// CHECK:STDOUT:   %PrintChar.d75: %PrintChar.type.089 = struct_value () [concrete]
// CHECK:STDOUT:   %.f8d: Core.CharLiteral = char_value U+0032 [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.1f7: <bound method> = bound_method %.f8d, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_50: %char = int_value 50 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Char = %Core.Char
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .PrintChar = %Core.PrintChar
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//io
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Char: type = import_ref Core//prelude/types/char, Char, loaded [concrete = constants.%char]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.543: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1f2 = impl_witness_table (%Core.import_ref.543), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.PrintChar: %PrintChar.type.089 = import_ref Core//io, PrintChar, loaded [concrete = constants.%PrintChar.d75]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %PrintChar.ref.loc19: %PrintChar.type.c95 = name_ref PrintChar, file.%PrintChar.decl [concrete = constants.%PrintChar.843]
// CHECK:STDOUT:   %.loc19_13.1: Core.CharLiteral = char_value U+0031 [concrete = constants.%.75c]
// CHECK:STDOUT:   %impl.elem0.loc19: %.8d6 = impl_witness_access constants.%ImplicitAs.impl_witness.892, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %.loc19_13.1, %impl.elem0.loc19 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.fd2]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %char = call %bound_method.loc19(%.loc19_13.1) [concrete = constants.%int_49]
// CHECK:STDOUT:   %.loc19_13.2: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_49]
// CHECK:STDOUT:   %.loc19_13.3: %char = converted %.loc19_13.1, %.loc19_13.2 [concrete = constants.%int_49]
// CHECK:STDOUT:   %PrintChar.call.loc19: init %i32 = call %PrintChar.ref.loc19(%.loc19_13.3)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %PrintChar.ref.loc20: %PrintChar.type.089 = name_ref PrintChar, imports.%Core.PrintChar [concrete = constants.%PrintChar.d75]
// CHECK:STDOUT:   %.loc20_18.1: Core.CharLiteral = char_value U+0032 [concrete = constants.%.f8d]
// CHECK:STDOUT:   %impl.elem0.loc20: %.8d6 = impl_witness_access constants.%ImplicitAs.impl_witness.892, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %.loc20_18.1, %impl.elem0.loc20 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.1f7]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %char = call %bound_method.loc20(%.loc20_18.1) [concrete = constants.%int_50]
// CHECK:STDOUT:   %.loc20_18.2: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_50]
// CHECK:STDOUT:   %.loc20_18.3: %char = converted %.loc20_18.1, %.loc20_18.2 [concrete = constants.%int_50]
// CHECK:STDOUT:   %PrintChar.call.loc20: init %i32 = call %PrintChar.ref.loc20(%.loc20_18.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
