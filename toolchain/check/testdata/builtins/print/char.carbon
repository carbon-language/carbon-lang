// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --include-carbon-core
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
// CHECK:STDOUT:   %PrintChar.type.7a7: type = fn_type @PrintChar.loc24 [concrete]
// CHECK:STDOUT:   %PrintChar.772: %PrintChar.type.7a7 = struct_value () [concrete]
// CHECK:STDOUT:   %.75c: Core.CharLiteral = char_value U+0031 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e81: type = facet_type <@ImplicitAs, @ImplicitAs(%char)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.780: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8a5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e81 = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness.780) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.0c6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%char, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1e6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.0c6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.17b: <bound method> = bound_method %.75c, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_49: %char = int_value 49 [concrete]
// CHECK:STDOUT:   %PrintChar.type.b3c: type = fn_type @PrintChar.1 [concrete]
// CHECK:STDOUT:   %PrintChar.063: %PrintChar.type.b3c = struct_value () [concrete]
// CHECK:STDOUT:   %.f8d: Core.CharLiteral = char_value U+0032 [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.e67: <bound method> = bound_method %.f8d, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8a5 = impl_witness_table (%Core.import_ref.06c), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.PrintChar: %PrintChar.type.b3c = import_ref Core//io, PrintChar, loaded [concrete = constants.%PrintChar.063]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %PrintChar.ref.loc28: %PrintChar.type.7a7 = name_ref PrintChar, file.%PrintChar.decl [concrete = constants.%PrintChar.772]
// CHECK:STDOUT:   %.loc28_13.1: Core.CharLiteral = char_value U+0031 [concrete = constants.%.75c]
// CHECK:STDOUT:   %impl.elem0.loc28: %.1e6 = impl_witness_access constants.%ImplicitAs.impl_witness.780, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %.loc28_13.1, %impl.elem0.loc28 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.17b]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %char = call %bound_method.loc28(%.loc28_13.1) [concrete = constants.%int_49]
// CHECK:STDOUT:   %.loc28_13.2: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_49]
// CHECK:STDOUT:   %.loc28_13.3: %char = converted %.loc28_13.1, %.loc28_13.2 [concrete = constants.%int_49]
// CHECK:STDOUT:   %PrintChar.call.loc28: init %i32 = call %PrintChar.ref.loc28(%.loc28_13.3)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %PrintChar.ref.loc29: %PrintChar.type.b3c = name_ref PrintChar, imports.%Core.PrintChar [concrete = constants.%PrintChar.063]
// CHECK:STDOUT:   %.loc29_18.1: Core.CharLiteral = char_value U+0032 [concrete = constants.%.f8d]
// CHECK:STDOUT:   %impl.elem0.loc29: %.1e6 = impl_witness_access constants.%ImplicitAs.impl_witness.780, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %.loc29_18.1, %impl.elem0.loc29 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound.e67]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc29: init %char = call %bound_method.loc29(%.loc29_18.1) [concrete = constants.%int_50]
// CHECK:STDOUT:   %.loc29_18.2: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call.loc29 [concrete = constants.%int_50]
// CHECK:STDOUT:   %.loc29_18.3: %char = converted %.loc29_18.1, %.loc29_18.2 [concrete = constants.%int_50]
// CHECK:STDOUT:   %PrintChar.call.loc29: init %i32 = call %PrintChar.ref.loc29(%.loc29_18.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
