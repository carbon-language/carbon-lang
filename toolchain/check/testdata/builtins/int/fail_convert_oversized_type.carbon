// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/fail_convert_oversized_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/fail_convert_oversized_type.carbon

// --- fail_width_not_multiple_of_8.carbon

library "[[@TEST_NAME]]";

// Initializing a binding whose type is an invalid integer type must not crash
// constant evaluation (#7278). The type is diagnosed; the conversion of the
// initializer should then produce an error value, not a CHECK failure.
// CHECK:STDERR:        ╭─┤ fail_width_not_multiple_of_8.carbon:[[@LINE+52]]:10
// CHECK:STDERR: ->     │ let bad: i8388609 = 0;
// CHECK:STDERR:        ·          ┳━━━━━━━
// CHECK:STDERR:        ·          ┗━━┫ error: bit width of integer type literal must be a multiple of 8;
// CHECK:STDERR:        ·             ┃        use `Core.Int(8388609)` instead
// CHECK:STDERR:        ·             ┃        [IntWidthNotMultipleOf8]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ {{.*}}/prelude/types/int.carbon:18:9
// CHECK:STDERR: ->     │   adapt MakeInt(N);
// CHECK:STDERR:        ·         ┳━━━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: integer type width of 8388609 is greater than the maximum
// CHECK:STDERR:        ·            ┃        supported width of 8388608
// CHECK:STDERR:        ·            ┃        [IntWidthTooLarge]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ fail_width_not_multiple_of_8.carbon
// CHECK:STDERR:        │ let bad: i8388609 = 0;
// CHECK:STDERR:        ·          ────┬───
// CHECK:STDERR:        ·              ╰──┤ in monomorphization of `i8388609` here [ResolvingSpecificHere]
// CHECK:STDERR:        ·              ╰────┤ in completion of type `i8388609` here [InTypeCompletion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ {{.*}}/prelude/types/int.carbon:18:9
// CHECK:STDERR: ->     │   adapt MakeInt(N);
// CHECK:STDERR:        ·         ┳━━━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: integer type width of 16777216 is greater than the maximum
// CHECK:STDERR:        ·            ┃        supported width of 8388608
// CHECK:STDERR:        ·            ┃        [IntWidthTooLarge]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ fail_width_too_large.carbon
// CHECK:STDERR:        │ let bad: i16777216 = 0;
// CHECK:STDERR:        ·          ────┬────
// CHECK:STDERR:        ·              ╰──┤ in monomorphization of `i16777216` here
// CHECK:STDERR:        ·              │  │ [ResolvingSpecificHere]
// CHECK:STDERR:        ·              ╰────┤ in completion of type `i16777216` here [InTypeCompletion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ {{.*}}/prelude/types/int.carbon:18:9
// CHECK:STDERR: ->     │   adapt MakeInt(N);
// CHECK:STDERR:        ·         ┳━━━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: integer type width of 8388609 is greater than the maximum
// CHECK:STDERR:        ·            ┃        supported width of 8388608
// CHECK:STDERR:        ·            ┃        [IntWidthTooLarge]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ fail_width_too_large_core_int.carbon
// CHECK:STDERR:        │ let bad: Core.Int(8388609) = 0;
// CHECK:STDERR:        ·          ────────┬────────
// CHECK:STDERR:        ·                  ╰──┤ in monomorphization of `i8388609` here
// CHECK:STDERR:        ·                  │  │ [ResolvingSpecificHere]
// CHECK:STDERR:        ·                  ╰────┤ in completion of type `i8388609` here [InTypeCompletion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let bad: i8388609 = 0;

// --- fail_width_too_large.carbon

library "[[@TEST_NAME]]";

let bad: i16777216 = 0;

// --- fail_width_too_large_core_int.carbon

library "[[@TEST_NAME]]";

let bad: Core.Int(8388609) = 0;
