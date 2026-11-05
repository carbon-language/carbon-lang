// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_extension.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_extension.carbon

// --- fail_main.incorrect
// CHECK:STDERR: fail_main.incorrect: error: file extension of `.carbon` required for api [IncorrectExtension]
// CHECK:STDERR:

// --- fail_main_redundant_with_swapped_ext.impl.carbon
// CHECK:STDERR: fail_main_redundant_with_swapped_ext.impl.carbon: error: `Main//default` previously provided by `fail_main.incorrect` [DuplicateMainApi]
// CHECK:STDERR:
// CHECK:STDERR: fail_main_redundant_with_swapped_ext.impl.carbon: error: file extension of `.carbon` required for api; `.impl.carbon` names an `impl` file [IncorrectExtension]
// CHECK:STDERR:

// --- fail_main_lib.incorrect

// CHECK:STDERR:        ╭─┤ fail_main_lib.incorrect:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ library "main_lib";
// CHECK:STDERR:        · ━━━━━━━━━┳━━━━━━━━━
// CHECK:STDERR:        ·          ┗━━┫ error: file extension of `.carbon` required for api
// CHECK:STDERR:        ·             ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
library "[[@TEST_NAME]]";

// --- fail_main_lib.impl.incorrect

// CHECK:STDERR:        ╭─┤ fail_main_lib.impl.incorrect:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl library "main_lib";
// CHECK:STDERR:        · ━━━━━━━━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        ·             ┗━━┫ error: file extension of `.impl.carbon` required for `impl`
// CHECK:STDERR:        ·                ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl library "[[@TEST_NAME]]";

// --- fail_package.incorrect

// CHECK:STDERR:        ╭─┤ fail_package.incorrect:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ package Package;
// CHECK:STDERR:        · ━━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: file extension of `.carbon` required for api
// CHECK:STDERR:        ·            ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
package Package;

// --- fail_package_impl.incorrect

// CHECK:STDERR:        ╭─┤ fail_package_impl.incorrect:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl package Package;
// CHECK:STDERR:        · ━━━━━━━━━━┳━━━━━━━━━━
// CHECK:STDERR:        ·           ┗━━┫ error: file extension of `.impl.carbon` required for `impl`
// CHECK:STDERR:        ·              ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl package Package;

// --- fail_package_lib.incorrect

// CHECK:STDERR:        ╭─┤ fail_package_lib.incorrect:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ package Package library "package_lib";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: file extension of `.carbon` required for api
// CHECK:STDERR:        ·                       ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
package Package library "[[@TEST_NAME]]";

// --- fail_package_lib.impl.incorrect

// CHECK:STDERR:        ╭─┤ fail_package_lib.impl.incorrect:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl package Package library "package_lib";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                 ┗━━┫ error: file extension of `.impl.carbon` required for `impl`
// CHECK:STDERR:        ·                    ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl package Package library "[[@TEST_NAME]]";

// --- fail_swapped_ext.impl.carbon

// CHECK:STDERR:        ╭─┤ fail_swapped_ext.impl.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ package SwappedExt;
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: file extension of `.carbon` required for api; `.impl.carbon` names
// CHECK:STDERR:        ·    ┃        an `impl` file
// CHECK:STDERR:        ·    ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
package SwappedExt;

// --- fail_swapped_ext.carbon

// CHECK:STDERR:        ╭─┤ fail_swapped_ext.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl package SwappedExt;
// CHECK:STDERR:        · ━━━━━━━━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        ·             ┗━━┫ error: file extension of `.impl.carbon` required for `impl`
// CHECK:STDERR:        ·                ┃        [IncorrectExtension]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl package SwappedExt;


// CHECK:STDOUT: --- fail_main.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_redundant_with_swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib.impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %default.import.loc9_24.1 = import <none>
// CHECK:STDOUT:   %default.import.loc9_24.2 = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %Package.import = import Package
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_lib.impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %Package.import = import Package
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_swapped_ext.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %SwappedExt.import = import SwappedExt
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
