// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_modifiers.carbon

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+51]]:25
// CHECK:STDERR: ->     │ private extern abstract base default final namespace A;
// CHECK:STDERR:        ·                ────┬─── ┳━━━
// CHECK:STDERR:        ·                    │    ┗━━┫ error: `base` not allowed on declaration with
// CHECK:STDERR:        ·                    │       ┃        `abstract`
// CHECK:STDERR:        ·                    │       ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·                    ╰─────────┤ `abstract` previously appeared here
// CHECK:STDERR:        ·                              │ [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+41]]:30
// CHECK:STDERR: ->     │ private extern abstract base default final namespace A;
// CHECK:STDERR:        ·                ────┬───      ┳━━━━━━
// CHECK:STDERR:        ·                    │         ┗━━┫ error: `default` not allowed on declaration
// CHECK:STDERR:        ·                    │            ┃        with `abstract`
// CHECK:STDERR:        ·                    │            ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·                    ╰──────────────┤ `abstract` previously appeared here
// CHECK:STDERR:        ·                                   │ [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+31]]:38
// CHECK:STDERR: ->     │ private extern abstract base default final namespace A;
// CHECK:STDERR:        ·                ────┬───              ┳━━━━
// CHECK:STDERR:        ·                    │                 ┗━━┫ error: `final` not allowed on
// CHECK:STDERR:        ·                    │                    ┃        declaration with `abstract`
// CHECK:STDERR:        ·                    │                    ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·                    ╰──────────────────────┤ `abstract` previously appeared here
// CHECK:STDERR:        ·                                           │ [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+21]]:1
// CHECK:STDERR: ->     │ private extern abstract base default final namespace A;
// CHECK:STDERR:        · ━━━┳━━━
// CHECK:STDERR:        ·    ┗━━┫ error: `private` not allowed on `namespace` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+14]]:9
// CHECK:STDERR: ->     │ private extern abstract base default final namespace A;
// CHECK:STDERR:        ·         ━━━┳━━
// CHECK:STDERR:        ·            ┗━━┫ error: `extern` not allowed on `namespace` declaration
// CHECK:STDERR:        ·               ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:16
// CHECK:STDERR: ->     │ private extern abstract base default final namespace A;
// CHECK:STDERR:        ·                ━━━━┳━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: `abstract` not allowed on `namespace` declaration
// CHECK:STDERR:        ·                       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
private extern abstract base default final namespace A;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl namespace B;
// CHECK:STDERR:        · ━━┳━
// CHECK:STDERR:        ·   ┗━━┫ error: `impl` not allowed on `namespace` declaration
// CHECK:STDERR:        ·      ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl namespace B;

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ extern namespace C;
// CHECK:STDERR:        · ━━━┳━━
// CHECK:STDERR:        ·    ┗━━┫ error: `extern` not allowed on `namespace` declaration
// CHECK:STDERR:        ·       ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern namespace C;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %B: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %C: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
