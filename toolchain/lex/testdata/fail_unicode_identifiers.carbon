// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_unicode_identifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_unicode_identifiers.carbon

// --- fail_unicode_identifiers.carbon
// CHECK:STDOUT: - filename: fail_unicode_identifiers.carbon
// CHECK:STDOUT:   tokens:

// Non-ASCII characters aren't yet supported in identifiers, so they end an
// identifier and are diagnosed as unrecognized characters.
// CHECK:STDERR:        ╭─┤ fail_unicode_identifiers.carbon:[[@LINE+7]]:4
// CHECK:STDERR: ->     │ café
// CHECK:STDERR:        ·    ┳
// CHECK:STDERR:        ·    ┗━━┫ error: encountered unrecognized characters while parsing
// CHECK:STDERR:        ·       ┃        [UnrecognizedCharacters]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
café
// CHECK:STDOUT:   - { index:  1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "caf", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Error", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "\xC3\xA9" }
// CHECK:STDERR:        ╭─┤ fail_unicode_identifiers.carbon:[[@LINE+7]]:2
// CHECK:STDERR: ->     │ a日本b
// CHECK:STDERR:        ·  ━━┳━
// CHECK:STDERR:        ·    ┗━━┫ error: encountered unrecognized characters while parsing
// CHECK:STDERR:        ·       ┃        [UnrecognizedCharacters]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
a日本b
// CHECK:STDOUT:   - { index:  3, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "a", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index:  4, kind:      "Error", line: {{ *}}[[@LINE-2]], column:   2, indent: 1, spelling: "\xE6\x97\xA5\xE6\x9C\xAC" }
// CHECK:STDOUT:   - { index:  5, kind: "Identifier", line: {{ *}}[[@LINE-3]], column:   8, indent: 1, spelling: "b", identifier: 2 }

// This applies to raw identifiers and long identifiers in the same way.
// CHECK:STDERR:        ╭─┤ fail_unicode_identifiers.carbon:[[@LINE+7]]:5
// CHECK:STDERR: ->     │ r#naïve
// CHECK:STDERR:        ·     ┳
// CHECK:STDERR:        ·     ┗━━┫ error: encountered unrecognized characters while parsing
// CHECK:STDERR:        ·        ┃        [UnrecognizedCharacters]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
r#naïve
// CHECK:STDOUT:   - { index:  6, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "r#na", identifier: 4, has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:      "Error", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "\xC3\xAF" }
// CHECK:STDOUT:   - { index:  8, kind: "Identifier", line: {{ *}}[[@LINE-3]], column:   7, indent: 1, spelling: "ve", identifier: 5 }
// CHECK:STDERR:        ╭─┤ fail_unicode_identifiers.carbon:[[@LINE+8]]:74
// CHECK:STDERR: ->     │ a_long_identifier_that_needs_more_than_one_word_of_the_identifier_bitmap_é_x
// CHECK:STDERR:        ·                                                                          ┳
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: encountered unrecognized characters while parsing
// CHECK:STDERR:        · ┃        [UnrecognizedCharacters]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
a_long_identifier_that_needs_more_than_one_word_of_the_identifier_bitmap_é_x
// CHECK:STDOUT:   - { index:  9, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "a_long_identifier_that_needs_more_than_one_word_of_the_identifier_bitmap_", identifier: 6, has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:      "Error", line: {{ *}}[[@LINE-2]], column:  74, indent: 1, spelling: "\xC3\xA9" }
// CHECK:STDOUT:   - { index: 11, kind: "Identifier", line: {{ *}}[[@LINE-3]], column:  76, indent: 1, spelling: "_x", identifier: 7 }
