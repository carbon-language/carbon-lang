// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/comments/wrap.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/comments/wrap.carbon

// --- test.carbon

// This is a very long top-level comment that goes well beyond the eighty column limit and so must wrap.
fn F() {
  // An indented over-long comment line inside a function body that also exceeds the limit and wraps.

  // A short line is left alone.

  // A line with a single enormous word such as aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa stays.
}

// Wrapping keeps the whitespace after `//` as each produced line's prefix, so
// an indented bullet stays a bullet, and keeps interior spacing verbatim:
//   - a bulleted list item long enough that it has to wrap onto the next line to fit
//   code    aligned    with    extra    interior    spaces    that    must    survive

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // This is a very long top-level comment that goes well beyond the eighty column
// CHECK:STDOUT: // limit and so must wrap.
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   // An indented over-long comment line inside a function body that also exceeds
// CHECK:STDOUT:   // the limit and wraps.
// CHECK:STDOUT:
// CHECK:STDOUT:   // A short line is left alone.
// CHECK:STDOUT:
// CHECK:STDOUT:   // A line with a single enormous word such as
// CHECK:STDOUT:   // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa stays.
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // Wrapping keeps the whitespace after `//` as each produced line's prefix, so
// CHECK:STDOUT: // an indented bullet stays a bullet, and keeps interior spacing verbatim:
// CHECK:STDOUT: //   - a bulleted list item long enough that it has to wrap onto the next line
// CHECK:STDOUT: //   to fit
// CHECK:STDOUT: //   code    aligned    with    extra    interior    spaces    that    must
// CHECK:STDOUT: //   survive
