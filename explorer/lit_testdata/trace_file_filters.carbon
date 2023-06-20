// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A lot of output is elided: this is only checking for a few things for simple
// sanity checking on --parser_debug --trace_file=- along with file context
// filter flags output.
//
// NOAUTOUPDATE
// RUN: %{explorer} --parser_debug --trace_file=- -trace_all -trace_file_main | %{FileCheck-allow-unmatched} --check-prefixes=MAIN,NO-PRELUDE
// RUN: %{explorer} --parser_debug --trace_file=- -trace_all -trace_file_prelude | %{FileCheck-allow-unmatched} --check-prefixes=NO-MAIN,PRELUDE
// RUN: %{explorer} --parser_debug --trace_file=- -trace_all -trace_file_all | %{FileCheck-allow-unmatched} --check-prefixes=MAIN,PRELUDE
// NO-PRELUDE-NOT:STDOUT: ** declaring interface As
//        PRELUDE:STDOUT: ** declaring interface As
// NO-MAIN-NOT:STDOUT: ** declaring function Main
//        MAIN:STDOUT: ** declaring function Main
// NO-PRELUDE-NOT:STDOUT: interface As {
//        PRELUDE:STDOUT: interface As {
// NO-MAIN-NOT:STDOUT: fn Main ()-> i32 {
//        MAIN:STDOUT: fn Main ()-> i32 {

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}
