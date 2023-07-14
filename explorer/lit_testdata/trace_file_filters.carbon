// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A lot of output is elided: this is only checking for a few things for simple
// sanity checking on --parser_debug --trace_file=- along with file context
// filter flags output.
//
// NOAUTOUPDATE
// RUN: %{explorer} --parser_debug --trace_file=- -trace_phase=all -trace_file_context=main | %{FileCheck-allow-unmatched} --check-prefixes=MAIN,NO-PRELUDE
// RUN: %{explorer} --parser_debug --trace_file=- -trace_phase=all -trace_file_context=prelude | %{FileCheck-allow-unmatched} --check-prefixes=NO-MAIN,PRELUDE
// RUN: %{explorer} --parser_debug --trace_file=- -trace_phase=all -trace_file_context=all | %{FileCheck-allow-unmatched} --check-prefixes=MAIN,PRELUDE
// RUN: %{explorer} --parser_debug --trace_file=- -trace_phase=all -trace_file_context=main,prelude | %{FileCheck-allow-unmatched} --check-prefixes=MAIN,PRELUDE
// NO-PRELUDE-NOT:STDOUT: ** declaring interface As
//        PRELUDE:STDOUT: ** declaring interface As
// NO-MAIN-NOT:STDOUT: ** declaring function Main
//        MAIN:STDOUT: ** declaring function Main
// NO-PRELUDE-NOT:STDOUT: interface As {
//        PRELUDE:STDOUT: interface As {
// NO-MAIN-NOT:STDOUT: fn Main ()-> i32 {
//        MAIN:STDOUT: fn Main ()-> i32 {
// NO-MAIN-NOT:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
//        MAIN:STDOUT: --- step exp Main() .0. (<Main()>:0) --->

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}
