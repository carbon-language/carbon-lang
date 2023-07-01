// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A lot of output is elided: this is only checking for a few things for simple
// sanity checking on --parser_debug --trace_file=- along with filter flags output.
//
// NOAUTOUPDATE
// RUN: %{explorer} --parser_debug --trace_file=- | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NO-NAMES,NO-PRELUDE,NO-FLOW,NO-TYPE,NO-UNFORMED,NO-DECLS,EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=execution | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NO-NAMES,NO-PRELUDE,NO-FLOW,NO-TYPE,NO-UNFORMED,NO-DECLS,EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=source_program | %{FileCheck-allow-unmatched} --check-prefixes=SOURCE,NO-NAMES,NO-PRELUDE,NO-FLOW,NO-TYPE,NO-UNFORMED,NO-DECLS,NO-EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=name_resolution | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NAMES,NO-PRELUDE,NO-FLOW,NO-TYPE,NO-UNFORMED,NO-DECLS,NO-EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=control_flow_resolution | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NO-NAMES,NO-PRELUDE,FLOW,NO-TYPE,NO-UNFORMED,NO-DECLS,NO-EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=type_checking | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NO-NAMES,NO-PRELUDE,NO-FLOW,TYPE,NO-UNFORMED,NO-DECLS,NO-EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=unformed_variables_resolution | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NO-NAMES,NO-PRELUDE,NO-FLOW,NO-TYPE,UNFORMED,NO-DECLS,NO-EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=declarations | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NO-NAMES,NO-PRELUDE,NO-FLOW,NO-TYPE,NO-UNFORMED,DECLS,NO-EXEC,NO-TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=timing | %{FileCheck-allow-unmatched} --check-prefixes=NO-SOURCE,NO-NAMES,NO-PRELUDE,NO-FLOW,NO-TYPE,NO-UNFORMED,NO-DECLS,NO-EXEC,TIMING
// RUN: %{explorer} --parser_debug --trace_file=- -trace_file_context=all -trace_phase=all | %{FileCheck-allow-unmatched} --check-prefixes=SOURCE,NAMES,NO-PRELUDE,FLOW,TYPE,UNFORMED,DECLS,EXEC,TIMING
// NO-SOURCE-NOT:STDOUT: ********** source program **********
//        SOURCE:STDOUT: ********** source program **********
// NO-SOURCE-NOT:STDOUT: interface TestInterface {
//        SOURCE:STDOUT: interface TestInterface {
// NO-NAMES-NOT:STDOUT: ********** resolving names **********
//        NAMES:STDOUT: ********** resolving names **********
// NO-PRELUDE-NOT:STDOUT: interface ImplicitAs {
// NO-FLOW-NOT:STDOUT: ********** resolving control flow **********
//        FLOW:STDOUT: ********** resolving control flow **********
// NO-TYPE-NOT:STDOUT: ********** type checking **********
//        TYPE:STDOUT: ********** type checking **********
// NO-TYPE-NOT:STDOUT: ** declaring interface TestInterface
//        TYPE:STDOUT: ** declaring interface TestInterface
// NO-UNFORMED-NOT:STDOUT: ********** resolving unformed variables **********
//        UNFORMED:STDOUT: ********** resolving unformed variables **********
// NO-DECLS-NOT:STDOUT: ********** printing declarations **********
//        DECLS:STDOUT: ********** printing declarations **********
// NO-DECLS-NOT:STDOUT: interface TestInterface {
//        DECLS:STDOUT: interface TestInterface {
// NO-EXEC-NOT:STDOUT: ********** starting execution **********
//        EXEC:STDOUT: ********** starting execution **********
// NO-EXEC-NOT:STDOUT: ********** initializing globals **********
//        EXEC:STDOUT: ********** initializing globals **********
// NO-EXEC-NOT:STDOUT: ********** calling main function **********
//        EXEC:STDOUT: ********** calling main function **********
// NO-EXEC-NOT:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
//        EXEC:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// NO-EXEC-NOT:STDOUT: interpreter result: 0
//        EXEC:STDOUT: interpreter result: 0
// NO-TIMING-NOT:STDOUT: ********** printing timing **********
//        TIMING:STDOUT: ********** printing timing **********
// NO-TIMING-NOT:STDOUT: Time elapsed in ExecProgram: {{[0-9]+}}ms
//        TIMING:STDOUT: Time elapsed in ExecProgram: {{[0-9]+}}ms
// NO-TIMING-NOT:STDOUT: Time elapsed in AnalyzeProgram: {{[0-9]+}}ms
//        TIMING:STDOUT: Time elapsed in AnalyzeProgram: {{[0-9]+}}ms
// NO-TIMING-NOT:STDOUT: Time elapsed in AddPrelude: {{[0-9]+}}ms
//        TIMING:STDOUT: Time elapsed in AddPrelude: {{[0-9]+}}ms
// NO-TIMING-NOT:STDOUT: Time elapsed in Parse: {{[0-9]+}}ms
//        TIMING:STDOUT: Time elapsed in Parse: {{[0-9]+}}ms

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}
