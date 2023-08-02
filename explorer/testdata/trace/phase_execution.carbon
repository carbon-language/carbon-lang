// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}

// Place checks after code so that line numbers are stable, reducing merge
// conflicts.
// ARGS: --trace_file=- --trace_phase=execution %s
// AUTOUPDATE
// CHECK:STDOUT: ********** starting execution **********
// CHECK:STDOUT: ********** initializing globals **********
// CHECK:STDOUT: (+) stack-push: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: --- step decl interface TestInterface .0. (phase_execution.carbon:7) --->
// CHECK:STDOUT: (-) stack-pop: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: (+) stack-push: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: --- step decl fn Main .0. (phase_execution.carbon:11) --->
// CHECK:STDOUT: (-) stack-pop: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: ********** calling main function **********
// CHECK:STDOUT: (+) stack-push: Main() .0.
// CHECK:STDOUT: (+) stack-push: Main() .0.
// CHECK:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// CHECK:STDOUT: (+) stack-push: Main .0.
// CHECK:STDOUT: (+) stack-push: Main .0.
// CHECK:STDOUT: --- step exp Main .0. (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop: Main .0.
// CHECK:STDOUT: (-) stack-pop: Main .1. {{\[\[}}fun<Main>]]
// CHECK:STDOUT: --- step exp Main() .1. (<Main()>:0) --->
// CHECK:STDOUT: calling function: fun<Main>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: from value expression with value ()
// CHECK:STDOUT: (+) stack-push: .0. {}
// CHECK:STDOUT: (+) stack-push: {return 0;} .0.
// CHECK:STDOUT: --- step stmt {return 0;} .0. (phase_execution.carbon:11) --->
// CHECK:STDOUT: (+) stack-push: return 0; .0.
// CHECK:STDOUT: --- step stmt return 0; .0. (phase_execution.carbon:10) --->
// CHECK:STDOUT: (+) stack-push: 0 .0.
// CHECK:STDOUT: (+) stack-push: 0 .0.
// CHECK:STDOUT: --- step exp 0 .0. (phase_execution.carbon:10) --->
// CHECK:STDOUT: (-) stack-pop: 0 .0.
// CHECK:STDOUT: (-) stack-pop: 0 .1. {{\[\[}}0]]
// CHECK:STDOUT: --- step stmt return 0; .1. (phase_execution.carbon:10) --->
// CHECK:STDOUT: (-) stack-pop: return 0; .1. {{\[\[}}0]]
// CHECK:STDOUT: (-) stack-pop: {return 0;} .1. {}
// CHECK:STDOUT: (-) stack-pop: .0. {}
// CHECK:STDOUT: (+) stack-push: clean up.0. {}
// CHECK:STDOUT: (+) stack-push: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: clean up.0. {}
// CHECK:STDOUT: --- step exp Main() .2. (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop: Main() .2. {{\[\[}}fun<Main>, 0]] {}
// CHECK:STDOUT: (+) stack-push: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: Main() .1. {{\[\[}}0]]
// CHECK:STDOUT: interpreter result: 0
// CHECK:STDOUT: result: 0
