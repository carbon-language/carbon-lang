//===- WebAssemblyInstrControl.td-WebAssembly control-flow ------*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly control-flow code-gen constructs.
///
//===----------------------------------------------------------------------===//

/*
 * TODO(jfb): Add the following.
 *
 * block: a fixed-length sequence of statements
 * if: if statement
 * do_while: do while statement, basically a loop with a conditional branch
 * forever: infinite loop statement (like while (1)), basically an unconditional
 *          branch (back to the top of the loop)
 * continue: continue to start of nested loop
 * break: break to end from nested loop or block
 * switch: switch statement with fallthrough
 */

let hasSideEffects = 1, isReturn = 1, isTerminator = 1, hasCtrlDep = 1,
    isBarrier = 1 in {
//FIXME return more than just int32.
def RETURN : I<(outs), (ins Int32:$val), [(WebAssemblyreturn Int32:$val)]>;
} // hasSideEffects = 1, isReturn = 1, isTerminator = 1, hasCtrlDep = 1,
  // isBarrier = 1
