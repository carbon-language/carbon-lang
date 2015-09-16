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

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
def BRIF : I<(outs), (ins bb_op:$dst, Int32:$a),
             [(brcond Int32:$a, bb:$dst)]>;
let isBarrier = 1 in {
def BR   : I<(outs), (ins bb_op:$dst),
             [(br bb:$dst)]>;
} // isBarrier = 1
} // isBranch = 1, isTerminator = 1, hasCtrlDep = 1

// TODO: SelectionDAG's lowering insists on using a pointer as the index for
// jump tables, so in practice we don't ever use SWITCH_I64 in wasm32 mode
// currently.
let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
def SWITCH_I32 : I<(outs), (ins Int32:$index, variable_ops),
                   [(WebAssemblyswitch Int32:$index)]>;
def SWITCH_I64 : I<(outs), (ins Int64:$index, variable_ops),
                   [(WebAssemblyswitch Int64:$index)]>;
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1

// Placemarkers to indicate the start of a block or loop scope.
def BLOCK     : I<(outs), (ins bb_op:$dst), []>;
def LOOP      : I<(outs), (ins bb_op:$dst), []>;

multiclass RETURN<WebAssemblyRegClass vt> {
  def RETURN_#vt : I<(outs), (ins vt:$val), [(WebAssemblyreturn vt:$val)]>;
}
let isReturn = 1, isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
  defm : RETURN<Int32>;
  defm : RETURN<Int64>;
  defm : RETURN<Float32>;
  defm : RETURN<Float64>;
  def RETURN_VOID : I<(outs), (ins), [(WebAssemblyreturn)]>;
} // isReturn = 1, isTerminator = 1, hasCtrlDep = 1, isBarrier = 1
