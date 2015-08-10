// WebAssemblyInstrInfo.td-Describe the WebAssembly Instructions-*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Instruction definitions.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//

def HasAddr32 : Predicate<"!Subtarget->hasAddr64()">;
def HasAddr64 : Predicate<"Subtarget->hasAddr64()">;
def HasSIMD128 : Predicate<"Subtarget->hasSIMD128()">,
                           AssemblerPredicate<"FeatureSIMD128", "simd128">;

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Node Types.
//===----------------------------------------------------------------------===//

def SDT_WebAssemblyArgument : SDTypeProfile<1, 1, [SDTCisVT<1, i32>]>;
def SDT_WebAssemblyReturn   : SDTypeProfile<0, -1, []>;

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Nodes.
//===----------------------------------------------------------------------===//

def WebAssemblyargument : SDNode<"WebAssemblyISD::ARGUMENT",
                                 SDT_WebAssemblyArgument>;
def WebAssemblyreturn   : SDNode<"WebAssemblyISD::RETURN",
                                 SDT_WebAssemblyReturn,
                                 [SDNPHasChain, SDNPSideEffect]>;

//===----------------------------------------------------------------------===//
// WebAssembly-specific Operands.
//===----------------------------------------------------------------------===//

/*
 * TODO(jfb): Add the following.
 *
 * get_local: read the current value of a local variable
 * set_local: set the current value of a local variable
*/

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Format Definitions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrFormats.td"

multiclass ARGUMENT<WebAssemblyRegClass vt> {
  def ARGUMENT_#vt : I<(outs vt:$res), (ins i32imm:$argno),
                       [(set vt:$res, (WebAssemblyargument timm:$argno))]>;
}
defm : ARGUMENT<Int32>;
defm : ARGUMENT<Int64>;
defm : ARGUMENT<Float32>;
defm : ARGUMENT<Float64>;


def Immediate_I32 : I<(outs Int32:$res), (ins i32imm:$imm),
                      [(set Int32:$res, imm:$imm)]>;
def Immediate_I64 : I<(outs Int64:$res), (ins i64imm:$imm),
                      [(set Int64:$res, imm:$imm)]>;
def Immediate_F32 : I<(outs Float32:$res), (ins f32imm:$imm),
                      [(set Float32:$res, fpimm:$imm)]>;
def Immediate_F64 : I<(outs Float64:$res), (ins f64imm:$imm),
                      [(set Float64:$res, fpimm:$imm)]>;

//===----------------------------------------------------------------------===//
// Additional sets of instructions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrMemory.td"
include "WebAssemblyInstrCall.td"
include "WebAssemblyInstrControl.td"
include "WebAssemblyInstrInteger.td"
include "WebAssemblyInstrFloat.td"
include "WebAssemblyInstrConv.td"
include "WebAssemblyInstrAtomics.td"
include "WebAssemblyInstrSIMD.td"
