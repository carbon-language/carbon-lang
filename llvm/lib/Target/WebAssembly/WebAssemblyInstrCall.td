//===- WebAssemblyInstrCall.td-WebAssembly Call codegen support -*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Call operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

// The call sequence start/end LLVM-isms isn't useful to WebAssembly since it's
// a virtual ISA.

// FIXME make noop?
//def : Pat<(WebAssemblycallseq_start timm), (i32 (IMPLICIT_DEF))>;
//def : Pat<(WebAssemblycallseq_end timm, timm), (i32 (IMPLICIT_DEF))>;

def SDT_WebAssemblyCallSeqStart : SDCallSeqStart<[SDTCisVT<0, iPTR>]>;
def SDT_WebAssemblyCallSeqEnd :
    SDCallSeqEnd<[SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;
def WebAssemblycallseq_start :
    SDNode<"ISD::CALLSEQ_START", SDT_WebAssemblyCallSeqStart,
           [SDNPHasChain, SDNPOutGlue]>;
def WebAssemblycallseq_end :
    SDNode<"ISD::CALLSEQ_END", SDT_WebAssemblyCallSeqEnd,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def : Pseudo<(outs), (ins i64imm:$amt),
             [(WebAssemblycallseq_start timm:$amt)],
             "#ADJCALLSTACKDOWN $amt">;
def : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
             [(WebAssemblycallseq_end timm:$amt1, timm:$amt2)],
	     "#ADJCALLSTACKUP $amt1 $amt2">;

/*
 * TODO(jfb): Add the following.
 *
 * call_direct: call function directly
 * call_indirect: call function indirectly
 * addressof: obtain a function pointer value for a given function
 */
