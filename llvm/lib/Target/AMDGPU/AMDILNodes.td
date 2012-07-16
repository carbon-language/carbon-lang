//===- AMDILNodes.td - AMD IL nodes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Flow Control DAG Nodes
//===----------------------------------------------------------------------===//
def IL_brcond      : SDNode<"AMDILISD::BRANCH_COND", SDTIL_BRCond, [SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// Comparison DAG Nodes
//===----------------------------------------------------------------------===//
def IL_cmp       : SDNode<"AMDILISD::CMP", SDTIL_Cmp>;

//===----------------------------------------------------------------------===//
// Call/Return DAG Nodes
//===----------------------------------------------------------------------===//
def IL_call      : SDNode<"AMDILISD::CALL", SDTIL_Call,
    [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def IL_retflag       : SDNode<"AMDILISD::RET_FLAG", SDTNone,
    [SDNPHasChain, SDNPOptInGlue]>;

//===--------------------------------------------------------------------===//
// Instructions
//===--------------------------------------------------------------------===//
// Floating point math functions
def IL_cmov_logical : SDNode<"AMDILISD::CMOVLOG", SDTIL_GenTernaryOp>;
def IL_div_inf      : SDNode<"AMDILISD::DIV_INF", SDTIL_GenBinaryOp>;
def IL_mad          : SDNode<"AMDILISD::MAD", SDTIL_GenTernaryOp>;

//===----------------------------------------------------------------------===//
// Integer functions
//===----------------------------------------------------------------------===//
def IL_umul        : SDNode<"AMDILISD::UMUL"    , SDTIntBinOp,
    [SDNPCommutative, SDNPAssociative]>;

//===----------------------------------------------------------------------===//
// Vector functions
//===----------------------------------------------------------------------===//
def IL_vbuild     : SDNode<"AMDILISD::VBUILD", SDTIL_GenVecBuild,
    []>;
