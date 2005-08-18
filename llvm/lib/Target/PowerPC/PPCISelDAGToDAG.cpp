//===-- PPC32ISelDAGToDAG.cpp - PPC32 pattern matching inst selector ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for 32 bit PowerPC,
// converting from a legalized dag to a PPC dag.
//
//===----------------------------------------------------------------------===//

#include "PowerPC.h"
#include "PPC32TargetMachine.h"
#include "PPC32ISelLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

namespace {
  Statistic<> Recorded("ppc-codegen", "Number of recording ops emitted");
  Statistic<> FusedFP ("ppc-codegen", "Number of fused fp operations");
  Statistic<> FrameOff("ppc-codegen", "Number of frame idx offsets collapsed");
    
  //===--------------------------------------------------------------------===//
  /// PPC32DAGToDAGISel - PPC32 specific code to select PPC32 machine
  /// instructions for SelectionDAG operations.
  ///
  class PPC32DAGToDAGISel : public SelectionDAGISel {
    PPC32TargetLowering PPC32Lowering;
    
    unsigned GlobalBaseReg;
    bool GlobalBaseInitialized;
  public:
    PPC32DAGToDAGISel(TargetMachine &TM)
      : SelectionDAGISel(PPC32Lowering), PPC32Lowering(TM) {}
    
    /// runOnFunction - Override this function in order to reset our
    /// per-function variables.
    virtual bool runOnFunction(Function &Fn) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseInitialized = false;
      return SelectionDAGISel::runOnFunction(Fn);
    }
    
    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDOperand getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }
    
    // Select - Convert the specified operand from a target-independent to a
    // target-specific node if it hasn't already been changed.
    SDOperand Select(SDOperand Op);
    
    SDNode *SelectIntImmediateExpr(SDOperand LHS, SDOperand RHS,
                                   unsigned OCHi, unsigned OCLo,
                                   bool IsArithmetic = false,
                                   bool Negate = false);
   
    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
      DEBUG(BB->dump());
      // Codegen the basic block.
      Select(DAG.getRoot());
      DAG.RemoveDeadNodes();
      DAG.viewGraph();
    }
 
    virtual const char *getPassName() const {
      return "PowerPC DAG->DAG Pattern Instruction Selection";
    } 
  };
}

// Immediate constant composers.
// Lo16 - grabs the lo 16 bits from a 32 bit constant.
// Hi16 - grabs the hi 16 bits from a 32 bit constant.
// HA16 - computes the hi bits required if the lo bits are add/subtracted in
// arithmethically.
static unsigned Lo16(unsigned x)  { return x & 0x0000FFFF; }
static unsigned Hi16(unsigned x)  { return Lo16(x >> 16); }
static unsigned HA16(unsigned x)  { return Hi16((signed)x - (signed short)x); }

// isIntImmediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isIntImmediate(SDOperand N, unsigned& Imm) {
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    Imm = (unsigned)CN->getSignExtended();
    return true;
  }
  return false;
}

// SelectIntImmediateExpr - Choose code for integer operations with an immediate
// operand.
SDNode *PPC32DAGToDAGISel::SelectIntImmediateExpr(SDOperand LHS, SDOperand RHS,
                                                  unsigned OCHi, unsigned OCLo,
                                                  bool IsArithmetic,
                                                  bool Negate) {
  // Check to make sure this is a constant.
  ConstantSDNode *CN = dyn_cast<ConstantSDNode>(RHS);
  // Exit if not a constant.
  if (!CN) return 0;
  // Extract immediate.
  unsigned C = (unsigned)CN->getValue();
  // Negate if required (ISD::SUB).
  if (Negate) C = -C;
  // Get the hi and lo portions of constant.
  unsigned Hi = IsArithmetic ? HA16(C) : Hi16(C);
  unsigned Lo = Lo16(C);

  // If two instructions are needed and usage indicates it would be better to
  // load immediate into a register, bail out.
  if (Hi && Lo && CN->use_size() > 2) return false;

  // Select the first operand.
  SDOperand Opr0 = Select(LHS);

  if (Lo)  // Add in the lo-part.
    Opr0 = CurDAG->getTargetNode(OCLo, MVT::i32, Opr0, getI32Imm(Lo));
  if (Hi)  // Add in the hi-part.
    Opr0 = CurDAG->getTargetNode(OCHi, MVT::i32, Opr0, getI32Imm(Hi));
  return Opr0.Val;
}


// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
SDOperand PPC32DAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END)
    return Op;   // Already selected.
  
  switch (N->getOpcode()) {
  default:
    std::cerr << "Cannot yet select: ";
    N->dump();
    std::cerr << "\n";
    abort();
  case ISD::EntryToken:       // These leaves remain the same.
  case ISD::UNDEF:
    return Op;
  case ISD::TokenFactor: {
    SDOperand New;
    if (N->getNumOperands() == 2) {
      SDOperand Op0 = Select(N->getOperand(0));
      SDOperand Op1 = Select(N->getOperand(1));
      New = CurDAG->getNode(ISD::TokenFactor, MVT::Other, Op0, Op1);
    } else {
      std::vector<SDOperand> Ops;
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        Ops.push_back(Select(N->getOperand(0)));
      New = CurDAG->getNode(ISD::TokenFactor, MVT::Other, Ops);
    }
    
    if (New.Val != N) {
      CurDAG->ReplaceAllUsesWith(N, New.Val);
      N = New.Val;
    }
    break;
  }
  case ISD::CopyFromReg: {
    SDOperand Chain = Select(N->getOperand(0));
    if (Chain == N->getOperand(0)) return Op; // No change
    SDOperand New = CurDAG->getCopyFromReg(Chain,
         cast<RegisterSDNode>(N->getOperand(1))->getReg(), N->getValueType(0));
    return New.getValue(Op.ResNo);
  }
  case ISD::CopyToReg: {
    SDOperand Chain = Select(N->getOperand(0));
    SDOperand Reg = N->getOperand(1);
    SDOperand Val = Select(N->getOperand(2));
    if (Chain != N->getOperand(0) || Val != N->getOperand(2)) {
      SDOperand New = CurDAG->getNode(ISD::CopyToReg, MVT::Other,
                                      Chain, Reg, Val);
      CurDAG->ReplaceAllUsesWith(N, New.Val);
      N = New.Val;
    }
    break;    
  }
  case ISD::Constant: {
    assert(N->getValueType(0) == MVT::i32);
    unsigned v = (unsigned)cast<ConstantSDNode>(N)->getValue();
    if ((unsigned)(short)v == v) {
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::LI, getI32Imm(v));
      break;
    } else {
      SDOperand Top = CurDAG->getTargetNode(PPC::LIS, MVT::i32,
                                            getI32Imm(unsigned(v) >> 16));
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::ORI, Top, getI32Imm(v & 0xFFFF));
      break;
    }
  }
  case ISD::ADD: {
    MVT::ValueType Ty = N->getValueType(0);
    if (Ty == MVT::i32) {
      if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), N->getOperand(1),
                                             PPC::ADDIS, PPC::ADDI, true)) {
        CurDAG->ReplaceAllUsesWith(N, I);
        N = I;
      } else {
        CurDAG->SelectNodeTo(N, Ty, PPC::ADD, Select(N->getOperand(0)),
                             Select(N->getOperand(1)));
      }
      break;
    }
    
    if (!NoExcessFPPrecision) {  // Match FMA ops
      if (N->getOperand(0).getOpcode() == ISD::MUL &&
          N->getOperand(0).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty, Ty == MVT::f64 ? PPC::FMADD : PPC::FMADDS,
                             Select(N->getOperand(0).getOperand(0)),
                             Select(N->getOperand(0).getOperand(1)),
                             Select(N->getOperand(1)));
        break;
      } else if (N->getOperand(1).getOpcode() == ISD::MUL &&
                 N->getOperand(1).hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty, Ty == MVT::f64 ? PPC::FMADD : PPC::FMADDS,
                             Select(N->getOperand(1).getOperand(0)),
                             Select(N->getOperand(1).getOperand(1)),
                             Select(N->getOperand(0)));
        break;
      }
    }
    
    CurDAG->SelectNodeTo(N, Ty, Ty == MVT::f64 ? PPC::FADD : PPC::FADDS,
                         Select(N->getOperand(0)), Select(N->getOperand(1)));
    break;
  }
  case ISD::SUB: {
    MVT::ValueType Ty = N->getValueType(0);
    if (Ty == MVT::i32) {
      unsigned Imm;
      if (isIntImmediate(N->getOperand(0), Imm) && isInt16(Imm)) {
        CurDAG->SelectNodeTo(N, Ty, PPC::SUBFIC, Select(N->getOperand(1)),
                             getI32Imm(Lo16(Imm)));
        break;
      }
      if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), N->getOperand(1),
                                          PPC::ADDIS, PPC::ADDI, true, true)) {
        CurDAG->ReplaceAllUsesWith(N, I);
        N = I;
      } else {
        CurDAG->SelectNodeTo(N, Ty, PPC::SUBF, Select(N->getOperand(1)),
                             Select(N->getOperand(0)));
      }
      break;
    }
    
    if (!NoExcessFPPrecision) {  // Match FMA ops
      if (N->getOperand(0).getOpcode() == ISD::MUL &&
          N->getOperand(0).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty, Ty == MVT::f64 ? PPC::FMSUB : PPC::FMSUBS,
                             Select(N->getOperand(0).getOperand(0)),
                             Select(N->getOperand(0).getOperand(1)),
                             Select(N->getOperand(1)));
        break;
      } else if (N->getOperand(1).getOpcode() == ISD::MUL &&
                 N->getOperand(1).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty, Ty == MVT::f64 ? PPC::FNMSUB : PPC::FNMSUBS,
                             Select(N->getOperand(1).getOperand(0)),
                             Select(N->getOperand(1).getOperand(1)),
                             Select(N->getOperand(0)));
        break;
      }
    }
    CurDAG->SelectNodeTo(N, Ty, Ty == MVT::f64 ? PPC::FSUB : PPC::FSUBS,
                         Select(N->getOperand(0)),
                         Select(N->getOperand(1)));
    break;
  }
  case ISD::MUL: {
    unsigned Imm, Opc;
    if (isIntImmediate(N->getOperand(1), Imm) && isInt16(Imm)) {
      CurDAG->SelectNodeTo(N, N->getValueType(0), PPC::MULLI, 
                           Select(N->getOperand(0)), getI32Imm(Lo16(Imm)));
      break;
    } 
    switch (N->getValueType(0)) {
      default: assert(0 && "Unhandled multiply type!");
      case MVT::i32: Opc = PPC::MULLW; break;
      case MVT::f32: Opc = PPC::FMULS; break;
      case MVT::f64: Opc = PPC::FMUL;  break;
    }
    CurDAG->SelectNodeTo(N, N->getValueType(0), Opc, Select(N->getOperand(0)), 
                         Select(N->getOperand(1)));
    break;
  }
  case ISD::MULHS: {
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, N->getValueType(0), PPC::MULHW, 
                         Select(N->getOperand(0)), Select(N->getOperand(1)));
    break;
  }
  case ISD::MULHU: {
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, N->getValueType(0), PPC::MULHWU, 
                         Select(N->getOperand(0)), Select(N->getOperand(1)));
    break;
  }
  case ISD::FABS: {
    CurDAG->SelectNodeTo(N, N->getValueType(0), PPC::FABS, 
                         Select(N->getOperand(0)));
    break;
  }
  case ISD::FNEG: {
    SDOperand Val = Select(N->getOperand(0));
    MVT::ValueType Ty = N->getValueType(0);
    if (Val.Val->hasOneUse()) {
      unsigned Opc;
      switch (Val.getTargetOpcode()) {
      default:          Opc = 0;            break;
      case PPC::FABS:   Opc = PPC::FNABS;   break;
      case PPC::FMADD:  Opc = PPC::FNMADD;  break;
      case PPC::FMADDS: Opc = PPC::FNMADDS; break;
      case PPC::FMSUB:  Opc = PPC::FNMSUB;  break;
      case PPC::FMSUBS: Opc = PPC::FNMSUBS; break;
      }
      // If we inverted the opcode, then emit the new instruction with the
      // inverted opcode and the original instruction's operands.  Otherwise, 
      // fall through and generate a fneg instruction.
      if (Opc) {
        if (PPC::FNABS == Opc)
          CurDAG->SelectNodeTo(N, Ty, Opc, Val.getOperand(0));
        else
          CurDAG->SelectNodeTo(N, Ty, Opc, Val.getOperand(0),
                               Val.getOperand(1), Val.getOperand(2));
        break;
      }
    }
    CurDAG->SelectNodeTo(N, Ty, PPC::FNEG, Val);
    break;
  }
  case ISD::FSQRT: {
    MVT::ValueType Ty = N->getValueType(0);
    CurDAG->SelectNodeTo(N, Ty, Ty == MVT::f64 ? PPC::FSQRT : PPC::FSQRTS,
                         Select(N->getOperand(0)));
    break;
  }
  case ISD::RET: {
    SDOperand Chain = Select(N->getOperand(0));     // Token chain.

    if (N->getNumOperands() > 1) {
      SDOperand Val = Select(N->getOperand(1));
      switch (N->getOperand(1).getValueType()) {
      default: assert(0 && "Unknown return type!");
      case MVT::f64:
      case MVT::f32:
        Chain = CurDAG->getCopyToReg(Chain, PPC::F1, Val);
        break;
      case MVT::i32:
        Chain = CurDAG->getCopyToReg(Chain, PPC::R3, Val);
        break;
      }

      if (N->getNumOperands() > 2) {
        assert(N->getOperand(1).getValueType() == MVT::i32 &&
               N->getOperand(2).getValueType() == MVT::i32 &&
               N->getNumOperands() == 2 && "Unknown two-register ret value!");
        Val = Select(N->getOperand(2));
        Chain = CurDAG->getCopyToReg(Chain, PPC::R4, Val);
      }
    }

    // Finally, select this to a blr (return) instruction.
    CurDAG->SelectNodeTo(N, MVT::Other, PPC::BLR, Chain);
    break;
  }
  }
  return SDOperand(N, 0);
}


/// createPPC32ISelDag - This pass converts a legalized DAG into a 
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createPPC32ISelDag(TargetMachine &TM) {
  return new PPC32DAGToDAGISel(TM);
}

