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
    
  public:
    PPC32DAGToDAGISel(TargetMachine &TM)
      : SelectionDAGISel(PPC32Lowering), PPC32Lowering(TM) {}
    
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
      // Select target instructions for the DAG.
      Select(DAG.getRoot());
      DAG.RemoveDeadNodes();
      
      // Emit machine code to BB. 
      ScheduleAndEmitDAG(DAG);
    }
 
    virtual const char *getPassName() const {
      return "PowerPC DAG->DAG Pattern Instruction Selection";
    } 
  };
}

// isIntImmediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isIntImmediate(SDNode *N, unsigned& Imm) {
  if (N->getOpcode() == ISD::Constant) {
    Imm = cast<ConstantSDNode>(N)->getValue();
    return true;
  }
  return false;
}

// isOprShiftImm - Returns true if the specified operand is a shift opcode with
// a immediate shift count less than 32.
static bool isOprShiftImm(SDNode *N, unsigned& Opc, unsigned& SH) {
  Opc = N->getOpcode();
  return (Opc == ISD::SHL || Opc == ISD::SRL || Opc == ISD::SRA) &&
    isIntImmediate(N->getOperand(1).Val, SH) && SH < 32;
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME) {
  if (isShiftedMask_32(Val)) {
    // look for the first non-zero bit
    MB = CountLeadingZeros_32(Val);
    // look for the first zero bit after the run of ones
    ME = CountLeadingZeros_32((Val - 1) ^ Val);
    return true;
  } else if (isShiftedMask_32(Val = ~Val)) { // invert mask
                                             // effectively look for the first zero bit
    ME = CountLeadingZeros_32(Val) - 1;
    // effectively look for the first one bit after the run of zeros
    MB = CountLeadingZeros_32((Val - 1) ^ Val) + 1;
    return true;
  }
  // no run present
  return false;
}

// isRotateAndMask - Returns true if Mask and Shift can be folded in to a rotate
// and mask opcode and mask operation.
static bool isRotateAndMask(SDNode *N, unsigned Mask, bool IsShiftMask,
                            unsigned &SH, unsigned &MB, unsigned &ME) {
  unsigned Shift  = 32;
  unsigned Indeterminant = ~0;  // bit mask marking indeterminant results
  unsigned Opcode = N->getOpcode();
  if (!isIntImmediate(N->getOperand(1).Val, Shift) || (Shift > 31))
    return false;
  
  if (Opcode == ISD::SHL) {
    // apply shift left to mask if it comes first
    if (IsShiftMask) Mask = Mask << Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu << Shift);
  } else if (Opcode == ISD::SRA || Opcode == ISD::SRL) { 
    // apply shift right to mask if it comes first
    if (IsShiftMask) Mask = Mask >> Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu >> Shift);
    // adjust for the left rotate
    Shift = 32 - Shift;
  } else {
    return false;
  }
  
  // if the mask doesn't intersect any Indeterminant bits
  if (Mask && !(Mask & Indeterminant)) {
    SH = Shift;
    // make sure the mask is still a mask (wrap arounds may not be)
    return isRunOfOnes(Mask, MB, ME);
  }
  return false;
}

// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(SDNode *N, unsigned Opc, unsigned& Imm) {
  return N->getOpcode() == Opc && isIntImmediate(N->getOperand(1).Val, Imm);
}

// isOprNot - Returns true if the specified operand is an xor with immediate -1.
static bool isOprNot(SDNode *N) {
  unsigned Imm;
  return isOpcWithIntImmediate(N, ISD::XOR, Imm) && (signed)Imm == -1;
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
    unsigned Hi = HA16(v);
    unsigned Lo = Lo16(v);
    if (Hi && Lo) {
      SDOperand Top = CurDAG->getTargetNode(PPC::LIS, MVT::i32, 
                                            getI32Imm(v >> 16));
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::ORI, Top, getI32Imm(v & 0xFFFF));
    } else if (Lo) {
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::LI, getI32Imm(v));
    } else {
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::LIS, getI32Imm(v >> 16));
    }
    break;
  }
  case ISD::SIGN_EXTEND_INREG:
    switch(cast<VTSDNode>(N->getOperand(1))->getVT()) {
    default: assert(0 && "Illegal type in SIGN_EXTEND_INREG"); break;
    case MVT::i16:
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::EXTSH, Select(N->getOperand(0)));
      break;
    case MVT::i8:
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::EXTSB, Select(N->getOperand(0)));
      break;
    }
    break;
  case ISD::CTLZ:
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, MVT::i32, PPC::CNTLZW, Select(N->getOperand(0)));
    break;
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
  case ISD::MULHS:
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, MVT::i32, PPC::MULHW, Select(N->getOperand(0)), 
                         Select(N->getOperand(1)));
    break;
  case ISD::MULHU:
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, MVT::i32, PPC::MULHWU, Select(N->getOperand(0)),
                         Select(N->getOperand(1)));
    break;
  case ISD::AND: {
    unsigned Imm;
    // If this is an and of a value rotated between 0 and 31 bits and then and'd
    // with a mask, emit rlwinm
    if (isIntImmediate(N->getOperand(1), Imm) && (isShiftedMask_32(Imm) ||
                                                  isShiftedMask_32(~Imm))) {
      SDOperand Val;
      unsigned SH, MB, ME;
      if (isRotateAndMask(N->getOperand(0).Val, Imm, false, SH, MB, ME)) {
        Val = Select(N->getOperand(0).getOperand(0));
      } else {
        Val = Select(N->getOperand(0));
        isRunOfOnes(Imm, MB, ME);
        SH = 0;
      }
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::RLWINM, Val, getI32Imm(SH),
                           getI32Imm(MB), getI32Imm(ME));
      break;
    }
    // If this is an and with an immediate that isn't a mask, then codegen it as
    // high and low 16 bit immediate ands.
    if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), 
                                           N->getOperand(1),
                                           PPC::ANDISo, PPC::ANDIo)) {
      CurDAG->ReplaceAllUsesWith(N, I); 
      N = I;
      break;
    }
    // Finally, check for the case where we are being asked to select
    // and (not(a), b) or and (a, not(b)) which can be selected as andc.
    if (isOprNot(N->getOperand(0).Val))
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::ANDC, Select(N->getOperand(1)),
                           Select(N->getOperand(0).getOperand(0)));
    else if (isOprNot(N->getOperand(1).Val))
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::ANDC, Select(N->getOperand(0)),
                           Select(N->getOperand(1).getOperand(0)));
    else
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::AND, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  }
  case ISD::XOR:
    // Check whether or not this node is a logical 'not'.  This is represented
    // by llvm as a xor with the constant value -1 (all bits set).  If this is a
    // 'not', then fold 'or' into 'nor', and so forth for the supported ops.
    if (isOprNot(N)) {
      unsigned Opc;
      SDOperand Val = Select(N->getOperand(0));
      switch (Val.getTargetOpcode()) {
      default:        Opc = 0;          break;
      case PPC::OR:   Opc = PPC::NOR;   break;
      case PPC::AND:  Opc = PPC::NAND;  break;
      case PPC::XOR:  Opc = PPC::EQV;   break;
      }
      if (Opc)
        CurDAG->SelectNodeTo(N, MVT::i32, Opc, Val.getOperand(0),
                             Val.getOperand(1));
      else
        CurDAG->SelectNodeTo(N, MVT::i32, PPC::NOR, Val, Val);
      break;
    }
    // If this is a xor with an immediate other than -1, then codegen it as high
    // and low 16 bit immediate xors.
    if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), 
                                           N->getOperand(1),
                                           PPC::XORIS, PPC::XORI)) {
      CurDAG->ReplaceAllUsesWith(N, I); 
      N = I;
      break;
    }
    // Finally, check for the case where we are being asked to select
    // xor (not(a), b) which is equivalent to not(xor a, b), which is eqv
    if (isOprNot(N->getOperand(0).Val))
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::EQV, 
                           Select(N->getOperand(0).getOperand(0)),
                           Select(N->getOperand(1)));
    else
      CurDAG->SelectNodeTo(N, MVT::i32, PPC::XOR, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  case ISD::FABS:
    CurDAG->SelectNodeTo(N, N->getValueType(0), PPC::FABS, 
                         Select(N->getOperand(0)));
    break;
  case ISD::FP_EXTEND:
    assert(MVT::f64 == N->getValueType(0) && 
           MVT::f32 == N->getOperand(0).getValueType() && "Illegal FP_EXTEND");
    CurDAG->SelectNodeTo(N, MVT::f64, PPC::FMR, Select(N->getOperand(0)));
    break;
  case ISD::FP_ROUND:
    assert(MVT::f32 == N->getValueType(0) && 
           MVT::f64 == N->getOperand(0).getValueType() && "Illegal FP_ROUND");
    CurDAG->SelectNodeTo(N, MVT::f32, PPC::FRSP, Select(N->getOperand(0)));
    break;
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

