//===-- PPC32ISelPattern.cpp - A pattern matching inst selector for PPC32 -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for 32 bit PowerPC.
// Magic number generation for integer divide from the PowerPC Compiler Writer's
// Guide, section 3.2.3.5
//
//===----------------------------------------------------------------------===//

#include "PowerPC.h"
#include "PowerPCInstrBuilder.h"
#include "PowerPCInstrInfo.h"
#include "PPC32TargetMachine.h"
#include "PPC32ISelLowering.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Statistic.h"
#include <set>
#include <algorithm>
using namespace llvm;

namespace {
Statistic<> Recorded("ppc-codegen", "Number of recording ops emitted");
Statistic<> FusedFP ("ppc-codegen", "Number of fused fp operations");
Statistic<> FrameOff("ppc-codegen", "Number of frame idx offsets collapsed");

//===--------------------------------------------------------------------===//
// ISel - PPC32 specific code to select PPC32 machine instructions for
// SelectionDAG operations.
//===--------------------------------------------------------------------===//

class ISel : public SelectionDAGISel {
  PPC32TargetLowering PPC32Lowering;
  SelectionDAG *ISelDAG;  // Hack to support us having a dag->dag transform
                          // for sdiv and udiv until it is put into the future
                          // dag combiner.

  /// ExprMap - As shared expressions are codegen'd, we keep track of which
  /// vreg the value is produced in, so we only emit one copy of each compiled
  /// tree.
  std::map<SDOperand, unsigned> ExprMap;

  unsigned GlobalBaseReg;
  bool GlobalBaseInitialized;
  bool RecordSuccess;
public:
  ISel(TargetMachine &TM) : SelectionDAGISel(PPC32Lowering), PPC32Lowering(TM),
                            ISelDAG(0) {}

  /// runOnFunction - Override this function in order to reset our per-function
  /// variables.
  virtual bool runOnFunction(Function &Fn) {
    // Make sure we re-emit a set of the global base reg if necessary
    GlobalBaseInitialized = false;
    return SelectionDAGISel::runOnFunction(Fn);
  }

  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
    DEBUG(BB->dump());
    // Codegen the basic block.
    ISelDAG = &DAG;
    Select(DAG.getRoot());

    // Clear state used for selection.
    ExprMap.clear();
    ISelDAG = 0;
  }

  // convenience functions for virtual register creation
  inline unsigned MakeIntReg() {
    return RegMap->createVirtualRegister(PPC32::GPRCRegisterClass);
  }
  inline unsigned MakeFPReg() {
    return RegMap->createVirtualRegister(PPC32::FPRCRegisterClass);
  }
  
  // dag -> dag expanders for integer divide by constant
  SDOperand BuildSDIVSequence(SDOperand N);
  SDOperand BuildUDIVSequence(SDOperand N);

  unsigned getGlobalBaseReg();
  void MoveCRtoGPR(unsigned CCReg, ISD::CondCode CC, unsigned Result);
  bool SelectBitfieldInsert(SDOperand OR, unsigned Result);
  unsigned FoldIfWideZeroExtend(SDOperand N);
  unsigned SelectCC(SDOperand LHS, SDOperand RHS, ISD::CondCode CC);
  bool SelectIntImmediateExpr(SDOperand N, unsigned Result,
                              unsigned OCHi, unsigned OCLo,
                              bool IsArithmetic = false, bool Negate = false);
  unsigned SelectExpr(SDOperand N, bool Recording=false);
  void Select(SDOperand N);

  unsigned SelectAddr(SDOperand N, unsigned& Reg, int& offset);
  void SelectBranchCC(SDOperand N);
  
  virtual const char *getPassName() const {
    return "PowerPC Pattern Instruction Selection";
  } 
};

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
static bool isRotateAndMask(unsigned Opcode, unsigned Shift, unsigned Mask,
                            bool IsShiftMask,
                            unsigned &SH, unsigned &MB, unsigned &ME) {
  if (Shift > 31) return false;
  unsigned Indeterminant = ~0;       // bit mask marking indeterminant results
  
  if (Opcode == ISD::SHL) { // shift left
    // apply shift to mask if it comes first
    if (IsShiftMask) Mask = Mask << Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu << Shift);
  } else if (Opcode == ISD::SRA || Opcode == ISD::SRL) { // shift rights
    // apply shift to mask if it comes first
    if (IsShiftMask) Mask = Mask >> Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu >> Shift);
    // adjust for the left rotate
    Shift = 32 - Shift;
  }
  
  // if the mask doesn't intersect any Indeterminant bits
  if (Mask && !(Mask & Indeterminant)) {
    SH = Shift;
    // make sure the mask is still a mask (wrap arounds may not be)
    return isRunOfOnes(Mask, MB, ME);
  }
  
  // can't do it
  return false;
}

// isIntImmediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isIntImmediate(SDOperand N, unsigned& Imm) {
  // test for constant
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    // retrieve value
    Imm = (unsigned)CN->getValue();
    // passes muster
    return true;
  }
  // not a constant
  return false;
}

// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(SDOperand N, unsigned Opc, unsigned& Imm) {
  return N.getOpcode() == Opc && isIntImmediate(N.getOperand(1), Imm);
}

// isOprShiftImm - Returns true if the specified operand is a shift opcode with
// a immediate shift count less than 32.
static bool isOprShiftImm(SDOperand N, unsigned& Opc, unsigned& SH) {
  Opc = N.getOpcode();
  return (Opc == ISD::SHL || Opc == ISD::SRL || Opc == ISD::SRA) &&
         isIntImmediate(N.getOperand(1), SH) && SH < 32;
}

// isOprNot - Returns true if the specified operand is an xor with immediate -1.
static bool isOprNot(SDOperand N) {
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

/// NodeHasRecordingVariant - If SelectExpr can always produce code for
/// NodeOpcode that also sets CR0 as a side effect, return true.  Otherwise,
/// return false.
static bool NodeHasRecordingVariant(unsigned NodeOpcode) {
  switch(NodeOpcode) {
  default: return false;
  case ISD::AND:
  case ISD::OR:
    return true;
  }
}

/// getBCCForSetCC - Returns the PowerPC condition branch mnemonic corresponding
/// to Condition.
static unsigned getBCCForSetCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETEQ:  return PPC::BEQ;
  case ISD::SETNE:  return PPC::BNE;
  case ISD::SETULT:
  case ISD::SETLT:  return PPC::BLT;
  case ISD::SETULE:
  case ISD::SETLE:  return PPC::BLE;
  case ISD::SETUGT:
  case ISD::SETGT:  return PPC::BGT;
  case ISD::SETUGE:
  case ISD::SETGE:  return PPC::BGE;
  }
  return 0;
}

/// getCRIdxForSetCC - Return the index of the condition register field
/// associated with the SetCC condition, and whether or not the field is
/// treated as inverted.  That is, lt = 0; ge = 0 inverted.
static unsigned getCRIdxForSetCC(ISD::CondCode CC, bool& Inv) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETULT:
  case ISD::SETLT:  Inv = false;  return 0;
  case ISD::SETUGE:
  case ISD::SETGE:  Inv = true;   return 0;
  case ISD::SETUGT:
  case ISD::SETGT:  Inv = false;  return 1;
  case ISD::SETULE:
  case ISD::SETLE:  Inv = true;   return 1;
  case ISD::SETEQ:  Inv = false;  return 2;
  case ISD::SETNE:  Inv = true;   return 2;
  }
  return 0;
}

/// IndexedOpForOp - Return the indexed variant for each of the PowerPC load
/// and store immediate instructions.
static unsigned IndexedOpForOp(unsigned Opcode) {
  switch(Opcode) {
  default: assert(0 && "Unknown opcode!"); abort();
  case PPC::LBZ: return PPC::LBZX;  case PPC::STB: return PPC::STBX;
  case PPC::LHZ: return PPC::LHZX;  case PPC::STH: return PPC::STHX;
  case PPC::LHA: return PPC::LHAX;  case PPC::STW: return PPC::STWX;
  case PPC::LWZ: return PPC::LWZX;  case PPC::STFS: return PPC::STFSX;
  case PPC::LFS: return PPC::LFSX;  case PPC::STFD: return PPC::STFDX;
  case PPC::LFD: return PPC::LFDX;
  }
  return 0;
}

// Structure used to return the necessary information to codegen an SDIV as
// a multiply.
struct ms {
  int m; // magic number
  int s; // shift amount
};

struct mu {
  unsigned int m; // magic number
  int a;          // add indicator
  int s;          // shift amount
};

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static struct ms magic(int d) {
  int p;
  unsigned int ad, anc, delta, q1, r1, q2, r2, t;
  const unsigned int two31 = 0x80000000U;
  struct ms mag;

  ad = abs(d);
  t = two31 + ((unsigned int)d >> 31);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 31;               // initialize p
  q1 = two31/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two31 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two31/ad;        // initialize q2 = 2p/abs(d)
  r2 = two31 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));

  mag.m = q2 + 1;
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 32;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static struct mu magicu(unsigned d)
{
  int p;
  unsigned int nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 31;                   // initialize p
  q1 = 0x80000000/nc;       // initialize q1 = 2p/nc
  r1 = 0x80000000 - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFF/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFF - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFF) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x80000000) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 32;  // resulting shift
  return magu;
}
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand ISel::BuildSDIVSequence(SDOperand N) {
  int d = (int)cast<ConstantSDNode>(N.getOperand(1))->getSignExtended();
  ms magics = magic(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = ISelDAG->getNode(ISD::MULHS, MVT::i32, N.getOperand(0),
                                 ISelDAG->getConstant(magics.m, MVT::i32));
  // If d > 0 and m < 0, add the numerator
  if (d > 0 && magics.m < 0)
    Q = ISelDAG->getNode(ISD::ADD, MVT::i32, Q, N.getOperand(0));
  // If d < 0 and m > 0, subtract the numerator.
  if (d < 0 && magics.m > 0)
    Q = ISelDAG->getNode(ISD::SUB, MVT::i32, Q, N.getOperand(0));
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0)
    Q = ISelDAG->getNode(ISD::SRA, MVT::i32, Q,
                         ISelDAG->getConstant(magics.s, MVT::i32));
  // Extract the sign bit and add it to the quotient
  SDOperand T =
    ISelDAG->getNode(ISD::SRL, MVT::i32, Q, ISelDAG->getConstant(31, MVT::i32));
  return ISelDAG->getNode(ISD::ADD, MVT::i32, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand ISel::BuildUDIVSequence(SDOperand N) {
  unsigned d =
    (unsigned)cast<ConstantSDNode>(N.getOperand(1))->getSignExtended();
  mu magics = magicu(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = ISelDAG->getNode(ISD::MULHU, MVT::i32, N.getOperand(0),
                                 ISelDAG->getConstant(magics.m, MVT::i32));
  if (magics.a == 0) {
    Q = ISelDAG->getNode(ISD::SRL, MVT::i32, Q,
                         ISelDAG->getConstant(magics.s, MVT::i32));
  } else {
    SDOperand NPQ = ISelDAG->getNode(ISD::SUB, MVT::i32, N.getOperand(0), Q);
    NPQ = ISelDAG->getNode(ISD::SRL, MVT::i32, NPQ,
                           ISelDAG->getConstant(1, MVT::i32));
    NPQ = ISelDAG->getNode(ISD::ADD, MVT::i32, NPQ, Q);
    Q = ISelDAG->getNode(ISD::SRL, MVT::i32, NPQ,
                           ISelDAG->getConstant(magics.s-1, MVT::i32));
  }
  return Q;
}

/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
unsigned ISel::getGlobalBaseReg() {
  if (!GlobalBaseInitialized) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    GlobalBaseReg = MakeIntReg();
    BuildMI(FirstMBB, MBBI, PPC::MovePCtoLR, 0, PPC::LR);
    BuildMI(FirstMBB, MBBI, PPC::MFLR, 1, GlobalBaseReg);
    GlobalBaseInitialized = true;
  }
  return GlobalBaseReg;
}

/// MoveCRtoGPR - Move CCReg[Idx] to the least significant bit of Result.  If
/// Inv is true, then invert the result.
void ISel::MoveCRtoGPR(unsigned CCReg, ISD::CondCode CC, unsigned Result){
  bool Inv;
  unsigned IntCR = MakeIntReg();
  unsigned Idx = getCRIdxForSetCC(CC, Inv);
  BuildMI(BB, PPC::MCRF, 1, PPC::CR7).addReg(CCReg);
  bool GPOpt =
    TLI.getTargetMachine().getSubtarget<PPCSubtarget>().isGigaProcessor();
  if (GPOpt)
    BuildMI(BB, PPC::MFOCRF, 1, IntCR).addReg(PPC::CR7);
  else
    BuildMI(BB, PPC::MFCR, 0, IntCR);
  if (Inv) {
    unsigned Tmp1 = MakeIntReg();
    BuildMI(BB, PPC::RLWINM, 4, Tmp1).addReg(IntCR).addImm(32-(3-Idx))
      .addImm(31).addImm(31);
    BuildMI(BB, PPC::XORI, 2, Result).addReg(Tmp1).addImm(1);
  } else {
    BuildMI(BB, PPC::RLWINM, 4, Result).addReg(IntCR).addImm(32-(3-Idx))
      .addImm(31).addImm(31);
  }
}

/// SelectBitfieldInsert - turn an or of two masked values into
/// the rotate left word immediate then mask insert (rlwimi) instruction.
/// Returns true on success, false if the caller still needs to select OR.
///
/// Patterns matched:
/// 1. or shl, and   5. or and, and
/// 2. or and, shl   6. or shl, shr
/// 3. or shr, and   7. or shr, shl
/// 4. or and, shr
bool ISel::SelectBitfieldInsert(SDOperand OR, unsigned Result) {
  bool IsRotate = false;
  unsigned TgtMask = 0xFFFFFFFF, InsMask = 0xFFFFFFFF, Amount = 0;
  unsigned Value;

  SDOperand Op0 = OR.getOperand(0);
  SDOperand Op1 = OR.getOperand(1);

  unsigned Op0Opc = Op0.getOpcode();
  unsigned Op1Opc = Op1.getOpcode();

  // Verify that we have the correct opcodes
  if (ISD::SHL != Op0Opc && ISD::SRL != Op0Opc && ISD::AND != Op0Opc)
    return false;
  if (ISD::SHL != Op1Opc && ISD::SRL != Op1Opc && ISD::AND != Op1Opc)
    return false;

  // Generate Mask value for Target
  if (isIntImmediate(Op0.getOperand(1), Value)) {
    switch(Op0Opc) {
    case ISD::SHL: TgtMask <<= Value; break;
    case ISD::SRL: TgtMask >>= Value; break;
    case ISD::AND: TgtMask &= Value; break;
    }
  } else {
    return false;
  }

  // Generate Mask value for Insert
  if (isIntImmediate(Op1.getOperand(1), Value)) {
    switch(Op1Opc) {
    case ISD::SHL:
      Amount = Value;
      InsMask <<= Amount;
      if (Op0Opc == ISD::SRL) IsRotate = true;
      break;
    case ISD::SRL:
      Amount = Value;
      InsMask >>= Amount;
      Amount = 32-Amount;
      if (Op0Opc == ISD::SHL) IsRotate = true;
      break;
    case ISD::AND:
      InsMask &= Value;
      break;
    }
  } else {
    return false;
  }

  unsigned Tmp3 = 0;

  // If both of the inputs are ANDs and one of them has a logical shift by
  // constant as its input, make that the inserted value so that we can combine
  // the shift into the rotate part of the rlwimi instruction
  if (Op0Opc == ISD::AND && Op1Opc == ISD::AND) {
    if (Op1.getOperand(0).getOpcode() == ISD::SHL ||
        Op1.getOperand(0).getOpcode() == ISD::SRL) {
      if (isIntImmediate(Op1.getOperand(0).getOperand(1), Value)) {
        Amount = Op1.getOperand(0).getOpcode() == ISD::SHL ?
          Value : 32 - Value;
        Tmp3 = SelectExpr(Op1.getOperand(0).getOperand(0));
      }
    } else if (Op0.getOperand(0).getOpcode() == ISD::SHL ||
               Op0.getOperand(0).getOpcode() == ISD::SRL) {
      if (isIntImmediate(Op0.getOperand(0).getOperand(1), Value)) {
        std::swap(Op0, Op1);
        std::swap(TgtMask, InsMask);
        Amount = Op1.getOperand(0).getOpcode() == ISD::SHL ?
          Value : 32 - Value;
        Tmp3 = SelectExpr(Op1.getOperand(0).getOperand(0));
      }
    }
  }

  // Verify that the Target mask and Insert mask together form a full word mask
  // and that the Insert mask is a run of set bits (which implies both are runs
  // of set bits).  Given that, Select the arguments and generate the rlwimi
  // instruction.
  unsigned MB, ME;
  if (((TgtMask & InsMask) == 0) && isRunOfOnes(InsMask, MB, ME)) {
    unsigned Tmp1, Tmp2;
    bool fullMask = (TgtMask ^ InsMask) == 0xFFFFFFFF;
    // Check for rotlwi / rotrwi here, a special case of bitfield insert
    // where both bitfield halves are sourced from the same value.
    if (IsRotate && fullMask &&
        OR.getOperand(0).getOperand(0) == OR.getOperand(1).getOperand(0)) {
      Tmp1 = SelectExpr(OR.getOperand(0).getOperand(0));
      BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(Amount)
        .addImm(0).addImm(31);
      return true;
    }
    if (Op0Opc == ISD::AND && fullMask)
      Tmp1 = SelectExpr(Op0.getOperand(0));
    else
      Tmp1 = SelectExpr(Op0);
    Tmp2 = Tmp3 ? Tmp3 : SelectExpr(Op1.getOperand(0));
    BuildMI(BB, PPC::RLWIMI, 5, Result).addReg(Tmp1).addReg(Tmp2)
      .addImm(Amount).addImm(MB).addImm(ME);
    return true;
  }
  return false;
}

/// FoldIfWideZeroExtend - 32 bit PowerPC implicit masks shift amounts to the
/// low six bits.  If the shift amount is an ISD::AND node with a mask that is
/// wider than the implicit mask, then we can get rid of the AND and let the
/// shift do the mask.
unsigned ISel::FoldIfWideZeroExtend(SDOperand N) {
  unsigned C;
  if (isOpcWithIntImmediate(N, ISD::AND, C) && isMask_32(C) && C > 63)
    return SelectExpr(N.getOperand(0));
  else
    return SelectExpr(N);
}

unsigned ISel::SelectCC(SDOperand LHS, SDOperand RHS, ISD::CondCode CC) {
  unsigned Result, Tmp1, Tmp2;
  bool AlreadySelected = false;
  static const unsigned CompareOpcodes[] =
    { PPC::FCMPU, PPC::FCMPU, PPC::CMPW, PPC::CMPLW };

  // Allocate a condition register for this expression
  Result = RegMap->createVirtualRegister(PPC32::CRRCRegisterClass);

  // Use U to determine whether the SETCC immediate range is signed or not.
  bool U = ISD::isUnsignedIntSetCC(CC);
  if (isIntImmediate(RHS, Tmp2) && 
      ((U && isUInt16(Tmp2)) || (!U && isInt16(Tmp2)))) {
    Tmp2 = Lo16(Tmp2);
    // For comparisons against zero, we can implicity set CR0 if a recording
    // variant (e.g. 'or.' instead of 'or') of the instruction that defines
    // operand zero of the SetCC node is available.
    if (Tmp2 == 0 &&
        NodeHasRecordingVariant(LHS.getOpcode()) && LHS.Val->hasOneUse()) {
      RecordSuccess = false;
      Tmp1 = SelectExpr(LHS, true);
      if (RecordSuccess) {
        ++Recorded;
        BuildMI(BB, PPC::MCRF, 1, Result).addReg(PPC::CR0);
        return Result;
      }
      AlreadySelected = true;
    }
    // If we could not implicitly set CR0, then emit a compare immediate
    // instead.
    if (!AlreadySelected) Tmp1 = SelectExpr(LHS);
    if (U)
      BuildMI(BB, PPC::CMPLWI, 2, Result).addReg(Tmp1).addImm(Tmp2);
    else
      BuildMI(BB, PPC::CMPWI, 2, Result).addReg(Tmp1).addSImm(Tmp2);
  } else {
    bool IsInteger = MVT::isInteger(LHS.getValueType());
    unsigned CompareOpc = CompareOpcodes[2 * IsInteger + U];
    Tmp1 = SelectExpr(LHS);
    Tmp2 = SelectExpr(RHS);
    BuildMI(BB, CompareOpc, 2, Result).addReg(Tmp1).addReg(Tmp2);
  }
  return Result;
}

/// Check to see if the load is a constant offset from a base register.
unsigned ISel::SelectAddr(SDOperand N, unsigned& Reg, int& offset)
{
  unsigned imm = 0, opcode = N.getOpcode();
  if (N.getOpcode() == ISD::ADD) {
    bool isFrame = N.getOperand(0).getOpcode() == ISD::FrameIndex;
    if (isIntImmediate(N.getOperand(1), imm) && isInt16(imm)) {
      offset = Lo16(imm);
      if (isFrame) {
        ++FrameOff;
        Reg = cast<FrameIndexSDNode>(N.getOperand(0))->getIndex();
        return 1;
      } else {
        Reg = SelectExpr(N.getOperand(0));
        return 0;
      }
    } else {
      Reg = SelectExpr(N.getOperand(0));
      offset = SelectExpr(N.getOperand(1));
      return 2;
    }
  }
  // Now check if we're dealing with a global, and whether or not we should emit
  // an optimized load or store for statics.
  if(GlobalAddressSDNode *GN = dyn_cast<GlobalAddressSDNode>(N)) {
    GlobalValue *GV = GN->getGlobal();
    if (!GV->hasWeakLinkage() && !GV->isExternal()) {
      unsigned GlobalHi = MakeIntReg();
      if (PICEnabled)
        BuildMI(BB, PPC::ADDIS, 2, GlobalHi).addReg(getGlobalBaseReg())
          .addGlobalAddress(GV);
      else
        BuildMI(BB, PPC::LIS, 1, GlobalHi).addGlobalAddress(GV);
      Reg = GlobalHi;
      offset = 0;
      return 3;
    }
  }
  Reg = SelectExpr(N);
  offset = 0;
  return 0;
}

void ISel::SelectBranchCC(SDOperand N)
{
  MachineBasicBlock *Dest =
    cast<BasicBlockSDNode>(N.getOperand(4))->getBasicBlock();

  Select(N.getOperand(0));  //chain
  ISD::CondCode CC = cast<CondCodeSDNode>(N.getOperand(1))->get();
  unsigned CCReg = SelectCC(N.getOperand(2), N.getOperand(3), CC);
  unsigned Opc = getBCCForSetCC(CC);

  // If this is a two way branch, then grab the fallthrough basic block argument
  // and build a PowerPC branch pseudo-op, suitable for long branch conversion
  // if necessary by the branch selection pass.  Otherwise, emit a standard
  // conditional branch.
  if (N.getOpcode() == ISD::BRTWOWAY_CC) {
    MachineBasicBlock *Fallthrough =
      cast<BasicBlockSDNode>(N.getOperand(5))->getBasicBlock();
    BuildMI(BB, PPC::COND_BRANCH, 4).addReg(CCReg).addImm(Opc)
      .addMBB(Dest).addMBB(Fallthrough);
    BuildMI(BB, PPC::B, 1).addMBB(Fallthrough);
  } else {
    // Iterate to the next basic block
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    // If the fallthrough path is off the end of the function, which would be
    // undefined behavior, set it to be the same as the current block because
    // we have nothing better to set it to, and leaving it alone will cause the
    // PowerPC Branch Selection pass to crash.
    if (It == BB->getParent()->end()) It = Dest;
    BuildMI(BB, PPC::COND_BRANCH, 4).addReg(CCReg).addImm(Opc)
      .addMBB(Dest).addMBB(It);
  }
  return;
}

// SelectIntImmediateExpr - Choose code for opcodes with immediate value.
bool ISel::SelectIntImmediateExpr(SDOperand N, unsigned Result,
                                  unsigned OCHi, unsigned OCLo,
                                  bool IsArithmetic, bool Negate) {
  // check constant
  ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1));
  // exit if not a constant
  if (!CN) return false;
  // extract immediate
  unsigned C = (unsigned)CN->getValue();
  // negate if required (ISD::SUB)
  if (Negate) C = -C;
  // get the hi and lo portions of constant
  unsigned Hi = IsArithmetic ? HA16(C) : Hi16(C);
  unsigned Lo = Lo16(C);
  // assume no intermediate result from lo instruction (same as final result)
  unsigned Tmp = Result;
  // check if two instructions are needed
  if (Hi && Lo) {
    // exit if usage indicates it would be better to load immediate into a 
    // register
    if (CN->use_size() > 2) return false;
    // need intermediate result for two instructions
    Tmp = MakeIntReg();
  }
  // get first operand
  unsigned Opr0 = SelectExpr(N.getOperand(0));
  // is a lo instruction needed
  if (Lo) {
    // generate instruction for lo portion
    BuildMI(BB, OCLo, 2, Tmp).addReg(Opr0).addImm(Lo);
    // need to switch out first operand for hi instruction
    Opr0 = Tmp;
  }
  // is a hi instruction needed
  if (Hi) {
    // generate instruction for hi portion
    BuildMI(BB, OCHi, 2, Result).addReg(Opr0).addImm(Hi);
  }
  return true;
}

unsigned ISel::SelectExpr(SDOperand N, bool Recording) {
  unsigned Result;
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  unsigned opcode = N.getOpcode();

  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();

  if (Node->getOpcode() == ISD::CopyFromReg) {
    unsigned Reg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    // Just use the specified register as our input.
    if (MRegisterInfo::isVirtualRegister(Reg) || Reg == PPC::R1)
      return Reg;
  }

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  switch (N.getOpcode()) {
  default:
    Reg = Result = (N.getValueType() != MVT::Other) ?
                            MakeReg(N.getValueType()) : 1;
    break;
  case ISD::TAILCALL:
  case ISD::CALL:
    // If this is a call instruction, make sure to prepare ALL of the result
    // values as well as the chain.
    if (Node->getNumValues() == 1)
      Reg = Result = 1;  // Void call, just a chain.
    else {
      Result = MakeReg(Node->getValueType(0));
      ExprMap[N.getValue(0)] = Result;
      for (unsigned i = 1, e = N.Val->getNumValues()-1; i != e; ++i)
        ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
      ExprMap[SDOperand(Node, Node->getNumValues()-1)] = 1;
    }
    break;
  case ISD::ADD_PARTS:
  case ISD::SUB_PARTS:
  case ISD::SHL_PARTS:
  case ISD::SRL_PARTS:
  case ISD::SRA_PARTS:
    Result = MakeReg(Node->getValueType(0));
    ExprMap[N.getValue(0)] = Result;
    for (unsigned i = 1, e = N.Val->getNumValues(); i != e; ++i)
      ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
    break;
  }

  switch (opcode) {
  default:
    Node->dump(); std::cerr << '\n';
    assert(0 && "Node not handled!\n");
  case ISD::BUILTIN_OP_END+PPC::FSEL:
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    Tmp3 = SelectExpr(N.getOperand(2));
    BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
    return Result;
  case ISD::UNDEF:
    if (Node->getValueType(0) == MVT::i32)
      BuildMI(BB, PPC::IMPLICIT_DEF_GPR, 0, Result);
    else
      BuildMI(BB, PPC::IMPLICIT_DEF_FP, 0, Result);
    return Result;
  case ISD::DYNAMIC_STACKALLOC:
    // Generate both result values.  FIXME: Need a better commment here?
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    // FIXME: We are currently ignoring the requested alignment for handling
    // greater than the stack alignment.  This will need to be revisited at some
    // point.  Align = N.getOperand(2);
    if (!isa<ConstantSDNode>(N.getOperand(2)) ||
        cast<ConstantSDNode>(N.getOperand(2))->getValue() != 0) {
      std::cerr << "Cannot allocate stack object with greater alignment than"
                << " the stack alignment yet!";
      abort();
    }
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    // Subtract size from stack pointer, thereby allocating some space.
    BuildMI(BB, PPC::SUBF, 2, PPC::R1).addReg(Tmp1).addReg(PPC::R1);
    // Put a pointer to the space into the result register by copying the SP
    BuildMI(BB, PPC::OR, 2, Result).addReg(PPC::R1).addReg(PPC::R1);
    return Result;

  case ISD::ConstantPool:
    Tmp1 = BB->getParent()->getConstantPool()->
               getConstantPoolIndex(cast<ConstantPoolSDNode>(N)->get());
    Tmp2 = MakeIntReg();
    if (PICEnabled)
      BuildMI(BB, PPC::ADDIS, 2, Tmp2).addReg(getGlobalBaseReg())
        .addConstantPoolIndex(Tmp1);
    else
      BuildMI(BB, PPC::LIS, 1, Tmp2).addConstantPoolIndex(Tmp1);
    BuildMI(BB, PPC::LA, 2, Result).addReg(Tmp2).addConstantPoolIndex(Tmp1);
    return Result;

  case ISD::FrameIndex:
    Tmp1 = cast<FrameIndexSDNode>(N)->getIndex();
    addFrameReference(BuildMI(BB, PPC::ADDI, 2, Result), (int)Tmp1, 0, false);
    return Result;

  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
    Tmp1 = MakeIntReg();
    if (PICEnabled)
      BuildMI(BB, PPC::ADDIS, 2, Tmp1).addReg(getGlobalBaseReg())
        .addGlobalAddress(GV);
    else
      BuildMI(BB, PPC::LIS, 1, Tmp1).addGlobalAddress(GV);
    if (GV->hasWeakLinkage() || GV->isExternal()) {
      BuildMI(BB, PPC::LWZ, 2, Result).addGlobalAddress(GV).addReg(Tmp1);
    } else {
      BuildMI(BB, PPC::LA, 2, Result).addReg(Tmp1).addGlobalAddress(GV);
    }
    return Result;
  }

  case ISD::LOAD:
  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD: {
    MVT::ValueType TypeBeingLoaded = (ISD::LOAD == opcode) ?
      Node->getValueType(0) : cast<VTSDNode>(Node->getOperand(3))->getVT();
    bool sext = (ISD::SEXTLOAD == opcode);

    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    SDOperand Chain   = N.getOperand(0);
    SDOperand Address = N.getOperand(1);
    Select(Chain);

    switch (TypeBeingLoaded) {
    default: Node->dump(); assert(0 && "Cannot load this type!");
    case MVT::i1:  Opc = PPC::LBZ; break;
    case MVT::i8:  Opc = PPC::LBZ; break;
    case MVT::i16: Opc = sext ? PPC::LHA : PPC::LHZ; break;
    case MVT::i32: Opc = PPC::LWZ; break;
    case MVT::f32: Opc = PPC::LFS; break;
    case MVT::f64: Opc = PPC::LFD; break;
    }

    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
      Tmp1 = MakeIntReg();
      unsigned CPI = BB->getParent()->getConstantPool()->
        getConstantPoolIndex(CP->get());
      if (PICEnabled)
        BuildMI(BB, PPC::ADDIS, 2, Tmp1).addReg(getGlobalBaseReg())
          .addConstantPoolIndex(CPI);
      else
        BuildMI(BB, PPC::LIS, 1, Tmp1).addConstantPoolIndex(CPI);
      BuildMI(BB, Opc, 2, Result).addConstantPoolIndex(CPI).addReg(Tmp1);
    } else if (Address.getOpcode() == ISD::FrameIndex) {
      Tmp1 = cast<FrameIndexSDNode>(Address)->getIndex();
      addFrameReference(BuildMI(BB, Opc, 2, Result), (int)Tmp1);
    } else {
      int offset;
      switch(SelectAddr(Address, Tmp1, offset)) {
      default: assert(0 && "Unhandled return value from SelectAddr");
      case 0:   // imm offset, no frame, no index
        BuildMI(BB, Opc, 2, Result).addSImm(offset).addReg(Tmp1);
        break;
      case 1:   // imm offset + frame index
        addFrameReference(BuildMI(BB, Opc, 2, Result), (int)Tmp1, offset);
        break;
      case 2:   // base+index addressing
        Opc = IndexedOpForOp(Opc);
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(offset);
        break;
      case 3: {
        GlobalAddressSDNode *GN = cast<GlobalAddressSDNode>(Address);
        GlobalValue *GV = GN->getGlobal();
        BuildMI(BB, Opc, 2, Result).addGlobalAddress(GV).addReg(Tmp1);
      }
      }
    }
    return Result;
  }

  case ISD::TAILCALL:
  case ISD::CALL: {
    unsigned GPR_idx = 0, FPR_idx = 0;
    static const unsigned GPR[] = {
      PPC::R3, PPC::R4, PPC::R5, PPC::R6,
      PPC::R7, PPC::R8, PPC::R9, PPC::R10,
    };
    static const unsigned FPR[] = {
      PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
      PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
    };

    // Lower the chain for this call.
    Select(N.getOperand(0));
    ExprMap[N.getValue(Node->getNumValues()-1)] = 1;

    MachineInstr *CallMI;
    // Emit the correct call instruction based on the type of symbol called.
    if (GlobalAddressSDNode *GASD =
        dyn_cast<GlobalAddressSDNode>(N.getOperand(1))) {
      CallMI = BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(GASD->getGlobal(),
                                                           true);
    } else if (ExternalSymbolSDNode *ESSDN =
               dyn_cast<ExternalSymbolSDNode>(N.getOperand(1))) {
      CallMI = BuildMI(PPC::CALLpcrel, 1).addExternalSymbol(ESSDN->getSymbol(),
                                                            true);
    } else {
      Tmp1 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::MTCTR, 1).addReg(Tmp1);
      BuildMI(BB, PPC::OR, 2, PPC::R12).addReg(Tmp1).addReg(Tmp1);
      CallMI = BuildMI(PPC::CALLindirect, 3).addImm(20).addImm(0)
        .addReg(PPC::R12);
    }

    // Load the register args to virtual regs
    std::vector<unsigned> ArgVR;
    for(int i = 2, e = Node->getNumOperands(); i < e; ++i)
      ArgVR.push_back(SelectExpr(N.getOperand(i)));

    // Copy the virtual registers into the appropriate argument register
    for(int i = 0, e = ArgVR.size(); i < e; ++i) {
      switch(N.getOperand(i+2).getValueType()) {
      default: Node->dump(); assert(0 && "Unknown value type for call");
      case MVT::i32:
        assert(GPR_idx < 8 && "Too many int args");
        if (N.getOperand(i+2).getOpcode() != ISD::UNDEF) {
          BuildMI(BB, PPC::OR,2,GPR[GPR_idx]).addReg(ArgVR[i]).addReg(ArgVR[i]);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
        }
        ++GPR_idx;
        break;
      case MVT::f64:
      case MVT::f32:
        assert(FPR_idx < 13 && "Too many fp args");
        BuildMI(BB, PPC::FMR, 1, FPR[FPR_idx]).addReg(ArgVR[i]);
        CallMI->addRegOperand(FPR[FPR_idx], MachineOperand::Use);
        ++FPR_idx;
        break;
      }
    }

    // Put the call instruction in the correct place in the MachineBasicBlock
    BB->push_back(CallMI);

    switch (Node->getValueType(0)) {
    default: assert(0 && "Unknown value type for call result!");
    case MVT::Other: return 1;
    case MVT::i32:
      if (Node->getValueType(1) == MVT::i32) {
        BuildMI(BB, PPC::OR, 2, Result+1).addReg(PPC::R3).addReg(PPC::R3);
        BuildMI(BB, PPC::OR, 2, Result).addReg(PPC::R4).addReg(PPC::R4);
      } else {
        BuildMI(BB, PPC::OR, 2, Result).addReg(PPC::R3).addReg(PPC::R3);
      }
      break;
    case MVT::f32:
    case MVT::f64:
      BuildMI(BB, PPC::FMR, 1, Result).addReg(PPC::F1);
      break;
    }
    return Result+N.ResNo;
  }

  case ISD::SIGN_EXTEND:
  case ISD::SIGN_EXTEND_INREG:
    Tmp1 = SelectExpr(N.getOperand(0));
    switch(cast<VTSDNode>(Node->getOperand(1))->getVT()) {
    default: Node->dump(); assert(0 && "Unhandled SIGN_EXTEND type"); break;
    case MVT::i16:
      BuildMI(BB, PPC::EXTSH, 1, Result).addReg(Tmp1);
      break;
    case MVT::i8:
      BuildMI(BB, PPC::EXTSB, 1, Result).addReg(Tmp1);
      break;
    }
    return Result;

  case ISD::CopyFromReg:
    DestType = N.getValue(0).getValueType();
    if (Result == 1)
      Result = ExprMap[N.getValue(0)] = MakeReg(DestType);
    Tmp1 = dyn_cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    if (MVT::isInteger(DestType))
      BuildMI(BB, PPC::OR, 2, Result).addReg(Tmp1).addReg(Tmp1);
    else
      BuildMI(BB, PPC::FMR, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::SHL:
    if (isIntImmediate(N.getOperand(1), Tmp2)) {
      unsigned SH, MB, ME;
      if (isOpcWithIntImmediate(N.getOperand(0), ISD::AND, Tmp3) &&
          isRotateAndMask(ISD::SHL, Tmp2, Tmp3, true, SH, MB, ME)) {
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(SH)
          .addImm(MB).addImm(ME);
        return Result;
      }
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 &= 0x1F;
      BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(Tmp2).addImm(0)
        .addImm(31-Tmp2);
    } else {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = FoldIfWideZeroExtend(N.getOperand(1));
      BuildMI(BB, PPC::SLW, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::SRL:
    if (isIntImmediate(N.getOperand(1), Tmp2)) {
      unsigned SH, MB, ME;
      if (isOpcWithIntImmediate(N.getOperand(0), ISD::AND, Tmp3) &&
          isRotateAndMask(ISD::SRL, Tmp2, Tmp3, true, SH, MB, ME)) {
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(SH)
          .addImm(MB).addImm(ME);
        return Result;
      }
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 &= 0x1F;
      BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(32-Tmp2)
        .addImm(Tmp2).addImm(31);
    } else {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = FoldIfWideZeroExtend(N.getOperand(1));
      BuildMI(BB, PPC::SRW, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::SRA:
    if (isIntImmediate(N.getOperand(1), Tmp2)) {
      unsigned SH, MB, ME;
      if (isOpcWithIntImmediate(N.getOperand(0), ISD::AND, Tmp3) &&
          isRotateAndMask(ISD::SRA, Tmp2, Tmp3, true, SH, MB, ME)) {
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(SH)
          .addImm(MB).addImm(ME);
        return Result;
      }
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 &= 0x1F;
      BuildMI(BB, PPC::SRAWI, 2, Result).addReg(Tmp1).addImm(Tmp2);
    } else {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = FoldIfWideZeroExtend(N.getOperand(1));
      BuildMI(BB, PPC::SRAW, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::CTLZ:
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, PPC::CNTLZW, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::ADD:
    if (!MVT::isInteger(DestType)) {
      if (!NoExcessFPPrecision && N.getOperand(0).getOpcode() == ISD::MUL &&
          N.getOperand(0).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        Tmp3 = SelectExpr(N.getOperand(1));
        Opc = DestType == MVT::f64 ? PPC::FMADD : PPC::FMADDS;
        BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
        return Result;
      }
      if (!NoExcessFPPrecision && N.getOperand(1).getOpcode() == ISD::MUL &&
          N.getOperand(1).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        Tmp1 = SelectExpr(N.getOperand(1).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(1).getOperand(1));
        Tmp3 = SelectExpr(N.getOperand(0));
        Opc = DestType == MVT::f64 ? PPC::FMADD : PPC::FMADDS;
        BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
        return Result;
      }
      Opc = DestType == MVT::f64 ? PPC::FADD : PPC::FADDS;
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
    if (SelectIntImmediateExpr(N, Result, PPC::ADDIS, PPC::ADDI, true))
      return Result;
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, PPC::ADD, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::AND:
    if (isIntImmediate(N.getOperand(1), Tmp2)) {
      if (isShiftedMask_32(Tmp2) || isShiftedMask_32(~Tmp2)) {
        unsigned SH, MB, ME;
        Opc = Recording ? PPC::RLWINMo : PPC::RLWINM;
        unsigned OprOpc;
        if (isOprShiftImm(N.getOperand(0), OprOpc, Tmp3) &&
            isRotateAndMask(OprOpc, Tmp3, Tmp2, false, SH, MB, ME)) {
          Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        } else {
          Tmp1 = SelectExpr(N.getOperand(0));
          isRunOfOnes(Tmp2, MB, ME);
          SH = 0;
        }
        BuildMI(BB, Opc, 4, Result).addReg(Tmp1).addImm(SH)
          .addImm(MB).addImm(ME);
        RecordSuccess = true;
        return Result;
      } else if (isUInt16(Tmp2)) {
        Tmp2 = Lo16(Tmp2);
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, PPC::ANDIo, 2, Result).addReg(Tmp1).addImm(Tmp2);
        RecordSuccess = true;
        return Result;
      } else if (isUInt16(Tmp2)) {
        Tmp2 = Hi16(Tmp2);
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, PPC::ANDISo, 2, Result).addReg(Tmp1).addImm(Tmp2);
        RecordSuccess = true;
       return Result;
      }
    }
    if (isOprNot(N.getOperand(1))) {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1).getOperand(0));
      BuildMI(BB, PPC::ANDC, 2, Result).addReg(Tmp1).addReg(Tmp2);
      RecordSuccess = false;
      return Result;
    }
    if (isOprNot(N.getOperand(0))) {
      Tmp1 = SelectExpr(N.getOperand(1));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(0));
      BuildMI(BB, PPC::ANDC, 2, Result).addReg(Tmp1).addReg(Tmp2);
      RecordSuccess = false;
      return Result;
    }
    // emit a regular and
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    Opc = Recording ? PPC::ANDo : PPC::AND;
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    RecordSuccess = true;
    return Result;

  case ISD::OR:
    if (SelectBitfieldInsert(N, Result))
      return Result;
    if (SelectIntImmediateExpr(N, Result, PPC::ORIS, PPC::ORI))
      return Result;
    if (isOprNot(N.getOperand(1))) {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1).getOperand(0));
      BuildMI(BB, PPC::ORC, 2, Result).addReg(Tmp1).addReg(Tmp2);
      RecordSuccess = false;
      return Result;
    }
    if (isOprNot(N.getOperand(0))) {
      Tmp1 = SelectExpr(N.getOperand(1));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(0));
      BuildMI(BB, PPC::ORC, 2, Result).addReg(Tmp1).addReg(Tmp2);
      RecordSuccess = false;
      return Result;
    }
    // emit regular or
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    Opc = Recording ? PPC::ORo : PPC::OR;
    RecordSuccess = true;
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::XOR: {
    // Check for EQV: xor, (xor a, -1), b
    if (isOprNot(N.getOperand(0))) {
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::EQV, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
    // Check for NOT, NOR, EQV, and NAND: xor (copy, or, xor, and), -1
    if (isOprNot(N)) {
      switch(N.getOperand(0).getOpcode()) {
      case ISD::OR:
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        BuildMI(BB, PPC::NOR, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case ISD::AND:
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        BuildMI(BB, PPC::NAND, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case ISD::XOR:
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        BuildMI(BB, PPC::EQV, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      default:
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, PPC::NOR, 2, Result).addReg(Tmp1).addReg(Tmp1);
        break;
      }
      return Result;
    }
    if (SelectIntImmediateExpr(N, Result, PPC::XORIS, PPC::XORI))
      return Result;
    // emit regular xor
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, PPC::XOR, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  }

   case ISD::SUB:
    if (!MVT::isInteger(DestType)) {
      if (!NoExcessFPPrecision && N.getOperand(0).getOpcode() == ISD::MUL &&
          N.getOperand(0).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        Tmp3 = SelectExpr(N.getOperand(1));
        Opc = DestType == MVT::f64 ? PPC::FMSUB : PPC::FMSUBS;
        BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
        return Result;
      }
      if (!NoExcessFPPrecision && N.getOperand(1).getOpcode() == ISD::MUL &&
          N.getOperand(1).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        Tmp1 = SelectExpr(N.getOperand(1).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(1).getOperand(1));
        Tmp3 = SelectExpr(N.getOperand(0));
        Opc = DestType == MVT::f64 ? PPC::FNMSUB : PPC::FNMSUBS;
        BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
        return Result;
      }
      Opc = DestType == MVT::f64 ? PPC::FSUB : PPC::FSUBS;
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
    if (isIntImmediate(N.getOperand(0), Tmp1) && isInt16(Tmp1)) {
      Tmp1 = Lo16(Tmp1);
      Tmp2 = SelectExpr(N.getOperand(1));
      if (0 == Tmp1)
        BuildMI(BB, PPC::NEG, 1, Result).addReg(Tmp2);
      else
        BuildMI(BB, PPC::SUBFIC, 2, Result).addReg(Tmp2).addSImm(Tmp1);
      return Result;
    }
    if (SelectIntImmediateExpr(N, Result, PPC::ADDIS, PPC::ADDI, true, true))
        return Result;
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, PPC::SUBF, 2, Result).addReg(Tmp2).addReg(Tmp1);
    return Result;

  case ISD::MUL:
    Tmp1 = SelectExpr(N.getOperand(0));
    if (isIntImmediate(N.getOperand(1), Tmp2) && isInt16(Tmp2)) {
      Tmp2 = Lo16(Tmp2);
      BuildMI(BB, PPC::MULLI, 2, Result).addReg(Tmp1).addSImm(Tmp2);
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      switch (DestType) {
      default: assert(0 && "Unknown type to ISD::MUL"); break;
      case MVT::i32: Opc = PPC::MULLW; break;
      case MVT::f32: Opc = PPC::FMULS; break;
      case MVT::f64: Opc = PPC::FMUL; break;
      }
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::MULHS:
  case ISD::MULHU:
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    Opc = (ISD::MULHU == opcode) ? PPC::MULHWU : PPC::MULHW;
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::SDIV:
    if (isIntImmediate(N.getOperand(1), Tmp3)) {
      if ((signed)Tmp3 > 0 && isPowerOf2_32(Tmp3)) {
        Tmp3 = Log2_32(Tmp3);
        Tmp1 = MakeIntReg();
        Tmp2 = SelectExpr(N.getOperand(0));
        BuildMI(BB, PPC::SRAWI, 2, Tmp1).addReg(Tmp2).addImm(Tmp3);
        BuildMI(BB, PPC::ADDZE, 1, Result).addReg(Tmp1);
        return Result;
      } else if ((signed)Tmp3 < 0 && isPowerOf2_32(-Tmp3)) {
        Tmp3 = Log2_32(-Tmp3);
        Tmp2 = SelectExpr(N.getOperand(0));
        Tmp1 = MakeIntReg();
        unsigned Tmp4 = MakeIntReg();
        BuildMI(BB, PPC::SRAWI, 2, Tmp1).addReg(Tmp2).addImm(Tmp3);
        BuildMI(BB, PPC::ADDZE, 1, Tmp4).addReg(Tmp1);
        BuildMI(BB, PPC::NEG, 1, Result).addReg(Tmp4);
        return Result;
      } else if (Tmp3) {
        ExprMap.erase(N);
        return SelectExpr(BuildSDIVSequence(N));
      }
    }
    // fall thru
  case ISD::UDIV:
    // If this is a divide by constant, we can emit code using some magic
    // constants to implement it as a multiply instead.
    if (isIntImmediate(N.getOperand(1), Tmp3) && Tmp3) {
      ExprMap.erase(N);
      return SelectExpr(BuildUDIVSequence(N));
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    switch (DestType) {
    default: assert(0 && "Unknown type to ISD::DIV"); break;
    case MVT::i32: Opc = (ISD::UDIV == opcode) ? PPC::DIVWU : PPC::DIVW; break;
    case MVT::f32: Opc = PPC::FDIVS; break;
    case MVT::f64: Opc = PPC::FDIV; break;
    }
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::ADD_PARTS:
  case ISD::SUB_PARTS: {
    assert(N.getNumOperands() == 4 && N.getValueType() == MVT::i32 &&
           "Not an i64 add/sub!");
    unsigned Tmp4 = 0;
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    
    if (N.getOpcode() == ISD::ADD_PARTS) {
      bool ME = false, ZE = false;
      if (isIntImmediate(N.getOperand(3), Tmp3)) {
        ME = (signed)Tmp3 == -1;
        ZE = Tmp3 == 0;
      }
      
      if (!ZE && !ME)
        Tmp4 = SelectExpr(N.getOperand(3));

      if (isIntImmediate(N.getOperand(2), Tmp3) &&
          ((signed)Tmp3 >= -32768 || (signed)Tmp3 < 32768)) {
        // Codegen the low 32 bits of the add.  Interestingly, there is no
        // shifted form of add immediate carrying.
        BuildMI(BB, PPC::ADDIC, 2, Result).addReg(Tmp1).addSImm(Tmp3);
      } else {
        Tmp3 = SelectExpr(N.getOperand(2));
        BuildMI(BB, PPC::ADDC, 2, Result).addReg(Tmp1).addReg(Tmp3);
      }
      
      // Codegen the high 32 bits, adding zero, minus one, or the full value
      // along with the carry flag produced by addc/addic to tmp2.
      if (ZE) {
        BuildMI(BB, PPC::ADDZE, 1, Result+1).addReg(Tmp2);
      } else if (ME) {
        BuildMI(BB, PPC::ADDME, 1, Result+1).addReg(Tmp2);
      } else {
        BuildMI(BB, PPC::ADDE, 2, Result+1).addReg(Tmp2).addReg(Tmp4);
      }
    } else {
      Tmp3 = SelectExpr(N.getOperand(2));
      Tmp4 = SelectExpr(N.getOperand(3));
      BuildMI(BB, PPC::SUBFC, 2, Result).addReg(Tmp3).addReg(Tmp1);
      BuildMI(BB, PPC::SUBFE, 2, Result+1).addReg(Tmp4).addReg(Tmp2);
    }
    return Result+N.ResNo;
  }

  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS: {
    assert(N.getNumOperands() == 3 && N.getValueType() == MVT::i32 &&
           "Not an i64 shift!");
    unsigned ShiftOpLo = SelectExpr(N.getOperand(0));
    unsigned ShiftOpHi = SelectExpr(N.getOperand(1));
    unsigned SHReg = FoldIfWideZeroExtend(N.getOperand(2));
    Tmp1 = MakeIntReg();
    Tmp2 = MakeIntReg();
    Tmp3 = MakeIntReg();
    unsigned Tmp4 = MakeIntReg();
    unsigned Tmp5 = MakeIntReg();
    unsigned Tmp6 = MakeIntReg();
    BuildMI(BB, PPC::SUBFIC, 2, Tmp1).addReg(SHReg).addSImm(32);
    if (ISD::SHL_PARTS == opcode) {
      BuildMI(BB, PPC::SLW, 2, Tmp2).addReg(ShiftOpHi).addReg(SHReg);
      BuildMI(BB, PPC::SRW, 2, Tmp3).addReg(ShiftOpLo).addReg(Tmp1);
      BuildMI(BB, PPC::OR, 2, Tmp4).addReg(Tmp2).addReg(Tmp3);
      BuildMI(BB, PPC::ADDI, 2, Tmp5).addReg(SHReg).addSImm(-32);
      BuildMI(BB, PPC::SLW, 2, Tmp6).addReg(ShiftOpLo).addReg(Tmp5);
      BuildMI(BB, PPC::OR, 2, Result+1).addReg(Tmp4).addReg(Tmp6);
      BuildMI(BB, PPC::SLW, 2, Result).addReg(ShiftOpLo).addReg(SHReg);
    } else if (ISD::SRL_PARTS == opcode) {
      BuildMI(BB, PPC::SRW, 2, Tmp2).addReg(ShiftOpLo).addReg(SHReg);
      BuildMI(BB, PPC::SLW, 2, Tmp3).addReg(ShiftOpHi).addReg(Tmp1);
      BuildMI(BB, PPC::OR, 2, Tmp4).addReg(Tmp2).addReg(Tmp3);
      BuildMI(BB, PPC::ADDI, 2, Tmp5).addReg(SHReg).addSImm(-32);
      BuildMI(BB, PPC::SRW, 2, Tmp6).addReg(ShiftOpHi).addReg(Tmp5);
      BuildMI(BB, PPC::OR, 2, Result).addReg(Tmp4).addReg(Tmp6);
      BuildMI(BB, PPC::SRW, 2, Result+1).addReg(ShiftOpHi).addReg(SHReg);
    } else {
      MachineBasicBlock *TmpMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *PhiMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *OldMBB = BB;
      MachineFunction *F = BB->getParent();
      ilist<MachineBasicBlock>::iterator It = BB; ++It;
      F->getBasicBlockList().insert(It, TmpMBB);
      F->getBasicBlockList().insert(It, PhiMBB);
      BB->addSuccessor(TmpMBB);
      BB->addSuccessor(PhiMBB);
      BuildMI(BB, PPC::SRW, 2, Tmp2).addReg(ShiftOpLo).addReg(SHReg);
      BuildMI(BB, PPC::SLW, 2, Tmp3).addReg(ShiftOpHi).addReg(Tmp1);
      BuildMI(BB, PPC::OR, 2, Tmp4).addReg(Tmp2).addReg(Tmp3);
      BuildMI(BB, PPC::ADDICo, 2, Tmp5).addReg(SHReg).addSImm(-32);
      BuildMI(BB, PPC::SRAW, 2, Tmp6).addReg(ShiftOpHi).addReg(Tmp5);
      BuildMI(BB, PPC::SRAW, 2, Result+1).addReg(ShiftOpHi).addReg(SHReg);
      BuildMI(BB, PPC::BLE, 2).addReg(PPC::CR0).addMBB(PhiMBB);
      // Select correct least significant half if the shift amount > 32
      BB = TmpMBB;
      unsigned Tmp7 = MakeIntReg();
      BuildMI(BB, PPC::OR, 2, Tmp7).addReg(Tmp6).addReg(Tmp6);
      TmpMBB->addSuccessor(PhiMBB);
      BB = PhiMBB;
      BuildMI(BB, PPC::PHI, 4, Result).addReg(Tmp4).addMBB(OldMBB)
        .addReg(Tmp7).addMBB(TmpMBB);
    }
    return Result+N.ResNo;
  }

  case ISD::FP_TO_SINT: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeFPReg();
    BuildMI(BB, PPC::FCTIWZ, 1, Tmp2).addReg(Tmp1);
    int FrameIdx = BB->getParent()->getFrameInfo()->CreateStackObject(8, 8);
    addFrameReference(BuildMI(BB, PPC::STFD, 3).addReg(Tmp2), FrameIdx);
    addFrameReference(BuildMI(BB, PPC::LWZ, 2, Result), FrameIdx, 4);
    return Result;
  }

  case ISD::SETCC: {
    ISD::CondCode CC = cast<CondCodeSDNode>(Node->getOperand(2))->get();
    if (isIntImmediate(Node->getOperand(1), Tmp3)) {
      // We can codegen setcc op, imm very efficiently compared to a brcond.
      // Check for those cases here.
      // setcc op, 0
      if (Tmp3 == 0) {
        Tmp1 = SelectExpr(Node->getOperand(0));
        switch (CC) {
        default: Node->dump(); assert(0 && "Unhandled SetCC condition");abort();
        case ISD::SETEQ:
          Tmp2 = MakeIntReg();
          BuildMI(BB, PPC::CNTLZW, 1, Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp2).addImm(27)
            .addImm(5).addImm(31);
          break;
        case ISD::SETNE:
          Tmp2 = MakeIntReg();
          BuildMI(BB, PPC::ADDIC, 2, Tmp2).addReg(Tmp1).addSImm(-1);
          BuildMI(BB, PPC::SUBFE, 2, Result).addReg(Tmp2).addReg(Tmp1);
          break;
        case ISD::SETLT:
          BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(1)
            .addImm(31).addImm(31);
          break;
        case ISD::SETGT:
          Tmp2 = MakeIntReg();
          Tmp3 = MakeIntReg();
          BuildMI(BB, PPC::NEG, 2, Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::ANDC, 2, Tmp3).addReg(Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp3).addImm(1)
            .addImm(31).addImm(31);
          break;
        }
        return Result;
      } else if (Tmp3 == ~0U) {        // setcc op, -1
        Tmp1 = SelectExpr(Node->getOperand(0));
        switch (CC) {
        default: assert(0 && "Unhandled SetCC condition"); abort();
        case ISD::SETEQ:
          Tmp2 = MakeIntReg();
          Tmp3 = MakeIntReg();
          BuildMI(BB, PPC::ADDIC, 2, Tmp2).addReg(Tmp1).addSImm(1);
          BuildMI(BB, PPC::LI, 1, Tmp3).addSImm(0);
          BuildMI(BB, PPC::ADDZE, 1, Result).addReg(Tmp3);
          break;
        case ISD::SETNE:
          Tmp2 = MakeIntReg();
          Tmp3 = MakeIntReg();
          BuildMI(BB, PPC::NOR, 2, Tmp2).addReg(Tmp1).addReg(Tmp1);
          BuildMI(BB, PPC::ADDIC, 2, Tmp3).addReg(Tmp2).addSImm(-1);
          BuildMI(BB, PPC::SUBFE, 2, Result).addReg(Tmp3).addReg(Tmp2);
          break;
        case ISD::SETLT:
          Tmp2 = MakeIntReg();
          Tmp3 = MakeIntReg();
          BuildMI(BB, PPC::ADDI, 2, Tmp2).addReg(Tmp1).addSImm(1);
          BuildMI(BB, PPC::AND, 2, Tmp3).addReg(Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp3).addImm(1)
            .addImm(31).addImm(31);
          break;
        case ISD::SETGT:
          Tmp2 = MakeIntReg();
          BuildMI(BB, PPC::RLWINM, 4, Tmp2).addReg(Tmp1).addImm(1)
            .addImm(31).addImm(31);
          BuildMI(BB, PPC::XORI, 2, Result).addReg(Tmp2).addImm(1);
          break;
        }
        return Result;
      }
    }

    unsigned CCReg = SelectCC(N.getOperand(0), N.getOperand(1), CC);
    MoveCRtoGPR(CCReg, CC, Result);
    return Result;
  }
    
  case ISD::SELECT_CC: {
    ISD::CondCode CC = cast<CondCodeSDNode>(N.getOperand(4))->get();

    // handle the setcc cases here.  select_cc lhs, 0, 1, 0, cc
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N.getOperand(2));
    ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N.getOperand(3));
    if (N1C && N2C && N3C && N1C->isNullValue() && N3C->isNullValue() &&
        N2C->getValue() == 1ULL && CC == ISD::SETNE) {
      Tmp1 = SelectExpr(Node->getOperand(0));
      Tmp2 = MakeIntReg();
      BuildMI(BB, PPC::ADDIC, 2, Tmp2).addReg(Tmp1).addSImm(-1);
      BuildMI(BB, PPC::SUBFE, 2, Result).addReg(Tmp2).addReg(Tmp1);
      return Result;
    }
    
    // If the False value only has one use, we can generate better code by
    // selecting it in the fallthrough basic block rather than here, which
    // increases register pressure.
    unsigned TrueValue = SelectExpr(N.getOperand(2));
    unsigned FalseValue;

    // If the false value is simple enough, evaluate it inline in the false
    // block.
    if (N.getOperand(3).Val->hasOneUse() &&
        (isa<ConstantSDNode>(N.getOperand(3)) ||
         isa<ConstantFPSDNode>(N.getOperand(3)) ||
         isa<GlobalAddressSDNode>(N.getOperand(3))))
      FalseValue = 0;
    else
      FalseValue = SelectExpr(N.getOperand(3));
    unsigned CCReg = SelectCC(N.getOperand(0), N.getOperand(1), CC);
    Opc = getBCCForSetCC(CC);
    
    // Create an iterator with which to insert the MBB for copying the false
    // value and the MBB to hold the PHI instruction for this SetCC.
    MachineBasicBlock *thisMBB = BB;
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    BuildMI(BB, Opc, 2).addReg(CCReg).addMBB(sinkMBB);
    MachineFunction *F = BB->getParent();
    F->getBasicBlockList().insert(It, copy0MBB);
    F->getBasicBlockList().insert(It, sinkMBB);
    // Update machine-CFG edges
    BB->addSuccessor(copy0MBB);
    BB->addSuccessor(sinkMBB);

    //  copy0MBB:
    //   %FalseValue = ...
    //   # fallthrough to sinkMBB
    BB = copy0MBB;

    // If the false value is simple enough, evaluate it here, to avoid it being
    // evaluated on the true edge.
    if (FalseValue == 0)
      FalseValue = SelectExpr(N.getOperand(3));

    // Update machine-CFG edges
    BB->addSuccessor(sinkMBB);

    //  sinkMBB:
    //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
    //  ...
    BB = sinkMBB;
    BuildMI(BB, PPC::PHI, 4, Result).addReg(FalseValue)
      .addMBB(copy0MBB).addReg(TrueValue).addMBB(thisMBB);
    return Result;
  }

  case ISD::Constant: {
    assert(N.getValueType() == MVT::i32 &&
           "Only i32 constants are legal on this target!");
    unsigned v = (unsigned)cast<ConstantSDNode>(N)->getValue();
    if (isInt16(v)) {
      BuildMI(BB, PPC::LI, 1, Result).addSImm(Lo16(v));
    } else {
      unsigned Hi = Hi16(v);
      unsigned Lo = Lo16(v);
      if (Lo) {
        Tmp1 = MakeIntReg();
        BuildMI(BB, PPC::LIS, 1, Tmp1).addSImm(Hi);
        BuildMI(BB, PPC::ORI, 2, Result).addReg(Tmp1).addImm(Lo);
      } else {
        BuildMI(BB, PPC::LIS, 1, Result).addSImm(Hi);
      }
    }
    return Result;
  }

  case ISD::FNEG:
    if (!NoExcessFPPrecision &&
        ISD::ADD == N.getOperand(0).getOpcode() &&
        N.getOperand(0).Val->hasOneUse() &&
        ISD::MUL == N.getOperand(0).getOperand(0).getOpcode() &&
        N.getOperand(0).getOperand(0).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(0).getOperand(1));
      Opc = DestType == MVT::f64 ? PPC::FNMADD : PPC::FNMADDS;
      BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
    } else if (!NoExcessFPPrecision &&
        ISD::ADD == N.getOperand(0).getOpcode() &&
        N.getOperand(0).Val->hasOneUse() &&
        ISD::MUL == N.getOperand(0).getOperand(1).getOpcode() &&
        N.getOperand(0).getOperand(1).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(1).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(1).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(0).getOperand(0));
      Opc = DestType == MVT::f64 ? PPC::FNMADD : PPC::FNMADDS;
      BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
    } else if (ISD::FABS == N.getOperand(0).getOpcode()) {
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      BuildMI(BB, PPC::FNABS, 1, Result).addReg(Tmp1);
    } else {
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, PPC::FNEG, 1, Result).addReg(Tmp1);
    }
    return Result;

  case ISD::FABS:
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, PPC::FABS, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::FSQRT:
    Tmp1 = SelectExpr(N.getOperand(0));
    Opc = DestType == MVT::f64 ? PPC::FSQRT : PPC::FSQRTS;
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::FP_ROUND:
    assert (DestType == MVT::f32 &&
            N.getOperand(0).getValueType() == MVT::f64 &&
            "only f64 to f32 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, PPC::FRSP, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::FP_EXTEND:
    assert (DestType == MVT::f64 &&
            N.getOperand(0).getValueType() == MVT::f32 &&
            "only f32 to f64 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, PPC::FMR, 1, Result).addReg(Tmp1);
    return Result;
  }
  return 0;
}

void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Tmp3, Opc;
  unsigned opcode = N.getOpcode();

  if (!ExprMap.insert(std::make_pair(N, 1)).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (Node->getOpcode()) {
  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");
  case ISD::EntryToken: return;  // Noop
  case ISD::TokenFactor:
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Select(Node->getOperand(i));
    return;
  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END:
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
    Opc = N.getOpcode() == ISD::CALLSEQ_START ? PPC::ADJCALLSTACKDOWN :
      PPC::ADJCALLSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;
  case ISD::BR: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();
    Select(N.getOperand(0));
    BuildMI(BB, PPC::B, 1).addMBB(Dest);
    return;
  }
  case ISD::BR_CC:
  case ISD::BRTWOWAY_CC:
    SelectBranchCC(N);
    return;
  case ISD::CopyToReg:
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(2));
    Tmp2 = cast<RegisterSDNode>(N.getOperand(1))->getReg();

    if (Tmp1 != Tmp2) {
      if (N.getOperand(2).getValueType() == MVT::f64 ||
          N.getOperand(2).getValueType() == MVT::f32)
        BuildMI(BB, PPC::FMR, 1, Tmp2).addReg(Tmp1);
      else
        BuildMI(BB, PPC::OR, 2, Tmp2).addReg(Tmp1).addReg(Tmp1);
    }
    return;
  case ISD::ImplicitDef:
    Select(N.getOperand(0));
    Tmp1 = cast<RegisterSDNode>(N.getOperand(1))->getReg();
    if (N.getOperand(1).getValueType() == MVT::i32)
      BuildMI(BB, PPC::IMPLICIT_DEF_GPR, 0, Tmp1);
    else
      BuildMI(BB, PPC::IMPLICIT_DEF_FP, 0, Tmp1);
    return;
  case ISD::RET:
    switch (N.getNumOperands()) {
    default:
      assert(0 && "Unknown return instruction!");
    case 3:
      assert(N.getOperand(1).getValueType() == MVT::i32 &&
             N.getOperand(2).getValueType() == MVT::i32 &&
             "Unknown two-register value!");
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1));
      Tmp2 = SelectExpr(N.getOperand(2));
      BuildMI(BB, PPC::OR, 2, PPC::R3).addReg(Tmp2).addReg(Tmp2);
      BuildMI(BB, PPC::OR, 2, PPC::R4).addReg(Tmp1).addReg(Tmp1);
      break;
    case 2:
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1));
      switch (N.getOperand(1).getValueType()) {
        default:
          assert(0 && "Unknown return type!");
        case MVT::f64:
        case MVT::f32:
          BuildMI(BB, PPC::FMR, 1, PPC::F1).addReg(Tmp1);
          break;
        case MVT::i32:
          BuildMI(BB, PPC::OR, 2, PPC::R3).addReg(Tmp1).addReg(Tmp1);
          break;
      }
    case 1:
      Select(N.getOperand(0));
      break;
    }
    BuildMI(BB, PPC::BLR, 0); // Just emit a 'ret' instruction
    return;
  case ISD::TRUNCSTORE:
  case ISD::STORE: {
    SDOperand Chain   = N.getOperand(0);
    SDOperand Value   = N.getOperand(1);
    SDOperand Address = N.getOperand(2);
    Select(Chain);

    Tmp1 = SelectExpr(Value); //value

    if (opcode == ISD::STORE) {
      switch(Value.getValueType()) {
      default: assert(0 && "unknown Type in store");
      case MVT::i32: Opc = PPC::STW; break;
      case MVT::f64: Opc = PPC::STFD; break;
      case MVT::f32: Opc = PPC::STFS; break;
      }
    } else { //ISD::TRUNCSTORE
      switch(cast<VTSDNode>(Node->getOperand(4))->getVT()) {
      default: assert(0 && "unknown Type in store");
      case MVT::i1:
      case MVT::i8: Opc  = PPC::STB; break;
      case MVT::i16: Opc = PPC::STH; break;
      }
    }

    if(Address.getOpcode() == ISD::FrameIndex) {
      Tmp2 = cast<FrameIndexSDNode>(Address)->getIndex();
      addFrameReference(BuildMI(BB, Opc, 3).addReg(Tmp1), (int)Tmp2);
    } else {
      int offset;
      switch(SelectAddr(Address, Tmp2, offset)) {
      default: assert(0 && "Unhandled return value from SelectAddr");
      case 0:   // imm offset, no frame, no index
        BuildMI(BB, Opc, 3).addReg(Tmp1).addSImm(offset).addReg(Tmp2);
        break;
      case 1:   // imm offset + frame index
        addFrameReference(BuildMI(BB, Opc, 3).addReg(Tmp1), (int)Tmp2, offset);
        break;
      case 2:   // base+index addressing
        Opc = IndexedOpForOp(Opc);
        BuildMI(BB, Opc, 3).addReg(Tmp1).addReg(Tmp2).addReg(offset);
        break;
      case 3: {
        GlobalAddressSDNode *GN = cast<GlobalAddressSDNode>(Address);
        GlobalValue *GV = GN->getGlobal();
        BuildMI(BB, Opc, 3).addReg(Tmp1).addGlobalAddress(GV).addReg(Tmp2);
      }
      }
    }
    return;
  }
  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::LOAD:
  case ISD::CopyFromReg:
  case ISD::TAILCALL:
  case ISD::CALL:
  case ISD::DYNAMIC_STACKALLOC:
    ExprMap.erase(N);
    SelectExpr(N);
    return;
  }
  assert(0 && "Should not be reached!");
}


/// createPPC32PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createPPC32ISelPattern(TargetMachine &TM) {
  return new ISel(TM);
}

