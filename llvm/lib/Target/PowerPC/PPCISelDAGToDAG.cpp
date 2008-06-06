//===-- PPCISelDAGToDAG.cpp - PPC --pattern matching inst selector --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for PowerPC,
// converting from a legalized dag to a PPC dag.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ppc-codegen"
#include "PPC.h"
#include "PPCPredicates.h"
#include "PPCTargetMachine.h"
#include "PPCISelLowering.h"
#include "PPCHazardRecognizers.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Compiler.h"
#include <queue>
#include <set>
using namespace llvm;

namespace {
  //===--------------------------------------------------------------------===//
  /// PPCDAGToDAGISel - PPC specific code to select PPC machine
  /// instructions for SelectionDAG operations.
  ///
  class VISIBILITY_HIDDEN PPCDAGToDAGISel : public SelectionDAGISel {
    PPCTargetMachine &TM;
    PPCTargetLowering PPCLowering;
    const PPCSubtarget &PPCSubTarget;
    unsigned GlobalBaseReg;
  public:
    PPCDAGToDAGISel(PPCTargetMachine &tm)
      : SelectionDAGISel(PPCLowering), TM(tm),
        PPCLowering(*TM.getTargetLowering()),
        PPCSubTarget(*TM.getSubtargetImpl()) {}
    
    virtual bool runOnFunction(Function &Fn) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseReg = 0;
      SelectionDAGISel::runOnFunction(Fn);
      
      InsertVRSaveCode(Fn);
      return true;
    }
   
    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDOperand getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }

    /// getI64Imm - Return a target constant with the specified value, of type
    /// i64.
    inline SDOperand getI64Imm(uint64_t Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i64);
    }
    
    /// getSmallIPtrImm - Return a target constant of pointer type.
    inline SDOperand getSmallIPtrImm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, PPCLowering.getPointerTy());
    }
    
    /// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s 
    /// with any number of 0s on either side.  The 1s are allowed to wrap from
    /// LSB to MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.
    /// 0x0F0F0000 is not, since all 1s are not contiguous.
    static bool isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME);


    /// isRotateAndMask - Returns true if Mask and Shift can be folded into a
    /// rotate and mask opcode and mask operation.
    static bool isRotateAndMask(SDNode *N, unsigned Mask, bool IsShiftMask,
                                unsigned &SH, unsigned &MB, unsigned &ME);
    
    /// getGlobalBaseReg - insert code into the entry mbb to materialize the PIC
    /// base register.  Return the virtual register that holds this value.
    SDNode *getGlobalBaseReg();
    
    // Select - Convert the specified operand from a target-independent to a
    // target-specific node if it hasn't already been changed.
    SDNode *Select(SDOperand Op);
    
    SDNode *SelectBitfieldInsert(SDNode *N);

    /// SelectCC - Select a comparison of the specified values with the
    /// specified condition code, returning the CR# of the expression.
    SDOperand SelectCC(SDOperand LHS, SDOperand RHS, ISD::CondCode CC);

    /// SelectAddrImm - Returns true if the address N can be represented by
    /// a base register plus a signed 16-bit displacement [r+imm].
    bool SelectAddrImm(SDOperand Op, SDOperand N, SDOperand &Disp,
                       SDOperand &Base) {
      return PPCLowering.SelectAddressRegImm(N, Disp, Base, *CurDAG);
    }
    
    /// SelectAddrImmOffs - Return true if the operand is valid for a preinc
    /// immediate field.  Because preinc imms have already been validated, just
    /// accept it.
    bool SelectAddrImmOffs(SDOperand Op, SDOperand N, SDOperand &Out) const {
      Out = N;
      return true;
    }
      
    /// SelectAddrIdx - Given the specified addressed, check to see if it can be
    /// represented as an indexed [r+r] operation.  Returns false if it can
    /// be represented by [r+imm], which are preferred.
    bool SelectAddrIdx(SDOperand Op, SDOperand N, SDOperand &Base,
                       SDOperand &Index) {
      return PPCLowering.SelectAddressRegReg(N, Base, Index, *CurDAG);
    }
    
    /// SelectAddrIdxOnly - Given the specified addressed, force it to be
    /// represented as an indexed [r+r] operation.
    bool SelectAddrIdxOnly(SDOperand Op, SDOperand N, SDOperand &Base,
                           SDOperand &Index) {
      return PPCLowering.SelectAddressRegRegOnly(N, Base, Index, *CurDAG);
    }

    /// SelectAddrImmShift - Returns true if the address N can be represented by
    /// a base register plus a signed 14-bit displacement [r+imm*4].  Suitable
    /// for use by STD and friends.
    bool SelectAddrImmShift(SDOperand Op, SDOperand N, SDOperand &Disp,
                            SDOperand &Base) {
      return PPCLowering.SelectAddressRegImmShift(N, Disp, Base, *CurDAG);
    }
      
    /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
    /// inline asm expressions.
    virtual bool SelectInlineAsmMemoryOperand(const SDOperand &Op,
                                              char ConstraintCode,
                                              std::vector<SDOperand> &OutOps,
                                              SelectionDAG &DAG) {
      SDOperand Op0, Op1;
      switch (ConstraintCode) {
      default: return true;
      case 'm':   // memory
        if (!SelectAddrIdx(Op, Op, Op0, Op1))
          SelectAddrImm(Op, Op, Op0, Op1);
        break;
      case 'o':   // offsetable
        if (!SelectAddrImm(Op, Op, Op0, Op1)) {
          Op0 = Op;
          AddToISelQueue(Op0);     // r+0.
          Op1 = getSmallIPtrImm(0);
        }
        break;
      case 'v':   // not offsetable
        SelectAddrIdxOnly(Op, Op, Op0, Op1);
        break;
      }
      
      OutOps.push_back(Op0);
      OutOps.push_back(Op1);
      return false;
    }
    
    SDOperand BuildSDIVSequence(SDNode *N);
    SDOperand BuildUDIVSequence(SDNode *N);
    
    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
    
    void InsertVRSaveCode(Function &Fn);

    virtual const char *getPassName() const {
      return "PowerPC DAG->DAG Pattern Instruction Selection";
    } 
    
    /// CreateTargetHazardRecognizer - Return the hazard recognizer to use for
    /// this target when scheduling the DAG.
    virtual HazardRecognizer *CreateTargetHazardRecognizer() {
      // Should use subtarget info to pick the right hazard recognizer.  For
      // now, always return a PPC970 recognizer.
      const TargetInstrInfo *II = PPCLowering.getTargetMachine().getInstrInfo();
      assert(II && "No InstrInfo?");
      return new PPCHazardRecognizer970(*II); 
    }

// Include the pieces autogenerated from the target description.
#include "PPCGenDAGISel.inc"
    
private:
    SDNode *SelectSETCC(SDOperand Op);
  };
}

/// InstructionSelectBasicBlock - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void PPCDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());

  // Select target instructions for the DAG.
  DAG.setRoot(SelectRoot(DAG.getRoot()));
  DAG.RemoveDeadNodes();
  
  // Emit machine code to BB.
  ScheduleAndEmitDAG(DAG);
}

/// InsertVRSaveCode - Once the entire function has been instruction selected,
/// all virtual registers are created and all machine instructions are built,
/// check to see if we need to save/restore VRSAVE.  If so, do it.
void PPCDAGToDAGISel::InsertVRSaveCode(Function &F) {
  // Check to see if this function uses vector registers, which means we have to
  // save and restore the VRSAVE register and update it with the regs we use.  
  //
  // In this case, there will be virtual registers of vector type type created
  // by the scheduler.  Detect them now.
  MachineFunction &Fn = MachineFunction::get(&F);
  bool HasVectorVReg = false;
  for (unsigned i = TargetRegisterInfo::FirstVirtualRegister, 
       e = RegInfo->getLastVirtReg()+1; i != e; ++i)
    if (RegInfo->getRegClass(i) == &PPC::VRRCRegClass) {
      HasVectorVReg = true;
      break;
    }
  if (!HasVectorVReg) return;  // nothing to do.
      
  // If we have a vector register, we want to emit code into the entry and exit
  // blocks to save and restore the VRSAVE register.  We do this here (instead
  // of marking all vector instructions as clobbering VRSAVE) for two reasons:
  //
  // 1. This (trivially) reduces the load on the register allocator, by not
  //    having to represent the live range of the VRSAVE register.
  // 2. This (more significantly) allows us to create a temporary virtual
  //    register to hold the saved VRSAVE value, allowing this temporary to be
  //    register allocated, instead of forcing it to be spilled to the stack.

  // Create two vregs - one to hold the VRSAVE register that is live-in to the
  // function and one for the value after having bits or'd into it.
  unsigned InVRSAVE = RegInfo->createVirtualRegister(&PPC::GPRCRegClass);
  unsigned UpdatedVRSAVE = RegInfo->createVirtualRegister(&PPC::GPRCRegClass);
  
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  MachineBasicBlock &EntryBB = *Fn.begin();
  // Emit the following code into the entry block:
  // InVRSAVE = MFVRSAVE
  // UpdatedVRSAVE = UPDATE_VRSAVE InVRSAVE
  // MTVRSAVE UpdatedVRSAVE
  MachineBasicBlock::iterator IP = EntryBB.begin();  // Insert Point
  BuildMI(EntryBB, IP, TII.get(PPC::MFVRSAVE), InVRSAVE);
  BuildMI(EntryBB, IP, TII.get(PPC::UPDATE_VRSAVE),
          UpdatedVRSAVE).addReg(InVRSAVE);
  BuildMI(EntryBB, IP, TII.get(PPC::MTVRSAVE)).addReg(UpdatedVRSAVE);
  
  // Find all return blocks, outputting a restore in each epilog.
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    if (!BB->empty() && BB->back().getDesc().isReturn()) {
      IP = BB->end(); --IP;
      
      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = IP;
      while (I2 != BB->begin() && (--I2)->getDesc().isTerminator())
        IP = I2;
      
      // Emit: MTVRSAVE InVRSave
      BuildMI(*BB, IP, TII.get(PPC::MTVRSAVE)).addReg(InVRSAVE);
    }        
  }
}


/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
SDNode *PPCDAGToDAGISel::getGlobalBaseReg() {
  if (!GlobalBaseReg) {
    const TargetInstrInfo &TII = *TM.getInstrInfo();
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();

    if (PPCLowering.getPointerTy() == MVT::i32) {
      GlobalBaseReg = RegInfo->createVirtualRegister(PPC::GPRCRegisterClass);
      BuildMI(FirstMBB, MBBI, TII.get(PPC::MovePCtoLR), PPC::LR);
      BuildMI(FirstMBB, MBBI, TII.get(PPC::MFLR), GlobalBaseReg);
    } else {
      GlobalBaseReg = RegInfo->createVirtualRegister(PPC::G8RCRegisterClass);
      BuildMI(FirstMBB, MBBI, TII.get(PPC::MovePCtoLR8), PPC::LR8);
      BuildMI(FirstMBB, MBBI, TII.get(PPC::MFLR8), GlobalBaseReg);
    }
  }
  return CurDAG->getRegister(GlobalBaseReg, PPCLowering.getPointerTy()).Val;
}

/// isIntS16Immediate - This method tests to see if the node is either a 32-bit
/// or 64-bit immediate, and if the value can be accurately represented as a
/// sign extension from a 16-bit value.  If so, this returns true and the
/// immediate.
static bool isIntS16Immediate(SDNode *N, short &Imm) {
  if (N->getOpcode() != ISD::Constant)
    return false;

  Imm = (short)cast<ConstantSDNode>(N)->getValue();
  if (N->getValueType(0) == MVT::i32)
    return Imm == (int32_t)cast<ConstantSDNode>(N)->getValue();
  else
    return Imm == (int64_t)cast<ConstantSDNode>(N)->getValue();
}

static bool isIntS16Immediate(SDOperand Op, short &Imm) {
  return isIntS16Immediate(Op.Val, Imm);
}


/// isInt32Immediate - This method tests to see if the node is a 32-bit constant
/// operand. If so Imm will receive the 32-bit value.
static bool isInt32Immediate(SDNode *N, unsigned &Imm) {
  if (N->getOpcode() == ISD::Constant && N->getValueType(0) == MVT::i32) {
    Imm = cast<ConstantSDNode>(N)->getValue();
    return true;
  }
  return false;
}

/// isInt64Immediate - This method tests to see if the node is a 64-bit constant
/// operand.  If so Imm will receive the 64-bit value.
static bool isInt64Immediate(SDNode *N, uint64_t &Imm) {
  if (N->getOpcode() == ISD::Constant && N->getValueType(0) == MVT::i64) {
    Imm = cast<ConstantSDNode>(N)->getValue();
    return true;
  }
  return false;
}

// isInt32Immediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isInt32Immediate(SDOperand N, unsigned &Imm) {
  return isInt32Immediate(N.Val, Imm);
}


// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(SDNode *N, unsigned Opc, unsigned& Imm) {
  return N->getOpcode() == Opc && isInt32Immediate(N->getOperand(1).Val, Imm);
}

bool PPCDAGToDAGISel::isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME) {
  if (isShiftedMask_32(Val)) {
    // look for the first non-zero bit
    MB = CountLeadingZeros_32(Val);
    // look for the first zero bit after the run of ones
    ME = CountLeadingZeros_32((Val - 1) ^ Val);
    return true;
  } else {
    Val = ~Val; // invert mask
    if (isShiftedMask_32(Val)) {
      // effectively look for the first zero bit
      ME = CountLeadingZeros_32(Val) - 1;
      // effectively look for the first one bit after the run of zeros
      MB = CountLeadingZeros_32((Val - 1) ^ Val) + 1;
      return true;
    }
  }
  // no run present
  return false;
}

bool PPCDAGToDAGISel::isRotateAndMask(SDNode *N, unsigned Mask, 
                                      bool IsShiftMask, unsigned &SH, 
                                      unsigned &MB, unsigned &ME) {
  // Don't even go down this path for i64, since different logic will be
  // necessary for rldicl/rldicr/rldimi.
  if (N->getValueType(0) != MVT::i32)
    return false;

  unsigned Shift  = 32;
  unsigned Indeterminant = ~0;  // bit mask marking indeterminant results
  unsigned Opcode = N->getOpcode();
  if (N->getNumOperands() != 2 ||
      !isInt32Immediate(N->getOperand(1).Val, Shift) || (Shift > 31))
    return false;
  
  if (Opcode == ISD::SHL) {
    // apply shift left to mask if it comes first
    if (IsShiftMask) Mask = Mask << Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu << Shift);
  } else if (Opcode == ISD::SRL) { 
    // apply shift right to mask if it comes first
    if (IsShiftMask) Mask = Mask >> Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu >> Shift);
    // adjust for the left rotate
    Shift = 32 - Shift;
  } else if (Opcode == ISD::ROTL) {
    Indeterminant = 0;
  } else {
    return false;
  }
  
  // if the mask doesn't intersect any Indeterminant bits
  if (Mask && !(Mask & Indeterminant)) {
    SH = Shift & 31;
    // make sure the mask is still a mask (wrap arounds may not be)
    return isRunOfOnes(Mask, MB, ME);
  }
  return false;
}

/// SelectBitfieldInsert - turn an or of two masked values into
/// the rotate left word immediate then mask insert (rlwimi) instruction.
SDNode *PPCDAGToDAGISel::SelectBitfieldInsert(SDNode *N) {
  SDOperand Op0 = N->getOperand(0);
  SDOperand Op1 = N->getOperand(1);
  
  APInt LKZ, LKO, RKZ, RKO;
  CurDAG->ComputeMaskedBits(Op0, APInt::getAllOnesValue(32), LKZ, LKO);
  CurDAG->ComputeMaskedBits(Op1, APInt::getAllOnesValue(32), RKZ, RKO);
  
  unsigned TargetMask = LKZ.getZExtValue();
  unsigned InsertMask = RKZ.getZExtValue();
  
  if ((TargetMask | InsertMask) == 0xFFFFFFFF) {
    unsigned Op0Opc = Op0.getOpcode();
    unsigned Op1Opc = Op1.getOpcode();
    unsigned Value, SH = 0;
    TargetMask = ~TargetMask;
    InsertMask = ~InsertMask;

    // If the LHS has a foldable shift and the RHS does not, then swap it to the
    // RHS so that we can fold the shift into the insert.
    if (Op0Opc == ISD::AND && Op1Opc == ISD::AND) {
      if (Op0.getOperand(0).getOpcode() == ISD::SHL ||
          Op0.getOperand(0).getOpcode() == ISD::SRL) {
        if (Op1.getOperand(0).getOpcode() != ISD::SHL &&
            Op1.getOperand(0).getOpcode() != ISD::SRL) {
          std::swap(Op0, Op1);
          std::swap(Op0Opc, Op1Opc);
          std::swap(TargetMask, InsertMask);
        }
      }
    } else if (Op0Opc == ISD::SHL || Op0Opc == ISD::SRL) {
      if (Op1Opc == ISD::AND && Op1.getOperand(0).getOpcode() != ISD::SHL &&
          Op1.getOperand(0).getOpcode() != ISD::SRL) {
        std::swap(Op0, Op1);
        std::swap(Op0Opc, Op1Opc);
        std::swap(TargetMask, InsertMask);
      }
    }
    
    unsigned MB, ME;
    if (InsertMask && isRunOfOnes(InsertMask, MB, ME)) {
      SDOperand Tmp1, Tmp2, Tmp3;
      bool DisjointMask = (TargetMask ^ InsertMask) == 0xFFFFFFFF;

      if ((Op1Opc == ISD::SHL || Op1Opc == ISD::SRL) &&
          isInt32Immediate(Op1.getOperand(1), Value)) {
        Op1 = Op1.getOperand(0);
        SH  = (Op1Opc == ISD::SHL) ? Value : 32 - Value;
      }
      if (Op1Opc == ISD::AND) {
        unsigned SHOpc = Op1.getOperand(0).getOpcode();
        if ((SHOpc == ISD::SHL || SHOpc == ISD::SRL) &&
            isInt32Immediate(Op1.getOperand(0).getOperand(1), Value)) {
          Op1 = Op1.getOperand(0).getOperand(0);
          SH  = (SHOpc == ISD::SHL) ? Value : 32 - Value;
        } else {
          Op1 = Op1.getOperand(0);
        }
      }
      
      Tmp3 = (Op0Opc == ISD::AND && DisjointMask) ? Op0.getOperand(0) : Op0;
      AddToISelQueue(Tmp3);
      AddToISelQueue(Op1);
      SH &= 31;
      SDOperand Ops[] = { Tmp3, Op1, getI32Imm(SH), getI32Imm(MB),
                          getI32Imm(ME) };
      return CurDAG->getTargetNode(PPC::RLWIMI, MVT::i32, Ops, 5);
    }
  }
  return 0;
}

/// SelectCC - Select a comparison of the specified values with the specified
/// condition code, returning the CR# of the expression.
SDOperand PPCDAGToDAGISel::SelectCC(SDOperand LHS, SDOperand RHS,
                                    ISD::CondCode CC) {
  // Always select the LHS.
  AddToISelQueue(LHS);
  unsigned Opc;
  
  if (LHS.getValueType() == MVT::i32) {
    unsigned Imm;
    if (CC == ISD::SETEQ || CC == ISD::SETNE) {
      if (isInt32Immediate(RHS, Imm)) {
        // SETEQ/SETNE comparison with 16-bit immediate, fold it.
        if (isUInt16(Imm))
          return SDOperand(CurDAG->getTargetNode(PPC::CMPLWI, MVT::i32, LHS,
                                                 getI32Imm(Imm & 0xFFFF)), 0);
        // If this is a 16-bit signed immediate, fold it.
        if (isInt16((int)Imm))
          return SDOperand(CurDAG->getTargetNode(PPC::CMPWI, MVT::i32, LHS,
                                                 getI32Imm(Imm & 0xFFFF)), 0);
        
        // For non-equality comparisons, the default code would materialize the
        // constant, then compare against it, like this:
        //   lis r2, 4660
        //   ori r2, r2, 22136 
        //   cmpw cr0, r3, r2
        // Since we are just comparing for equality, we can emit this instead:
        //   xoris r0,r3,0x1234
        //   cmplwi cr0,r0,0x5678
        //   beq cr0,L6
        SDOperand Xor(CurDAG->getTargetNode(PPC::XORIS, MVT::i32, LHS,
                                            getI32Imm(Imm >> 16)), 0);
        return SDOperand(CurDAG->getTargetNode(PPC::CMPLWI, MVT::i32, Xor,
                                               getI32Imm(Imm & 0xFFFF)), 0);
      }
      Opc = PPC::CMPLW;
    } else if (ISD::isUnsignedIntSetCC(CC)) {
      if (isInt32Immediate(RHS, Imm) && isUInt16(Imm))
        return SDOperand(CurDAG->getTargetNode(PPC::CMPLWI, MVT::i32, LHS,
                                               getI32Imm(Imm & 0xFFFF)), 0);
      Opc = PPC::CMPLW;
    } else {
      short SImm;
      if (isIntS16Immediate(RHS, SImm))
        return SDOperand(CurDAG->getTargetNode(PPC::CMPWI, MVT::i32, LHS,
                                               getI32Imm((int)SImm & 0xFFFF)),
                         0);
      Opc = PPC::CMPW;
    }
  } else if (LHS.getValueType() == MVT::i64) {
    uint64_t Imm;
    if (CC == ISD::SETEQ || CC == ISD::SETNE) {
      if (isInt64Immediate(RHS.Val, Imm)) {
        // SETEQ/SETNE comparison with 16-bit immediate, fold it.
        if (isUInt16(Imm))
          return SDOperand(CurDAG->getTargetNode(PPC::CMPLDI, MVT::i64, LHS,
                                                 getI32Imm(Imm & 0xFFFF)), 0);
        // If this is a 16-bit signed immediate, fold it.
        if (isInt16(Imm))
          return SDOperand(CurDAG->getTargetNode(PPC::CMPDI, MVT::i64, LHS,
                                                 getI32Imm(Imm & 0xFFFF)), 0);
        
        // For non-equality comparisons, the default code would materialize the
        // constant, then compare against it, like this:
        //   lis r2, 4660
        //   ori r2, r2, 22136 
        //   cmpd cr0, r3, r2
        // Since we are just comparing for equality, we can emit this instead:
        //   xoris r0,r3,0x1234
        //   cmpldi cr0,r0,0x5678
        //   beq cr0,L6
        if (isUInt32(Imm)) {
          SDOperand Xor(CurDAG->getTargetNode(PPC::XORIS8, MVT::i64, LHS,
                                              getI64Imm(Imm >> 16)), 0);
          return SDOperand(CurDAG->getTargetNode(PPC::CMPLDI, MVT::i64, Xor,
                                                 getI64Imm(Imm & 0xFFFF)), 0);
        }
      }
      Opc = PPC::CMPLD;
    } else if (ISD::isUnsignedIntSetCC(CC)) {
      if (isInt64Immediate(RHS.Val, Imm) && isUInt16(Imm))
        return SDOperand(CurDAG->getTargetNode(PPC::CMPLDI, MVT::i64, LHS,
                                               getI64Imm(Imm & 0xFFFF)), 0);
      Opc = PPC::CMPLD;
    } else {
      short SImm;
      if (isIntS16Immediate(RHS, SImm))
        return SDOperand(CurDAG->getTargetNode(PPC::CMPDI, MVT::i64, LHS,
                                               getI64Imm(SImm & 0xFFFF)),
                         0);
      Opc = PPC::CMPD;
    }
  } else if (LHS.getValueType() == MVT::f32) {
    Opc = PPC::FCMPUS;
  } else {
    assert(LHS.getValueType() == MVT::f64 && "Unknown vt!");
    Opc = PPC::FCMPUD;
  }
  AddToISelQueue(RHS);
  return SDOperand(CurDAG->getTargetNode(Opc, MVT::i32, LHS, RHS), 0);
}

static PPC::Predicate getPredicateForSetCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETOEQ:    // FIXME: This is incorrect see PR642.
  case ISD::SETUEQ:
  case ISD::SETEQ:  return PPC::PRED_EQ;
  case ISD::SETONE:    // FIXME: This is incorrect see PR642.
  case ISD::SETUNE:
  case ISD::SETNE:  return PPC::PRED_NE;
  case ISD::SETOLT:    // FIXME: This is incorrect see PR642.
  case ISD::SETULT:
  case ISD::SETLT:  return PPC::PRED_LT;
  case ISD::SETOLE:    // FIXME: This is incorrect see PR642.
  case ISD::SETULE:
  case ISD::SETLE:  return PPC::PRED_LE;
  case ISD::SETOGT:    // FIXME: This is incorrect see PR642.
  case ISD::SETUGT:
  case ISD::SETGT:  return PPC::PRED_GT;
  case ISD::SETOGE:    // FIXME: This is incorrect see PR642.
  case ISD::SETUGE:
  case ISD::SETGE:  return PPC::PRED_GE;
    
  case ISD::SETO:   return PPC::PRED_NU;
  case ISD::SETUO:  return PPC::PRED_UN;
  }
}

/// getCRIdxForSetCC - Return the index of the condition register field
/// associated with the SetCC condition, and whether or not the field is
/// treated as inverted.  That is, lt = 0; ge = 0 inverted.
///
/// If this returns with Other != -1, then the returned comparison is an or of
/// two simpler comparisons.  In this case, Invert is guaranteed to be false.
static unsigned getCRIdxForSetCC(ISD::CondCode CC, bool &Invert, int &Other) {
  Invert = false;
  Other = -1;
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETOLT:
  case ISD::SETLT:  return 0;                  // Bit #0 = SETOLT
  case ISD::SETOGT:
  case ISD::SETGT:  return 1;                  // Bit #1 = SETOGT
  case ISD::SETOEQ:
  case ISD::SETEQ:  return 2;                  // Bit #2 = SETOEQ
  case ISD::SETUO:  return 3;                  // Bit #3 = SETUO
  case ISD::SETUGE:
  case ISD::SETGE:  Invert = true; return 0;   // !Bit #0 = SETUGE
  case ISD::SETULE:
  case ISD::SETLE:  Invert = true; return 1;   // !Bit #1 = SETULE
  case ISD::SETUNE:
  case ISD::SETNE:  Invert = true; return 2;   // !Bit #2 = SETUNE
  case ISD::SETO:   Invert = true; return 3;   // !Bit #3 = SETO
  case ISD::SETULT: Other = 0; return 3;       // SETOLT | SETUO
  case ISD::SETUGT: Other = 1; return 3;       // SETOGT | SETUO
  case ISD::SETUEQ: Other = 2; return 3;       // SETOEQ | SETUO
  case ISD::SETOGE: Other = 1; return 2;       // SETOGT | SETOEQ
  case ISD::SETOLE: Other = 0; return 2;       // SETOLT | SETOEQ
  case ISD::SETONE: Other = 0; return 1;       // SETOLT | SETOGT
  }
  return 0;
}

SDNode *PPCDAGToDAGISel::SelectSETCC(SDOperand Op) {
  SDNode *N = Op.Val;
  unsigned Imm;
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  if (isInt32Immediate(N->getOperand(1), Imm)) {
    // We can codegen setcc op, imm very efficiently compared to a brcond.
    // Check for those cases here.
    // setcc op, 0
    if (Imm == 0) {
      SDOperand Op = N->getOperand(0);
      AddToISelQueue(Op);
      switch (CC) {
      default: break;
      case ISD::SETEQ: {
        Op = SDOperand(CurDAG->getTargetNode(PPC::CNTLZW, MVT::i32, Op), 0);
        SDOperand Ops[] = { Op, getI32Imm(27), getI32Imm(5), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      case ISD::SETNE: {
        SDOperand AD =
          SDOperand(CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                          Op, getI32Imm(~0U)), 0);
        return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, AD, Op, 
                                    AD.getValue(1));
      }
      case ISD::SETLT: {
        SDOperand Ops[] = { Op, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      case ISD::SETGT: {
        SDOperand T =
          SDOperand(CurDAG->getTargetNode(PPC::NEG, MVT::i32, Op), 0);
        T = SDOperand(CurDAG->getTargetNode(PPC::ANDC, MVT::i32, T, Op), 0);
        SDOperand Ops[] = { T, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      }
    } else if (Imm == ~0U) {        // setcc op, -1
      SDOperand Op = N->getOperand(0);
      AddToISelQueue(Op);
      switch (CC) {
      default: break;
      case ISD::SETEQ:
        Op = SDOperand(CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                             Op, getI32Imm(1)), 0);
        return CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32, 
                              SDOperand(CurDAG->getTargetNode(PPC::LI, MVT::i32,
                                                              getI32Imm(0)), 0),
                                    Op.getValue(1));
      case ISD::SETNE: {
        Op = SDOperand(CurDAG->getTargetNode(PPC::NOR, MVT::i32, Op, Op), 0);
        SDNode *AD = CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                           Op, getI32Imm(~0U));
        return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, SDOperand(AD, 0),
                                    Op, SDOperand(AD, 1));
      }
      case ISD::SETLT: {
        SDOperand AD = SDOperand(CurDAG->getTargetNode(PPC::ADDI, MVT::i32, Op,
                                                       getI32Imm(1)), 0);
        SDOperand AN = SDOperand(CurDAG->getTargetNode(PPC::AND, MVT::i32, AD,
                                                       Op), 0);
        SDOperand Ops[] = { AN, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      case ISD::SETGT: {
        SDOperand Ops[] = { Op, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        Op = SDOperand(CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, Ops, 4), 0);
        return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Op, 
                                    getI32Imm(1));
      }
      }
    }
  }
  
  bool Inv;
  int OtherCondIdx;
  unsigned Idx = getCRIdxForSetCC(CC, Inv, OtherCondIdx);
  SDOperand CCReg = SelectCC(N->getOperand(0), N->getOperand(1), CC);
  SDOperand IntCR;
  
  // Force the ccreg into CR7.
  SDOperand CR7Reg = CurDAG->getRegister(PPC::CR7, MVT::i32);
  
  SDOperand InFlag(0, 0);  // Null incoming flag value.
  CCReg = CurDAG->getCopyToReg(CurDAG->getEntryNode(), CR7Reg, CCReg, 
                               InFlag).getValue(1);
  
  if (PPCSubTarget.isGigaProcessor() && OtherCondIdx == -1)
    IntCR = SDOperand(CurDAG->getTargetNode(PPC::MFOCRF, MVT::i32, CR7Reg,
                                            CCReg), 0);
  else
    IntCR = SDOperand(CurDAG->getTargetNode(PPC::MFCR, MVT::i32, CCReg), 0);
  
  SDOperand Ops[] = { IntCR, getI32Imm((32-(3-Idx)) & 31),
                      getI32Imm(31), getI32Imm(31) };
  if (OtherCondIdx == -1 && !Inv)
    return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);

  // Get the specified bit.
  SDOperand Tmp =
    SDOperand(CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, Ops, 4), 0);
  if (Inv) {
    assert(OtherCondIdx == -1 && "Can't have split plus negation");
    return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Tmp, getI32Imm(1));
  }

  // Otherwise, we have to turn an operation like SETONE -> SETOLT | SETOGT.
  // We already got the bit for the first part of the comparison (e.g. SETULE).

  // Get the other bit of the comparison.
  Ops[1] = getI32Imm((32-(3-OtherCondIdx)) & 31);
  SDOperand OtherCond = 
    SDOperand(CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, Ops, 4), 0);

  return CurDAG->SelectNodeTo(N, PPC::OR, MVT::i32, Tmp, OtherCond);
}


// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
SDNode *PPCDAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&
      N->getOpcode() < PPCISD::FIRST_NUMBER)
    return NULL;   // Already selected.

  switch (N->getOpcode()) {
  default: break;
  
  case ISD::Constant: {
    if (N->getValueType(0) == MVT::i64) {
      // Get 64 bit value.
      int64_t Imm = cast<ConstantSDNode>(N)->getValue();
      // Assume no remaining bits.
      unsigned Remainder = 0;
      // Assume no shift required.
      unsigned Shift = 0;
      
      // If it can't be represented as a 32 bit value.
      if (!isInt32(Imm)) {
        Shift = CountTrailingZeros_64(Imm);
        int64_t ImmSh = static_cast<uint64_t>(Imm) >> Shift;
        
        // If the shifted value fits 32 bits.
        if (isInt32(ImmSh)) {
          // Go with the shifted value.
          Imm = ImmSh;
        } else {
          // Still stuck with a 64 bit value.
          Remainder = Imm;
          Shift = 32;
          Imm >>= 32;
        }
      }
      
      // Intermediate operand.
      SDNode *Result;

      // Handle first 32 bits.
      unsigned Lo = Imm & 0xFFFF;
      unsigned Hi = (Imm >> 16) & 0xFFFF;
      
      // Simple value.
      if (isInt16(Imm)) {
       // Just the Lo bits.
        Result = CurDAG->getTargetNode(PPC::LI8, MVT::i64, getI32Imm(Lo));
      } else if (Lo) {
        // Handle the Hi bits.
        unsigned OpC = Hi ? PPC::LIS8 : PPC::LI8;
        Result = CurDAG->getTargetNode(OpC, MVT::i64, getI32Imm(Hi));
        // And Lo bits.
        Result = CurDAG->getTargetNode(PPC::ORI8, MVT::i64,
                                       SDOperand(Result, 0), getI32Imm(Lo));
      } else {
       // Just the Hi bits.
        Result = CurDAG->getTargetNode(PPC::LIS8, MVT::i64, getI32Imm(Hi));
      }
      
      // If no shift, we're done.
      if (!Shift) return Result;

      // Shift for next step if the upper 32-bits were not zero.
      if (Imm) {
        Result = CurDAG->getTargetNode(PPC::RLDICR, MVT::i64,
                                       SDOperand(Result, 0),
                                       getI32Imm(Shift), getI32Imm(63 - Shift));
      }

      // Add in the last bits as required.
      if ((Hi = (Remainder >> 16) & 0xFFFF)) {
        Result = CurDAG->getTargetNode(PPC::ORIS8, MVT::i64,
                                       SDOperand(Result, 0), getI32Imm(Hi));
      } 
      if ((Lo = Remainder & 0xFFFF)) {
        Result = CurDAG->getTargetNode(PPC::ORI8, MVT::i64,
                                       SDOperand(Result, 0), getI32Imm(Lo));
      }
      
      return Result;
    }
    break;
  }
  
  case ISD::SETCC:
    return SelectSETCC(Op);
  case PPCISD::GlobalBaseReg:
    return getGlobalBaseReg();
    
  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDOperand TFI = CurDAG->getTargetFrameIndex(FI, Op.getValueType());
    unsigned Opc = Op.getValueType() == MVT::i32 ? PPC::ADDI : PPC::ADDI8;
    if (N->hasOneUse())
      return CurDAG->SelectNodeTo(N, Opc, Op.getValueType(), TFI,
                                  getSmallIPtrImm(0));
    return CurDAG->getTargetNode(Opc, Op.getValueType(), TFI,
                                 getSmallIPtrImm(0));
  }

  case PPCISD::MFCR: {
    SDOperand InFlag = N->getOperand(1);
    AddToISelQueue(InFlag);
    // Use MFOCRF if supported.
    if (PPCSubTarget.isGigaProcessor())
      return CurDAG->getTargetNode(PPC::MFOCRF, MVT::i32,
                                   N->getOperand(0), InFlag);
    else
      return CurDAG->getTargetNode(PPC::MFCR, MVT::i32, InFlag);
  }
    
  case ISD::SDIV: {
    // FIXME: since this depends on the setting of the carry flag from the srawi
    //        we should really be making notes about that for the scheduler.
    // FIXME: It sure would be nice if we could cheaply recognize the 
    //        srl/add/sra pattern the dag combiner will generate for this as
    //        sra/addze rather than having to handle sdiv ourselves.  oh well.
    unsigned Imm;
    if (isInt32Immediate(N->getOperand(1), Imm)) {
      SDOperand N0 = N->getOperand(0);
      AddToISelQueue(N0);
      if ((signed)Imm > 0 && isPowerOf2_32(Imm)) {
        SDNode *Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                N0, getI32Imm(Log2_32(Imm)));
        return CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32, 
                                    SDOperand(Op, 0), SDOperand(Op, 1));
      } else if ((signed)Imm < 0 && isPowerOf2_32(-Imm)) {
        SDNode *Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                N0, getI32Imm(Log2_32(-Imm)));
        SDOperand PT =
          SDOperand(CurDAG->getTargetNode(PPC::ADDZE, MVT::i32,
                                          SDOperand(Op, 0), SDOperand(Op, 1)),
                    0);
        return CurDAG->SelectNodeTo(N, PPC::NEG, MVT::i32, PT);
      }
    }
    
    // Other cases are autogenerated.
    break;
  }
    
  case ISD::LOAD: {
    // Handle preincrement loads.
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    MVT LoadedVT = LD->getMemoryVT();
    
    // Normal loads are handled by code generated from the .td file.
    if (LD->getAddressingMode() != ISD::PRE_INC)
      break;
    
    SDOperand Offset = LD->getOffset();
    if (isa<ConstantSDNode>(Offset) ||
        Offset.getOpcode() == ISD::TargetGlobalAddress) {
      
      unsigned Opcode;
      bool isSExt = LD->getExtensionType() == ISD::SEXTLOAD;
      if (LD->getValueType(0) != MVT::i64) {
        // Handle PPC32 integer and normal FP loads.
        assert((!isSExt || LoadedVT == MVT::i16) && "Invalid sext update load");
        switch (LoadedVT.getSimpleVT()) {
          default: assert(0 && "Invalid PPC load type!");
          case MVT::f64: Opcode = PPC::LFDU; break;
          case MVT::f32: Opcode = PPC::LFSU; break;
          case MVT::i32: Opcode = PPC::LWZU; break;
          case MVT::i16: Opcode = isSExt ? PPC::LHAU : PPC::LHZU; break;
          case MVT::i1:
          case MVT::i8:  Opcode = PPC::LBZU; break;
        }
      } else {
        assert(LD->getValueType(0) == MVT::i64 && "Unknown load result type!");
        assert((!isSExt || LoadedVT == MVT::i16) && "Invalid sext update load");
        switch (LoadedVT.getSimpleVT()) {
          default: assert(0 && "Invalid PPC load type!");
          case MVT::i64: Opcode = PPC::LDU; break;
          case MVT::i32: Opcode = PPC::LWZU8; break;
          case MVT::i16: Opcode = isSExt ? PPC::LHAU8 : PPC::LHZU8; break;
          case MVT::i1:
          case MVT::i8:  Opcode = PPC::LBZU8; break;
        }
      }
      
      SDOperand Chain = LD->getChain();
      SDOperand Base = LD->getBasePtr();
      AddToISelQueue(Chain);
      AddToISelQueue(Base);
      AddToISelQueue(Offset);
      SDOperand Ops[] = { Offset, Base, Chain };
      // FIXME: PPC64
      return CurDAG->getTargetNode(Opcode, MVT::i32, MVT::i32,
                                   MVT::Other, Ops, 3);
    } else {
      assert(0 && "R+R preindex loads not supported yet!");
    }
  }
    
  case ISD::AND: {
    unsigned Imm, Imm2, SH, MB, ME;

    // If this is an and of a value rotated between 0 and 31 bits and then and'd
    // with a mask, emit rlwinm
    if (isInt32Immediate(N->getOperand(1), Imm) &&
        isRotateAndMask(N->getOperand(0).Val, Imm, false, SH, MB, ME)) {
      SDOperand Val = N->getOperand(0).getOperand(0);
      AddToISelQueue(Val);
      SDOperand Ops[] = { Val, getI32Imm(SH), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }
    // If this is just a masked value where the input is not handled above, and
    // is not a rotate-left (handled by a pattern in the .td file), emit rlwinm
    if (isInt32Immediate(N->getOperand(1), Imm) &&
        isRunOfOnes(Imm, MB, ME) && 
        N->getOperand(0).getOpcode() != ISD::ROTL) {
      SDOperand Val = N->getOperand(0);
      AddToISelQueue(Val);
      SDOperand Ops[] = { Val, getI32Imm(0), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }
    // AND X, 0 -> 0, not "rlwinm 32".
    if (isInt32Immediate(N->getOperand(1), Imm) && (Imm == 0)) {
      AddToISelQueue(N->getOperand(1));
      ReplaceUses(SDOperand(N, 0), N->getOperand(1));
      return NULL;
    }
    // ISD::OR doesn't get all the bitfield insertion fun.
    // (and (or x, c1), c2) where isRunOfOnes(~(c1^c2)) is a bitfield insert
    if (isInt32Immediate(N->getOperand(1), Imm) && 
        N->getOperand(0).getOpcode() == ISD::OR &&
        isInt32Immediate(N->getOperand(0).getOperand(1), Imm2)) {
      unsigned MB, ME;
      Imm = ~(Imm^Imm2);
      if (isRunOfOnes(Imm, MB, ME)) {
        AddToISelQueue(N->getOperand(0).getOperand(0));
        AddToISelQueue(N->getOperand(0).getOperand(1));
        SDOperand Ops[] = { N->getOperand(0).getOperand(0),
                            N->getOperand(0).getOperand(1),
                            getI32Imm(0), getI32Imm(MB),getI32Imm(ME) };
        return CurDAG->getTargetNode(PPC::RLWIMI, MVT::i32, Ops, 5);
      }
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::OR:
    if (N->getValueType(0) == MVT::i32)
      if (SDNode *I = SelectBitfieldInsert(N))
        return I;
      
    // Other cases are autogenerated.
    break;
  case ISD::SHL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) {
      AddToISelQueue(N->getOperand(0).getOperand(0));
      SDOperand Ops[] = { N->getOperand(0).getOperand(0),
                          getI32Imm(SH), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::SRL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) { 
      AddToISelQueue(N->getOperand(0).getOperand(0));
      SDOperand Ops[] = { N->getOperand(0).getOperand(0),
                          getI32Imm(SH), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::SELECT_CC: {
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(4))->get();
    
    // Handle the setcc cases here.  select_cc lhs, 0, 1, 0, cc
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N->getOperand(1)))
      if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N->getOperand(2)))
        if (ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N->getOperand(3)))
          if (N1C->isNullValue() && N3C->isNullValue() &&
              N2C->getValue() == 1ULL && CC == ISD::SETNE &&
              // FIXME: Implement this optzn for PPC64.
              N->getValueType(0) == MVT::i32) {
            AddToISelQueue(N->getOperand(0));
            SDNode *Tmp =
              CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                    N->getOperand(0), getI32Imm(~0U));
            return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32,
                                        SDOperand(Tmp, 0), N->getOperand(0),
                                        SDOperand(Tmp, 1));
          }

    SDOperand CCReg = SelectCC(N->getOperand(0), N->getOperand(1), CC);
    unsigned BROpc = getPredicateForSetCC(CC);

    unsigned SelectCCOp;
    if (N->getValueType(0) == MVT::i32)
      SelectCCOp = PPC::SELECT_CC_I4;
    else if (N->getValueType(0) == MVT::i64)
      SelectCCOp = PPC::SELECT_CC_I8;
    else if (N->getValueType(0) == MVT::f32)
      SelectCCOp = PPC::SELECT_CC_F4;
    else if (N->getValueType(0) == MVT::f64)
      SelectCCOp = PPC::SELECT_CC_F8;
    else
      SelectCCOp = PPC::SELECT_CC_VRRC;

    AddToISelQueue(N->getOperand(2));
    AddToISelQueue(N->getOperand(3));
    SDOperand Ops[] = { CCReg, N->getOperand(2), N->getOperand(3),
                        getI32Imm(BROpc) };
    return CurDAG->SelectNodeTo(N, SelectCCOp, N->getValueType(0), Ops, 4);
  }
  case PPCISD::COND_BRANCH: {
    AddToISelQueue(N->getOperand(0));  // Op #0 is the Chain.
    // Op #1 is the PPC::PRED_* number.
    // Op #2 is the CR#
    // Op #3 is the Dest MBB
    AddToISelQueue(N->getOperand(4));  // Op #4 is the Flag.
    // Prevent PPC::PRED_* from being selected into LI.
    SDOperand Pred =
      getI32Imm(cast<ConstantSDNode>(N->getOperand(1))->getValue());
    SDOperand Ops[] = { Pred, N->getOperand(2), N->getOperand(3),
      N->getOperand(0), N->getOperand(4) };
    return CurDAG->SelectNodeTo(N, PPC::BCC, MVT::Other, Ops, 5);
  }
  case ISD::BR_CC: {
    AddToISelQueue(N->getOperand(0));
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
    SDOperand CondCode = SelectCC(N->getOperand(2), N->getOperand(3), CC);
    SDOperand Ops[] = { getI32Imm(getPredicateForSetCC(CC)), CondCode, 
                        N->getOperand(4), N->getOperand(0) };
    return CurDAG->SelectNodeTo(N, PPC::BCC, MVT::Other, Ops, 4);
  }
  case ISD::BRIND: {
    // FIXME: Should custom lower this.
    SDOperand Chain = N->getOperand(0);
    SDOperand Target = N->getOperand(1);
    AddToISelQueue(Chain);
    AddToISelQueue(Target);
    unsigned Opc = Target.getValueType() == MVT::i32 ? PPC::MTCTR : PPC::MTCTR8;
    Chain = SDOperand(CurDAG->getTargetNode(Opc, MVT::Other, Target,
                                            Chain), 0);
    return CurDAG->SelectNodeTo(N, PPC::BCTR, MVT::Other, Chain);
  }
  }
  
  return SelectCode(Op);
}



/// createPPCISelDag - This pass converts a legalized DAG into a 
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createPPCISelDag(PPCTargetMachine &TM) {
  return new PPCDAGToDAGISel(TM);
}

