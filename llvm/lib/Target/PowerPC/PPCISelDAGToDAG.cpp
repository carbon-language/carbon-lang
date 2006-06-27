//===-- PPCISelDAGToDAG.cpp - PPC --pattern matching inst selector --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for PowerPC,
// converting from a legalized dag to a PPC dag.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCTargetMachine.h"
#include "PPCISelLowering.h"
#include "PPCHazardRecognizers.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <iostream>
#include <set>
using namespace llvm;

namespace {
  Statistic<> FrameOff("ppc-codegen", "Number of frame idx offsets collapsed");
    
  //===--------------------------------------------------------------------===//
  /// PPCDAGToDAGISel - PPC specific code to select PPC machine
  /// instructions for SelectionDAG operations.
  ///
  class PPCDAGToDAGISel : public SelectionDAGISel {
    PPCTargetMachine &TM;
    PPCTargetLowering PPCLowering;
    unsigned GlobalBaseReg;
  public:
    PPCDAGToDAGISel(PPCTargetMachine &tm)
      : SelectionDAGISel(PPCLowering), TM(tm),
        PPCLowering(*TM.getTargetLowering()) {}
    
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
    
    
    /// getGlobalBaseReg - insert code into the entry mbb to materialize the PIC
    /// base register.  Return the virtual register that holds this value.
    SDOperand getGlobalBaseReg();
    
    // Select - Convert the specified operand from a target-independent to a
    // target-specific node if it hasn't already been changed.
    void Select(SDOperand &Result, SDOperand Op);
    
    SDNode *SelectBitfieldInsert(SDNode *N);

    /// SelectCC - Select a comparison of the specified values with the
    /// specified condition code, returning the CR# of the expression.
    SDOperand SelectCC(SDOperand LHS, SDOperand RHS, ISD::CondCode CC);

    /// SelectAddrImm - Returns true if the address N can be represented by
    /// a base register plus a signed 16-bit displacement [r+imm].
    bool SelectAddrImm(SDOperand N, SDOperand &Disp, SDOperand &Base);
      
    /// SelectAddrIdx - Given the specified addressed, check to see if it can be
    /// represented as an indexed [r+r] operation.  Returns false if it can
    /// be represented by [r+imm], which are preferred.
    bool SelectAddrIdx(SDOperand N, SDOperand &Base, SDOperand &Index);
    
    /// SelectAddrIdxOnly - Given the specified addressed, force it to be
    /// represented as an indexed [r+r] operation.
    bool SelectAddrIdxOnly(SDOperand N, SDOperand &Base, SDOperand &Index);

    /// SelectAddrImmShift - Returns true if the address N can be represented by
    /// a base register plus a signed 14-bit displacement [r+imm*4].  Suitable
    /// for use by STD and friends.
    bool SelectAddrImmShift(SDOperand N, SDOperand &Disp, SDOperand &Base);
    
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
        if (!SelectAddrIdx(Op, Op0, Op1))
          SelectAddrImm(Op, Op0, Op1);
        break;
      case 'o':   // offsetable
        if (!SelectAddrImm(Op, Op0, Op1)) {
          Select(Op0, Op);     // r+0.
          Op1 = getSmallIPtrImm(0);
        }
        break;
      case 'v':   // not offsetable
        SelectAddrIdxOnly(Op, Op0, Op1);
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
    SDOperand SelectSETCC(SDOperand Op);
    void MySelect_PPCbctrl(SDOperand &Result, SDOperand N);
    void MySelect_PPCcall(SDOperand &Result, SDOperand N);
  };
}

/// InstructionSelectBasicBlock - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void PPCDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());
  
  // The selection process is inherently a bottom-up recursive process (users
  // select their uses before themselves).  Given infinite stack space, we
  // could just start selecting on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, select the entry node, then all its uses,
  // iteratively instead of recursively.
  std::vector<SDOperand> Worklist;
  Worklist.push_back(DAG.getEntryNode());
  
  // Note that we can do this in the PPC target (scanning forward across token
  // chain edges) because no nodes ever get folded across these edges.  On a
  // target like X86 which supports load/modify/store operations, this would
  // have to be more careful.
  while (!Worklist.empty()) {
    SDOperand Node = Worklist.back();
    Worklist.pop_back();
    
    // Chose from the least deep of the top two nodes.
    if (!Worklist.empty() &&
        Worklist.back().Val->getNodeDepth() < Node.Val->getNodeDepth())
      std::swap(Worklist.back(), Node);
    
    if ((Node.Val->getOpcode() >= ISD::BUILTIN_OP_END &&
         Node.Val->getOpcode() < PPCISD::FIRST_NUMBER) ||
        CodeGenMap.count(Node)) continue;
    
    for (SDNode::use_iterator UI = Node.Val->use_begin(),
         E = Node.Val->use_end(); UI != E; ++UI) {
      // Scan the values.  If this use has a value that is a token chain, add it
      // to the worklist.
      SDNode *User = *UI;
      for (unsigned i = 0, e = User->getNumValues(); i != e; ++i)
        if (User->getValueType(i) == MVT::Other) {
          Worklist.push_back(SDOperand(User, i));
          break; 
        }
    }

    // Finally, legalize this node.
    SDOperand Dummy;
    Select(Dummy, Node);
  }
    
  // Select target instructions for the DAG.
  DAG.setRoot(SelectRoot(DAG.getRoot()));
  assert(InFlightSet.empty() && "ISel InFlightSet has not been emptied!");
  CodeGenMap.clear();
  HandleMap.clear();
  ReplaceMap.clear();
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
  SSARegMap *RegMap = Fn.getSSARegMap();
  bool HasVectorVReg = false;
  for (unsigned i = MRegisterInfo::FirstVirtualRegister, 
       e = RegMap->getLastVirtReg()+1; i != e; ++i)
    if (RegMap->getRegClass(i) == &PPC::VRRCRegClass) {
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
  unsigned InVRSAVE = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
  unsigned UpdatedVRSAVE = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
  
  MachineBasicBlock &EntryBB = *Fn.begin();
  // Emit the following code into the entry block:
  // InVRSAVE = MFVRSAVE
  // UpdatedVRSAVE = UPDATE_VRSAVE InVRSAVE
  // MTVRSAVE UpdatedVRSAVE
  MachineBasicBlock::iterator IP = EntryBB.begin();  // Insert Point
  BuildMI(EntryBB, IP, PPC::MFVRSAVE, 0, InVRSAVE);
  BuildMI(EntryBB, IP, PPC::UPDATE_VRSAVE, 1, UpdatedVRSAVE).addReg(InVRSAVE);
  BuildMI(EntryBB, IP, PPC::MTVRSAVE, 1).addReg(UpdatedVRSAVE);
  
  // Find all return blocks, outputting a restore in each epilog.
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    if (!BB->empty() && TII.isReturn(BB->back().getOpcode())) {
      IP = BB->end(); --IP;
      
      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = IP;
      while (I2 != BB->begin() && TII.isTerminatorInstr((--I2)->getOpcode()))
        IP = I2;
      
      // Emit: MTVRSAVE InVRSave
      BuildMI(*BB, IP, PPC::MTVRSAVE, 1).addReg(InVRSAVE);
    }        
  }
}


/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
SDOperand PPCDAGToDAGISel::getGlobalBaseReg() {
  if (!GlobalBaseReg) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    SSARegMap *RegMap = BB->getParent()->getSSARegMap();

    if (PPCLowering.getPointerTy() == MVT::i32)
      GlobalBaseReg = RegMap->createVirtualRegister(PPC::GPRCRegisterClass);
    else
      GlobalBaseReg = RegMap->createVirtualRegister(PPC::G8RCRegisterClass);
    
    BuildMI(FirstMBB, MBBI, PPC::MovePCtoLR, 0, PPC::LR);
    BuildMI(FirstMBB, MBBI, PPC::MFLR, 1, GlobalBaseReg);
  }
  return CurDAG->getRegister(GlobalBaseReg, PPCLowering.getPointerTy());
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
  if (N->getOpcode() == ISD::Constant && N->getValueType(0) == MVT::i32) {
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

// isRotateAndMask - Returns true if Mask and Shift can be folded into a rotate
// and mask opcode and mask operation.
static bool isRotateAndMask(SDNode *N, unsigned Mask, bool IsShiftMask,
                            unsigned &SH, unsigned &MB, unsigned &ME) {
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
  
  uint64_t LKZ, LKO, RKZ, RKO;
  TLI.ComputeMaskedBits(Op0, 0xFFFFFFFFULL, LKZ, LKO);
  TLI.ComputeMaskedBits(Op1, 0xFFFFFFFFULL, RKZ, RKO);
  
  unsigned TargetMask = LKZ;
  unsigned InsertMask = RKZ;
  
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
      Select(Tmp1, Tmp3);
      Select(Tmp2, Op1);
      SH &= 31;
      return CurDAG->getTargetNode(PPC::RLWIMI, MVT::i32, Tmp1, Tmp2,
                                   getI32Imm(SH), getI32Imm(MB), getI32Imm(ME));
    }
  }
  return 0;
}

/// SelectAddrImm - Returns true if the address N can be represented by
/// a base register plus a signed 16-bit displacement [r+imm].
bool PPCDAGToDAGISel::SelectAddrImm(SDOperand N, SDOperand &Disp, 
                                    SDOperand &Base) {
  // If this can be more profitably realized as r+r, fail.
  if (SelectAddrIdx(N, Disp, Base))
    return false;

  if (N.getOpcode() == ISD::ADD) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm)) {
      Disp = getI32Imm((int)imm & 0xFFFF);
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N.getOperand(0))) {
        Base = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
      } else {
        Base = N.getOperand(0);
      }
      return true; // [r+i]
    } else if (N.getOperand(1).getOpcode() == PPCISD::Lo) {
      // Match LOAD (ADD (X, Lo(G))).
      assert(!cast<ConstantSDNode>(N.getOperand(1).getOperand(1))->getValue()
             && "Cannot handle constant offsets yet!");
      Disp = N.getOperand(1).getOperand(0);  // The global address.
      assert(Disp.getOpcode() == ISD::TargetGlobalAddress ||
             Disp.getOpcode() == ISD::TargetConstantPool ||
             Disp.getOpcode() == ISD::TargetJumpTable);
      Base = N.getOperand(0);
      return true;  // [&g+r]
    }
  } else if (N.getOpcode() == ISD::OR) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm)) {
      // If this is an or of disjoint bitfields, we can codegen this as an add
      // (for better address arithmetic) if the LHS and RHS of the OR are
      // provably disjoint.
      uint64_t LHSKnownZero, LHSKnownOne;
      PPCLowering.ComputeMaskedBits(N.getOperand(0), ~0U,
                                    LHSKnownZero, LHSKnownOne);
      if ((LHSKnownZero|~(unsigned)imm) == ~0U) {
        // If all of the bits are known zero on the LHS or RHS, the add won't
        // carry.
        Base = N.getOperand(0);
        Disp = getI32Imm((int)imm & 0xFFFF);
        return true;
      }
    }
  } else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    // Loading from a constant address.

    // If this address fits entirely in a 16-bit sext immediate field, codegen
    // this as "d, 0"
    short Imm;
    if (isIntS16Immediate(CN, Imm)) {
      Disp = CurDAG->getTargetConstant(Imm, CN->getValueType(0));
      Base = CurDAG->getRegister(PPC::R0, CN->getValueType(0));
      return true;
    }

    // FIXME: Handle small sext constant offsets in PPC64 mode also!
    if (CN->getValueType(0) == MVT::i32) {
      int Addr = (int)CN->getValue();
    
      // Otherwise, break this down into an LIS + disp.
      Disp = getI32Imm((short)Addr);
      Base = CurDAG->getConstant(Addr - (signed short)Addr, MVT::i32);
      return true;
    }
  }
  
  Disp = getSmallIPtrImm(0);
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N))
    Base = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
  else
    Base = N;
  return true;      // [r+0]
}

/// SelectAddrIdx - Given the specified addressed, check to see if it can be
/// represented as an indexed [r+r] operation.  Returns false if it can
/// be represented by [r+imm], which are preferred.
bool PPCDAGToDAGISel::SelectAddrIdx(SDOperand N, SDOperand &Base, 
                                    SDOperand &Index) {
  short imm = 0;
  if (N.getOpcode() == ISD::ADD) {
    if (isIntS16Immediate(N.getOperand(1), imm))
      return false;    // r+i
    if (N.getOperand(1).getOpcode() == PPCISD::Lo)
      return false;    // r+i
    
    Base = N.getOperand(0);
    Index = N.getOperand(1);
    return true;
  } else if (N.getOpcode() == ISD::OR) {
    if (isIntS16Immediate(N.getOperand(1), imm))
      return false;    // r+i can fold it if we can.
    
    // If this is an or of disjoint bitfields, we can codegen this as an add
    // (for better address arithmetic) if the LHS and RHS of the OR are provably
    // disjoint.
    uint64_t LHSKnownZero, LHSKnownOne;
    uint64_t RHSKnownZero, RHSKnownOne;
    PPCLowering.ComputeMaskedBits(N.getOperand(0), ~0U,
                                  LHSKnownZero, LHSKnownOne);
    
    if (LHSKnownZero) {
      PPCLowering.ComputeMaskedBits(N.getOperand(1), ~0U,
                                    RHSKnownZero, RHSKnownOne);
      // If all of the bits are known zero on the LHS or RHS, the add won't
      // carry.
      if ((LHSKnownZero | RHSKnownZero) == ~0U) {
        Base = N.getOperand(0);
        Index = N.getOperand(1);
        return true;
      }
    }
  }
  
  return false;
}

/// SelectAddrIdxOnly - Given the specified addressed, force it to be
/// represented as an indexed [r+r] operation.
bool PPCDAGToDAGISel::SelectAddrIdxOnly(SDOperand N, SDOperand &Base, 
                                        SDOperand &Index) {
  // Check to see if we can easily represent this as an [r+r] address.  This
  // will fail if it thinks that the address is more profitably represented as
  // reg+imm, e.g. where imm = 0.
  if (SelectAddrIdx(N, Base, Index))
    return true;
  
  // If the operand is an addition, always emit this as [r+r], since this is
  // better (for code size, and execution, as the memop does the add for free)
  // than emitting an explicit add.
  if (N.getOpcode() == ISD::ADD) {
    Base = N.getOperand(0);
    Index = N.getOperand(1);
    return true;
  }
  
  // Otherwise, do it the hard way, using R0 as the base register.
  Base = CurDAG->getRegister(PPC::R0, N.getValueType());
  Index = N;
  return true;
}

/// SelectAddrImmShift - Returns true if the address N can be represented by
/// a base register plus a signed 14-bit displacement [r+imm*4].  Suitable
/// for use by STD and friends.
bool PPCDAGToDAGISel::SelectAddrImmShift(SDOperand N, SDOperand &Disp, 
                                         SDOperand &Base) {
  // If this can be more profitably realized as r+r, fail.
  if (SelectAddrIdx(N, Disp, Base))
    return false;
  
  if (N.getOpcode() == ISD::ADD) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm) && (imm & 3) == 0) {
      Disp = getI32Imm(((int)imm & 0xFFFF) >> 2);
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N.getOperand(0))) {
        Base = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
      } else {
        Base = N.getOperand(0);
      }
      return true; // [r+i]
    } else if (N.getOperand(1).getOpcode() == PPCISD::Lo) {
      // Match LOAD (ADD (X, Lo(G))).
      assert(!cast<ConstantSDNode>(N.getOperand(1).getOperand(1))->getValue()
             && "Cannot handle constant offsets yet!");
      Disp = N.getOperand(1).getOperand(0);  // The global address.
      assert(Disp.getOpcode() == ISD::TargetGlobalAddress ||
             Disp.getOpcode() == ISD::TargetConstantPool ||
             Disp.getOpcode() == ISD::TargetJumpTable);
      Base = N.getOperand(0);
      return true;  // [&g+r]
    }
  } else if (N.getOpcode() == ISD::OR) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm) && (imm & 3) == 0) {
      // If this is an or of disjoint bitfields, we can codegen this as an add
      // (for better address arithmetic) if the LHS and RHS of the OR are
      // provably disjoint.
      uint64_t LHSKnownZero, LHSKnownOne;
      PPCLowering.ComputeMaskedBits(N.getOperand(0), ~0U,
                                    LHSKnownZero, LHSKnownOne);
      if ((LHSKnownZero|~(unsigned)imm) == ~0U) {
        // If all of the bits are known zero on the LHS or RHS, the add won't
        // carry.
        Base = N.getOperand(0);
        Disp = getI32Imm(((int)imm & 0xFFFF) >> 2);
        return true;
      }
    }
  } else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    // Loading from a constant address.
    
    // If this address fits entirely in a 14-bit sext immediate field, codegen
    // this as "d, 0"
    short Imm;
    if (isIntS16Immediate(CN, Imm)) {
      Disp = getSmallIPtrImm((unsigned short)Imm >> 2);
      Base = CurDAG->getRegister(PPC::R0, CN->getValueType(0));
      return true;
    }
    
    // FIXME: Handle small sext constant offsets in PPC64 mode also!
    if (CN->getValueType(0) == MVT::i32) {
      int Addr = (int)CN->getValue();
      
      // Otherwise, break this down into an LIS + disp.
      Disp = getI32Imm((short)Addr >> 2);
      Base = CurDAG->getConstant(Addr - (signed short)Addr, MVT::i32);
      return true;
    }
  }
  
  Disp = getSmallIPtrImm(0);
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N))
    Base = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
  else
    Base = N;
  return true;      // [r+0]
}


/// SelectCC - Select a comparison of the specified values with the specified
/// condition code, returning the CR# of the expression.
SDOperand PPCDAGToDAGISel::SelectCC(SDOperand LHS, SDOperand RHS,
                                    ISD::CondCode CC) {
  // Always select the LHS.
  Select(LHS, LHS);
  unsigned Opc;
  
  if (LHS.getValueType() == MVT::i32) {
    unsigned Imm;
    if (ISD::isUnsignedIntSetCC(CC)) {
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
    if (ISD::isUnsignedIntSetCC(CC)) {
      if (isInt64Immediate(RHS.Val, Imm) && isUInt16(Imm))
        return SDOperand(CurDAG->getTargetNode(PPC::CMPLDI, MVT::i64, LHS,
                                               getI64Imm(Imm & 0xFFFF)), 0);
      Opc = PPC::CMPLD;
    } else {
      short SImm;
      if (isIntS16Immediate(RHS, SImm))
        return SDOperand(CurDAG->getTargetNode(PPC::CMPDI, MVT::i64, LHS,
                                               getI64Imm((int)SImm & 0xFFFF)),
                         0);
      Opc = PPC::CMPD;
    }
  } else if (LHS.getValueType() == MVT::f32) {
    Opc = PPC::FCMPUS;
  } else {
    assert(LHS.getValueType() == MVT::f64 && "Unknown vt!");
    Opc = PPC::FCMPUD;
  }
  Select(RHS, RHS);
  return SDOperand(CurDAG->getTargetNode(Opc, MVT::i32, LHS, RHS), 0);
}

/// getBCCForSetCC - Returns the PowerPC condition branch mnemonic corresponding
/// to Condition.
static unsigned getBCCForSetCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETOEQ:    // FIXME: This is incorrect see PR642.
  case ISD::SETUEQ:
  case ISD::SETEQ:  return PPC::BEQ;
  case ISD::SETONE:    // FIXME: This is incorrect see PR642.
  case ISD::SETUNE:
  case ISD::SETNE:  return PPC::BNE;
  case ISD::SETOLT:    // FIXME: This is incorrect see PR642.
  case ISD::SETULT:
  case ISD::SETLT:  return PPC::BLT;
  case ISD::SETOLE:    // FIXME: This is incorrect see PR642.
  case ISD::SETULE:
  case ISD::SETLE:  return PPC::BLE;
  case ISD::SETOGT:    // FIXME: This is incorrect see PR642.
  case ISD::SETUGT:
  case ISD::SETGT:  return PPC::BGT;
  case ISD::SETOGE:    // FIXME: This is incorrect see PR642.
  case ISD::SETUGE:
  case ISD::SETGE:  return PPC::BGE;
    
  case ISD::SETO:   return PPC::BUN;
  case ISD::SETUO:  return PPC::BNU;
  }
  return 0;
}

/// getCRIdxForSetCC - Return the index of the condition register field
/// associated with the SetCC condition, and whether or not the field is
/// treated as inverted.  That is, lt = 0; ge = 0 inverted.
static unsigned getCRIdxForSetCC(ISD::CondCode CC, bool& Inv) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETOLT:  // FIXME: This is incorrect see PR642.
  case ISD::SETULT:
  case ISD::SETLT:  Inv = false;  return 0;
  case ISD::SETOGE:  // FIXME: This is incorrect see PR642.
  case ISD::SETUGE:
  case ISD::SETGE:  Inv = true;   return 0;
  case ISD::SETOGT:  // FIXME: This is incorrect see PR642.
  case ISD::SETUGT:
  case ISD::SETGT:  Inv = false;  return 1;
  case ISD::SETOLE:  // FIXME: This is incorrect see PR642.
  case ISD::SETULE:
  case ISD::SETLE:  Inv = true;   return 1;
  case ISD::SETOEQ:  // FIXME: This is incorrect see PR642.
  case ISD::SETUEQ:
  case ISD::SETEQ:  Inv = false;  return 2;
  case ISD::SETONE:  // FIXME: This is incorrect see PR642.
  case ISD::SETUNE:
  case ISD::SETNE:  Inv = true;   return 2;
  case ISD::SETO:   Inv = true;   return 3;
  case ISD::SETUO:  Inv = false;  return 3;
  }
  return 0;
}

SDOperand PPCDAGToDAGISel::SelectSETCC(SDOperand Op) {
  SDNode *N = Op.Val;
  unsigned Imm;
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  if (isInt32Immediate(N->getOperand(1), Imm)) {
    // We can codegen setcc op, imm very efficiently compared to a brcond.
    // Check for those cases here.
    // setcc op, 0
    if (Imm == 0) {
      SDOperand Op;
      Select(Op, N->getOperand(0));
      switch (CC) {
      default: break;
      case ISD::SETEQ:
        Op = SDOperand(CurDAG->getTargetNode(PPC::CNTLZW, MVT::i32, Op), 0);
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Op, getI32Imm(27),
                                    getI32Imm(5), getI32Imm(31));
      case ISD::SETNE: {
        SDOperand AD =
          SDOperand(CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                          Op, getI32Imm(~0U)), 0);
        return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, AD, Op, 
                                    AD.getValue(1));
      }
      case ISD::SETLT:
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Op, getI32Imm(1),
                                    getI32Imm(31), getI32Imm(31));
      case ISD::SETGT: {
        SDOperand T =
          SDOperand(CurDAG->getTargetNode(PPC::NEG, MVT::i32, Op), 0);
        T = SDOperand(CurDAG->getTargetNode(PPC::ANDC, MVT::i32, T, Op), 0);
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, T, getI32Imm(1),
                                    getI32Imm(31), getI32Imm(31));
      }
      }
    } else if (Imm == ~0U) {        // setcc op, -1
      SDOperand Op;
      Select(Op, N->getOperand(0));
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
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, AN, getI32Imm(1),
                                    getI32Imm(31), getI32Imm(31));
      }
      case ISD::SETGT:
        Op = SDOperand(CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, Op,
                                             getI32Imm(1), getI32Imm(31),
                                             getI32Imm(31)), 0);
        return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Op, getI32Imm(1));
      }
    }
  }
  
  bool Inv;
  unsigned Idx = getCRIdxForSetCC(CC, Inv);
  SDOperand CCReg = SelectCC(N->getOperand(0), N->getOperand(1), CC);
  SDOperand IntCR;
  
  // Force the ccreg into CR7.
  SDOperand CR7Reg = CurDAG->getRegister(PPC::CR7, MVT::i32);
  
  SDOperand InFlag(0, 0);  // Null incoming flag value.
  CCReg = CurDAG->getCopyToReg(CurDAG->getEntryNode(), CR7Reg, CCReg, 
                               InFlag).getValue(1);
  
  if (TLI.getTargetMachine().getSubtarget<PPCSubtarget>().isGigaProcessor())
    IntCR = SDOperand(CurDAG->getTargetNode(PPC::MFOCRF, MVT::i32, CR7Reg,
                                            CCReg), 0);
  else
    IntCR = SDOperand(CurDAG->getTargetNode(PPC::MFCR, MVT::i32, CCReg), 0);
  
  if (!Inv) {
    return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, IntCR,
                                getI32Imm((32-(3-Idx)) & 31),
                                getI32Imm(31), getI32Imm(31));
  } else {
    SDOperand Tmp =
      SDOperand(CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, IntCR,
                                      getI32Imm((32-(3-Idx)) & 31),
                                      getI32Imm(31),getI32Imm(31)), 0);
    return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Tmp, getI32Imm(1));
  }
}


// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
void PPCDAGToDAGISel::Select(SDOperand &Result, SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&
      N->getOpcode() < PPCISD::FIRST_NUMBER) {
    Result = Op;
    return;   // Already selected.
  }

  // If this has already been converted, use it.
  std::map<SDOperand, SDOperand>::iterator CGMI = CodeGenMap.find(Op);
  if (CGMI != CodeGenMap.end()) {
    Result = CGMI->second;
    return;
  }
  
  switch (N->getOpcode()) {
  default: break;
  case ISD::SETCC:
    Result = SelectSETCC(Op);
    return;
  case PPCISD::GlobalBaseReg:
    Result = getGlobalBaseReg();
    return;
    
  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDOperand TFI = CurDAG->getTargetFrameIndex(FI, Op.getValueType());
    unsigned Opc = Op.getValueType() == MVT::i32 ? PPC::ADDI : PPC::ADDI8;
    if (N->hasOneUse()) {
      Result = CurDAG->SelectNodeTo(N, Opc, Op.getValueType(), TFI,
                                    getSmallIPtrImm(0));
      return;
    }
    Result = CodeGenMap[Op] = 
      SDOperand(CurDAG->getTargetNode(Opc, Op.getValueType(), TFI,
                                      getSmallIPtrImm(0)), 0);
    return;
  }

  case PPCISD::MFCR: {
    SDOperand InFlag;
    Select(InFlag, N->getOperand(1));
    // Use MFOCRF if supported.
    if (TLI.getTargetMachine().getSubtarget<PPCSubtarget>().isGigaProcessor())
      Result = SDOperand(CurDAG->getTargetNode(PPC::MFOCRF, MVT::i32,
                                               N->getOperand(0), InFlag), 0);
    else
      Result = SDOperand(CurDAG->getTargetNode(PPC::MFCR, MVT::i32, InFlag), 0);
    CodeGenMap[Op] = Result;
    return;
  }
    
  case ISD::SDIV: {
    // FIXME: since this depends on the setting of the carry flag from the srawi
    //        we should really be making notes about that for the scheduler.
    // FIXME: It sure would be nice if we could cheaply recognize the 
    //        srl/add/sra pattern the dag combiner will generate for this as
    //        sra/addze rather than having to handle sdiv ourselves.  oh well.
    unsigned Imm;
    if (isInt32Immediate(N->getOperand(1), Imm)) {
      SDOperand N0;
      Select(N0, N->getOperand(0));
      if ((signed)Imm > 0 && isPowerOf2_32(Imm)) {
        SDNode *Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                N0, getI32Imm(Log2_32(Imm)));
        Result = CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32, 
                                      SDOperand(Op, 0), SDOperand(Op, 1));
      } else if ((signed)Imm < 0 && isPowerOf2_32(-Imm)) {
        SDNode *Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                N0, getI32Imm(Log2_32(-Imm)));
        SDOperand PT =
          SDOperand(CurDAG->getTargetNode(PPC::ADDZE, MVT::i32,
                                          SDOperand(Op, 0), SDOperand(Op, 1)),
                    0);
        Result = CurDAG->SelectNodeTo(N, PPC::NEG, MVT::i32, PT);
      }
      return;
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::AND: {
    unsigned Imm, Imm2;
    // If this is an and of a value rotated between 0 and 31 bits and then and'd
    // with a mask, emit rlwinm
    if (isInt32Immediate(N->getOperand(1), Imm) &&
        (isShiftedMask_32(Imm) || isShiftedMask_32(~Imm))) {
      SDOperand Val;
      unsigned SH, MB, ME;
      if (isRotateAndMask(N->getOperand(0).Val, Imm, false, SH, MB, ME)) {
        Select(Val, N->getOperand(0).getOperand(0));
      } else if (Imm == 0) {
        // AND X, 0 -> 0, not "rlwinm 32".
        Select(Result, N->getOperand(1));
        return ;
      } else {        
        Select(Val, N->getOperand(0));
        isRunOfOnes(Imm, MB, ME);
        SH = 0;
      }
      Result = CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Val,
                                    getI32Imm(SH), getI32Imm(MB),
                                    getI32Imm(ME));
      return;
    }
    // ISD::OR doesn't get all the bitfield insertion fun.
    // (and (or x, c1), c2) where isRunOfOnes(~(c1^c2)) is a bitfield insert
    if (isInt32Immediate(N->getOperand(1), Imm) && 
        N->getOperand(0).getOpcode() == ISD::OR &&
        isInt32Immediate(N->getOperand(0).getOperand(1), Imm2)) {
      unsigned MB, ME;
      Imm = ~(Imm^Imm2);
      if (isRunOfOnes(Imm, MB, ME)) {
        SDOperand Tmp1, Tmp2;
        Select(Tmp1, N->getOperand(0).getOperand(0));
        Select(Tmp2, N->getOperand(0).getOperand(1));
        Result = SDOperand(CurDAG->getTargetNode(PPC::RLWIMI, MVT::i32,
                                                 Tmp1, Tmp2,
                                                 getI32Imm(0), getI32Imm(MB),
                                                 getI32Imm(ME)), 0);
        return;
      }
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::OR:
    if (SDNode *I = SelectBitfieldInsert(N)) {
      Result = CodeGenMap[Op] = SDOperand(I, 0);
      return;
    }
      
    // Other cases are autogenerated.
    break;
  case ISD::SHL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) {
      SDOperand Val;
      Select(Val, N->getOperand(0).getOperand(0));
      Result = CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, 
                                    Val, getI32Imm(SH), getI32Imm(MB),
                                    getI32Imm(ME));
      return;
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::SRL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) { 
      SDOperand Val;
      Select(Val, N->getOperand(0).getOperand(0));
      Result = CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, 
                                    Val, getI32Imm(SH), getI32Imm(MB),
                                    getI32Imm(ME));
      return;
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
            SDOperand LHS;
            Select(LHS, N->getOperand(0));
            SDNode *Tmp =
              CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                    LHS, getI32Imm(~0U));
            Result = CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32,
                                          SDOperand(Tmp, 0), LHS,
                                          SDOperand(Tmp, 1));
            return;
          }

    SDOperand CCReg = SelectCC(N->getOperand(0), N->getOperand(1), CC);
    unsigned BROpc = getBCCForSetCC(CC);

    bool isFP = MVT::isFloatingPoint(N->getValueType(0));
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

    SDOperand N2, N3;
    Select(N2, N->getOperand(2));
    Select(N3, N->getOperand(3));
    Result = CurDAG->SelectNodeTo(N, SelectCCOp, N->getValueType(0), CCReg,
                                  N2, N3, getI32Imm(BROpc));
    return;
  }
  case ISD::BR_CC: {
    SDOperand Chain;
    Select(Chain, N->getOperand(0));
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
    SDOperand CondCode = SelectCC(N->getOperand(2), N->getOperand(3), CC);
    Result = CurDAG->SelectNodeTo(N, PPC::COND_BRANCH, MVT::Other, 
                                  CondCode, getI32Imm(getBCCForSetCC(CC)), 
                                  N->getOperand(4), Chain);
    return;
  }
  case ISD::BRIND: {
    // FIXME: Should custom lower this.
    SDOperand Chain, Target;
    Select(Chain, N->getOperand(0));
    Select(Target,N->getOperand(1));
    unsigned Opc = Target.getValueType() == MVT::i32 ? PPC::MTCTR : PPC::MTCTR8;
    Chain = SDOperand(CurDAG->getTargetNode(Opc, MVT::Other, Target,
                                            Chain), 0);
    Result = CurDAG->SelectNodeTo(N, PPC::BCTR, MVT::Other, Chain);
    return;
  }
  // FIXME: These are manually selected because tblgen isn't handling varargs
  // nodes correctly.
  case PPCISD::BCTRL:            MySelect_PPCbctrl(Result, Op); return;
  case PPCISD::CALL:             MySelect_PPCcall(Result, Op); return;
  }
  
  SelectCode(Result, Op);
}


// FIXME: This is manually selected because tblgen isn't handling varargs nodes
// correctly.
void PPCDAGToDAGISel::MySelect_PPCbctrl(SDOperand &Result, SDOperand N) {
  SDOperand Chain(0, 0);
  SDOperand InFlag(0, 0);
  SDNode *ResNode;
  
  bool hasFlag =
    N.getOperand(N.getNumOperands()-1).getValueType() == MVT::Flag;

  std::vector<SDOperand> Ops;
  // Push varargs arguments, including optional flag.
  for (unsigned i = 1, e = N.getNumOperands()-hasFlag; i != e; ++i) {
    Select(Chain, N.getOperand(i));
    Ops.push_back(Chain);
  }

  Select(Chain, N.getOperand(0));
  Ops.push_back(Chain);

  if (hasFlag) {
    Select(Chain, N.getOperand(N.getNumOperands()-1));
    Ops.push_back(Chain);
  }
  
  ResNode = CurDAG->getTargetNode(PPC::BCTRL, MVT::Other, MVT::Flag, Ops);
  Chain = SDOperand(ResNode, 0);
  InFlag = SDOperand(ResNode, 1);
  SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 0, Chain.Val,
                                   Chain.ResNo);
  SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 1, InFlag.Val,
                                   InFlag.ResNo);
  Result = SDOperand(ResNode, N.ResNo);
  return;
}

// FIXME: This is manually selected because tblgen isn't handling varargs nodes
// correctly.
void PPCDAGToDAGISel::MySelect_PPCcall(SDOperand &Result, SDOperand N) {
  SDOperand Chain(0, 0);
  SDOperand InFlag(0, 0);
  SDOperand N1(0, 0);
  SDOperand Tmp0(0, 0);
  SDNode *ResNode;
  Chain = N.getOperand(0);
  N1 = N.getOperand(1);
  
  // Pattern: (PPCcall:void (imm:i32):$func)
  // Emits: (BLA:void (imm:i32):$func)
  // Pattern complexity = 4  cost = 1
  if (N1.getOpcode() == ISD::Constant) {
    unsigned Tmp0C = (unsigned)cast<ConstantSDNode>(N1)->getValue();
    
    std::vector<SDOperand> Ops;
    Ops.push_back(CurDAG->getTargetConstant(Tmp0C, MVT::i32));

    bool hasFlag =
      N.getOperand(N.getNumOperands()-1).getValueType() == MVT::Flag;
    
    // Push varargs arguments, not including optional flag.
    for (unsigned i = 2, e = N.getNumOperands()-hasFlag; i != e; ++i) {
      Select(Chain, N.getOperand(i));
      Ops.push_back(Chain);
    }
    Select(Chain, N.getOperand(0));
    Ops.push_back(Chain);
    if (hasFlag) {
      Select(Chain, N.getOperand(N.getNumOperands()-1));
      Ops.push_back(Chain);
    }
    ResNode = CurDAG->getTargetNode(PPC::BLA, MVT::Other, MVT::Flag, Ops);
    
    Chain = SDOperand(ResNode, 0);
    InFlag = SDOperand(ResNode, 1);
    SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 0, Chain.Val, 
                                     Chain.ResNo);
    SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 1, InFlag.Val, 
                                     InFlag.ResNo);
    Result = SDOperand(ResNode, N.ResNo);
    return;
  }
  
  // Pattern: (PPCcall:void (tglobaladdr:i32):$dst)
  // Emits: (BL:void (tglobaladdr:i32):$dst)
  // Pattern complexity = 4  cost = 1
  if (N1.getOpcode() == ISD::TargetGlobalAddress) {
    std::vector<SDOperand> Ops;
    Ops.push_back(N1);
    
    bool hasFlag =
      N.getOperand(N.getNumOperands()-1).getValueType() == MVT::Flag;

    // Push varargs arguments, not including optional flag.
    for (unsigned i = 2, e = N.getNumOperands()-hasFlag; i != e; ++i) {
      Select(Chain, N.getOperand(i));
      Ops.push_back(Chain);
    }
    Select(Chain, N.getOperand(0));
    Ops.push_back(Chain);
    if (hasFlag) {
      Select(Chain, N.getOperand(N.getNumOperands()-1));
      Ops.push_back(Chain);
    }
    
    ResNode = CurDAG->getTargetNode(PPC::BL, MVT::Other, MVT::Flag, Ops);
    
    Chain = SDOperand(ResNode, 0);
    InFlag = SDOperand(ResNode, 1);
    SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 0, Chain.Val,
                                     Chain.ResNo);
    SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 1, InFlag.Val, 
                                     InFlag.ResNo);
    Result = SDOperand(ResNode, N.ResNo);
    return;
  }
  
  // Pattern: (PPCcall:void (texternalsym:i32):$dst)
  // Emits: (BL:void (texternalsym:i32):$dst)
  // Pattern complexity = 4  cost = 1
  if (N1.getOpcode() == ISD::TargetExternalSymbol) {
    std::vector<SDOperand> Ops;
    Ops.push_back(N1);
    
    bool hasFlag =
      N.getOperand(N.getNumOperands()-1).getValueType() == MVT::Flag;

    // Push varargs arguments, not including optional flag.
    for (unsigned i = 2, e = N.getNumOperands()-hasFlag; i != e; ++i) {
      Select(Chain, N.getOperand(i));
      Ops.push_back(Chain);
    }
    Select(Chain, N.getOperand(0));
    Ops.push_back(Chain);
    if (hasFlag) {
      Select(Chain, N.getOperand(N.getNumOperands()-1));
      Ops.push_back(Chain);
    }
    
    ResNode = CurDAG->getTargetNode(PPC::BL, MVT::Other, MVT::Flag, Ops);

    Chain = SDOperand(ResNode, 0);
    InFlag = SDOperand(ResNode, 1);
    SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 0, Chain.Val,
                                     Chain.ResNo);
    SelectionDAG::InsertISelMapEntry(CodeGenMap, N.Val, 1, InFlag.Val,
                                     InFlag.ResNo);
    Result = SDOperand(ResNode, N.ResNo);
    return;
  }
  std::cerr << "Cannot yet select: ";
  N.Val->dump(CurDAG);
  std::cerr << '\n';
  abort();
}


/// createPPCISelDag - This pass converts a legalized DAG into a 
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createPPCISelDag(PPCTargetMachine &TM) {
  return new PPCDAGToDAGISel(TM);
}

