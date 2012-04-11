//===-- MipsISelLowering.h - Mips DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Mips uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef MipsISELLOWERING_H
#define MipsISELLOWERING_H

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace MipsISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      // Jump and link (call)
      JmpLink,

      // Get the Higher 16 bits from a 32-bit immediate
      // No relation with Mips Hi register
      Hi,

      // Get the Lower 16 bits from a 32-bit immediate
      // No relation with Mips Lo register
      Lo,

      // Handle gp_rel (small data/bss sections) relocation.
      GPRel,

      // Thread Pointer
      ThreadPointer,

      // Floating Point Branch Conditional
      FPBrcond,

      // Floating Point Compare
      FPCmp,

      // Floating Point Conditional Moves
      CMovFP_T,
      CMovFP_F,

      // Floating Point Rounding
      FPRound,

      // Return
      Ret,

      // MAdd/Sub nodes
      MAdd,
      MAddu,
      MSub,
      MSubu,

      // DivRem(u)
      DivRem,
      DivRemU,

      BuildPairF64,
      ExtractElementF64,

      Wrapper,

      DynAlloc,

      Sync,

      Ext,
      Ins
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//

  class MipsTargetLowering : public TargetLowering  {
  public:
    explicit MipsTargetLowering(MipsTargetMachine &TM);

    virtual MVT getShiftAmountTy(EVT LHSTy) const { return MVT::i32; }

    virtual bool allowsUnalignedMemoryAccesses (EVT VT) const;

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    /// getTargetNodeName - This method returns the name of a target specific
    //  DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - get the ISD::SETCC result ValueType
    EVT getSetCCResultType(EVT VT) const;

    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  private:
    // Subtarget Info
    const MipsSubtarget *Subtarget;

    bool HasMips64, IsN64, IsO32;

    // Lower Operand helpers
    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;

    // Lower Operand specifics
    SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFABS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerMEMBARRIER(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerCall(SDValue Chain, SDValue Callee,
                CallingConv::ID CallConv, bool isVarArg,
                bool doesNotRet, bool &isTailCall,
                const SmallVectorImpl<ISD::OutputArg> &Outs,
                const SmallVectorImpl<SDValue> &OutVals,
                const SmallVectorImpl<ISD::InputArg> &Ins,
                DebugLoc dl, SelectionDAG &DAG,
                SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl, SelectionDAG &DAG) const;

    virtual MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const;

    // Inline asm support
    ConstraintType getConstraintType(const std::string &Constraint) const;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

    std::pair<unsigned, const TargetRegisterClass*>
              getRegForInlineAsmConstraint(const std::string &Constraint,
              EVT VT) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

    virtual unsigned getJumpTableEncoding() const;

    MachineBasicBlock *EmitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                    unsigned Size, unsigned BinOpcode, bool Nand = false) const;
    MachineBasicBlock *EmitAtomicBinaryPartword(MachineInstr *MI,
                    MachineBasicBlock *BB, unsigned Size, unsigned BinOpcode,
                    bool Nand = false) const;
    MachineBasicBlock *EmitAtomicCmpSwap(MachineInstr *MI,
                                  MachineBasicBlock *BB, unsigned Size) const;
    MachineBasicBlock *EmitAtomicCmpSwapPartword(MachineInstr *MI,
                                  MachineBasicBlock *BB, unsigned Size) const;
  };
}

#endif // MipsISELLOWERING_H
