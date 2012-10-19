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

      // Tail call
      TailCall,

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
      Ins,

      // EXTR.W instrinsic nodes.
      EXTP,
      EXTPDP,
      EXTR_S_H,
      EXTR_W,
      EXTR_R_W,
      EXTR_RS_W,
      SHILO,
      MTHLIP,

      // DPA.W intrinsic nodes.
      MULSAQ_S_W_PH,
      MAQ_S_W_PHL,
      MAQ_S_W_PHR,
      MAQ_SA_W_PHL,
      MAQ_SA_W_PHR,
      DPAU_H_QBL,
      DPAU_H_QBR,
      DPSU_H_QBL,
      DPSU_H_QBR,
      DPAQ_S_W_PH,
      DPSQ_S_W_PH,
      DPAQ_SA_L_W,
      DPSQ_SA_L_W,
      DPA_W_PH,
      DPS_W_PH,
      DPAQX_S_W_PH,
      DPAQX_SA_W_PH,
      DPAX_W_PH,
      DPSX_W_PH,
      DPSQX_S_W_PH,
      DPSQX_SA_W_PH,
      MULSA_W_PH,

      MULT,
      MULTU,
      MADD_DSP,
      MADDU_DSP,
      MSUB_DSP,
      MSUBU_DSP,

      // Load/Store Left/Right nodes.
      LWL = ISD::FIRST_TARGET_MEMORY_OPCODE,
      LWR,
      SWL,
      SWR,
      LDL,
      LDR,
      SDL,
      SDR
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

    virtual void LowerOperationWrapper(SDNode *N,
                                       SmallVectorImpl<SDValue> &Results,
                                       SelectionDAG &DAG) const;

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    /// ReplaceNodeResults - Replace the results of node with an illegal result
    /// type with new values built out of custom code.
    ///
    virtual void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                                    SelectionDAG &DAG) const;

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
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFABS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerMEMBARRIER(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerShiftLeftParts(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerShiftRightParts(SDValue Op, SelectionDAG& DAG,
                                 bool IsSRA) const;
    SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const;

    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization.
    bool IsEligibleForTailCallOptimization(CallingConv::ID CalleeCC,
                                           unsigned NextStackOffset) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const;

    virtual bool
      CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                     bool isVarArg,
                     const SmallVectorImpl<ISD::OutputArg> &Outs,
                     LLVMContext &Context) const;

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

    /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
    /// vector.  If it is invalid, don't add anything to Ops. If hasMemory is
    /// true it means one of the asm constraint of the inline asm instruction
    /// being processed is 'm'.
    virtual void LowerAsmOperandForConstraint(SDValue Op,
                                              std::string &Constraint,
                                              std::vector<SDValue> &Ops,
                                              SelectionDAG &DAG) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    virtual EVT getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                                    unsigned SrcAlign, bool IsZeroVal,
                                    bool MemcpyStrSrc,
                                    MachineFunction &MF) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

    virtual unsigned getJumpTableEncoding() const;

    MachineBasicBlock *EmitBPOSGE32(MachineInstr *MI,
                                    MachineBasicBlock *BB) const;
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
