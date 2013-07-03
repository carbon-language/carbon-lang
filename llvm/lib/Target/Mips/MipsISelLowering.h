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
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetLowering.h"
#include <deque>
#include <string>

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

      // FP-to-int truncation node.
      TruncIntFP,

      // Return
      Ret,

      EH_RETURN,

      // Node used to extract integer from accumulator.
      ExtractLOHI,

      // Node used to insert integers to accumulator.
      InsertLOHI,

      // Mult nodes.
      Mult,
      Multu,

      // MAdd/Sub nodes
      MAdd,
      MAddu,
      MSub,
      MSubu,

      // DivRem(u)
      DivRem,
      DivRemU,
      DivRem16,
      DivRemU16,

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

      // DSP shift nodes.
      SHLL_DSP,
      SHRA_DSP,
      SHRL_DSP,

      // DSP setcc and select_cc nodes.
      SETCC_DSP,
      SELECT_CC_DSP,

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
  class MipsFunctionInfo;

  class MipsTargetLowering : public TargetLowering  {
  public:
    explicit MipsTargetLowering(MipsTargetMachine &TM);

    static const MipsTargetLowering *create(MipsTargetMachine &TM);

    virtual MVT getScalarShiftAmountTy(EVT LHSTy) const { return MVT::i32; }

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
    EVT getSetCCResultType(LLVMContext &Context, EVT VT) const;

    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr *MI, MachineBasicBlock *MBB) const;

    struct LTStr {
      bool operator()(const char *S1, const char *S2) const {
        return strcmp(S1, S2) < 0;
      }
    };

  protected:
    SDValue getGlobalReg(SelectionDAG &DAG, EVT Ty) const;

    SDValue getAddrLocal(SDValue Op, SelectionDAG &DAG, bool HasMips64) const;

    SDValue getAddrGlobal(SDValue Op, SelectionDAG &DAG, unsigned Flag) const;

    SDValue getAddrGlobalLargeGOT(SDValue Op, SelectionDAG &DAG,
                                  unsigned HiFlag, unsigned LoFlag) const;

    /// This function fills Ops, which is the list of operands that will later
    /// be used when a function call node is created. It also generates
    /// copyToReg nodes to set up argument registers.
    virtual void
    getOpndList(SmallVectorImpl<SDValue> &Ops,
                std::deque< std::pair<unsigned, SDValue> > &RegsToPass,
                bool IsPICCall, bool GlobalOrExternal, bool InternalLinkage,
                CallLoweringInfo &CLI, SDValue Callee, SDValue Chain) const;

    /// ByValArgInfo - Byval argument information.
    struct ByValArgInfo {
      unsigned FirstIdx; // Index of the first register used.
      unsigned NumRegs;  // Number of registers used for this argument.
      unsigned Address;  // Offset of the stack area used to pass this argument.

      ByValArgInfo() : FirstIdx(0), NumRegs(0), Address(0) {}
    };

    /// MipsCC - This class provides methods used to analyze formal and call
    /// arguments and inquire about calling convention information.
    class MipsCC {
    public:
      enum SpecialCallingConvType {
        Mips16RetHelperConv, NoSpecialCallingConv
      };

      MipsCC(
        CallingConv::ID CallConv, bool IsO32, CCState &Info,
        SpecialCallingConvType SpecialCallingConv = NoSpecialCallingConv);


      void analyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                               bool IsVarArg, bool IsSoftFloat,
                               const SDNode *CallNode,
                               std::vector<ArgListEntry> &FuncArgs);
      void analyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                                  bool IsSoftFloat,
                                  Function::const_arg_iterator FuncArg);

      void analyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                             bool IsSoftFloat, const SDNode *CallNode,
                             const Type *RetTy) const;

      void analyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                         bool IsSoftFloat, const Type *RetTy) const;

      const CCState &getCCInfo() const { return CCInfo; }

      /// hasByValArg - Returns true if function has byval arguments.
      bool hasByValArg() const { return !ByValArgs.empty(); }

      /// regSize - Size (in number of bits) of integer registers.
      unsigned regSize() const { return IsO32 ? 4 : 8; }

      /// numIntArgRegs - Number of integer registers available for calls.
      unsigned numIntArgRegs() const;

      /// reservedArgArea - The size of the area the caller reserves for
      /// register arguments. This is 16-byte if ABI is O32.
      unsigned reservedArgArea() const;

      /// Return pointer to array of integer argument registers.
      const uint16_t *intArgRegs() const;

      typedef SmallVectorImpl<ByValArgInfo>::const_iterator byval_iterator;
      byval_iterator byval_begin() const { return ByValArgs.begin(); }
      byval_iterator byval_end() const { return ByValArgs.end(); }

    private:
      void handleByValArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                          CCValAssign::LocInfo LocInfo,
                          ISD::ArgFlagsTy ArgFlags);

      /// useRegsForByval - Returns true if the calling convention allows the
      /// use of registers to pass byval arguments.
      bool useRegsForByval() const { return CallConv != CallingConv::Fast; }

      /// Return the function that analyzes fixed argument list functions.
      llvm::CCAssignFn *fixedArgFn() const;

      /// Return the function that analyzes variable argument list functions.
      llvm::CCAssignFn *varArgFn() const;

      const uint16_t *shadowRegs() const;

      void allocateRegs(ByValArgInfo &ByVal, unsigned ByValSize,
                        unsigned Align);

      /// Return the type of the register which is used to pass an argument or
      /// return a value. This function returns f64 if the argument is an i64
      /// value which has been generated as a result of softening an f128 value.
      /// Otherwise, it just returns VT.
      MVT getRegVT(MVT VT, const Type *OrigTy, const SDNode *CallNode,
                   bool IsSoftFloat) const;

      template<typename Ty>
      void analyzeReturn(const SmallVectorImpl<Ty> &RetVals, bool IsSoftFloat,
                         const SDNode *CallNode, const Type *RetTy) const;

      CCState &CCInfo;
      CallingConv::ID CallConv;
      bool IsO32;
      SpecialCallingConvType SpecialCallingConv;
      SmallVector<ByValArgInfo, 2> ByValArgs;
    };
  protected:
    // Subtarget Info
    const MipsSubtarget *Subtarget;

    bool HasMips64, IsN64, IsO32;

  private:

    MipsCC::SpecialCallingConvType getSpecialCallingConv(SDValue Callee) const;
    // Lower Operand helpers
    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            SDLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals,
                            const SDNode *CallNode, const Type *RetTy) const;

    // Lower Operand specifics
    SDValue lowerBR_JT(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerFABS(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerEH_RETURN(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;
    SDValue lowerShiftLeftParts(SDValue Op, SelectionDAG& DAG) const;
    SDValue lowerShiftRightParts(SDValue Op, SelectionDAG& DAG,
                                 bool IsSRA) const;
    SDValue lowerLOAD(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerSTORE(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerADD(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG) const;

    /// isEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization.
    virtual bool
    isEligibleForTailCallOptimization(const MipsCC &MipsCCInfo,
                                      unsigned NextStackOffset,
                                      const MipsFunctionInfo& FI) const = 0;

    /// copyByValArg - Copy argument registers which were used to pass a byval
    /// argument to the stack. Create a stack frame object for the byval
    /// argument.
    void copyByValRegs(SDValue Chain, SDLoc DL,
                       std::vector<SDValue> &OutChains, SelectionDAG &DAG,
                       const ISD::ArgFlagsTy &Flags,
                       SmallVectorImpl<SDValue> &InVals,
                       const Argument *FuncArg,
                       const MipsCC &CC, const ByValArgInfo &ByVal) const;

    /// passByValArg - Pass a byval argument in registers or on stack.
    void passByValArg(SDValue Chain, SDLoc DL,
                      std::deque< std::pair<unsigned, SDValue> > &RegsToPass,
                      SmallVector<SDValue, 8> &MemOpChains, SDValue StackPtr,
                      MachineFrameInfo *MFI, SelectionDAG &DAG, SDValue Arg,
                      const MipsCC &CC, const ByValArgInfo &ByVal,
                      const ISD::ArgFlagsTy &Flags, bool isLittle) const;

    /// writeVarArgRegs - Write variable function arguments passed in registers
    /// to the stack. Also create a stack frame object for the first variable
    /// argument.
    void writeVarArgRegs(std::vector<SDValue> &OutChains, const MipsCC &CC,
                         SDValue Chain, SDLoc DL, SelectionDAG &DAG) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           SDLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    SDValue passArgOnStack(SDValue StackPtr, unsigned Offset, SDValue Chain,
                           SDValue Arg, SDLoc DL, bool IsTailCall,
                           SelectionDAG &DAG) const;

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
                  SDLoc dl, SelectionDAG &DAG) const;

    // Inline asm support
    ConstraintType getConstraintType(const std::string &Constraint) const;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

    std::pair<unsigned, const TargetRegisterClass*>
              getRegForInlineAsmConstraint(const std::string &Constraint,
                                           MVT VT) const;

    /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
    /// vector.  If it is invalid, don't add anything to Ops. If hasMemory is
    /// true it means one of the asm constraint of the inline asm instruction
    /// being processed is 'm'.
    virtual void LowerAsmOperandForConstraint(SDValue Op,
                                              std::string &Constraint,
                                              std::vector<SDValue> &Ops,
                                              SelectionDAG &DAG) const;

    virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    virtual EVT getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                                    unsigned SrcAlign,
                                    bool IsMemset, bool ZeroMemset,
                                    bool MemcpyStrSrc,
                                    MachineFunction &MF) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

    virtual unsigned getJumpTableEncoding() const;

    MachineBasicBlock *emitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                    unsigned Size, unsigned BinOpcode, bool Nand = false) const;
    MachineBasicBlock *emitAtomicBinaryPartword(MachineInstr *MI,
                    MachineBasicBlock *BB, unsigned Size, unsigned BinOpcode,
                    bool Nand = false) const;
    MachineBasicBlock *emitAtomicCmpSwap(MachineInstr *MI,
                                  MachineBasicBlock *BB, unsigned Size) const;
    MachineBasicBlock *emitAtomicCmpSwapPartword(MachineInstr *MI,
                                  MachineBasicBlock *BB, unsigned Size) const;
  };

  /// Create MipsTargetLowering objects.
  const MipsTargetLowering *createMips16TargetLowering(MipsTargetMachine &TM);
  const MipsTargetLowering *createMipsSETargetLowering(MipsTargetMachine &TM);
}

#endif // MipsISELLOWERING_H
