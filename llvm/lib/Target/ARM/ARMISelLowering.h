//===-- ARMISelLowering.h - ARM DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that ARM uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef ARMISELLOWERING_H
#define ARMISELLOWERING_H

#include "ARMSubtarget.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include <vector>

namespace llvm {
  class ARMConstantPoolValue;

  namespace ARMISD {
    // ARM Specific DAG Nodes
    enum NodeType {
      // Start the numbering where the builtin ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      Wrapper,      // Wrapper - A wrapper node for TargetConstantPool,
                    // TargetExternalSymbol, and TargetGlobalAddress.
      WrapperJT,    // WrapperJT - A wrapper node for TargetJumpTable

      CALL,         // Function call.
      CALL_PRED,    // Function call that's predicable.
      CALL_NOLINK,  // Function call with branch not branch-and-link.
      tCALL,        // Thumb function call.
      BRCOND,       // Conditional branch.
      BR_JT,        // Jumptable branch.
      BR2_JT,       // Jumptable branch (2 level - jumptable entry is a jump).
      RET_FLAG,     // Return with a flag operand.

      PIC_ADD,      // Add with a PC operand and a PIC label.

      CMP,          // ARM compare instructions.
      CMPZ,         // ARM compare that sets only Z flag.
      CMPFP,        // ARM VFP compare instruction, sets FPSCR.
      CMPFPw0,      // ARM VFP compare against zero instruction, sets FPSCR.
      FMSTAT,       // ARM fmstat instruction.
      CMOV,         // ARM conditional move instructions.
      CNEG,         // ARM conditional negate instructions.

      RBIT,         // ARM bitreverse instruction

      FTOSI,        // FP to sint within a FP register.
      FTOUI,        // FP to uint within a FP register.
      SITOF,        // sint to FP within a FP register.
      UITOF,        // uint to FP within a FP register.

      SRL_FLAG,     // V,Flag = srl_flag X -> srl X, 1 + save carry out.
      SRA_FLAG,     // V,Flag = sra_flag X -> sra X, 1 + save carry out.
      RRX,          // V = RRX X, Flag     -> srl X, 1 + shift in carry flag.

      VMOVRRD,      // double to two gprs.
      VMOVDRR,      // Two gprs to double.

      EH_SJLJ_SETJMP,    // SjLj exception handling setjmp.
      EH_SJLJ_LONGJMP,   // SjLj exception handling longjmp.

      THREAD_POINTER,

      DYN_ALLOC,    // Dynamic allocation on the stack.

      MEMBARRIER,   // Memory barrier
      SYNCBARRIER,  // Memory sync barrier

      VCEQ,         // Vector compare equal.
      VCGE,         // Vector compare greater than or equal.
      VCGEU,        // Vector compare unsigned greater than or equal.
      VCGT,         // Vector compare greater than.
      VCGTU,        // Vector compare unsigned greater than.
      VTST,         // Vector test bits.

      // Vector shift by immediate:
      VSHL,         // ...left
      VSHRs,        // ...right (signed)
      VSHRu,        // ...right (unsigned)
      VSHLLs,       // ...left long (signed)
      VSHLLu,       // ...left long (unsigned)
      VSHLLi,       // ...left long (with maximum shift count)
      VSHRN,        // ...right narrow

      // Vector rounding shift by immediate:
      VRSHRs,       // ...right (signed)
      VRSHRu,       // ...right (unsigned)
      VRSHRN,       // ...right narrow

      // Vector saturating shift by immediate:
      VQSHLs,       // ...left (signed)
      VQSHLu,       // ...left (unsigned)
      VQSHLsu,      // ...left (signed to unsigned)
      VQSHRNs,      // ...right narrow (signed)
      VQSHRNu,      // ...right narrow (unsigned)
      VQSHRNsu,     // ...right narrow (signed to unsigned)

      // Vector saturating rounding shift by immediate:
      VQRSHRNs,     // ...right narrow (signed)
      VQRSHRNu,     // ...right narrow (unsigned)
      VQRSHRNsu,    // ...right narrow (signed to unsigned)

      // Vector shift and insert:
      VSLI,         // ...left
      VSRI,         // ...right

      // Vector get lane (VMOV scalar to ARM core register)
      // (These are used for 8- and 16-bit element types only.)
      VGETLANEu,    // zero-extend vector extract element
      VGETLANEs,    // sign-extend vector extract element

      // Vector duplicate:
      VDUP,
      VDUPLANE,

      // Vector shuffles:
      VEXT,         // extract
      VREV64,       // reverse elements within 64-bit doublewords
      VREV32,       // reverse elements within 32-bit words
      VREV16,       // reverse elements within 16-bit halfwords
      VZIP,         // zip (interleave)
      VUZP,         // unzip (deinterleave)
      VTRN,         // transpose

      // Floating-point max and min:
      FMAX,
      FMIN
    };
  }

  /// Define some predicates that are used for node matching.
  namespace ARM {
    /// getVMOVImm - If this is a build_vector of constants which can be
    /// formed by using a VMOV instruction of the specified element size,
    /// return the constant being splatted.  The ByteSize field indicates the
    /// number of bytes of each element [1248].
    SDValue getVMOVImm(SDNode *N, unsigned ByteSize, SelectionDAG &DAG);

    /// getVFPf32Imm / getVFPf64Imm - If the given fp immediate can be
    /// materialized with a VMOV.f32 / VMOV.f64 (i.e. fconsts / fconstd)
    /// instruction, returns its 8-bit integer representation. Otherwise,
    /// returns -1.
    int getVFPf32Imm(const APFloat &FPImm);
    int getVFPf64Imm(const APFloat &FPImm);
  }

  //===--------------------------------------------------------------------===//
  //  ARMTargetLowering - ARM Implementation of the TargetLowering interface

  class ARMTargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
  public:
    explicit ARMTargetLowering(TargetMachine &TM);

    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);

    /// ReplaceNodeResults - Replace the results of node with an illegal result
    /// type with new values built out of custom code.
    ///
    virtual void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                                    SelectionDAG &DAG);

    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual const char *getTargetNodeName(unsigned Opcode) const;

    virtual MachineBasicBlock *EmitInstrWithCustomInserter(MachineInstr *MI,
                                                         MachineBasicBlock *MBB,
                       DenseMap<MachineBasicBlock*, MachineBasicBlock*>*) const;

    /// allowsUnalignedMemoryAccesses - Returns true if the target allows
    /// unaligned memory accesses. of the specified type.
    /// FIXME: Add getOptimalMemOpType to implement memcpy with NEON?
    virtual bool allowsUnalignedMemoryAccesses(EVT VT) const;

    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    virtual bool isLegalAddressingMode(const AddrMode &AM, const Type *Ty)const;
    bool isLegalT2ScaledAddressingMode(const AddrMode &AM, EVT VT) const;

    /// isLegalICmpImmediate - Return true if the specified immediate is legal
    /// icmp immediate, that is the target has icmp instructions which can compare
    /// a register against the immediate without having to materialize the
    /// immediate into a register.
    virtual bool isLegalICmpImmediate(int64_t Imm) const;

    /// getPreIndexedAddressParts - returns true by value, base pointer and
    /// offset pointer and addressing mode by reference if the node's address
    /// can be legally represented as pre-indexed load / store address.
    virtual bool getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                           SDValue &Offset,
                                           ISD::MemIndexedMode &AM,
                                           SelectionDAG &DAG) const;

    /// getPostIndexedAddressParts - returns true by value, base pointer and
    /// offset pointer and addressing mode by reference if this node can be
    /// combined with a load / store to form a post-indexed load / store.
    virtual bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                            SDValue &Base, SDValue &Offset,
                                            ISD::MemIndexedMode &AM,
                                            SelectionDAG &DAG) const;

    virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                                const APInt &Mask,
                                                APInt &KnownZero,
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth) const;


    ConstraintType getConstraintType(const std::string &Constraint) const;
    std::pair<unsigned, const TargetRegisterClass*>
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   EVT VT) const;
    std::vector<unsigned>
    getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                      EVT VT) const;

    /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
    /// vector.  If it is invalid, don't add anything to Ops. If hasMemory is
    /// true it means one of the asm constraint of the inline asm instruction
    /// being processed is 'm'.
    virtual void LowerAsmOperandForConstraint(SDValue Op,
                                              char ConstraintLetter,
                                              bool hasMemory,
                                              std::vector<SDValue> &Ops,
                                              SelectionDAG &DAG) const;

    virtual const ARMSubtarget* getSubtarget() {
      return Subtarget;
    }

    /// getFunctionAlignment - Return the Log2 alignment of this function.
    virtual unsigned getFunctionAlignment(const Function *F) const;

    bool isShuffleMaskLegal(const SmallVectorImpl<int> &M, EVT VT) const;
    bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

  private:
    /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
    /// make the right decision when generating code for different targets.
    const ARMSubtarget *Subtarget;

    /// ARMPCLabelIndex - Keep track of the number of ARM PC labels created.
    ///
    unsigned ARMPCLabelIndex;

    void addTypeForNEON(EVT VT, EVT PromotedLdStVT, EVT PromotedBitwiseVT);
    void addDRTypeForNEON(EVT VT);
    void addQRTypeForNEON(EVT VT);

    typedef SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPassVector;
    void PassF64ArgInRegs(DebugLoc dl, SelectionDAG &DAG,
                          SDValue Chain, SDValue &Arg,
                          RegsToPassVector &RegsToPass,
                          CCValAssign &VA, CCValAssign &NextVA,
                          SDValue &StackPtr,
                          SmallVector<SDValue, 8> &MemOpChains,
                          ISD::ArgFlagsTy Flags);
    SDValue GetF64FormalArgument(CCValAssign &VA, CCValAssign &NextVA,
                                 SDValue &Root, SelectionDAG &DAG, DebugLoc dl);

    CCAssignFn *CCAssignFnForNode(CallingConv::ID CC, bool Return, bool isVarArg) const;
    SDValue LowerMemOpCallTo(SDValue Chain, SDValue StackPtr, SDValue Arg,
                             DebugLoc dl, SelectionDAG &DAG,
                             const CCValAssign &VA,
                             ISD::ArgFlagsTy Flags);
    SDValue LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG);
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG,
                                    const ARMSubtarget *Subtarget);
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalAddressDarwin(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalAddressELF(SDValue Op, SelectionDAG &DAG);
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
                                            SelectionDAG &DAG);
    SDValue LowerToTLSExecModels(GlobalAddressSDNode *GA,
                                   SelectionDAG &DAG);
    SDValue LowerGLOBAL_OFFSET_TABLE(SDValue Op, SelectionDAG &DAG);
    SDValue LowerBR_JT(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG);
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerShiftRightParts(SDValue Op, SelectionDAG &DAG);
    SDValue LowerShiftLeftParts(SDValue Op, SelectionDAG &DAG);

    SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                      SDValue Chain,
                                      SDValue Dst, SDValue Src,
                                      SDValue Size, unsigned Align,
                                      bool AlwaysInline,
                                      const Value *DstSV, uint64_t DstSVOff,
                                      const Value *SrcSV, uint64_t SrcSVOff);
    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals);

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals);

    virtual SDValue
      LowerCall(SDValue Chain, SDValue Callee,
                CallingConv::ID CallConv, bool isVarArg,
                bool &isTailCall,
                const SmallVectorImpl<ISD::OutputArg> &Outs,
                const SmallVectorImpl<ISD::InputArg> &Ins,
                DebugLoc dl, SelectionDAG &DAG,
                SmallVectorImpl<SDValue> &InVals);

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  DebugLoc dl, SelectionDAG &DAG);

    SDValue getARMCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                      SDValue &ARMCC, SelectionDAG &DAG, DebugLoc dl);

    MachineBasicBlock *EmitAtomicCmpSwap(MachineInstr *MI,
                                         MachineBasicBlock *BB,
                                         unsigned Size) const;
    MachineBasicBlock *EmitAtomicBinary(MachineInstr *MI,
                                        MachineBasicBlock *BB,
                                        unsigned Size,
                                        unsigned BinOpcode) const;

  };
}

#endif  // ARMISELLOWERING_H
