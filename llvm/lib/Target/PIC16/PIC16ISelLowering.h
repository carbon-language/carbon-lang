//===-- PIC16ISelLowering.h - PIC16 DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PIC16 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16ISELLOWERING_H
#define PIC16ISELLOWERING_H

#include "PIC16.h"
#include "PIC16Subtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
#include <map>

namespace llvm {
  namespace PIC16ISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      Lo,            // Low 8-bits of GlobalAddress.
      Hi,            // High 8-bits of GlobalAddress.
      PIC16Load,
      PIC16LdArg,   // This is replica of PIC16Load but used to load function 
                    // arguments and is being used for facilitating for some 
                    // store removal optimizations. 

      PIC16LdWF,
      PIC16Store,
      PIC16StWF,
      Banksel,
      MTLO,          // Move to low part of FSR
      MTHI,          // Move to high part of FSR
      MTPCLATH,      // Move to PCLATCH
      PIC16Connect,  // General connector for PIC16 nodes
      BCF,
      LSLF,          // PIC16 Logical shift left
      LRLF,          // PIC16 Logical shift right
      RLF,           // Rotate left through carry
      RRF,           // Rotate right through carry
      CALL,          // PIC16 Call instruction 
      CALLW,         // PIC16 CALLW instruction 
      SUBCC,         // Compare for equality or inequality.
      SELECT_ICC,    // Pseudo to be caught in scheduler and expanded to brcond.
      BRCOND,        // Conditional branch.
      RET,           // Return.
      Dummy
    };

    // Keep track of different address spaces. 
    enum AddressSpace {
      RAM_SPACE = 0,   // RAM address space
      ROM_SPACE = 1    // ROM address space number is 1
    };
    enum PIC16Libcall {
      MUL_I8 = RTLIB::UNKNOWN_LIBCALL + 1,
      SRA_I8,
      SLL_I8,
      SRL_I8,
      PIC16UnknownCall
    };
  }


  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class PIC16TargetLowering : public TargetLowering {
  public:
    explicit PIC16TargetLowering(PIC16TargetMachine &TM);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;
    /// getSetCCResultType - Return the ISD::SETCC ValueType
    virtual MVT::SimpleValueType getSetCCResultType(EVT ValType) const;
    virtual MVT::SimpleValueType getCmpLibcallReturnType() const;
    SDValue LowerShift(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerMUL(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADD(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSUB(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBinOp(SDValue Op, SelectionDAG &DAG) const;
    // Call returns
    SDValue 
    LowerDirectCallReturn(SDValue RetLabel, SDValue Chain, SDValue InFlag,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          DebugLoc dl, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals) const;
    SDValue 
    LowerIndirectCallReturn(SDValue Chain, SDValue InFlag,
                             SDValue DataAddr_Lo, SDValue DataAddr_Hi,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;

    // Call arguments
    SDValue 
    LowerDirectCallArguments(SDValue ArgLabel, SDValue Chain, SDValue InFlag,
                             const SmallVectorImpl<ISD::OutputArg> &Outs,
                             const SmallVectorImpl<SDValue> &OutVals,
                             DebugLoc dl, SelectionDAG &DAG) const;

    SDValue 
    LowerIndirectCallArguments(SDValue Chain, SDValue InFlag,
                               SDValue DataAddr_Lo, SDValue DataAddr_Hi, 
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               DebugLoc dl, SelectionDAG &DAG) const;

    SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue getPIC16Cmp(SDValue LHS, SDValue RHS, unsigned OrigCC, SDValue &CC,
                        SelectionDAG &DAG, DebugLoc dl) const;
    virtual MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const;

    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;
    virtual void ReplaceNodeResults(SDNode *N,
                                    SmallVectorImpl<SDValue> &Results,
                                    SelectionDAG &DAG) const;
    virtual void LowerOperationWrapper(SDNode *N,
                                       SmallVectorImpl<SDValue> &Results,
                                       SelectionDAG &DAG) const;

    virtual SDValue
    LowerFormalArguments(SDValue Chain,
                         CallingConv::ID CallConv,
                         bool isVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         DebugLoc dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerCall(SDValue Chain, SDValue Callee,
                CallingConv::ID CallConv, bool isVarArg, bool &isTailCall,
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

    SDValue ExpandStore(SDNode *N, SelectionDAG &DAG) const;
    SDValue ExpandLoad(SDNode *N, SelectionDAG &DAG) const;
    SDValue ExpandGlobalAddress(SDNode *N, SelectionDAG &DAG) const;
    SDValue ExpandExternalSymbol(SDNode *N, SelectionDAG &DAG) const;
    SDValue ExpandFrameIndex(SDNode *N, SelectionDAG &DAG) const;

    SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const; 
    SDValue PerformPIC16LoadCombine(SDNode *N, DAGCombinerInfo &DCI) const; 
    SDValue PerformStoreCombine(SDNode *N, DAGCombinerInfo &DCI) const; 

    // This function returns the Tmp Offset for FrameIndex. If any TmpOffset 
    // already exists for the FI then it returns the same else it creates the 
    // new offset and returns.
    unsigned GetTmpOffsetForFI(unsigned FI, unsigned slot_size,
                               MachineFunction &MF) const;
    void ResetTmpOffsetMap(SelectionDAG &DAG) const;
    void InitReservedFrameCount(const Function *F,
                                SelectionDAG &DAG) const;

    /// getFunctionAlignment - Return the Log2 alignment of this function.
    virtual unsigned getFunctionAlignment(const Function *) const {
      // FIXME: The function never seems to be aligned.
      return 1;
    }
  protected:
    std::pair<const TargetRegisterClass*, uint8_t>
    findRepresentativeClass(EVT VT) const;
  private:
    // If the Node is a BUILD_PAIR representing a direct Address,
    // then this function will return true.
    bool isDirectAddress(const SDValue &Op) const;

    // If the Node is a DirectAddress in ROM_SPACE then this 
    // function will return true
    bool isRomAddress(const SDValue &Op) const;

    // Extract the Lo and Hi component of Op. 
    void GetExpandedParts(SDValue Op, SelectionDAG &DAG, SDValue &Lo, 
                          SDValue &Hi) const;


    // Load pointer can be a direct or indirect address. In PIC16 direct
    // addresses need Banksel and Indirect addresses need to be loaded to
    // FSR first. Handle address specific cases here.
    void LegalizeAddress(SDValue Ptr, SelectionDAG &DAG, SDValue &Chain, 
                         SDValue &NewPtr, unsigned &Offset, DebugLoc dl) const;

    // FrameIndex should be broken down into ExternalSymbol and FrameOffset. 
    void LegalizeFrameIndex(SDValue Op, SelectionDAG &DAG, SDValue &ES, 
                            int &Offset) const;

    // For indirect calls data address of the callee frame need to be
    // extracted. This function fills the arguments DataAddr_Lo and 
    // DataAddr_Hi with the address of the callee frame.
    void GetDataAddress(DebugLoc dl, SDValue Callee, SDValue &Chain,
                        SDValue &DataAddr_Lo, SDValue &DataAddr_Hi,
                        SelectionDAG &DAG) const;

    // We can not have both operands of a binary operation in W.
    // This function is used to put one operand on stack and generate a load.
    SDValue ConvertToMemOperand(SDValue Op, SelectionDAG &DAG,
                                DebugLoc dl) const; 

    // This function checks if we need to put an operand of an operation on
    // stack and generate a load or not.
    // DAG parameter is required to access DAG information during
    // analysis.
    bool NeedToConvertToMemOp(SDValue Op, unsigned &MemOp,
                              SelectionDAG &DAG) const;

    /// Subtarget - Keep a pointer to the PIC16Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const PIC16Subtarget *Subtarget;


    // Extending the LIB Call framework of LLVM
    // to hold the names of PIC16Libcalls.
    const char *PIC16LibcallNames[PIC16ISD::PIC16UnknownCall]; 

    // To set and retrieve the lib call names.
    void setPIC16LibcallName(PIC16ISD::PIC16Libcall Call, const char *Name);
    const char *getPIC16LibcallName(PIC16ISD::PIC16Libcall Call) const;

    // Make PIC16 Libcall.
    SDValue MakePIC16Libcall(PIC16ISD::PIC16Libcall Call, EVT RetVT, 
                             const SDValue *Ops, unsigned NumOps, bool isSigned,
                             SelectionDAG &DAG, DebugLoc dl) const;

    // Check if operation has a direct load operand.
    inline bool isDirectLoad(const SDValue Op) const;
  };
} // namespace llvm

#endif // PIC16ISELLOWERING_H
