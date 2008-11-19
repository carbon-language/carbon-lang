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

namespace llvm {
  namespace PIC16ISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      Lo,            // Low 8-bits of GlobalAddress.
      Hi,            // High 8-bits of GlobalAddress.
      PIC16Load,
      PIC16Store,
      Banksel,
      MTLO,
      MTHI,
      BCF,
      LSLF,          // PIC16 Logical shift left
      LRLF,          // PIC16 Logical shift right
      RLF,           // Rotate left through carry
      RRF,           // Rotate right through carry
      Dummy
    };

    // Keep track of different address spaces. 
    enum AddressSpace {
      RAM_SPACE = 0,   // RAM address space
      ROM_SPACE = 1    // ROM address space number is 1
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
    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerADDE(SDValue Op, SelectionDAG &DAG);
    SDValue LowerADDC(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSUBE(SDValue Op, SelectionDAG &DAG);
    SDValue LowerSUBC(SDValue Op, SelectionDAG &DAG);

    SDNode *ReplaceNodeResults(SDNode *N, SelectionDAG &DAG);
    SDNode *ExpandStore(SDNode *N, SelectionDAG &DAG);
    SDNode *ExpandLoad(SDNode *N, SelectionDAG &DAG);
    SDNode *ExpandAdd(SDNode *N, SelectionDAG &DAG);
    SDNode *ExpandGlobalAddress(SDNode *N, SelectionDAG &DAG);
    SDNode *ExpandShift(SDNode *N, SelectionDAG &DAG);

    SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const; 
    SDValue PerformPIC16LoadCombine(SDNode *N, DAGCombinerInfo &DCI) const; 

  private:
    // If the Node is a BUILD_PAIR representing representing an Address
    // then this function will return true
    bool isDirectAddress(const SDValue &Op);

    // If the Node is a DirectAddress in ROM_SPACE then this 
    // function will return true
    bool isRomAddress(const SDValue &Op);

    // To extract chain value from the SDValue Nodes
    // This function will help to maintain the chain extracting
    // code at one place. In case of any change in future it will
    // help maintain the code
    SDValue getChain(SDValue &Op);


    // Extract the Lo and Hi component of Op. 
    void GetExpandedParts(SDValue Op, SelectionDAG &DAG, SDValue &Lo, 
                          SDValue &Hi); 


    // Load pointer can be a direct or indirect address. In PIC16 direct
    // addresses need Banksel and Indirect addresses need to be loaded to
    // FSR first. Handle address specific cases here.
    void LegalizeAddress(SDValue Ptr, SelectionDAG &DAG, SDValue &Chain, 
                         SDValue &NewPtr, unsigned &Offset);

    // We can not have both operands of a binary operation in W.
    // This function is used to put one operand on stack and generate a load.
    SDValue ConvertToMemOperand(SDValue Op, SelectionDAG &DAG); 

    /// Subtarget - Keep a pointer to the PIC16Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const PIC16Subtarget *Subtarget;
  };
} // namespace llvm

#endif // PIC16ISELLOWERING_H
