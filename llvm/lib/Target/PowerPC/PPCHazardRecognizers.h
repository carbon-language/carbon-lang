//===-- PPCHazardRecognizers.h - PowerPC Hazard Recognizers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling on PowerPC processors.
//
//===----------------------------------------------------------------------===//

#ifndef PPCHAZRECS_H
#define PPCHAZRECS_H

#include "llvm/CodeGen/ScheduleDAG.h"

namespace llvm {
  
/// PPCHazardRecognizer970 - This class defines a finite state automata that
/// models the dispatch logic on the PowerPC 970 (aka G5) processor.  This
/// promotes good dispatch group formation and implements noop insertion to
/// avoid structural hazards that cause significant performance penalties (e.g.
/// setting the CTR register then branching through it within a dispatch group),
/// or storing then loading from the same address within a dispatch group.
class PPCHazardRecognizer970 : public HazardRecognizer {
  unsigned NumIssued;  // Number of insts issued, including advanced cycles.
  
  // Number of various types of instructions in the current dispatch group.
  unsigned NumFXU;     // Number of Fixed Point (integer) instructions
  unsigned NumLSU;     // Number of Load/Store instructions
  unsigned NumFPU;     // Number of Floating Point instructions
  bool     HasCR;      // True if Condition Register instruction issued
  bool     HasVALU;    // True if Vector Arithmetic instruction issued
  bool     HasVPERM;   // True if Vector Permute instruction issued
  
  // Various things that can cause a structural hazard.
  
  // HasCTRSet - If the CTR register is set in this group, disallow BCTRL.
  bool HasCTRSet;
  
  // StoredPtr - Keep track of the address of any store.  If we see a load from
  // the same address (or one that aliases it), disallow the store.  We only
  // need one pointer here, because there can only be two LSU operations and we
  // only get an LSU reject if the first is a store and the second is a load.
  //
  // This is null if we haven't seen a store yet.  We keep track of both
  // operands of the store here, since we support [r+r] and [r+i] addressing.
  SDOperand StorePtr1, StorePtr2;
  unsigned  StoreSize;
  
public:
  virtual void StartBasicBlock();
  virtual HazardType getHazardType(SDNode *Node);
  virtual void EmitInstruction(SDNode *Node);
  virtual void AdvanceCycle();
  virtual void EmitNoop();
  
private:
  /// EndDispatchGroup - Called when we are finishing a new dispatch group.
  ///
  void EndDispatchGroup();
  
  enum PPC970InstrType {
    FXU, LSU_LD, LSU_ST, FPU, CR, VALU, VPERM, BR, PseudoInst
  };
  
  /// GetInstrType - Classify the specified powerpc opcode according to its
  /// pipeline.
  PPC970InstrType GetInstrType(unsigned Opcode);
  
  bool isLoadOfStoredAddress(unsigned LoadSize,
                             SDOperand Ptr1, SDOperand Ptr2) const;
};

} // end namespace llvm

#endif