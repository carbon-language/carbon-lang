//===-- PPCMachineFunctionInfo.h - Private data used for PowerPC --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of MachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_MACHINE_FUNCTION_INFO_H
#define PPC_MACHINE_FUNCTION_INFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// PPCFunctionInfo - This class is derived from MachineFunction private
/// PowerPC target-specific information for each MachineFunction.
class PPCFunctionInfo : public MachineFunctionInfo {
private:
  /// FramePointerSaveIndex - Frame index of where the old frame pointer is
  /// stored.  Also used as an anchor for instructions that need to be altered
  /// when using frame pointers (dyna_add, dyna_sub.)
  int FramePointerSaveIndex;
  
  /// ReturnAddrSaveIndex - Frame index of where the return address is stored.
  ///
  int ReturnAddrSaveIndex;

  /// UsesLR - Indicates whether LR is used in the current function.  This is
  /// only valid after the initial scan of the function by PEI.
  bool UsesLR;

  /// LRStoreRequired - The bool indicates whether there is some explicit use of
  /// the LR/LR8 stack slot that is not obvious from scanning the code.  This
  /// requires that the code generator produce a store of LR to the stack on
  /// entry, even though LR may otherwise apparently not be used.
  bool LRStoreRequired;
public:
  PPCFunctionInfo(MachineFunction &MF) 
    : FramePointerSaveIndex(0), ReturnAddrSaveIndex(0), LRStoreRequired(false){}

  int getFramePointerSaveIndex() const { return FramePointerSaveIndex; }
  void setFramePointerSaveIndex(int Idx) { FramePointerSaveIndex = Idx; }
  
  int getReturnAddrSaveIndex() const { return ReturnAddrSaveIndex; }
  void setReturnAddrSaveIndex(int idx) { ReturnAddrSaveIndex = idx; }
  
  /// UsesLR - This is set when the prolog/epilog inserter does its initial scan
  /// of the function, it is true if the LR/LR8 register is ever explicitly
  /// accessed/clobbered in the machine function (e.g. by calls and movpctolr,
  /// which is used in PIC generation).
  void setUsesLR(bool U) { UsesLR = U; }
  bool usesLR() const    { return UsesLR; }

  void setLRStoreRequired() { LRStoreRequired = true; }
  bool isLRStoreRequired() const { return LRStoreRequired; }
  
};

} // end of namespace llvm


#endif
