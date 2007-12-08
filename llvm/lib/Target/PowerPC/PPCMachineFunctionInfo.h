//===-- PPCMachineFunctionInfo.h - Private data used for PowerPC --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
  
  /// UsesLR - Indicates whether LR is used in the current function.
  ///
  bool UsesLR;

public:
  PPCFunctionInfo(MachineFunction& MF) 
  : FramePointerSaveIndex(0)
  {}

  int getFramePointerSaveIndex() const { return FramePointerSaveIndex; }
  void setFramePointerSaveIndex(int Idx) { FramePointerSaveIndex = Idx; }
  
  /// UsesLR - This is set when the prolog/epilog inserter does its initial scan
  /// of the function, it is true if the LR/LR8 register is ever explicitly
  /// accessed/clobbered in the machine function (e.g. by calls and movpctolr,
  /// which is used in PIC generation).
  void setUsesLR(bool U) { UsesLR = U; }
  bool usesLR() const    { return UsesLR; }

};

} // end of namespace llvm


#endif
