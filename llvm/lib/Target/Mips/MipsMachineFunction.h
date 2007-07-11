//===-- MipsMachineFunctionInfo.h - Private data used for Mips ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Mips specific subclass of MachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MIPS_MACHINE_FUNCTION_INFO_H
#define MIPS_MACHINE_FUNCTION_INFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// MipsFunctionInfo - This class is derived from MachineFunction private
/// Mips target-specific information for each MachineFunction.
class MipsFunctionInfo : public MachineFunctionInfo {

private:
  /// Holds for each function where on the stack 
  /// the Frame Pointer must be saved
  int FPStackOffset;

  /// Holds for each function where on the stack 
  /// the Return Address must be saved
  int RAStackOffset;

public:
  MipsFunctionInfo(MachineFunction& MF) 
  : FPStackOffset(0), RAStackOffset(0)
  {}

  int getFPStackOffset() const { return FPStackOffset; }
  void setFPStackOffset(int Off) { FPStackOffset = Off; }

  int getRAStackOffset() const { return RAStackOffset; }
  void setRAStackOffset(int Off) { RAStackOffset = Off; }

  int getTopSavedRegOffset() const { 
    return (RAStackOffset > FPStackOffset) ? 
           (RAStackOffset) : (FPStackOffset);
  }
};

} // end of namespace llvm


#endif
