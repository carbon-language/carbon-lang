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

#include "llvm/ADT/VectorExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

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

  /// When PIC is used the GP must be saved on the stack
  /// on the function prologue, so a reference to its stack
  /// location must be kept.
  int GPStackOffset;

  /// MipsFIHolder - Holds a FrameIndex and it's Stack Pointer Offset
  struct MipsFIHolder {

    int FI;
    int SPOffset;

    MipsFIHolder(int FrameIndex, int StackPointerOffset)
      : FI(FrameIndex), SPOffset(StackPointerOffset) {}
  };

  // On LowerFORMAL_ARGUMENTS the stack size is unknown,
  // so the Stack Pointer Offset calculation of "not in 
  // register arguments" must be postponed to emitPrologue. 
  SmallVector<MipsFIHolder, 16> FnLoadArgs;
  bool HasLoadArgs;

  // When VarArgs, we must write registers back to caller
  // stack, preserving on register arguments. Since the 
  // stack size is unknown on LowerFORMAL_ARGUMENTS,
  // the Stack Pointer Offset calculation must be
  // postponed to emitPrologue. 
  SmallVector<MipsFIHolder, 4> FnStoreVarArgs;
  bool HasStoreVarArgs;

public:
  MipsFunctionInfo(MachineFunction& MF) 
  : FPStackOffset(0), RAStackOffset(0), 
    HasLoadArgs(false), HasStoreVarArgs(false)
  {}

  int getFPStackOffset() const { return FPStackOffset; }
  void setFPStackOffset(int Off) { FPStackOffset = Off; }

  int getRAStackOffset() const { return RAStackOffset; }
  void setRAStackOffset(int Off) { RAStackOffset = Off; }

  int getGPStackOffset() const { return GPStackOffset; }
  void setGPStackOffset(int Off) { GPStackOffset = Off; }

  int getTopSavedRegOffset() const { 
    return (RAStackOffset > FPStackOffset) ? 
           (RAStackOffset) : (FPStackOffset);
  }

  bool hasLoadArgs() const { return HasLoadArgs; }
  bool hasStoreVarArgs() const { return HasStoreVarArgs; } 

  void recordLoadArgsFI(int FI, int SPOffset) {
    if (!HasLoadArgs) HasLoadArgs=true;
    FnLoadArgs.push_back(MipsFIHolder(FI, SPOffset));
  }
  void recordStoreVarArgsFI(int FI, int SPOffset) {
    if (!HasStoreVarArgs) HasStoreVarArgs=true;
    FnStoreVarArgs.push_back(MipsFIHolder(FI, SPOffset));
  }

  void adjustLoadArgsFI(MachineFrameInfo *MFI) const {
    if (!hasLoadArgs()) return;
    for (unsigned i = 0, e = FnLoadArgs.size(); i != e; ++i) 
      MFI->setObjectOffset( FnLoadArgs[i].FI, FnLoadArgs[i].SPOffset );
  }
  void adjustStoreVarArgsFI(MachineFrameInfo *MFI) const {
    if (!hasStoreVarArgs()) return; 
    for (unsigned i = 0, e = FnStoreVarArgs.size(); i != e; ++i) 
      MFI->setObjectOffset( FnStoreVarArgs[i].FI, FnStoreVarArgs[i].SPOffset );
  }

};

} // end of namespace llvm

#endif // MIPS_MACHINE_FUNCTION_INFO_H
