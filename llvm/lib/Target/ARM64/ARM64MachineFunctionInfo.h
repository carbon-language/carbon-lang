//===- ARM64MachineFuctionInfo.h - ARM64 machine function info --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares ARM64-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef ARM64MACHINEFUNCTIONINFO_H
#define ARM64MACHINEFUNCTIONINFO_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/MCLinkerOptimizationHint.h"

namespace llvm {

/// ARM64FunctionInfo - This class is derived from MachineFunctionInfo and
/// contains private ARM64-specific information for each MachineFunction.
class ARM64FunctionInfo : public MachineFunctionInfo {

  /// HasStackFrame - True if this function has a stack frame. Set by
  /// processFunctionBeforeCalleeSavedScan().
  bool HasStackFrame;

  /// \brief Amount of stack frame size, not including callee-saved registers.
  unsigned LocalStackSize;

  /// \brief Number of TLS accesses using the special (combinable)
  /// _TLS_MODULE_BASE_ symbol.
  unsigned NumLocalDynamicTLSAccesses;

  /// \brief FrameIndex for start of varargs area for arguments passed on the
  /// stack.
  int VarArgsStackIndex;

  /// \brief FrameIndex for start of varargs area for arguments passed in
  /// general purpose registers.
  int VarArgsGPRIndex;

  /// \brief Size of the varargs area for arguments passed in general purpose
  /// registers.
  unsigned VarArgsGPRSize;

  /// \brief FrameIndex for start of varargs area for arguments passed in
  /// floating-point registers.
  int VarArgsFPRIndex;

  /// \brief Size of the varargs area for arguments passed in floating-point
  /// registers.
  unsigned VarArgsFPRSize;

public:
  ARM64FunctionInfo()
      : HasStackFrame(false), NumLocalDynamicTLSAccesses(0),
        VarArgsStackIndex(0), VarArgsGPRIndex(0), VarArgsGPRSize(0),
        VarArgsFPRIndex(0), VarArgsFPRSize(0) {}

  explicit ARM64FunctionInfo(MachineFunction &MF)
      : HasStackFrame(false), NumLocalDynamicTLSAccesses(0),
        VarArgsStackIndex(0), VarArgsGPRIndex(0), VarArgsGPRSize(0),
        VarArgsFPRIndex(0), VarArgsFPRSize(0) {
    (void)MF;
  }

  bool hasStackFrame() const { return HasStackFrame; }
  void setHasStackFrame(bool s) { HasStackFrame = s; }

  void setLocalStackSize(unsigned Size) { LocalStackSize = Size; }
  unsigned getLocalStackSize() const { return LocalStackSize; }

  void incNumLocalDynamicTLSAccesses() { ++NumLocalDynamicTLSAccesses; }
  unsigned getNumLocalDynamicTLSAccesses() const {
    return NumLocalDynamicTLSAccesses;
  }

  int getVarArgsStackIndex() const { return VarArgsStackIndex; }
  void setVarArgsStackIndex(int Index) { VarArgsStackIndex = Index; }

  int getVarArgsGPRIndex() const { return VarArgsGPRIndex; }
  void setVarArgsGPRIndex(int Index) { VarArgsGPRIndex = Index; }

  unsigned getVarArgsGPRSize() const { return VarArgsGPRSize; }
  void setVarArgsGPRSize(unsigned Size) { VarArgsGPRSize = Size; }

  int getVarArgsFPRIndex() const { return VarArgsFPRIndex; }
  void setVarArgsFPRIndex(int Index) { VarArgsFPRIndex = Index; }

  unsigned getVarArgsFPRSize() const { return VarArgsFPRSize; }
  void setVarArgsFPRSize(unsigned Size) { VarArgsFPRSize = Size; }

  typedef SmallPtrSet<const MachineInstr *, 16> SetOfInstructions;

  const SetOfInstructions &getLOHRelated() const { return LOHRelated; }

  // Shortcuts for LOH related types.
  typedef LOHDirective<const MachineInstr> MILOHDirective;
  typedef MILOHDirective::LOHArgs MILOHArgs;

  typedef LOHContainer<const MachineInstr> MILOHContainer;
  typedef MILOHContainer::LOHDirectives MILOHDirectives;

  const MILOHContainer &getLOHContainer() const { return LOHContainerSet; }

  /// Add a LOH directive of this @p Kind and this @p Args.
  void addLOHDirective(MCLOHType Kind, const MILOHArgs &Args) {
    LOHContainerSet.addDirective(Kind, Args);
    for (MILOHArgs::const_iterator It = Args.begin(), EndIt = Args.end();
         It != EndIt; ++It)
      LOHRelated.insert(*It);
  }

private:
  // Hold the lists of LOHs.
  MILOHContainer LOHContainerSet;
  SetOfInstructions LOHRelated;
};
} // End llvm namespace

#endif // ARM64MACHINEFUNCTIONINFO_H
