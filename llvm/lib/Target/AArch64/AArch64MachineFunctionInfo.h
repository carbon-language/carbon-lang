//=- AArch64MachineFuctionInfo.h - AArch64 machine function info -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares AArch64-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef AARCH64MACHINEFUNCTIONINFO_H
#define AARCH64MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// This class is derived from MachineFunctionInfo and contains private AArch64
/// target-specific information for each MachineFunction.
class AArch64MachineFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

  /// Number of bytes of arguments this function has on the stack. If the callee
  /// is expected to restore the argument stack this should be a multiple of 16,
  /// all usable during a tail call.
  ///
  /// The alternative would forbid tail call optimisation in some cases: if we
  /// want to transfer control from a function with 8-bytes of stack-argument
  /// space to a function with 16-bytes then misalignment of this value would
  /// make a stack adjustment necessary, which could not be undone by the
  /// callee.
  unsigned BytesInStackArgArea;

  /// The number of bytes to restore to deallocate space for incoming
  /// arguments. Canonically 0 in the C calling convention, but non-zero when
  /// callee is expected to pop the args.
  unsigned ArgumentStackToRestore;

  /// If the stack needs to be adjusted on frame entry in two stages, this
  /// records the size of the first adjustment just prior to storing
  /// callee-saved registers. The callee-saved slots are addressed assuming
  /// SP == <incoming-SP> - InitialStackAdjust.
  unsigned InitialStackAdjust;

  /// Number of local-dynamic TLS accesses.
  unsigned NumLocalDynamics;

  /// @see AArch64 Procedure Call Standard, B.3
  ///
  /// The Frame index of the area where LowerFormalArguments puts the
  /// general-purpose registers that might contain variadic parameters.
  int VariadicGPRIdx;

  /// @see AArch64 Procedure Call Standard, B.3
  ///
  /// The size of the frame object used to store the general-purpose registers
  /// which might contain variadic arguments. This is the offset from
  /// VariadicGPRIdx to what's stored in __gr_top.
  unsigned VariadicGPRSize;

  /// @see AArch64 Procedure Call Standard, B.3
  ///
  /// The Frame index of the area where LowerFormalArguments puts the
  /// floating-point registers that might contain variadic parameters.
  int VariadicFPRIdx;

  /// @see AArch64 Procedure Call Standard, B.3
  ///
  /// The size of the frame object used to store the floating-point registers
  /// which might contain variadic arguments. This is the offset from
  /// VariadicFPRIdx to what's stored in __vr_top.
  unsigned VariadicFPRSize;

  /// @see AArch64 Procedure Call Standard, B.3
  ///
  /// The Frame index of an object pointing just past the last known stacked
  /// argument on entry to a variadic function. This goes into the __stack field
  /// of the va_list type.
  int VariadicStackIdx;

  /// The offset of the frame pointer from the stack pointer on function
  /// entry. This is expected to be negative.
  int FramePointerOffset;

public:
  AArch64MachineFunctionInfo()
    : BytesInStackArgArea(0),
      ArgumentStackToRestore(0),
      InitialStackAdjust(0),
      NumLocalDynamics(0),
      VariadicGPRIdx(0),
      VariadicGPRSize(0),
      VariadicFPRIdx(0),
      VariadicFPRSize(0),
      VariadicStackIdx(0),
      FramePointerOffset(0) {}

  explicit AArch64MachineFunctionInfo(MachineFunction &MF)
    : BytesInStackArgArea(0),
      ArgumentStackToRestore(0),
      InitialStackAdjust(0),
      NumLocalDynamics(0),
      VariadicGPRIdx(0),
      VariadicGPRSize(0),
      VariadicFPRIdx(0),
      VariadicFPRSize(0),
      VariadicStackIdx(0),
      FramePointerOffset(0) {}

  unsigned getBytesInStackArgArea() const { return BytesInStackArgArea; }
  void setBytesInStackArgArea (unsigned bytes) { BytesInStackArgArea = bytes;}

  unsigned getArgumentStackToRestore() const { return ArgumentStackToRestore; }
  void setArgumentStackToRestore(unsigned bytes) {
    ArgumentStackToRestore = bytes;
  }

  unsigned getInitialStackAdjust() const { return InitialStackAdjust; }
  void setInitialStackAdjust(unsigned bytes) { InitialStackAdjust = bytes; }

  unsigned getNumLocalDynamicTLSAccesses() const { return NumLocalDynamics; }
  void incNumLocalDynamicTLSAccesses() { ++NumLocalDynamics; }

  int getVariadicGPRIdx() const { return VariadicGPRIdx; }
  void setVariadicGPRIdx(int Idx) { VariadicGPRIdx = Idx; }

  unsigned getVariadicGPRSize() const { return VariadicGPRSize; }
  void setVariadicGPRSize(unsigned Size) { VariadicGPRSize = Size; }

  int getVariadicFPRIdx() const { return VariadicFPRIdx; }
  void setVariadicFPRIdx(int Idx) { VariadicFPRIdx = Idx; }

  unsigned getVariadicFPRSize() const { return VariadicFPRSize; }
  void setVariadicFPRSize(unsigned Size) { VariadicFPRSize = Size; }

  int getVariadicStackIdx() const { return VariadicStackIdx; }
  void setVariadicStackIdx(int Idx) { VariadicStackIdx = Idx; }

  int getFramePointerOffset() const { return FramePointerOffset; }
  void setFramePointerOffset(int Idx) { FramePointerOffset = Idx; }

};

} // End llvm namespace

#endif
