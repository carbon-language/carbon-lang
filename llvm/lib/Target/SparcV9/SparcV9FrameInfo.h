//===-- SparcV9FrameInfo.h - Define TargetFrameInfo for SparcV9 -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to stack frame layout info for the UltraSPARC.
//
//----------------------------------------------------------------------------

#ifndef SPARC_FRAMEINFO_H
#define SPARC_FRAMEINFO_H

#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "SparcV9RegInfo.h"

namespace llvm {

class SparcV9FrameInfo: public TargetFrameInfo {
  const TargetMachine &target;
public:
  SparcV9FrameInfo(const TargetMachine &TM)
    : TargetFrameInfo(StackGrowsDown, StackFrameSizeAlignment, 0), target(TM) {}

  // This method adjusts a stack offset to meet alignment rules of target.
  // The fixed OFFSET (0x7ff) must be subtracted and the result aligned.
  virtual int  adjustAlignment(int unalignedOffset, bool growUp,
                               unsigned int align) const {
    return unalignedOffset + (growUp? +1:-1)*((unalignedOffset-OFFSET) % align);
  }

  // These methods compute offsets using the frame contents for a
  // particular function.  The frame contents are obtained from the
  // MachineCodeInfoForMethod object for the given function.
  //
  int getFirstAutomaticVarOffset(MachineFunction& mcInfo, bool& growUp) const {
    growUp = false;
    return StaticAreaOffsetFromFP;
  }
  int getRegSpillAreaOffset(MachineFunction& mcInfo, bool& growUp) const;
  int getTmpAreaOffset(MachineFunction& mcInfo, bool& growUp) const;
  int getDynamicAreaOffset(MachineFunction& mcInfo, bool& growUp) const;

  virtual int getIncomingArgOffset(MachineFunction& mcInfo,
                                   unsigned argNum) const {
    unsigned relativeOffset = argNum * SizeOfEachArgOnStack;
    int firstArg = FirstIncomingArgOffsetFromFP;
    return firstArg + relativeOffset;
  }

  virtual int getOutgoingArgOffset(MachineFunction& mcInfo,
				   unsigned argNum) const {
    return FirstOutgoingArgOffsetFromSP + argNum * SizeOfEachArgOnStack;
  }

  /*----------------------------------------------------------------------
    This diagram shows the stack frame layout used by llc on SparcV9 V9.
    Note that only the location of automatic variables, spill area,
    temporary storage, and dynamically allocated stack area are chosen
    by us.  The rest conform to the SparcV9 V9 ABI.
    All stack addresses are offset by OFFSET = 0x7ff (2047).

    Alignment assumptions and other invariants:
    (1) %sp+OFFSET and %fp+OFFSET are always aligned on 16-byte boundary
    (2) Variables in automatic, spill, temporary, or dynamic regions
        are aligned according to their size as in all memory accesses.
    (3) Everything below the dynamically allocated stack area is only used
        during a call to another function, so it is never needed when
        the current function is active.  This is why space can be allocated
        dynamically by incrementing %sp any time within the function.

    STACK FRAME LAYOUT:

       ...
       %fp+OFFSET+176      Optional extra incoming arguments# 1..N
       %fp+OFFSET+168      Incoming argument #6
       ...                 ...
       %fp+OFFSET+128      Incoming argument #1
       ...                 ...
    ---%fp+OFFSET-0--------Bottom of caller's stack frame--------------------
       %fp+OFFSET-8        Automatic variables <-- ****TOP OF STACK FRAME****
                           Spill area
                           Temporary storage
       ...

       %sp+OFFSET+176+8N   Bottom of dynamically allocated stack area
       %sp+OFFSET+168+8N   Optional extra outgoing argument# N
       ...                 ...
       %sp+OFFSET+176      Optional extra outgoing argument# 1
       %sp+OFFSET+168      Outgoing argument #6
       ...                 ...
       %sp+OFFSET+128      Outgoing argument #1
       %sp+OFFSET+120      Save area for %i7
       ...                 ...
       %sp+OFFSET+0        Save area for %l0 <-- ****BOTTOM OF STACK FRAME****

   *----------------------------------------------------------------------*/

  // All stack addresses must be offset by 0x7ff (2047) on SparcV9 V9.
  static const int OFFSET                                  = (int) 0x7ff;
  static const int StackFrameSizeAlignment                 =  16;
  static const int MinStackFrameSize                       = 176;
  static const int SizeOfEachArgOnStack                    =   8;
  static const int FirstIncomingArgOffsetFromFP            = 128 + OFFSET;
  static const int FirstOptionalIncomingArgOffsetFromFP    = 176 + OFFSET;
  static const int StaticAreaOffsetFromFP                  =   0 + OFFSET;
  static const int FirstOutgoingArgOffsetFromSP            = 128 + OFFSET;
  static const int FirstOptionalOutgoingArgOffsetFromSP    = 176 + OFFSET;
};

} // End llvm namespace

#endif
