//===-- SPUFrameInfo.h - Top-level interface for Cell SPU Target -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains CellSPU frame information that doesn't fit anywhere else
// cleanly...
//
//===----------------------------------------------------------------------===//

#ifndef SPU_FRAMEINFO_H
#define SPU_FRAMEINFO_H

#include "SPURegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class SPUSubtarget;

  class SPUFrameInfo: public TargetFrameInfo {
    const SPUSubtarget &Subtarget;
    std::pair<unsigned, int> LR[1];

  public:
    SPUFrameInfo(const SPUSubtarget &sti);

    //! Determine the frame's layour
    void determineFrameLayout(MachineFunction &MF) const;

    /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
    /// the function.
    void emitPrologue(MachineFunction &MF) const;
    void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

    //! Return a function's saved spill slots
    /*!
      For CellSPU, a function's saved spill slots is just the link register.
     */
    const std::pair<unsigned, int> *
    getCalleeSaveSpillSlots(unsigned &NumEntries) const;

    //! Stack slot size (16 bytes)
    static int stackSlotSize() {
      return 16;
    }
    //! Maximum frame offset representable by a signed 10-bit integer
    /*!
      This is the maximum frame offset that can be expressed as a 10-bit
      integer, used in D-form addresses.
     */
    static int maxFrameOffset() {
      return ((1 << 9) - 1) * stackSlotSize();
    }
    //! Minimum frame offset representable by a signed 10-bit integer
    static int minFrameOffset() {
      return -(1 << 9) * stackSlotSize();
    }
    //! Minimum frame size (enough to spill LR + SP)
    static int minStackSize() {
      return (2 * stackSlotSize());
    }
    //! Convert frame index to stack offset
    static int FItoStackOffset(int frame_index) {
      return frame_index * stackSlotSize();
    }
    //! Number of instructions required to overcome hint-for-branch latency
    /*!
      HBR (hint-for-branch) instructions can be inserted when, for example,
      we know that a given function is going to be called, such as printf(),
      in the control flow graph. HBRs are only inserted if a sufficient number
      of instructions occurs between the HBR and the target. Currently, HBRs
      take 6 cycles, ergo, the magic number 6.
     */
    static int branchHintPenalty() {
      return 6;
    }
  };
}

#endif
