//===- SPURegisterInfo.h - Cell SPU Register Information Impl ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by a team from the Computer Systems Research
// Department at The Aerospace Corporation and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Cell SPU implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_REGISTERINFO_H
#define SPU_REGISTERINFO_H

#include "SPU.h"
#include "SPUGenRegisterInfo.h.inc"

namespace llvm {
  class SPUSubtarget;
  class TargetInstrInfo;
  class Type;

  class SPURegisterInfo : public SPUGenRegisterInfo {
  private:
    const SPUSubtarget &Subtarget;
    const TargetInstrInfo &TII;

    //! Predicate: Does the machine function use the link register?
    bool usesLR(MachineFunction &MF) const;

  public:
    SPURegisterInfo(const SPUSubtarget &subtarget, const TargetInstrInfo &tii);
    
    //! Translate a register's enum value to a register number
    /*!
      This method translates a register's enum value to it's regiser number,
      e.g. SPU::R14 -> 14.
     */
    static unsigned getRegisterNumbering(unsigned RegEnum);

    //! Store a register to a stack slot, based on its register class.
    void storeRegToStackSlot(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MBBI,
                             unsigned SrcReg, int FrameIndex,
                             const TargetRegisterClass *RC) const;

    //! Store a register to an address, based on its register class
    void storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
			SmallVectorImpl<MachineOperand> &Addr,
			const TargetRegisterClass *RC,
			SmallVectorImpl<MachineInstr*> &NewMIs) const;

    //! Load a register from a stack slot, based on its register class.
    void loadRegFromStackSlot(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              unsigned DestReg, int FrameIndex,
                              const TargetRegisterClass *RC) const;

    //! Loqad a register from an address, based on its register class
    virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
				 SmallVectorImpl<MachineOperand> &Addr,
				 const TargetRegisterClass *RC,
				 SmallVectorImpl<MachineInstr*> &NewMIs) const;

    //! Copy a register to another
    void copyRegToReg(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator MI,
                      unsigned DestReg, unsigned SrcReg,
                      const TargetRegisterClass *DestRC,
                      const TargetRegisterClass *SrcRC) const;

    void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
		       unsigned DestReg, const MachineInstr *Orig) const;

    //! Fold spills into load/store instructions
    virtual MachineInstr* foldMemoryOperand(MachineInstr* MI, unsigned OpNum,
                                            int FrameIndex) const;

    //! Fold any load/store to an operand
    virtual MachineInstr* foldMemoryOperand(MachineInstr* MI, unsigned OpNum,
                                            MachineInstr* LoadMI) const;
    
    //! Return the array of callee-saved registers
    virtual const unsigned* getCalleeSavedRegs(const MachineFunction *MF) const;

    //! Return the register class array of the callee-saved registers
    virtual const TargetRegisterClass* const *
      getCalleeSavedRegClasses(const MachineFunction *MF) const;

    //! Return the reserved registers
    BitVector getReservedRegs(const MachineFunction &MF) const;

    //! Prediate: Target has dedicated frame pointer
    bool hasFP(const MachineFunction &MF) const;
    //! Eliminate the call frame setup pseudo-instructions
    void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) const;
    //! Convert frame indicies into machine operands
    void eliminateFrameIndex(MachineBasicBlock::iterator II, int,
                             RegScavenger *RS) const;
    //! Determine the frame's layour
    void determineFrameLayout(MachineFunction &MF) const;

    void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
					      RegScavenger *RS = NULL) const;
    //! Emit the function prologue
    void emitPrologue(MachineFunction &MF) const;
    //! Emit the function epilogue
    void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
    //! Get return address register (LR, aka R0)
    unsigned getRARegister() const;
    //! Get the stack frame register (SP, aka R1)
    unsigned getFrameRegister(MachineFunction &MF) const;
    //! Perform target-specific stack frame setup.
    void getInitialFrameState(std::vector<MachineMove> &Moves) const;

    //------------------------------------------------------------------------
    // New methods added:
    //------------------------------------------------------------------------

    //! Return the array of argument passing registers
    /*!
      \note The size of this array is returned by getArgRegsSize().
     */
    static const unsigned *getArgRegs();

    //! Return the size of the argument passing register array
    static const unsigned getNumArgRegs();

    //! Get DWARF debugging register number
    int getDwarfRegNum(unsigned RegNum, bool isEH) const;
  };
} // end namespace llvm

#endif
