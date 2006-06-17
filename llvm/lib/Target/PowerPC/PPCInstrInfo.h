//===- PPCInstrInfo.h - PowerPC Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC32_INSTRUCTIONINFO_H
#define POWERPC32_INSTRUCTIONINFO_H

#include "PPC.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "PPCRegisterInfo.h"

namespace llvm {

/// PPCII - This namespace holds all of the PowerPC target-specific
/// per-instruction flags.  These must match the corresponding definitions in
/// PPC.td and PPCInstrFormats.td.
namespace PPCII {
enum {
  // PPC970 Instruction Flags.  These flags describe the characteristics of the
  // PowerPC 970 (aka G5) dispatch groups and how they are formed out of
  // raw machine instructions.

  /// PPC970_First - This instruction starts a new dispatch group, so it will
  /// always be the first one in the group.
  PPC970_First = 0x1,
  
  /// PPC970_Single - This instruction starts a new dispatch group and
  /// terminates it, so it will be the sole instruction in the group.
  PPC970_Single = 0x2,

  /// PPC970_Cracked - This instruction is cracked into two pieces, requiring
  /// two dispatch pipes to be available to issue.
  PPC970_Cracked = 0x4,
  
  /// PPC970_Mask/Shift - This is a bitmask that selects the pipeline type that
  /// an instruction is issued to.
  PPC970_Shift = 3,
  PPC970_Mask = 0x07 << PPC970_Shift
};
enum PPC970_Unit {
  /// These are the various PPC970 execution unit pipelines.  Each instruction
  /// is one of these.
  PPC970_Pseudo = 0 << PPC970_Shift,   // Pseudo instruction
  PPC970_FXU    = 1 << PPC970_Shift,   // Fixed Point (aka Integer/ALU) Unit
  PPC970_LSU    = 2 << PPC970_Shift,   // Load Store Unit
  PPC970_FPU    = 3 << PPC970_Shift,   // Floating Point Unit
  PPC970_CRU    = 4 << PPC970_Shift,   // Control Register Unit
  PPC970_VALU   = 5 << PPC970_Shift,   // Vector ALU
  PPC970_VPERM  = 6 << PPC970_Shift,   // Vector Permute Unit
  PPC970_BRU    = 7 << PPC970_Shift    // Branch Unit
};
}
  
  
class PPCInstrInfo : public TargetInstrInfo {
  PPCTargetMachine &TM;
  const PPCRegisterInfo RI;
public:
  PPCInstrInfo(PPCTargetMachine &TM);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *getPointerRegClass() const;  

  // Return true if the instruction is a register to register move and
  // leave the source and dest operands in the passed parameters.
  //
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& sourceReg,
                           unsigned& destReg) const;

  unsigned isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const;
  unsigned isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const;

  // commuteInstruction - We can commute rlwimi instructions, but only if the
  // rotate amt is zero.  We also have to munge the immediates a bit.
  virtual MachineInstr *commuteInstruction(MachineInstr *MI) const;
  
  virtual void insertNoop(MachineBasicBlock &MBB, 
                          MachineBasicBlock::iterator MI) const;

  static unsigned invertPPCBranchOpcode(unsigned Opcode) {
    switch (Opcode) {
    default: assert(0 && "Unknown PPC branch opcode!");
    case PPC::BEQ: return PPC::BNE;
    case PPC::BNE: return PPC::BEQ;
    case PPC::BLT: return PPC::BGE;
    case PPC::BGE: return PPC::BLT;
    case PPC::BGT: return PPC::BLE;
    case PPC::BLE: return PPC::BGT;
    case PPC::BNU: return PPC::BUN;
    case PPC::BUN: return PPC::BNU;
    }
  }
};

}

#endif
