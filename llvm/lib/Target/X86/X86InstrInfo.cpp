//===- X86InstrInfo.cpp - X86 Instruction Information -----------*- C++ -*-===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

#define I(ENUM, NAME, BASEOPCODE, FLAGS, TSFLAGS, IMPDEFS, IMPUSES)
#define IMPREGSLIST(NAME, ...) \
  static const unsigned NAME[] = { __VA_ARGS__ };
#include "X86InstrInfo.def"


// X86Insts - Turn the InstrInfo.def file into a bunch of instruction
// descriptors
//
static const TargetInstrDescriptor X86Insts[] = {
#define I(ENUM, NAME, BASEOPCODE, FLAGS, TSFLAGS, IMPUSES, IMPDEFS)   \
             { NAME,                    \
               -1, /* Always vararg */  \
               ((TSFLAGS) & X86II::Void) ? -1 : 0,  /* Result is in 0 */ \
               0,                                   /* maxImmedConst field */\
               false,                               /* immedIsSignExtended */\
               0,                                   /* numDelaySlots */\
               0,                                   /* latency */\
               0,                                   /* schedClass */\
               FLAGS,                               /* Flags */\
               TSFLAGS,                             /* TSFlags */\
               IMPUSES,                             /* ImplicitUses */\
               IMPDEFS },                           /* ImplicitDefs */
#include "X86InstrInfo.def"
};

X86InstrInfo::X86InstrInfo()
  : TargetInstrInfo(X86Insts, sizeof(X86Insts)/sizeof(X86Insts[0]), 0) {
}


// createNOPinstr - returns the target's implementation of NOP, which is
// usually a pseudo-instruction, implemented by a degenerate version of
// another instruction, e.g. X86: `xchg ax, ax'; SparcV9: `sethi r0, r0, r0'
//
MachineInstr* X86InstrInfo::createNOPinstr() const {
  return BuildMI(X86::XCHGrr16, 2).addReg(X86::AX).addReg(X86::AX);
}


// isNOPinstr - since we no longer have a special NOP opcode, we need to know
// if a given instruction is interpreted as an `official' NOP instr, i.e.,
// there may be more than one way to `do nothing' but only one canonical
// way to slack off.
//
bool X86InstrInfo::isNOPinstr(const MachineInstr &MI) const {
  // Make sure the instruction is EXACTLY `xchg ax, ax'
  if (MI.getOpcode() == X86::XCHGrr16 && MI.getNumOperands() == 2) {
    const MachineOperand &op0 = MI.getOperand(0), &op1 = MI.getOperand(1);
    if (op0.isMachineRegister() && op0.getMachineRegNum() == X86::AX &&
        op1.isMachineRegister() && op1.getMachineRegNum() == X86::AX)
    {
      return true;
    }
  }
  return false;
}


static unsigned char BaseOpcodes[] = {
#define I(ENUM, NAME, BASEOPCODE, FLAGS, TSFLAGS, IMPDEFS, IMPUSES) BASEOPCODE,
#include "X86InstrInfo.def"
};

// getBaseOpcodeFor - This function returns the "base" X86 opcode for the
// specified opcode number.
//
unsigned char X86InstrInfo::getBaseOpcodeFor(unsigned Opcode) const {
  assert(Opcode < sizeof(BaseOpcodes)/sizeof(BaseOpcodes[0]) &&
         "Opcode out of range!");
  return BaseOpcodes[Opcode];
}
