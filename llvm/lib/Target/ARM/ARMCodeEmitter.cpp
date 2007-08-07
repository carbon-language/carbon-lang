//===-- ARM/ARMCodeEmitter.cpp - Convert ARM code to machine code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Raul Herbster and is distributed under the 
// University of Illinois Open Source License.  See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the pass that transforms the ARM machine instructions into
// relocatable machine code.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-emitter"
#include "ARMInstrInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "ARMRelocations.h"
#include "ARMAddressingModes.h"
#include "ARM.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumEmitted, "Number of machine instructions emitted");

namespace {
  class VISIBILITY_HIDDEN Emitter : public MachineFunctionPass {
    const ARMInstrInfo  *II;
    const TargetData    *TD;
    TargetMachine       &TM;
    MachineCodeEmitter  &MCE;
  public:
    static char ID;
    explicit Emitter(TargetMachine &tm, MachineCodeEmitter &mce)
      : MachineFunctionPass((intptr_t)&ID), II(0), TD(0), TM(tm), 
      MCE(mce) {}
    Emitter(TargetMachine &tm, MachineCodeEmitter &mce,
            const ARMInstrInfo &ii, const TargetData &td)
      : MachineFunctionPass((intptr_t)&ID), II(&ii), TD(&td), TM(tm), 
      MCE(mce) {}

    bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "ARM Machine Code Emitter";
    }

    void emitInstruction(const MachineInstr &MI);
    unsigned getBinaryCodeForInstr(const MachineInstr &MI);
    int getMachineOpValue(const MachineInstr &MI, unsigned OpIndex);
    unsigned getBaseOpcodeFor(const TargetInstrDescriptor *TID);

    void emitGlobalAddressForCall(GlobalValue *GV, bool DoesntNeedStub);
    void emitExternalSymbolAddress(const char *ES, unsigned Reloc);
    void emitConstPoolAddress(unsigned CPI, unsigned Reloc,
                              int Disp = 0, unsigned PCAdj = 0 );
    void emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                              unsigned PCAdj = 0);

  private:
    int getShiftOp(const MachineOperand &MO);

  };
  char Emitter::ID = 0;
}

/// createARMCodeEmitterPass - Return a pass that emits the collected ARM code
/// to the specified MCE object.
FunctionPass *llvm::createARMCodeEmitterPass(ARMTargetMachine &TM,
                                             MachineCodeEmitter &MCE) {
  return new Emitter(TM, MCE);
}

bool Emitter::runOnMachineFunction(MachineFunction &MF) {
  assert((MF.getTarget().getRelocationModel() != Reloc::Default ||
          MF.getTarget().getRelocationModel() != Reloc::Static) &&
         "JIT relocation model must be set to static or default!");
  II = ((ARMTargetMachine&)MF.getTarget()).getInstrInfo();
  TD = ((ARMTargetMachine&)MF.getTarget()).getTargetData();

  do {
    MCE.startFunction(MF);
    for (MachineFunction::iterator MBB = MF.begin(), E = MF.end(); 
         MBB != E; ++MBB) {
      MCE.StartMachineBasicBlock(MBB);
      for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
           I != E; ++I)
        emitInstruction(*I);
    }
  } while (MCE.finishFunction(MF));

  return false;
}

unsigned Emitter::getBaseOpcodeFor(const TargetInstrDescriptor *TID) {
  return (TID->TSFlags & ARMII::OpcodeMask) >> ARMII::OpcodeShift;
}

int Emitter::getShiftOp(const MachineOperand &MO) {
  unsigned ShiftOp = 0x0;
  switch(ARM_AM::getAM2ShiftOpc(MO.getImmedValue())) {
  default: assert(0 && "Unknown shift opc!");
  case ARM_AM::asr:
    ShiftOp = 0X2;
    break;
  case ARM_AM::lsl:
    ShiftOp = 0X0;
    break;
  case ARM_AM::lsr:
    ShiftOp = 0X1;
    break;
  case ARM_AM::ror:
  case ARM_AM::rrx:
    ShiftOp = 0X3;
    break;
  }
  return ShiftOp;
}

int Emitter::getMachineOpValue(const MachineInstr &MI, unsigned OpIndex) {
  intptr_t rv = 0;
  const MachineOperand &MO = MI.getOperand(OpIndex);
  if (MO.isRegister()) {
    assert(MRegisterInfo::isPhysicalRegister(MO.getReg()));
    rv = ARMRegisterInfo::getRegisterNumbering(MO.getReg());
  } else if (MO.isImmediate()) {
    rv = MO.getImmedValue();
  } else if (MO.isGlobalAddress() || MO.isExternalSymbol() ||
             MO.isConstantPoolIndex() || MO.isJumpTableIndex()) {

    if (MO.isGlobalAddress()) {
      emitGlobalAddressForCall(MO.getGlobal(), true);
    } else if (MO.isExternalSymbol()) {
      emitExternalSymbolAddress(MO.getSymbolName(), ARM::reloc_arm_relative);
    } else if (MO.isConstantPoolIndex()) {
      emitConstPoolAddress(MO.getConstantPoolIndex(), ARM::reloc_arm_relative);
    } else if (MO.isJumpTableIndex()) {
      emitJumpTableAddress(MO.getJumpTableIndex(), ARM::reloc_arm_relative);
    }

  }
  return rv;
}

/// emitGlobalAddressForCall - Emit the specified address to the code stream
/// assuming this is part of a function call, which is PC relative.
///
void Emitter::emitGlobalAddressForCall(GlobalValue *GV, bool DoesntNeedStub) {
  MCE.addRelocation(MachineRelocation::getGV(MCE.getCurrentPCOffset(),
                                      ARM::reloc_arm_branch, GV, 0,
                                      DoesntNeedStub));
}

/// emitExternalSymbolAddress - Arrange for the address of an external symbol to
/// be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitExternalSymbolAddress(const char *ES, unsigned Reloc) {
  MCE.addRelocation(MachineRelocation::getExtSym(MCE.getCurrentPCOffset(),
                                                 Reloc, ES));
}

/// emitConstPoolAddress - Arrange for the address of an constant pool
/// to be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitConstPoolAddress(unsigned CPI, unsigned Reloc,
                                   int Disp /* = 0 */,
                                   unsigned PCAdj /* = 0 */) {
  MCE.addRelocation(MachineRelocation::getConstPool(MCE.getCurrentPCOffset(),
                                                    Reloc, CPI, PCAdj));
}

/// emitJumpTableAddress - Arrange for the address of a jump table to
/// be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                                   unsigned PCAdj /* = 0 */) {
  MCE.addRelocation(MachineRelocation::getJumpTable(MCE.getCurrentPCOffset(),
                                                    Reloc, JTI, PCAdj));
}



void Emitter::emitInstruction(const MachineInstr &MI) {
  NumEmitted++;  // Keep track of the # of mi's emitted
  MCE.emitWordLE(getBinaryCodeForInstr(MI));
}

unsigned Emitter::getBinaryCodeForInstr(const MachineInstr &MI) {
  const TargetInstrDescriptor *Desc = MI.getInstrDescriptor();
  const unsigned opcode = MI.getOpcode();
  unsigned Value = 0xE0000000;
  unsigned op;

  switch (Desc->TSFlags & ARMII::AddrModeMask) {
  case ARMII::AddrModeNone: {
    switch(Desc->TSFlags & ARMII::FormMask) {
    default: {
      assert(0 && "Unknown instruction subtype!");
      if(opcode == ARM::CLZ) {
        // set first operand
        op = getMachineOpValue(MI,0);
        Value |= op << 12;

        // set second operand
        op = getMachineOpValue(MI,1);
        Value |= op;
      }
      break;
    }
    case ARMII::MulFrm: {
      Value |= 9 << 4;

      unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
      Value |= BaseOpcode << 20;

      bool isMUL = opcode == ARM::MUL;
      bool isMLA = opcode == ARM::MLA;

      // set first operand
      op = getMachineOpValue(MI,0);
      Value |= op << (isMUL || isMLA ? 16 : 12);

      // set second operand
      op = getMachineOpValue(MI,1);
      Value |= op << (isMUL || isMLA ? 0 : 16);

      // set third operand
      op = getMachineOpValue(MI,2);
      Value |= op << (isMUL || isMLA ? 8 : 0);

      if (!isMUL) {
        op = getMachineOpValue(MI,3);
        Value |= op << (isMLA ? 12 : 8);
      }

      break;
    }
    case ARMII::Branch: {
      unsigned BaseOpcode = getBaseOpcodeFor(Desc);
      Value |= BaseOpcode << 24;

      op = getMachineOpValue(MI,0);
      Value |= op;

      break;
    }
    case ARMII::BranchMisc: {
      unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
      Value |= BaseOpcode << 4;
      Value |= 0x12fff << 8;

      if (opcode == ARM::BX_RET)
        op = 0xe;
      else 
        op = getMachineOpValue(MI,0);
      Value |= op;

      break;
    }
    case ARMII::Pseudo:
      break;
    }

    break;
  }
  case ARMII::AddrMode1: {
    unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
    Value |= BaseOpcode << 21;

    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    if (Format == ARMII::DPRdMisc) {
      Value |= getMachineOpValue(MI,0) << 12;
      Value |= getMachineOpValue(MI,1);
      switch(opcode) {
      case ARM::MOVsra_flag: {
        Value |= 0x1 << 6;
        Value |= 0x1 << 7;
        break;
      }
      case ARM::MOVsrl_flag: {
        Value |= 0x1 << 5;
        Value |= 0x1 << 7;
        break;
      }
      case ARM::MOVrx: {
        Value |= 0x3 << 5;
        break;
      }
      }
      break;
    }

    bool IsDataProcessing3 = false;

    if (Format == ARMII::DPRImS || Format == ARMII::DPRRegS ||
        Format == ARMII::DPRSoRegS) {
      Value |= 1 << 20;
      IsDataProcessing3 = true;
    }

    bool IsDataProcessing1 = Format == ARMII::DPRdIm    ||
                             Format == ARMII::DPRdReg   ||
                             Format == ARMII::DPRdSoReg;
    bool IsDataProcessing2 = Format == ARMII::DPRnIm    ||
                             Format == ARMII::DPRnReg   ||
                             Format == ARMII::DPRnSoReg;
    IsDataProcessing3 = Format == ARMII::DPRIm     ||
                        Format == ARMII::DPRReg    ||
                        Format == ARMII::DPRSoReg  ||
                        IsDataProcessing3;

    // set first operand
    op = getMachineOpValue(MI,0);
    if (IsDataProcessing1 || IsDataProcessing3) {
      Value |= op << 12;
    } else if (IsDataProcessing2) {
      Value |= op << 16;
    }

    if (IsDataProcessing3) {
      op = getMachineOpValue(MI,1);
      Value |= op << 16;
    }

    unsigned OperandIndex = IsDataProcessing3 ? 2 : 1;
    // set shift operand
    switch (Format) {
    case ARMII::DPRdIm: case ARMII::DPRnIm:
    case ARMII::DPRIm:  case ARMII::DPRImS: {
      Value |= 1 << 25;
      const MachineOperand &MO = MI.getOperand(OperandIndex);
      op = ARM_AM::getSOImmVal(MO.getImmedValue());
      Value |= op;

      break;
    }
    case ARMII::DPRdReg: case ARMII::DPRnReg:
    case ARMII::DPRReg:  case ARMII::DPRRegS: {
      op = getMachineOpValue(MI,OperandIndex);
      Value |= op;

      break;
    }
    case ARMII::DPRdSoReg: case ARMII::DPRnSoReg:
    case ARMII::DPRSoReg:  case ARMII::DPRSoRegS: {
      op = getMachineOpValue(MI,OperandIndex);
      Value |= op;

      const MachineOperand &MO1 = MI.getOperand(OperandIndex + 1);
      const MachineOperand &MO2 = MI.getOperand(OperandIndex + 2);
      bool IsShiftByRegister = MO1.getReg() > 0;
      switch(ARM_AM::getSORegShOp(MO2.getImmedValue())) {
        default: assert(0 && "Unknown shift opc!");
        case ARM_AM::asr: {
          if(IsShiftByRegister)
            Value |= 0x5 << 4;
          else
            Value |= 0x1 << 6;
          break;
        }
        case ARM_AM::lsl: {
          if(IsShiftByRegister)
            Value |= 0x1 << 4;
          break;
        }
        case ARM_AM::lsr: {
          if(IsShiftByRegister)
            Value |= 0x3 << 4;
          else
            Value |= 0x1 << 5;
          break;
        }
        case ARM_AM::ror: {
          if(IsShiftByRegister)
            Value |= 0x7 << 4;
          else
            Value |= 0x3 << 5;
          break;
        }
        case ARM_AM::rrx: {
          Value |= 0x3 << 5;
          break;
        }
      }
      if(ARM_AM::getSORegShOp(MO2.getImmedValue()) != ARM_AM::rrx)
        if(IsShiftByRegister) {
          assert(MRegisterInfo::isPhysicalRegister(MO1.getReg()));
          op = ARMRegisterInfo::getRegisterNumbering(MO1.getReg());
          assert(ARM_AM::getSORegOffset(MO2.getImm()) == 0);
          Value |= op << 8;
        } else {
          op = ARM_AM::getSORegOffset(MO2.getImm());
          Value |= op << 7;
        }
      break;
    }
    default: assert(false && "Unknown operand type!");
      break;
    }

    break;
  }
  case ARMII::AddrMode2: {
    Value |= 1 << 26;

    unsigned Index = (Desc->TSFlags & ARMII::IndexModeMask);
    if (Index == ARMII::IndexModePre || Index == 0)
      Value |= 1 << 24;
    if (Index == ARMII::IndexModePre)
      Value |= 1 << 21;

    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    if (Format == ARMII::LdFrm)
      Value |= 1 << 20;

    unsigned BitByte = getBaseOpcodeFor(Desc);
    Value |= BitByte << 22;

    // set first operand
    op = getMachineOpValue(MI,0);
    Value |= op << 12;

    // addressing mode
    op = getMachineOpValue(MI,1);
    Value |= op << 16;

    const MachineOperand &MO2 = MI.getOperand(2);
    const MachineOperand &MO3 = MI.getOperand(3);

    Value |= (ARM_AM::getAM2Op(MO3.getImm()) == ARM_AM::add ? 1 : 0) << 23;
    if (!MO2.getReg()) { // is immediate
      if (ARM_AM::getAM2Offset(MO3.getImm()))
        Value |= ARM_AM::getAM2Offset(MO3.getImm());
      break;
    }

    Value |= 1 << 25;
    assert(MRegisterInfo::isPhysicalRegister(MO2.getReg()));
    Value |= ARMRegisterInfo::getRegisterNumbering(MO2.getReg());

    if (unsigned ShImm = ARM_AM::getAM2Offset(MO3.getImm())) {
      unsigned ShiftOp = getShiftOp(MO3);
      Value |= ShiftOp << 5;
      Value |= ShImm << 7;
    }

    break;
  }
  case ARMII::AddrMode3: {
    unsigned Index = (Desc->TSFlags & ARMII::IndexModeMask);
    if (Index == ARMII::IndexModePre || Index == 0)
      Value |= 1 << 24;

    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    if (Format == ARMII::LdFrm)
      Value |= 1 << 20;

    unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
    Value |= BaseOpcode << 4;

    // set first operand
    op = getMachineOpValue(MI,0);
    Value |= op << 12;

    // addressing mode
    op = getMachineOpValue(MI,1);
    Value |= op << 16;

    const MachineOperand &MO2 = MI.getOperand(2);
    const MachineOperand &MO3 = MI.getOperand(3);

    Value |= (ARM_AM::getAM2Op(MO3.getImm()) == ARM_AM::add ? 1 : 0) << 23;

    if (MO2.getReg()) {
      Value |= ARMRegisterInfo::getRegisterNumbering(MO2.getReg());
      break;
    }

    if (unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm())) {
      Value |= 1 << 22;
      Value |= (ImmOffs >> 4) << 8; // immedH
      Value |= (ImmOffs & ~0xF); // immedL
    }

    break;
  }
  case ARMII::AddrMode4: {
    Value |= 1 << 27;

    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    if (Format == ARMII::LdFrm)
      Value |= 1 << 20;

    unsigned OpIndex = 0;

    // set first operand
    op = getMachineOpValue(MI,OpIndex);
    Value |= op << 16;

    // set addressing mode
    const MachineOperand &MO = MI.getOperand(OpIndex + 1);
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO.getImm());
    switch(Mode) {
    default: assert(0 && "Unknown addressing sub-mode!");
    case ARM_AM::ia: Value |= 0x1 << 23; break;
    case ARM_AM::ib: Value |= 0x2 << 23; break;
    case ARM_AM::da: break;
    case ARM_AM::db: Value |= 0x1 << 24; break;
    }

    // set flag W
    if (ARM_AM::getAM4WBFlag(MO.getImm()))
      Value |= 0x1 << 21;

    // set registers
    for (unsigned i = OpIndex + 4, e = MI.getNumOperands(); i != e; ++i) {
      const MachineOperand &MOR = MI.getOperand(i);
      unsigned RegNumber = ARMRegisterInfo::getRegisterNumbering(MOR.getReg());
      assert(MRegisterInfo::isPhysicalRegister(MOR.getReg()) && RegNumber < 16);
      Value |= 0x1 << RegNumber;
    }

    break;
  }
  }

  return Value;
}
