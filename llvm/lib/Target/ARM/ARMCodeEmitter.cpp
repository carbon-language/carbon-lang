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
    int getMachineOpValue(const MachineInstr &MI, unsigned OpIndex);
    unsigned getBaseOpcodeFor(const TargetInstrDescriptor *TID);
    unsigned getBinaryCodeForInstr(const MachineInstr &MI);

    void emitGlobalAddressForCall(GlobalValue *GV, bool DoesntNeedStub);
    void emitExternalSymbolAddress(const char *ES, unsigned Reloc);
    void emitConstPoolAddress(unsigned CPI, unsigned Reloc,
                              int Disp = 0, unsigned PCAdj = 0 );
    void emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                              unsigned PCAdj = 0);
    void emitGlobalConstant(const Constant *CV);
    void emitMachineBasicBlock(MachineBasicBlock *BB);

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

/// getBaseOpcodeFor - Return the opcode value
unsigned Emitter::getBaseOpcodeFor(const TargetInstrDescriptor *TID) {
  return (TID->TSFlags & ARMII::OpcodeMask) >> ARMII::OpcodeShift;
}

/// getShiftOp - Verify which is the shift opcode (bit[6:5]) of the
/// machine operand.
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
  } else if (MO.isGlobalAddress()) {
    emitGlobalAddressForCall(MO.getGlobal(), false);
  } else if (MO.isExternalSymbol()) {
    emitExternalSymbolAddress(MO.getSymbolName(), ARM::reloc_arm_relative);
  } else if (MO.isConstantPoolIndex()) {
    emitConstPoolAddress(MO.getConstantPoolIndex(), ARM::reloc_arm_relative);
  } else if (MO.isJumpTableIndex()) {
    emitJumpTableAddress(MO.getJumpTableIndex(), ARM::reloc_arm_relative);
  } else if (MO.isMachineBasicBlock()) {
    emitMachineBasicBlock(MO.getMachineBasicBlock());
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

/// emitMachineBasicBlock - Emit the specified address basic block.
void Emitter::emitMachineBasicBlock(MachineBasicBlock *BB) {
  MCE.addRelocation(MachineRelocation::getBB(MCE.getCurrentPCOffset(),
                                      ARM::reloc_arm_branch, BB));
}

void Emitter::emitInstruction(const MachineInstr &MI) {
  NumEmitted++;  // Keep track of the # of mi's emitted
  MCE.emitWordLE(getBinaryCodeForInstr(MI));
}

unsigned Emitter::getBinaryCodeForInstr(const MachineInstr &MI) {
  const TargetInstrDescriptor *Desc = MI.getInstrDescriptor();
  const unsigned opcode = MI.getOpcode();
  // initial instruction mask
  unsigned Value = 0xE0000000;
  unsigned op;

  switch (Desc->TSFlags & ARMII::AddrModeMask) {
  case ARMII::AddrModeNone: {
    switch(Desc->TSFlags & ARMII::FormMask) {
    default: {
      assert(0 && "Unknown instruction subtype!");
      // treat special instruction CLZ
      if(opcode == ARM::CLZ) {
        // set first operand
        op = getMachineOpValue(MI,0);
        Value |= op << ARMII::RegRdShift;

        // set second operand
        op = getMachineOpValue(MI,1);
        Value |= op;
      }
      break;
    }
    case ARMII::MulSMLAW:
    case ARMII::MulSMULW:
      // set bit W(21)
      Value |= 1 << 21;
    case ARMII::MulSMLA:
    case ARMII::MulSMUL: {
      // set bit W(21)
      Value |= 1 << 24;

      // set opcode (bit[7:4]). For more information, see ARM-ARM page A3-31
      // SMLA<x><y>  - 1yx0
      // SMLAW<y>    - 1y00
      // SMULW<y>    - 1y10
      // SMUL<x><y>  - 1yx0
      unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
      Value |= BaseOpcode << 4;

      unsigned Format = (Desc->TSFlags & ARMII::FormMask);
      if (Format == ARMII::MulSMUL)
        Value |= 1 << 22;

      // set first operand
      op = getMachineOpValue(MI,0);
      Value |= op << ARMII::RegRnShift;

      // set second operand
      op = getMachineOpValue(MI,1);
      Value |= op;

      // set third operand
      op = getMachineOpValue(MI,2);
      Value |= op << ARMII::RegRsShift;

      // instructions SMLA and SMLAW have a fourth operand
      if (Format != ARMII::MulSMULW && Format != ARMII::MulSMUL) {
        op = getMachineOpValue(MI,3);
        Value |= op << ARMII::RegRdShift;
      }

      break;
    }
    case ARMII::MulFrm: {
      // bit[7:4] is always 9
      Value |= 9 << 4;
      // set opcode (bit[23:20])
      unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
      Value |= BaseOpcode << 20;

      bool isMUL = opcode == ARM::MUL;
      bool isMLA = opcode == ARM::MLA;

      // set first operand
      op = getMachineOpValue(MI,0);
      Value |= op << (isMUL || isMLA ? ARMII::RegRnShift : ARMII::RegRdShift);

      // set second operand
      op = getMachineOpValue(MI,1);
      Value |= op << (isMUL || isMLA ? 0 : ARMII::RegRnShift);

      // set third operand
      op = getMachineOpValue(MI,2);
      Value |= op << (isMUL || isMLA ? ARMII::RegRsShift : 0);

      // multiply instructions (except MUL), have a fourth operand
      if (!isMUL) {
        op = getMachineOpValue(MI,3);
        Value |= op << (isMLA ? ARMII::RegRdShift : ARMII::RegRsShift);
      }

      break;
    }
    case ARMII::Branch: {
      // set opcode (bit[27:24])
      unsigned BaseOpcode = getBaseOpcodeFor(Desc);
      Value |= BaseOpcode << 24;

      // set signed_immed_24 field
      op = getMachineOpValue(MI,0);
      Value |= op;

      // if it is a conditional branch, set cond field
      if (opcode == ARM::Bcc) {
        op = getMachineOpValue(MI,1);
        Value &= 0x0FFFFFFF; // clear conditional field
        Value |= op << 28;   // set conditional field
      }

      break;
    }
    case ARMII::BranchMisc: {
      // set opcode (bit[7:4])
      unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
      Value |= BaseOpcode << 4;
      // set bit[27:24] to 1, set bit[23:20] to 2 and set bit[19:8] to 0xFFF
      Value |= 0x12fff << 8;

      if (opcode == ARM::BX_RET)
        op = 0xe; // the return register is LR
      else 
        // otherwise, set the return register
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
    // set opcode (bit[24:21]) of data-processing instructions
    unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
    Value |= BaseOpcode << 21;

    // treat 3 special instructions: MOVsra_flag, MOVsrl_flag and
    // MOVrx.
    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    if (Format == ARMII::DPRdMisc) {
      Value |= getMachineOpValue(MI,0) << ARMII::RegRdShift;
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

    // Data processing operand instructions has 3 possible encodings (for more
    // information, see ARM-ARM page A3-10):
    // 1. <instr> <Rd>,<shifter_operand>
    // 2. <instr> <Rn>,<shifter_operand>
    // 3. <instr> <Rd>,<Rn>,<shifter_operand>
    bool IsDataProcessing1 = Format == ARMII::DPRdIm    ||
                             Format == ARMII::DPRdReg   ||
                             Format == ARMII::DPRdSoReg;
    bool IsDataProcessing2 = Format == ARMII::DPRnIm    ||
                             Format == ARMII::DPRnReg   ||
                             Format == ARMII::DPRnSoReg;
    bool IsDataProcessing3 = false;

    // set bit S(20)
    if (Format == ARMII::DPRImS || Format == ARMII::DPRRegS ||
        Format == ARMII::DPRSoRegS || IsDataProcessing2) {
      Value |= 1 << ARMII::S_BitShift;
      IsDataProcessing3 = !IsDataProcessing2;
    }

    IsDataProcessing3 = Format == ARMII::DPRIm     ||
                        Format == ARMII::DPRReg    ||
                        Format == ARMII::DPRSoReg  ||
                        IsDataProcessing3;

    // set first operand
    op = getMachineOpValue(MI,0);
    if (IsDataProcessing1 || IsDataProcessing3) {
      Value |= op << ARMII::RegRdShift;
    } else if (IsDataProcessing2) {
      Value |= op << ARMII::RegRnShift;
    }

    // set second operand of data processing #3 instructions
    if (IsDataProcessing3) {
      op = getMachineOpValue(MI,1);
      Value |= op << ARMII::RegRnShift;
    }

    unsigned OperandIndex = IsDataProcessing3 ? 2 : 1;
    switch (Format) {
    case ARMII::DPRdIm: case ARMII::DPRnIm:
    case ARMII::DPRIm:  case ARMII::DPRImS: {
      // set bit I(25) to identify this is the immediate form of <shifter_op>
      Value |= 1 << ARMII::I_BitShift;
      // set immed_8 field
      const MachineOperand &MO = MI.getOperand(OperandIndex);
      op = ARM_AM::getSOImmVal(MO.getImmedValue());
      Value |= op;

      break;
    }
    case ARMII::DPRdReg: case ARMII::DPRnReg:
    case ARMII::DPRReg:  case ARMII::DPRRegS: {
      // set last operand (register Rm)
      op = getMachineOpValue(MI,OperandIndex);
      Value |= op;

      break;
    }
    case ARMII::DPRdSoReg: case ARMII::DPRnSoReg:
    case ARMII::DPRSoReg:  case ARMII::DPRSoRegS: {
      // set last operand (register Rm)
      op = getMachineOpValue(MI,OperandIndex);
      Value |= op;

      const MachineOperand &MO1 = MI.getOperand(OperandIndex + 1);
      const MachineOperand &MO2 = MI.getOperand(OperandIndex + 2);
      // identify it the instr is in immed or register shifts encoding
      bool IsShiftByRegister = MO1.getReg() > 0;
      // set shift operand (bit[6:4]).
      // ASR - 101 if it is in register shifts encoding; 100, otherwise.
      // LSL - 001 if it is in register shifts encoding; 000, otherwise.
      // LSR - 011 if it is in register shifts encoding; 010, otherwise.
      // ROR - 111 if it is in register shifts encoding; 110, otherwise.
      // RRX - 110 and bit[11:7] clear.
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
      // set the field related to shift operations (except rrx).
      if(ARM_AM::getSORegShOp(MO2.getImmedValue()) != ARM_AM::rrx)
        if(IsShiftByRegister) {
          // set the value of bit[11:8] (register Rs).
          assert(MRegisterInfo::isPhysicalRegister(MO1.getReg()));
          op = ARMRegisterInfo::getRegisterNumbering(MO1.getReg());
          assert(ARM_AM::getSORegOffset(MO2.getImm()) == 0);
          Value |= op << ARMII::RegRsShift;
        } else {
          // set the value of bit [11:7] (shift_immed field).
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
    // bit 26 is always 1
    Value |= 1 << 26;

    unsigned Index = (Desc->TSFlags & ARMII::IndexModeMask);
    // if the instruction uses offset addressing or pre-indexed addressing,
    // set bit P(24) to 1
    if (Index == ARMII::IndexModePre || Index == 0)
      Value |= 1 << ARMII::IndexShift;
    // if the instruction uses post-indexed addressing, set bit W(21) to 1
    if (Index == ARMII::IndexModePre)
      Value |= 1 << 21;

    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    // If it is a load instruction (except LDRD), set bit L(20) to 1
    if (Format == ARMII::LdFrm)
      Value |= 1 << ARMII::L_BitShift;

    // set bit B(22)
    unsigned BitByte = getBaseOpcodeFor(Desc);
    Value |= BitByte << 22;

    // set first operand
    op = getMachineOpValue(MI,0);
    Value |= op << ARMII::RegRdShift;

    // set second operand
    op = getMachineOpValue(MI,1);
    Value |= op << ARMII::RegRnShift;

    const MachineOperand &MO2 = MI.getOperand(2);
    const MachineOperand &MO3 = MI.getOperand(3);

    // set bit U(23) according to signal of immed value (positive or negative)
    Value |= (ARM_AM::getAM2Op(MO3.getImm()) == ARM_AM::add ? 1 : 0) <<
                                                ARMII::U_BitShift;
    if (!MO2.getReg()) { // is immediate
      if (ARM_AM::getAM2Offset(MO3.getImm()))
        // set the value of offset_12 field
        Value |= ARM_AM::getAM2Offset(MO3.getImm());
      break;
    }

    // set bit I(25), because this is not in immediate enconding.
    Value |= 1 << ARMII::I_BitShift;
    assert(MRegisterInfo::isPhysicalRegister(MO2.getReg()));
    // set bit[3:0] to the corresponding Rm register
    Value |= ARMRegisterInfo::getRegisterNumbering(MO2.getReg());

    // if this instr is in scaled register offset/index instruction, set
    // shift_immed(bit[11:7]) and shift(bit[6:5]) fields.
    if (unsigned ShImm = ARM_AM::getAM2Offset(MO3.getImm())) {
      unsigned ShiftOp = getShiftOp(MO3);
      Value |= ShiftOp << 5; // shift
      Value |= ShImm << 7;   // shift_immed
    }

    break;
  }
  case ARMII::AddrMode3: {

    unsigned Index = (Desc->TSFlags & ARMII::IndexModeMask);
    // if the instruction uses offset addressing or pre-indexed addressing,
    // set bit P(24) to 1
    if (Index == ARMII::IndexModePre || Index == 0)
      Value |= 1 << ARMII::IndexShift;

    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    // If it is a load instruction (except LDRD), set bit L(20) to 1
    if (Format == ARMII::LdFrm && opcode != ARM::LDRD)
      Value |= 1 << ARMII::L_BitShift;

    // bit[7:4] is the opcode of this instruction class (bits S and H).
    unsigned char BaseOpcode = getBaseOpcodeFor(Desc);
    Value |= BaseOpcode << 4;

    // set first operand
    op = getMachineOpValue(MI,0);
    Value |= op << ARMII::RegRdShift;

    // set second operand
    op = getMachineOpValue(MI,1);
    Value |= op << ARMII::RegRnShift;

    const MachineOperand &MO2 = MI.getOperand(2);
    const MachineOperand &MO3 = MI.getOperand(3);

    // set bit U(23) according to signal of immed value (positive or negative)
    Value |= (ARM_AM::getAM2Op(MO3.getImm()) == ARM_AM::add ? 1 : 0) <<
                                                ARMII::U_BitShift;

    // if this instr is in register offset/index encoding, set bit[3:0]
    // to the corresponding Rm register.
    if (MO2.getReg()) {
      Value |= ARMRegisterInfo::getRegisterNumbering(MO2.getReg());
      break;
    }

    // if this instr is in immediate offset/index encoding, set bit 22 to 1
    if (unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm())) {
      Value |= 1 << 22;
      // set operands
      Value |= (ImmOffs >> 4) << 8; // immedH
      Value |= (ImmOffs & ~0xF); // immedL
    }

    break;
  }
  case ARMII::AddrMode4: {
    // bit 27 is always 1
    Value |= 1 << 27;

    unsigned Format = (Desc->TSFlags & ARMII::FormMask);
    // if it is a load instr, set bit L(20) to 1
    if (Format == ARMII::LdFrm)
      Value |= 1 << ARMII::L_BitShift;

    unsigned OpIndex = 0;

    // set first operand
    op = getMachineOpValue(MI,OpIndex);
    Value |= op << ARMII::RegRnShift;

    // set addressing mode by modifying bits U(23) and P(24)
    // IA - Increment after  - bit U = 1 and bit P = 0
    // IB - Increment before - bit U = 1 and bit P = 1
    // DA - Decrement after  - bit U = 0 and bit P = 0
    // DB - Decrement before - bit U = 0 and bit P = 1
    const MachineOperand &MO = MI.getOperand(OpIndex + 1);
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO.getImm());
    switch(Mode) {
    default: assert(0 && "Unknown addressing sub-mode!");
    case ARM_AM::ia: Value |= 0x1 << 23; break;
    case ARM_AM::ib: Value |= 0x3 << 23; break;
    case ARM_AM::da: break;
    case ARM_AM::db: Value |= 0x1 << 24; break;
    }

    // set bit W(21)
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
