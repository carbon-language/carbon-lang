//===-- X86/MachineCodeEmitter.cpp - Convert X86 code to machine code -----===//
//
// This file contains the pass that transforms the X86 machine instructions into
// actual executable machine code.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Value.h"

namespace {
  class Emitter : public FunctionPass {
    X86TargetMachine    &TM;
    const X86InstrInfo  &II;
    MachineCodeEmitter  &MCE;
  public:

    Emitter(X86TargetMachine &tm, MachineCodeEmitter &mce)
      : TM(tm), II(TM.getInstrInfo()), MCE(mce) {}

    bool runOnFunction(Function &F);

  private:
    void emitBasicBlock(MachineBasicBlock &MBB);
    void emitInstruction(MachineInstr &MI);

    void emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeField);
    void emitSIBByte(unsigned SS, unsigned Index, unsigned Base);
    void emitConstant(unsigned Val, unsigned Size);

    void emitMemModRMByte(const MachineInstr &MI,
                          unsigned Op, unsigned RegOpcodeField);

  };
}


/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MAchineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool X86TargetMachine::addPassesToEmitMachineCode(PassManager &PM,
                                                  MachineCodeEmitter &MCE) {
  PM.add(new Emitter(*this, MCE));
  return false;
}

bool Emitter::runOnFunction(Function &F) {
  MachineFunction &MF = MachineFunction::get(&F);

  MCE.startFunction(MF);
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);
  return false;
}

void Emitter::emitBasicBlock(MachineBasicBlock &MBB) {
  MCE.startBasicBlock(MBB);
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    emitInstruction(**I);
}


namespace N86 {  // Native X86 Register numbers...
  enum {
    EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7
  };
}


// getX86RegNum - This function maps LLVM register identifiers to their X86
// specific numbering, which is used in various places encoding instructions.
//
static unsigned getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::ESP: case X86::SP: case X86::AH: return N86::ESP;
  case X86::EBP: case X86::BP: case X86::CH: return N86::EBP;
  case X86::ESI: case X86::SI: case X86::DH: return N86::ESI;
  case X86::EDI: case X86::DI: case X86::BH: return N86::EDI;
  default:
    assert(RegNo >= MRegisterInfo::FirstVirtualRegister &&
           "Unknown physical register!");
    assert(0 && "Register allocator hasn't allocated reg correctly yet!");
    return 0;
  }
}

inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                      unsigned RM) {
  assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
  return RM | (RegOpcode << 3) | (Mod << 6);
}

void Emitter::emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeFld){
  MCE.emitByte(ModRMByte(3, RegOpcodeFld, getX86RegNum(ModRMReg)));
}

void Emitter::emitSIBByte(unsigned SS, unsigned Index, unsigned Base) {
  // SIB byte is in the same format as the ModRMByte...
  MCE.emitByte(ModRMByte(SS, Index, Base));
}

void Emitter::emitConstant(unsigned Val, unsigned Size) {
  // Output the constant in little endian byte order...
  for (unsigned i = 0; i != Size; ++i) {
    MCE.emitByte(Val & 255);
    Val >>= 8;
  }
}

static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

void Emitter::emitMemModRMByte(const MachineInstr &MI,
                               unsigned Op, unsigned RegOpcodeField) {
  const MachineOperand &BaseReg  = MI.getOperand(Op);
  const MachineOperand &Scale    = MI.getOperand(Op+1);
  const MachineOperand &IndexReg = MI.getOperand(Op+2);
  const MachineOperand &Disp     = MI.getOperand(Op+3);

  // Is a SIB byte needed?
  if (IndexReg.getReg() == 0 && BaseReg.getReg() != X86::ESP) {
    if (BaseReg.getReg() == 0) {  // Just a displacement?
      // Emit special case [disp32] encoding
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      emitConstant(Disp.getImmedValue(), 4);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg.getReg());
      if (Disp.getImmedValue() == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        MCE.emitByte(ModRMByte(0, RegOpcodeField, BaseRegNo));
      } else if (isDisp8(Disp.getImmedValue())) {
        // Emit the disp8 encoding... [REG+disp8]
        MCE.emitByte(ModRMByte(1, RegOpcodeField, BaseRegNo));
        emitConstant(Disp.getImmedValue(), 1);
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        MCE.emitByte(ModRMByte(2, RegOpcodeField, BaseRegNo));
        emitConstant(Disp.getImmedValue(), 4);
      }
    }

  } else {  // We need a SIB byte, so start by outputting the ModR/M byte first
    assert(IndexReg.getReg() != X86::ESP && "Cannot use ESP as index reg!");

    bool ForceDisp32 = false;
    bool ForceDisp8  = false;
    if (BaseReg.getReg() == 0) {
      // If there is no base register, we emit the special case SIB byte with
      // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (Disp.getImmedValue() == 0 && BaseReg.getReg() != X86::EBP) {
      // Emit no displacement ModR/M byte
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
    } else if (isDisp8(Disp.getImmedValue())) {
      // Emit the disp8 encoding...
      MCE.emitByte(ModRMByte(1, RegOpcodeField, 4));
      ForceDisp8 = true;           // Make sure to force 8 bit disp if Base=EBP
    } else {
      // Emit the normal disp32 encoding...
      MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
    }

    // Calculate what the SS field value should be...
    static const unsigned SSTable[] = { ~0, 0, 1, ~0, 2, ~0, ~0, ~0, 3 };
    unsigned SS = SSTable[Scale.getImmedValue()];

    if (BaseReg.getReg() == 0) {
      // Handle the SIB byte for the case where there is no base.  The
      // displacement has already been output.
      assert(IndexReg.getReg() && "Index register must be specified!");
      emitSIBByte(SS, getX86RegNum(IndexReg.getReg()), 5);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg.getReg());
      unsigned IndexRegNo = getX86RegNum(IndexReg.getReg());
      emitSIBByte(SS, IndexRegNo, BaseRegNo);
    }

    // Do we need to output a displacement?
    if (Disp.getImmedValue() != 0 || ForceDisp32 || ForceDisp8) {
      if (!ForceDisp32 && isDisp8(Disp.getImmedValue()))
        emitConstant(Disp.getImmedValue(), 1);
      else
        emitConstant(Disp.getImmedValue(), 4);
    }
  }
}

unsigned sizeOfPtr (const MachineInstrDescriptor &Desc) {
  switch (Desc.TSFlags & X86II::ArgMask) {
  case X86II::Arg8:   return 1;
  case X86II::Arg16:  return 2;
  case X86II::Arg32:  return 4;
  case X86II::Arg64:  return 8;
  case X86II::Arg80:  return 10;
  case X86II::Arg128: return 16;
  default: assert(0 && "Memory size not set!");
  }
}


void Emitter::emitInstruction(MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  const MachineInstrDescriptor &Desc = II.get(Opcode);

  // Emit instruction prefixes if neccesary
  if (Desc.TSFlags & X86II::OpSize) MCE.emitByte(0x66);// Operand size...
  if (Desc.TSFlags & X86II::TB)     MCE.emitByte(0x0F);// Two-byte opcode prefix

  unsigned char BaseOpcode = II.getBaseOpcodeFor(Opcode);
  switch (Desc.TSFlags & X86II::FormMask) {
  case X86II::RawFrm:
    MCE.emitByte(BaseOpcode);

    if (MI.getNumOperands() == 1) {
      assert(MI.getOperand(0).getType() == MachineOperand::MO_PCRelativeDisp);
      MCE.emitPCRelativeDisp(MI.getOperand(0).getVRegValue());
    }
    break;
  case X86II::AddRegFrm:
    MCE.emitByte(BaseOpcode + getX86RegNum(MI.getOperand(0).getReg()));
    if (MI.getNumOperands() == 2) {
      unsigned Size = sizeOfPtr(Desc);
      if (Value *V = MI.getOperand(1).getVRegValueOrNull()) {
        assert(Size == 4 && "Don't know how to emit non-pointer values!");
        MCE.emitGlobalAddress(cast<GlobalValue>(V));
      } else {
        emitConstant(MI.getOperand(1).getImmedValue(), Size);
      }
    }
    break;
  case X86II::MRMDestReg:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(0).getReg(),
               getX86RegNum(MI.getOperand(MI.getNumOperands()-1).getReg()));
    break;    
  case X86II::MRMDestMem:
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, 0, getX86RegNum(MI.getOperand(4).getReg()));
    break;
  case X86II::MRMSrcReg:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(MI.getNumOperands()-1).getReg(),
                     getX86RegNum(MI.getOperand(0).getReg()));
    break;
  case X86II::MRMSrcMem:
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, MI.getNumOperands()-4,
                     getX86RegNum(MI.getOperand(0).getReg()));
    break;

  case X86II::MRMS0r: case X86II::MRMS1r:
  case X86II::MRMS2r: case X86II::MRMS3r:
  case X86II::MRMS4r: case X86II::MRMS5r:
  case X86II::MRMS6r: case X86II::MRMS7r:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(0).getReg(),
                     (Desc.TSFlags & X86II::FormMask)-X86II::MRMS0r);

    if (MI.getOperand(MI.getNumOperands()-1).isImmediate()) {
      unsigned Size = sizeOfPtr(Desc);
      emitConstant(MI.getOperand(MI.getNumOperands()-1).getImmedValue(), Size);
    }
    break;
  }
}
