//===-- X86/X86CodeEmitter.cpp - Convert X86 code to machine code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the pass that transforms the X86 machine instructions into
// relocatable machine code.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "X86Relocations.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Visibility.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<>
  NumEmitted("x86-emitter", "Number of machine instructions emitted");
}

namespace {
  class VISIBILITY_HIDDEN Emitter : public MachineFunctionPass {
    const X86InstrInfo  *II;
    TargetMachine &TM;
    MachineCodeEmitter  &MCE;
  public:
    explicit Emitter(TargetMachine &tm, MachineCodeEmitter &mce)
      : II(0), TM(tm), MCE(mce) {}
    Emitter(TargetMachine &tm, MachineCodeEmitter &mce,
            const X86InstrInfo& ii)
      : II(&ii), TM(tm), MCE(mce) {}

    bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "X86 Machine Code Emitter";
    }

    void emitInstruction(const MachineInstr &MI);

  private:
    void emitPCRelativeBlockAddress(MachineBasicBlock *MBB);
    void emitPCRelativeValue(unsigned Address);
    void emitGlobalAddressForCall(GlobalValue *GV, bool isTailCall);
    void emitGlobalAddressForPtr(GlobalValue *GV, int Disp = 0);
    void emitExternalSymbolAddress(const char *ES, bool isPCRelative);

    void emitDisplacementField(const MachineOperand *RelocOp, int DispVal);

    void emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeField);
    void emitSIBByte(unsigned SS, unsigned Index, unsigned Base);
    void emitConstant(unsigned Val, unsigned Size);

    void emitMemModRMByte(const MachineInstr &MI,
                          unsigned Op, unsigned RegOpcodeField);

  };
}

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *llvm::createX86CodeEmitterPass(X86TargetMachine &TM,
                                             MachineCodeEmitter &MCE) {
  return new Emitter(TM, MCE);
}

bool Emitter::runOnMachineFunction(MachineFunction &MF) {
  assert((MF.getTarget().getRelocationModel() != Reloc::Default ||
          MF.getTarget().getRelocationModel() != Reloc::Static) &&
         "JIT relocation model must be set to static or default!");
  II = ((X86TargetMachine&)MF.getTarget()).getInstrInfo();

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

/// emitPCRelativeValue - Emit a 32-bit PC relative address.
///
void Emitter::emitPCRelativeValue(unsigned Address) {
  MCE.emitWordLE(Address-MCE.getCurrentPCValue()-4);
}

/// emitPCRelativeBlockAddress - This method keeps track of the information
/// necessary to resolve the address of this block later and emits a dummy
/// value.
///
void Emitter::emitPCRelativeBlockAddress(MachineBasicBlock *MBB) {
  // Remember where this reference was and where it is to so we can
  // deal with it later.
  TM.getJITInfo()->addBBRef(MBB, MCE.getCurrentPCValue());
  MCE.emitWordLE(0);
}

/// emitGlobalAddressForCall - Emit the specified address to the code stream
/// assuming this is part of a function call, which is PC relative.
///
void Emitter::emitGlobalAddressForCall(GlobalValue *GV, bool isTailCall) {
  MCE.addRelocation(MachineRelocation::getGV(MCE.getCurrentPCOffset(),
                                      X86::reloc_pcrel_word, GV, 0,
                                      !isTailCall /*Doesn'tNeedStub*/));
  MCE.emitWordLE(0);
}

/// emitGlobalAddress - Emit the specified address to the code stream assuming
/// this is part of a "take the address of a global" instruction, which is not
/// PC relative.
///
void Emitter::emitGlobalAddressForPtr(GlobalValue *GV, int Disp /* = 0 */) {
  MCE.addRelocation(MachineRelocation::getGV(MCE.getCurrentPCOffset(),
                                      X86::reloc_absolute_word, GV));
  MCE.emitWordLE(Disp); // The relocated value will be added to the displacement
}

/// emitExternalSymbolAddress - Arrange for the address of an external symbol to
/// be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitExternalSymbolAddress(const char *ES, bool isPCRelative) {
  MCE.addRelocation(MachineRelocation::getExtSym(MCE.getCurrentPCOffset(),
          isPCRelative ? X86::reloc_pcrel_word : X86::reloc_absolute_word, ES));
  MCE.emitWordLE(0);
}

/// N86 namespace - Native X86 Register numbers... used by X86 backend.
///
namespace N86 {
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

  case X86::ST0: case X86::ST1: case X86::ST2: case X86::ST3:
  case X86::ST4: case X86::ST5: case X86::ST6: case X86::ST7:
    return RegNo-X86::ST0;

  case X86::XMM0: case X86::XMM1: case X86::XMM2: case X86::XMM3:
  case X86::XMM4: case X86::XMM5: case X86::XMM6: case X86::XMM7:
    return RegNo-X86::XMM0;

  default:
    assert(MRegisterInfo::isVirtualRegister(RegNo) &&
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

/// isDisp8 - Return true if this signed displacement fits in a 8-bit 
/// sign-extended field. 
static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

void Emitter::emitDisplacementField(const MachineOperand *RelocOp,
                                    int DispVal) {
  // If this is a simple integer displacement that doesn't require a relocation,
  // emit it now.
  if (!RelocOp) {
    emitConstant(DispVal, 4);
    return;
  }
  
  // Otherwise, this is something that requires a relocation.  Emit it as such
  // now.
  if (RelocOp->isGlobalAddress()) {
    emitGlobalAddressForPtr(RelocOp->getGlobal(), RelocOp->getOffset());
  } else {
    assert(0 && "Unknown value to relocate!");
  }
}

void Emitter::emitMemModRMByte(const MachineInstr &MI,
                               unsigned Op, unsigned RegOpcodeField) {
  const MachineOperand &Op3 = MI.getOperand(Op+3);
  int DispVal = 0;
  const MachineOperand *DispForReloc = 0;
  
  // Figure out what sort of displacement we have to handle here.
  if (Op3.isGlobalAddress()) {
    DispForReloc = &Op3;
  } else if (Op3.isConstantPoolIndex()) {
    DispVal += MCE.getConstantPoolEntryAddress(Op3.getConstantPoolIndex());
    DispVal += Op3.getOffset();
  } else if (Op3.isJumpTableIndex()) {
    DispVal += MCE.getJumpTableEntryAddress(Op3.getJumpTableIndex());
  } else {
    DispVal = Op3.getImmedValue();
  }

  const MachineOperand &Base     = MI.getOperand(Op);
  const MachineOperand &Scale    = MI.getOperand(Op+1);
  const MachineOperand &IndexReg = MI.getOperand(Op+2);

  unsigned BaseReg = Base.getReg();

  // Is a SIB byte needed?
  if (IndexReg.getReg() == 0 && BaseReg != X86::ESP) {
    if (BaseReg == 0) {  // Just a displacement?
      // Emit special case [disp32] encoding
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      
      emitDisplacementField(DispForReloc, DispVal);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg);
      if (!DispForReloc && DispVal == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        MCE.emitByte(ModRMByte(0, RegOpcodeField, BaseRegNo));
      } else if (!DispForReloc && isDisp8(DispVal)) {
        // Emit the disp8 encoding... [REG+disp8]
        MCE.emitByte(ModRMByte(1, RegOpcodeField, BaseRegNo));
        emitConstant(DispVal, 1);
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        MCE.emitByte(ModRMByte(2, RegOpcodeField, BaseRegNo));
        emitDisplacementField(DispForReloc, DispVal);
      }
    }

  } else {  // We need a SIB byte, so start by outputting the ModR/M byte first
    assert(IndexReg.getReg() != X86::ESP && "Cannot use ESP as index reg!");

    bool ForceDisp32 = false;
    bool ForceDisp8  = false;
    if (BaseReg == 0) {
      // If there is no base register, we emit the special case SIB byte with
      // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (DispForReloc) {
      // Emit the normal disp32 encoding.
      MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (DispVal == 0 && BaseReg != X86::EBP) {
      // Emit no displacement ModR/M byte
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
    } else if (isDisp8(DispVal)) {
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

    if (BaseReg == 0) {
      // Handle the SIB byte for the case where there is no base.  The
      // displacement has already been output.
      assert(IndexReg.getReg() && "Index register must be specified!");
      emitSIBByte(SS, getX86RegNum(IndexReg.getReg()), 5);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg);
      unsigned IndexRegNo;
      if (IndexReg.getReg())
        IndexRegNo = getX86RegNum(IndexReg.getReg());
      else
        IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
      emitSIBByte(SS, IndexRegNo, BaseRegNo);
    }

    // Do we need to output a displacement?
    if (ForceDisp8) {
      emitConstant(DispVal, 1);
    } else if (DispVal != 0 || ForceDisp32) {
      emitDisplacementField(DispForReloc, DispVal);
    }
  }
}

static unsigned sizeOfImm(const TargetInstrDescriptor &Desc) {
  switch (Desc.TSFlags & X86II::ImmMask) {
  case X86II::Imm8:   return 1;
  case X86II::Imm16:  return 2;
  case X86II::Imm32:  return 4;
  default: assert(0 && "Immediate size not set!");
    return 0;
  }
}

void Emitter::emitInstruction(const MachineInstr &MI) {
  NumEmitted++;  // Keep track of the # of mi's emitted

  unsigned Opcode = MI.getOpcode();
  const TargetInstrDescriptor &Desc = II->get(Opcode);

  // Emit the repeat opcode prefix as needed.
  if ((Desc.TSFlags & X86II::Op0Mask) == X86II::REP) MCE.emitByte(0xF3);

  // Emit the operand size opcode prefix as needed.
  if (Desc.TSFlags & X86II::OpSize) MCE.emitByte(0x66);

  switch (Desc.TSFlags & X86II::Op0Mask) {
  case X86II::TB:
    MCE.emitByte(0x0F);   // Two-byte opcode prefix
    break;
  case X86II::REP: break; // already handled.
  case X86II::XS:   // F3 0F
    MCE.emitByte(0xF3);
    MCE.emitByte(0x0F);
    break;
  case X86II::XD:   // F2 0F
    MCE.emitByte(0xF2);
    MCE.emitByte(0x0F);
    break;
  case X86II::D8: case X86II::D9: case X86II::DA: case X86II::DB:
  case X86II::DC: case X86II::DD: case X86II::DE: case X86II::DF:
    MCE.emitByte(0xD8+
                 (((Desc.TSFlags & X86II::Op0Mask)-X86II::D8)
                                   >> X86II::Op0Shift));
    break; // Two-byte opcode prefix
  default: assert(0 && "Invalid prefix!");
  case 0: break;  // No prefix!
  }

  unsigned char BaseOpcode = II->getBaseOpcodeFor(Opcode);
  switch (Desc.TSFlags & X86II::FormMask) {
  default: assert(0 && "Unknown FormMask value in X86 MachineCodeEmitter!");
  case X86II::Pseudo:
#ifndef NDEBUG
    switch (Opcode) {
    default: 
      assert(0 && "psuedo instructions should be removed before code emission");
    case X86::IMPLICIT_USE:
    case X86::IMPLICIT_DEF:
    case X86::IMPLICIT_DEF_GR8:
    case X86::IMPLICIT_DEF_GR16:
    case X86::IMPLICIT_DEF_GR32:
    case X86::IMPLICIT_DEF_FR32:
    case X86::IMPLICIT_DEF_FR64:
    case X86::IMPLICIT_DEF_VR64:
    case X86::IMPLICIT_DEF_VR128:
    case X86::FP_REG_KILL:
      break;
    }
#endif
    break;

  case X86II::RawFrm:
    MCE.emitByte(BaseOpcode);
    if (Desc.numOperands == 1) {
      const MachineOperand &MO = MI.getOperand(0);
      if (MO.isMachineBasicBlock()) {
        emitPCRelativeBlockAddress(MO.getMachineBasicBlock());
      } else if (MO.isGlobalAddress()) {
        bool isTailCall = Opcode == X86::TAILJMPd ||
                          Opcode == X86::TAILJMPr || Opcode == X86::TAILJMPm;
        emitGlobalAddressForCall(MO.getGlobal(), isTailCall);
      } else if (MO.isExternalSymbol()) {
        emitExternalSymbolAddress(MO.getSymbolName(), true);
      } else if (MO.isImmediate()) {
        emitConstant(MO.getImmedValue(), sizeOfImm(Desc));
      } else {
        assert(0 && "Unknown RawFrm operand!");
      }
    }
    break;

  case X86II::AddRegFrm:
    MCE.emitByte(BaseOpcode + getX86RegNum(MI.getOperand(0).getReg()));
    if (MI.getNumOperands() == 2) {
      const MachineOperand &MO1 = MI.getOperand(1);
      if (MO1.isGlobalAddress()) {
        assert(sizeOfImm(Desc) == 4 &&
               "Don't know how to emit non-pointer values!");
        emitGlobalAddressForPtr(MO1.getGlobal(), MO1.getOffset());
      } else if (MO1.isExternalSymbol()) {
        assert(sizeOfImm(Desc) == 4 &&
               "Don't know how to emit non-pointer values!");
        emitExternalSymbolAddress(MO1.getSymbolName(), false);
      } else if (MO1.isJumpTableIndex()) {
        assert(sizeOfImm(Desc) == 4 &&
               "Don't know how to emit non-pointer values!");
        emitConstant(MCE.getJumpTableEntryAddress(MO1.getJumpTableIndex()), 4);
      } else {
        emitConstant(MO1.getImmedValue(), sizeOfImm(Desc));
      }
    }
    break;

  case X86II::MRMDestReg: {
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(0).getReg(),
                     getX86RegNum(MI.getOperand(1).getReg()));
    if (MI.getNumOperands() == 3)
      emitConstant(MI.getOperand(2).getImmedValue(), sizeOfImm(Desc));
    break;
  }
  case X86II::MRMDestMem:
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, 0, getX86RegNum(MI.getOperand(4).getReg()));
    if (MI.getNumOperands() == 6)
      emitConstant(MI.getOperand(5).getImmedValue(), sizeOfImm(Desc));
    break;

  case X86II::MRMSrcReg:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(1).getReg(),
                     getX86RegNum(MI.getOperand(0).getReg()));
    if (MI.getNumOperands() == 3)
      emitConstant(MI.getOperand(2).getImmedValue(), sizeOfImm(Desc));
    break;

  case X86II::MRMSrcMem:
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, 1, getX86RegNum(MI.getOperand(0).getReg()));
    if (MI.getNumOperands() == 2+4)
      emitConstant(MI.getOperand(5).getImmedValue(), sizeOfImm(Desc));
    break;

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(0).getReg(),
                     (Desc.TSFlags & X86II::FormMask)-X86II::MRM0r);

    if (MI.getOperand(MI.getNumOperands()-1).isImmediate()) {
      emitConstant(MI.getOperand(MI.getNumOperands()-1).getImmedValue(),
                   sizeOfImm(Desc));
    }
    break;

  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m:
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, 0, (Desc.TSFlags & X86II::FormMask)-X86II::MRM0m);

    if (MI.getNumOperands() == 5) {
      if (MI.getOperand(4).isImmediate())
        emitConstant(MI.getOperand(4).getImmedValue(), sizeOfImm(Desc));
      else if (MI.getOperand(4).isGlobalAddress())
        emitGlobalAddressForPtr(MI.getOperand(4).getGlobal(),
                                MI.getOperand(4).getOffset());
      else if (MI.getOperand(4).isJumpTableIndex())
        emitConstant(MCE.getJumpTableEntryAddress(MI.getOperand(4)
                                                    .getJumpTableIndex()), 4);
      else
        assert(0 && "Unknown operand!");
    }
    break;

  case X86II::MRMInitReg:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(0).getReg(),
                     getX86RegNum(MI.getOperand(0).getReg()));
    break;
  }
}
