//===-- SystemZAsmPrinter.cpp - SystemZ LLVM assembly writer ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the SystemZ assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "SystemZ.h"
#include "SystemZInstrInfo.h"
#include "SystemZTargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  class SystemZAsmPrinter : public AsmPrinter {
  public:
    SystemZAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer) {}

    virtual const char *getPassName() const {
      return "SystemZ Assembly Printer";
    }

    void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                      const char* Modifier = 0);
    void printPCRelImmOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
    void printRIAddrOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                            const char* Modifier = 0);
    void printRRIAddrOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                             const char* Modifier = 0);
    void printS16ImmOperand(const MachineInstr *MI, int OpNum, raw_ostream &O) {
      O << (int16_t)MI->getOperand(OpNum).getImm();
    }
    void printU16ImmOperand(const MachineInstr *MI, int OpNum, raw_ostream &O) {
      O << (uint16_t)MI->getOperand(OpNum).getImm();
    }
    void printS32ImmOperand(const MachineInstr *MI, int OpNum, raw_ostream &O) {
      O << (int32_t)MI->getOperand(OpNum).getImm();
    }
    void printU32ImmOperand(const MachineInstr *MI, int OpNum, raw_ostream &O) {
      O << (uint32_t)MI->getOperand(OpNum).getImm();
    }

    void printInstruction(const MachineInstr *MI, raw_ostream &O);
    static const char *getRegisterName(unsigned RegNo);

    void EmitInstruction(const MachineInstr *MI);
  };
} // end of anonymous namespace

#include "SystemZGenAsmWriter.inc"

void SystemZAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  printInstruction(MI, OS);
  OutStreamer.EmitRawText(OS.str());
}

void SystemZAsmPrinter::printPCRelImmOperand(const MachineInstr *MI, int OpNum,
                                             raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    O << *Mang->getSymbol(GV);

    // Assemble calls via PLT for externally visible symbols if PIC.
    if (TM.getRelocationModel() == Reloc::PIC_ &&
        !GV->hasHiddenVisibility() && !GV->hasProtectedVisibility() &&
        !GV->hasLocalLinkage())
      O << "@PLT";

    printOffset(MO.getOffset(), O);
    return;
  }
  case MachineOperand::MO_ExternalSymbol: {
    std::string Name(MAI->getGlobalPrefix());
    Name += MO.getSymbolName();
    O << Name;

    if (TM.getRelocationModel() == Reloc::PIC_)
      O << "@PLT";

    return;
  }
  default:
    assert(0 && "Not implemented yet!");
  }
}


void SystemZAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                     raw_ostream &O, const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    assert (TargetRegisterInfo::isPhysicalRegister(MO.getReg()) &&
            "Virtual registers should be already mapped!");
    unsigned Reg = MO.getReg();
    if (Modifier && strncmp(Modifier, "subreg", 6) == 0) {
      if (strncmp(Modifier + 7, "even", 4) == 0)
        Reg = TM.getRegisterInfo()->getSubReg(Reg, SystemZ::subreg_32bit);
      else if (strncmp(Modifier + 7, "odd", 3) == 0)
        Reg = TM.getRegisterInfo()->getSubReg(Reg, SystemZ::subreg_odd32);
      else
        assert(0 && "Invalid subreg modifier");
    }

    O << '%' << getRegisterName(Reg);
    return;
  }
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_JumpTableIndex:
    O << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() << '_'
      << MO.getIndex();

    return;
  case MachineOperand::MO_ConstantPoolIndex:
    O << MAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
      << MO.getIndex();

    printOffset(MO.getOffset(), O);
    break;
  case MachineOperand::MO_GlobalAddress:
    O << *Mang->getSymbol(MO.getGlobal());
    break;
  case MachineOperand::MO_ExternalSymbol: {
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    break;
  }
  default:
    assert(0 && "Not implemented yet!");
  }

  switch (MO.getTargetFlags()) {
  default: assert(0 && "Unknown target flag on GV operand");
  case SystemZII::MO_NO_FLAG:
    break;
  case SystemZII::MO_GOTENT:    O << "@GOTENT";    break;
  case SystemZII::MO_PLT:       O << "@PLT";       break;
  }

  printOffset(MO.getOffset(), O);
}

void SystemZAsmPrinter::printRIAddrOperand(const MachineInstr *MI, int OpNum,
                                           raw_ostream &O,
                                           const char *Modifier) {
  const MachineOperand &Base = MI->getOperand(OpNum);

  // Print displacement operand.
  printOperand(MI, OpNum+1, O);

  // Print base operand (if any)
  if (Base.getReg()) {
    O << '(';
    printOperand(MI, OpNum, O);
    O << ')';
  }
}

void SystemZAsmPrinter::printRRIAddrOperand(const MachineInstr *MI, int OpNum,
                                            raw_ostream &O,
                                            const char *Modifier) {
  const MachineOperand &Base = MI->getOperand(OpNum);
  const MachineOperand &Index = MI->getOperand(OpNum+2);

  // Print displacement operand.
  printOperand(MI, OpNum+1, O);

  // Print base operand (if any)
  if (Base.getReg()) {
    O << '(';
    printOperand(MI, OpNum, O);
    if (Index.getReg()) {
      O << ',';
      printOperand(MI, OpNum+2, O);
    }
    O << ')';
  } else
    assert(!Index.getReg() && "Should allocate base register first!");
}

// Force static initialization.
extern "C" void LLVMInitializeSystemZAsmPrinter() {
  RegisterAsmPrinter<SystemZAsmPrinter> X(TheSystemZTarget);
}
