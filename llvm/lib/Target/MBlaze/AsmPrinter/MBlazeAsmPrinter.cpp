//===-- MBlazeAsmPrinter.cpp - MBlaze LLVM assembly writer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format MBlaze assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mblaze-asm-printer"

#include "MBlaze.h"
#include "MBlazeSubtarget.h"
#include "MBlazeInstrInfo.h"
#include "MBlazeTargetMachine.h"
#include "MBlazeMachineFunction.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>

using namespace llvm;

namespace {
  class MBlazeAsmPrinter : public AsmPrinter {
    const MBlazeSubtarget *Subtarget;
  public:
    explicit MBlazeAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer) {
      Subtarget = &TM.getSubtarget<MBlazeSubtarget>();
    }

    virtual const char *getPassName() const {
      return "MBlaze Assembly Printer";
    }

    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode,
                         raw_ostream &O);
    void printOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
    void printUnsignedImm(const MachineInstr *MI, int opNum, raw_ostream &O);
    void printFSLImm(const MachineInstr *MI, int opNum, raw_ostream &O);
    void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                         const char *Modifier = 0);
    void printFCCOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                         const char *Modifier = 0);
    void printSavedRegsBitmask(raw_ostream &OS);

    const char *emitCurrentABIString();
    void emitFrameDirective();

    void printInstruction(const MachineInstr *MI, raw_ostream &O);
    void EmitInstruction(const MachineInstr *MI) { 
      SmallString<128> Str;
      raw_svector_ostream OS(Str);
      printInstruction(MI, OS);
      OutStreamer.EmitRawText(OS.str());
    }
    virtual void EmitFunctionBodyStart();
    virtual void EmitFunctionBodyEnd();
    static const char *getRegisterName(unsigned RegNo);

    virtual void EmitFunctionEntryLabel();
  };
} // end of anonymous namespace

#include "MBlazeGenAsmWriter.inc"

//===----------------------------------------------------------------------===//
//
//  MBlaze Asm Directives
//
//  -- Frame directive "frame Stackpointer, Stacksize, RARegister"
//  Describe the stack frame.
//
//  -- Mask directives "mask  bitmask, offset"
//  Tells the assembler which registers are saved and where.
//  bitmask - contain a little endian bitset indicating which registers are
//            saved on function prologue (e.g. with a 0x80000000 mask, the
//            assembler knows the register 31 (RA) is saved at prologue.
//  offset  - the position before stack pointer subtraction indicating where
//            the first saved register on prologue is located. (e.g. with a
//
//  Consider the following function prologue:
//
//    .frame  R19,48,R15
//    .mask   0xc0000000,-8
//       addiu R1, R1, -48
//       sw R15, 40(R1)
//       sw R19, 36(R1)
//
//    With a 0xc0000000 mask, the assembler knows the register 15 (R15) and
//    19 (R19) are saved at prologue. As the save order on prologue is from
//    left to right, R15 is saved first. A -8 offset means that after the
//    stack pointer subtration, the first register in the mask (R15) will be
//    saved at address 48-8=40.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mask directives
//===----------------------------------------------------------------------===//

// Print a 32 bit hex number with all numbers.
static void printHex32(unsigned int Value, raw_ostream &O) {
  O << "0x";
  for (int i = 7; i >= 0; i--)
    O << utohexstr((Value & (0xF << (i*4))) >> (i*4));
}


// Create a bitmask with all callee saved registers for CPU or Floating Point
// registers. For CPU registers consider RA, GP and FP for saving if necessary.
void MBlazeAsmPrinter::printSavedRegsBitmask(raw_ostream &O) {
  const TargetRegisterInfo &RI = *TM.getRegisterInfo();
  const MBlazeFunctionInfo *MBlazeFI = MF->getInfo<MBlazeFunctionInfo>();

  // CPU Saved Registers Bitmasks
  unsigned int CPUBitmask = 0;

  // Set the CPU Bitmasks
  const MachineFrameInfo *MFI = MF->getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned RegNum = MBlazeRegisterInfo::getRegisterNumbering(CSI[i].getReg());
    if (CSI[i].getRegClass() == MBlaze::CPURegsRegisterClass)
      CPUBitmask |= (1 << RegNum);
  }

  // Return Address and Frame registers must also be set in CPUBitmask.
  if (RI.hasFP(*MF))
    CPUBitmask |= (1 << MBlazeRegisterInfo::
                getRegisterNumbering(RI.getFrameRegister(*MF)));

  if (MFI->hasCalls())
    CPUBitmask |= (1 << MBlazeRegisterInfo::
                getRegisterNumbering(RI.getRARegister()));

  // Print CPUBitmask
  O << "\t.mask \t"; printHex32(CPUBitmask, O);
  O << ',' << MBlazeFI->getCPUTopSavedRegOff() << '\n';
}

//===----------------------------------------------------------------------===//
// Frame and Set directives
//===----------------------------------------------------------------------===//

/// Frame Directive
void MBlazeAsmPrinter::emitFrameDirective() {
  const TargetRegisterInfo &RI = *TM.getRegisterInfo();

  unsigned stackReg  = RI.getFrameRegister(*MF);
  unsigned returnReg = RI.getRARegister();
  unsigned stackSize = MF->getFrameInfo()->getStackSize();


  OutStreamer.EmitRawText("\t.frame\t" + Twine(getRegisterName(stackReg)) +
                          "," + Twine(stackSize) + "," +
                          Twine(getRegisterName(returnReg)));
}

void MBlazeAsmPrinter::EmitFunctionEntryLabel() {
  OutStreamer.EmitRawText("\t.ent\t" + Twine(CurrentFnSym->getName()));
  OutStreamer.EmitLabel(CurrentFnSym);
}

/// EmitFunctionBodyStart - Targets can override this to emit stuff before
/// the first basic block in the function.
void MBlazeAsmPrinter::EmitFunctionBodyStart() {
  emitFrameDirective();
  
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  printSavedRegsBitmask(OS);
  OutStreamer.EmitRawText(OS.str());
}

/// EmitFunctionBodyEnd - Targets can override this to emit stuff after
/// the last basic block in the function.
void MBlazeAsmPrinter::EmitFunctionBodyEnd() {
  OutStreamer.EmitRawText("\t.end\t" + Twine(CurrentFnSym->getName()));
}

// Print out an operand for an inline asm expression.
bool MBlazeAsmPrinter::
PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                unsigned AsmVariant,const char *ExtraCode, raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.

  printOperand(MI, OpNo, O);
  return false;
}

void MBlazeAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                    raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << getRegisterName(MO.getReg());
    break;

  case MachineOperand::MO_Immediate:
    O << (int)MO.getImm();
    break;

  case MachineOperand::MO_FPImmediate: {
    const ConstantFP *fp = MO.getFPImm();
    printHex32(fp->getValueAPF().bitcastToAPInt().getZExtValue(), O);
    O << ";\t# immediate = " << *fp;
    break;
  }

  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;

  case MachineOperand::MO_GlobalAddress:
    O << *Mang->getSymbol(MO.getGlobal());
    break;

  case MachineOperand::MO_ExternalSymbol:
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    break;

  case MachineOperand::MO_JumpTableIndex:
    O << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
    << '_' << MO.getIndex();
    break;

  case MachineOperand::MO_ConstantPoolIndex:
    O << MAI->getPrivateGlobalPrefix() << "CPI"
      << getFunctionNumber() << "_" << MO.getIndex();
    if (MO.getOffset())
      O << "+" << MO.getOffset();
    break;

  default:
    llvm_unreachable("<unknown operand type>");
  }
}

void MBlazeAsmPrinter::printUnsignedImm(const MachineInstr *MI, int opNum,
                                        raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.getType() == MachineOperand::MO_Immediate)
    O << (unsigned int)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void MBlazeAsmPrinter::printFSLImm(const MachineInstr *MI, int opNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.getType() == MachineOperand::MO_Immediate)
    O << "rfsl" << (unsigned int)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void MBlazeAsmPrinter::
printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                const char *Modifier) {
  printOperand(MI, opNum+1, O);
  O << ", ";
  printOperand(MI, opNum, O);
}

void MBlazeAsmPrinter::
printFCCOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                const char *Modifier) {
  const MachineOperand& MO = MI->getOperand(opNum);
  O << MBlaze::MBlazeFCCToString((MBlaze::CondCode)MO.getImm());
}

// Force static initialization.
extern "C" void LLVMInitializeMBlazeAsmPrinter() {
  RegisterAsmPrinter<MBlazeAsmPrinter> X(TheMBlazeTarget);
}
