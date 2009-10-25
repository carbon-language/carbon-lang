//===-- AlphaAsmPrinter.cpp - Alpha LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format Alpha assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "Alpha.h"
#include "AlphaInstrInfo.h"
#include "AlphaTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

namespace {
  struct AlphaAsmPrinter : public AsmPrinter {
    /// Unique incrementer for label values for referencing Global values.
    ///

    explicit AlphaAsmPrinter(formatted_raw_ostream &o, TargetMachine &tm,
                             const MCAsmInfo *T, bool V)
      : AsmPrinter(o, tm, T, V) {}

    virtual const char *getPassName() const {
      return "Alpha Assembly Printer";
    }
    void printInstruction(const MachineInstr *MI);
    static const char *getRegisterName(unsigned RegNo);

    void printOp(const MachineOperand &MO, bool IsCallOp = false);
    void printOperand(const MachineInstr *MI, int opNum);
    void printBaseOffsetPair(const MachineInstr *MI, int i, bool brackets=true);
    void PrintGlobalVariable(const GlobalVariable *GVar);
    bool runOnMachineFunction(MachineFunction &F);
    void EmitStartOfAsmFile(Module &M);

    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode);
    bool PrintAsmMemoryOperand(const MachineInstr *MI,
                               unsigned OpNo,
                               unsigned AsmVariant,
                               const char *ExtraCode);
  };
} // end of anonymous namespace

#include "AlphaGenAsmWriter.inc"

void AlphaAsmPrinter::printOperand(const MachineInstr *MI, int opNum)
{
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.getType() == MachineOperand::MO_Register) {
    assert(TargetRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Not physreg??");
    O << getRegisterName(MO.getReg());
  } else if (MO.isImm()) {
    O << MO.getImm();
    assert(MO.getImm() < (1 << 30));
  } else {
    printOp(MO);
  }
}


void AlphaAsmPrinter::printOp(const MachineOperand &MO, bool IsCallOp) {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << getRegisterName(MO.getReg());
    return;

  case MachineOperand::MO_Immediate:
    llvm_unreachable("printOp() does not handle immediate values");
    return;

  case MachineOperand::MO_MachineBasicBlock:
    GetMBBSymbol(MO.getMBB()->getNumber())->print(O, MAI);
    return;

  case MachineOperand::MO_ConstantPoolIndex:
    O << MAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << "_"
      << MO.getIndex();
    return;

  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    return;

  case MachineOperand::MO_GlobalAddress:
    O << Mang->getMangledName(MO.getGlobal());
    return;

  case MachineOperand::MO_JumpTableIndex:
    O << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << '_' << MO.getIndex();
    return;

  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool AlphaAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;

  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out jump tables referenced by the function
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);

  // Print out labels for the function.
  const Function *F = MF.getFunction();
  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F, Mang, TM));

  EmitAlignment(MF.getAlignment(), F);
  switch (F->getLinkage()) {
  default: llvm_unreachable("Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
  case Function::PrivateLinkage:
  case Function::LinkerPrivateLinkage:
    break;
   case Function::ExternalLinkage:
     O << "\t.globl " << CurrentFnName << "\n";
     break;
  case Function::WeakAnyLinkage:
  case Function::WeakODRLinkage:
  case Function::LinkOnceAnyLinkage:
  case Function::LinkOnceODRLinkage:
    O << MAI->getWeakRefDirective() << CurrentFnName << "\n";
    break;
  }

  printVisibility(CurrentFnName, F->getVisibility());

  O << "\t.ent " << CurrentFnName << "\n";

  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    if (I != MF.begin()) {
      EmitBasicBlockStart(I);
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      ++EmittedInsts;
      processDebugLoc(II, true);
      printInstruction(II);
      
      if (VerboseAsm && !II->getDebugLoc().isUnknown())
        EmitComments(*II);
      O << '\n';
      processDebugLoc(II, false);
    }
  }

  O << "\t.end " << CurrentFnName << "\n";

  // We didn't modify anything.
  return false;
}

void AlphaAsmPrinter::EmitStartOfAsmFile(Module &M) {
  if (TM.getSubtarget<AlphaSubtarget>().hasCT())
    O << "\t.arch ev6\n"; //This might need to be ev67, so leave this test here
  else
    O << "\t.arch ev6\n";
  O << "\t.set noat\n";
}

void AlphaAsmPrinter::PrintGlobalVariable(const GlobalVariable *GVar) {
  const TargetData *TD = TM.getTargetData();

  if (!GVar->hasInitializer()) return;  // External global require no code

  // Check to see if this is a special global used by LLVM, if so, emit it.
  if (EmitSpecialLLVMGlobal(GVar))
    return;

  std::string name = Mang->getMangledName(GVar);
  Constant *C = GVar->getInitializer();
  unsigned Size = TD->getTypeAllocSize(C->getType());
  unsigned Align = TD->getPreferredAlignmentLog(GVar);

  // 0: Switch to section
  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(GVar, Mang,
                                                                  TM));

  // 1: Check visibility
  printVisibility(name, GVar->getVisibility());

  // 2: Kind
  switch (GVar->getLinkage()) {
   case GlobalValue::LinkOnceAnyLinkage:
   case GlobalValue::LinkOnceODRLinkage:
   case GlobalValue::WeakAnyLinkage:
   case GlobalValue::WeakODRLinkage:
   case GlobalValue::CommonLinkage:
    O << MAI->getWeakRefDirective() << name << '\n';
    break;
   case GlobalValue::AppendingLinkage:
   case GlobalValue::ExternalLinkage:
      O << MAI->getGlobalDirective() << name << "\n";
      break;
    case GlobalValue::InternalLinkage:
    case GlobalValue::PrivateLinkage:
    case GlobalValue::LinkerPrivateLinkage:
      break;
    default:
      llvm_unreachable("Unknown linkage type!");
    }

  // 3: Type, Size, Align
  if (MAI->hasDotTypeDotSizeDirective()) {
    O << "\t.type\t" << name << ", @object\n";
    O << "\t.size\t" << name << ", " << Size << "\n";
  }

  EmitAlignment(Align, GVar);

  O << name << ":\n";

  EmitGlobalConstant(C);
  O << '\n';
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool AlphaAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      unsigned AsmVariant,
                                      const char *ExtraCode) {
  printOperand(MI, OpNo);
  return false;
}

bool AlphaAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo,
                                            unsigned AsmVariant,
                                            const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.
  O << "0(";
  printOperand(MI, OpNo);
  O << ")";
  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeAlphaAsmPrinter() { 
  RegisterAsmPrinter<AlphaAsmPrinter> X(TheAlphaTarget);
}
