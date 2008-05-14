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
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

namespace {
  struct VISIBILITY_HIDDEN AlphaAsmPrinter : public AsmPrinter {

    /// Unique incrementer for label values for referencing Global values.
    ///

    AlphaAsmPrinter(std::ostream &o, TargetMachine &tm, const TargetAsmInfo *T)
      : AsmPrinter(o, tm, T) {
    }

    virtual const char *getPassName() const {
      return "Alpha Assembly Printer";
    }
    bool printInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool IsCallOp = false);
    void printOperand(const MachineInstr *MI, int opNum);
    void printBaseOffsetPair (const MachineInstr *MI, int i, bool brackets=true);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
    
    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode);
    bool PrintAsmMemoryOperand(const MachineInstr *MI, 
                               unsigned OpNo,
                               unsigned AsmVariant, 
                               const char *ExtraCode);
  };
} // end of anonymous namespace

/// createAlphaCodePrinterPass - Returns a pass that prints the Alpha
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *llvm::createAlphaCodePrinterPass(std::ostream &o,
                                               TargetMachine &tm) {
  return new AlphaAsmPrinter(o, tm, tm.getTargetAsmInfo());
}

#include "AlphaGenAsmWriter.inc"

void AlphaAsmPrinter::printOperand(const MachineInstr *MI, int opNum)
{
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.getType() == MachineOperand::MO_Register) {
    assert(TargetRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Not physreg??");
    O << TM.getRegisterInfo()->get(MO.getReg()).AsmName;
  } else if (MO.isImmediate()) {
    O << MO.getImm();
    assert(MO.getImm() < (1 << 30));
  } else {
    printOp(MO);
  }
}


void AlphaAsmPrinter::printOp(const MachineOperand &MO, bool IsCallOp) {
  const TargetRegisterInfo &RI = *TM.getRegisterInfo();

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << RI.get(MO.getReg()).AsmName;
    return;

  case MachineOperand::MO_Immediate:
    cerr << "printOp() does not handle immediate values\n";
    abort();
    return;

  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB());
    return;

  case MachineOperand::MO_ConstantPoolIndex:
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << "_"
      << MO.getIndex();
    return;

  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    return;

  case MachineOperand::MO_GlobalAddress: {
    GlobalValue *GV = MO.getGlobal();
    O << Mang->getValueName(GV);
    if (GV->isDeclaration() && GV->hasExternalWeakLinkage())
      ExtWeakSymbols.insert(GV);
    return;
  }

  case MachineOperand::MO_JumpTableIndex:
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
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
  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out jump tables referenced by the function
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);

  // Print out labels for the function.
  const Function *F = MF.getFunction();
  SwitchToTextSection(getSectionForFunction(*F).c_str(), F);
  
  EmitAlignment(4, F);
  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
    break;
   case Function::ExternalLinkage:
     O << "\t.globl " << CurrentFnName << "\n";
     break;
  case Function::WeakLinkage:
  case Function::LinkOnceLinkage:
    O << TAI->getWeakRefDirective() << CurrentFnName << "\n";
    break;
  }

  O << "\t.ent " << CurrentFnName << "\n";

  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    if (I != MF.begin()) {
      printBasicBlockLabel(I, true, true);
      O << '\n';
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      ++EmittedInsts;
      if (!printInstruction(II)) {
        assert(0 && "Unhandled instruction in asm writer!");
        abort();
      }
    }
  }

  O << "\t.end " << CurrentFnName << "\n";

  // We didn't modify anything.
  return false;
}

bool AlphaAsmPrinter::doInitialization(Module &M)
{
  if(TM.getSubtarget<AlphaSubtarget>().hasCT())
    O << "\t.arch ev6\n"; //This might need to be ev67, so leave this test here
  else
    O << "\t.arch ev6\n";
  O << "\t.set noat\n";
  return AsmPrinter::doInitialization(M);
}

bool AlphaAsmPrinter::doFinalization(Module &M) {
  const TargetData *TD = TM.getTargetData();

  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I) {

    if (!I->hasInitializer()) continue;  // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I))
      continue;
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD->getABITypeSize(C->getType());
    unsigned Align = TD->getPreferredAlignmentLog(I);
    
    //1: hidden?
    if (I->hasHiddenVisibility())
      O << TAI->getHiddenDirective() << name << "\n";
    
    //2: kind
    switch (I->getLinkage()) {
    case GlobalValue::LinkOnceLinkage:
    case GlobalValue::WeakLinkage:
    case GlobalValue::CommonLinkage:
      O << TAI->getWeakRefDirective() << name << '\n';
      break;
    case GlobalValue::AppendingLinkage:
    case GlobalValue::ExternalLinkage:
      O << "\t.globl " << name << "\n";
      break;
    case GlobalValue::InternalLinkage:
      break;
    default:
      assert(0 && "Unknown linkage type!");
      cerr << "Unknown linkage type!\n";
      abort();
    }
    
    //3: Section (if changed)
    if (I->hasSection() &&
        (I->getSection() == ".ctors" ||
         I->getSection() == ".dtors")) {
      std::string SectionName = ".section\t" + I->getSection()
        + ",\"aw\",@progbits";
      SwitchToDataSection(SectionName.c_str());
    } else {
      if (C->isNullValue())
        SwitchToDataSection("\t.section\t.bss", I);
      else
        SwitchToDataSection("\t.section\t.data", I);
    }
    
    //4: Type, Size, Align
    O << "\t.type\t" << name << ", @object\n";
    O << "\t.size\t" << name << ", " << Size << "\n";
    EmitAlignment(Align, I);
    
    O << name << ":\n";
    
    // If the initializer is a extern weak symbol, remember to emit the weak
    // reference!
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
      if (GV->hasExternalWeakLinkage())
        ExtWeakSymbols.insert(GV);
    
    EmitGlobalConstant(C);
    O << '\n';
  }

  return AsmPrinter::doFinalization(M);
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
