//===-- AlphaAsmPrinter.cpp - Alpha LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format Alpha assembly language.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaInstrInfo.h"
#include "AlphaTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

  struct AlphaAsmPrinter : public AsmPrinter {

    /// Unique incrementer for label values for referencing Global values.
    ///
    unsigned LabelNumber;

     AlphaAsmPrinter(std::ostream &o, TargetMachine &tm)
       : AsmPrinter(o, tm), LabelNumber(0) {
      AlignmentIsInBytes = false;
      PrivateGlobalPrefix = "$";
    }

    /// We name each basic block in a Function with a unique number, so
    /// that we can consistently refer to them later. This is cleared
    /// at the beginning of each call to runOnMachineFunction().
    ///
    typedef std::map<const Value *, unsigned> ValueMapTy;
    ValueMapTy NumberForBB;
    std::string CurSection;

    virtual const char *getPassName() const {
      return "Alpha Assembly Printer";
    }
    bool printInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool IsCallOp = false);
    void printOperand(const MachineInstr *MI, int opNum);
    void printBaseOffsetPair (const MachineInstr *MI, int i, bool brackets=true);
    void printMachineInstruction(const MachineInstr *MI);
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
FunctionPass *llvm::createAlphaCodePrinterPass (std::ostream &o,
                                                  TargetMachine &tm) {
  return new AlphaAsmPrinter(o, tm);
}

#include "AlphaGenAsmWriter.inc"

void AlphaAsmPrinter::printOperand(const MachineInstr *MI, int opNum)
{
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.getType() == MachineOperand::MO_Register) {
    assert(MRegisterInfo::isPhysicalRegister(MO.getReg())&&"Not physreg??");
    O << TM.getRegisterInfo()->get(MO.getReg()).Name;
  } else if (MO.isImmediate()) {
    O << MO.getImmedValue();
    assert(MO.getImmedValue() < (1 << 30));
  } else {
    printOp(MO);
  }
}


void AlphaAsmPrinter::printOp(const MachineOperand &MO, bool IsCallOp) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  int new_symbol;

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << RI.get(MO.getReg()).Name;
    return;

  case MachineOperand::MO_Immediate:
    std::cerr << "printOp() does not handle immediate values\n";
    abort();
    return;

  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMachineBasicBlock());
    return;

  case MachineOperand::MO_ConstantPoolIndex:
    O << PrivateGlobalPrefix << "CPI" << getFunctionNumber() << "_"
      << MO.getConstantPoolIndex();
    return;

  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    return;

  case MachineOperand::MO_GlobalAddress:
    O << Mang->getValueName(MO.getGlobal());
    return;

  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

/// printMachineInstruction -- Print out a single Alpha MI to
/// the current output stream.
///
void AlphaAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;
  if (printInstruction(MI))
    return; // Printer was automatically generated

  assert(0 && "Unhandled instruction in asm writer!");
  abort();
  return;
}


/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool AlphaAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  const Function *F = MF.getFunction();
  SwitchToTextSection(".text", F);
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
    O << "\t.weak " << CurrentFnName << "\n";
    break;
  }

  O << "\t.ent " << CurrentFnName << "\n";

  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    printBasicBlockLabel(I, true);
    O << '\n';
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }
  ++LabelNumber;

  O << "\t.end " << CurrentFnName << "\n";

  // We didn't modify anything.
  return false;
}

bool AlphaAsmPrinter::doInitialization(Module &M)
{
  AsmPrinter::doInitialization(M);
  if(TM.getSubtarget<AlphaSubtarget>().hasF2I() 
     || TM.getSubtarget<AlphaSubtarget>().hasCT())
    O << "\t.arch ev6\n";
  else
    O << "\t.arch ev56\n";
  O << "\t.set noat\n";
  return false;
}

bool AlphaAsmPrinter::doFinalization(Module &M) {
  const TargetData *TD = TM.getTargetData();

  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
    if (I->hasInitializer()) {   // External global require no code
      // Check to see if this is a special global used by LLVM, if so, emit it.
      if (EmitSpecialLLVMGlobal(I))
        continue;
      
      O << "\n\n";
      std::string name = Mang->getValueName(I);
      Constant *C = I->getInitializer();
      unsigned Size = TD->getTypeSize(C->getType());
      //      unsigned Align = TD->getTypeAlignmentShift(C->getType());
      unsigned Align = getPreferredAlignmentLog(I);

      if (C->isNullValue() &&
          (I->hasLinkOnceLinkage() || I->hasInternalLinkage() ||
           I->hasWeakLinkage() /* FIXME: Verify correct */)) {
        SwitchToDataSection("\t.section .data", I);
        if (I->hasInternalLinkage())
          O << "\t.local " << name << "\n";

        O << "\t.comm " << name << "," << TD->getTypeSize(C->getType())
          << "," << (1 << Align)
          <<  "\n";
      } else {
        switch (I->getLinkage()) {
        case GlobalValue::LinkOnceLinkage:
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
          SwitchToDataSection("", I);
          break;
        case GlobalValue::AppendingLinkage:
          // FIXME: appending linkage variables should go into a section of
          // their name or something.  For now, just emit them as external.
        case GlobalValue::ExternalLinkage:
          // If external or appending, declare as a global symbol
          O << "\t.globl " << name << "\n";
          // FALL THROUGH
        case GlobalValue::InternalLinkage:
          SwitchToDataSection(C->isNullValue() ? "\t.section .bss" : 
                              "\t.section .data", I);
          break;
        case GlobalValue::GhostLinkage:
          std::cerr << "GhostLinkage cannot appear in AlphaAsmPrinter!\n";
          abort();
        }

        EmitAlignment(Align);
        O << "\t.type " << name << ",@object\n";
        O << "\t.size " << name << "," << Size << "\n";
        O << name << ":\n";
        EmitGlobalConstant(C);
      }
    }

  AsmPrinter::doFinalization(M);
  return false;
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
  printOperand(MI, OpNo);
  return false;
}
