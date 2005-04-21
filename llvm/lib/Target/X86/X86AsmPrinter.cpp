//===-- X86AsmPrinter.cpp - Convert X86 LLVM code to Intel assembly -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to Intel and AT&T format assembly
// language. This printer is the output mechanism used by `llc' and `lli
// -print-machineinstrs' on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");
  enum AsmWriterFlavor { att, intel };

  cl::opt<AsmWriterFlavor>
  AsmWriterFlavor("x86-asm-syntax",
                  cl::desc("Choose style of code to emit from X86 backend:"),
                  cl::values(
                             clEnumVal(att,   "  Emit AT&T-style assembly"),
                             clEnumVal(intel, "  Emit Intel-style assembly"),
                             clEnumValEnd),
                  cl::init(att));

  struct X86SharedAsmPrinter : public AsmPrinter {
    X86SharedAsmPrinter(std::ostream &O, TargetMachine &TM)
      : AsmPrinter(O, TM), forCygwin(false) { }

    bool doInitialization(Module &M);
    void printConstantPool(MachineConstantPool *MCP);
    bool doFinalization(Module &M);
    bool forCygwin;
  };
}

static bool isScale(const MachineOperand &MO) {
  return MO.isImmediate() &&
    (MO.getImmedValue() == 1 || MO.getImmedValue() == 2 ||
     MO.getImmedValue() == 4 || MO.getImmedValue() == 8);
}

static bool isMem(const MachineInstr *MI, unsigned Op) {
  if (MI->getOperand(Op).isFrameIndex()) return true;
  if (MI->getOperand(Op).isConstantPoolIndex()) return true;
  return Op+4 <= MI->getNumOperands() &&
    MI->getOperand(Op  ).isRegister() && isScale(MI->getOperand(Op+1)) &&
    MI->getOperand(Op+2).isRegister() && (MI->getOperand(Op+3).isImmediate() ||
        MI->getOperand(Op+3).isGlobalAddress());
}

// SwitchSection - Switch to the specified section of the executable if we are
// not already in it!
//
static void SwitchSection(std::ostream &OS, std::string &CurSection,
                          const char *NewSection) {
  if (CurSection != NewSection) {
    CurSection = NewSection;
    if (!CurSection.empty())
      OS << "\t" << NewSection << "\n";
  }
}

/// doInitialization - determine
bool X86SharedAsmPrinter::doInitialization(Module& M) {
  forCygwin = false;
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5)
    forCygwin = TT.find("cygwin") != std::string::npos ||
                TT.find("mingw")  != std::string::npos;
  else if (TT.empty()) {
#if defined(__CYGWIN__) || defined(__MINGW32__)
    forCygwin = true;
#else
    forCygwin = false;
#endif
  }
  if (forCygwin)
    GlobalPrefix = "_";
  return AsmPrinter::doInitialization(M);
}

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void X86SharedAsmPrinter::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();

  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.section .rodata\n";
    emitAlignment(TD.getTypeAlignmentShift(CP[i]->getType()));
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t" << CommentString
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
    if (I->hasInitializer()) {   // External global require no code
      O << "\n\n";
      std::string name = Mang->getValueName(I);
      Constant *C = I->getInitializer();
      unsigned Size = TD.getTypeSize(C->getType());
      unsigned Align = TD.getTypeAlignmentShift(C->getType());

      if (C->isNullValue() &&
          (I->hasLinkOnceLinkage() || I->hasInternalLinkage() ||
           I->hasWeakLinkage() /* FIXME: Verify correct */)) {
        SwitchSection(O, CurSection, ".data");
        if (!forCygwin && I->hasInternalLinkage())
          O << "\t.local " << name << "\n";

        O << "\t.comm " << name << "," << TD.getTypeSize(C->getType());
        if (!forCygwin)
          O << "," << (1 << Align);
        O << "\t\t# ";
        WriteAsOperand(O, I, true, true, &M);
        O << "\n";
      } else {
        switch (I->getLinkage()) {
        case GlobalValue::LinkOnceLinkage:
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          SwitchSection(O, CurSection, "");
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
          break;
        case GlobalValue::AppendingLinkage:
          // FIXME: appending linkage variables should go into a section of
          // their name or something.  For now, just emit them as external.
        case GlobalValue::ExternalLinkage:
          // If external or appending, declare as a global symbol
          O << "\t.globl " << name << "\n";
          // FALL THROUGH
        case GlobalValue::InternalLinkage:
          if (C->isNullValue())
            SwitchSection(O, CurSection, ".bss");
          else
            SwitchSection(O, CurSection, ".data");
          break;
        case GlobalValue::GhostLinkage:
          std::cerr << "GhostLinkage cannot appear in X86AsmPrinter!\n";
          abort();
        }

        emitAlignment(Align);
	if (!forCygwin) {
	  O << "\t.type " << name << ",@object\n";
	  O << "\t.size " << name << "," << Size << "\n";
        }
        O << name << ":\t\t\t\t# ";
        WriteAsOperand(O, I, true, true, &M);
        O << " = ";
        WriteAsOperand(O, C, false, false, &M);
        O << "\n";
        emitGlobalConstant(C);
      }
    }

  AsmPrinter::doFinalization(M);
  return false; // success
}

namespace {
  struct X86IntelAsmPrinter : public X86SharedAsmPrinter {
    X86IntelAsmPrinter(std::ostream &O, TargetMachine &TM)
      : X86SharedAsmPrinter(O, TM) { }

    virtual const char *getPassName() const {
      return "X86 Intel-Style Assembly Printer";
    }

    /// printInstruction - This method is automatically generated by tablegen
    /// from the instruction set description.  This method returns true if the
    /// machine instruction was sufficiently described to print it, otherwise it
    /// returns false.
    bool printInstruction(const MachineInstr *MI);

    // This method is used by the tablegen'erated instruction printer.
    void printOperand(const MachineInstr *MI, unsigned OpNo, MVT::ValueType VT){
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (MO.getType() == MachineOperand::MO_MachineRegister) {
        assert(MRegisterInfo::isPhysicalRegister(MO.getReg())&&"Not physref??");
        // Bug Workaround: See note in Printer::doInitialization about %.
        O << "%" << TM.getRegisterInfo()->get(MO.getReg()).Name;
      } else {
        printOp(MO);
      }
    }

    void printCallOperand(const MachineInstr *MI, unsigned OpNo,
                          MVT::ValueType VT) {
      printOp(MI->getOperand(OpNo), true); // Don't print "OFFSET".
    }

    void printMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      switch (VT) {
      default: assert(0 && "Unknown arg size!");
      case MVT::i8:   O << "BYTE PTR "; break;
      case MVT::i16:  O << "WORD PTR "; break;
      case MVT::i32:
      case MVT::f32:  O << "DWORD PTR "; break;
      case MVT::i64:
      case MVT::f64:  O << "QWORD PTR "; break;
      case MVT::f80:  O << "XWORD PTR "; break;
      }
      printMemReference(MI, OpNo);
    }

    void printMachineInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool elideOffsetKeyword = false);
    void printMemReference(const MachineInstr *MI, unsigned Op);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
  };
} // end of anonymous namespace


// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter1.inc"


/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86IntelAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  setupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\t.text\n";
  emitAlignment(4);
  O << "\t.globl\t" << CurrentFnName << "\n";
  if (!forCygwin)
    O << "\t.type\t" << CurrentFnName << ", @function\n";
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block if there are any predecessors.
    if (I->pred_begin() != I->pred_end())
      O << ".LBB" << CurrentFnName << "_" << I->getNumber() << ":\t"
        << CommentString << " " << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  // We didn't modify anything.
  return false;
}

void X86IntelAsmPrinter::printOp(const MachineOperand &MO,
                                 bool elideOffsetKeyword /* = false */) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
    if (MRegisterInfo::isPhysicalRegister(MO.getReg()))
      // Bug Workaround: See note in Printer::doInitialization about %.
      O << "%" << RI.get(MO.getReg()).Name;
    else
      O << "%reg" << MO.getReg();
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber () << "\t# "
      << MBBOp->getBasicBlock ()->getName ();
    return;
  }
  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building X86 MachineInstrs";
    abort ();
    return;
  case MachineOperand::MO_GlobalAddress: {
    if (!elideOffsetKeyword)
      O << "OFFSET ";
    O << Mang->getValueName(MO.getGlobal());
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << " + " << Offset;
    else if (Offset < 0)
      O << " - " << -Offset;
    return;
  }
  case MachineOperand::MO_ExternalSymbol:
    O << GlobalPrefix << MO.getSymbolName();
    return;
  default:
    O << "<unknown operand type>"; return;
  }
}

void X86IntelAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op){
  assert(isMem(MI, Op) && "Invalid memory reference!");

  const MachineOperand &BaseReg  = MI->getOperand(Op);
  int ScaleVal                   = MI->getOperand(Op+1).getImmedValue();
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &DispSpec = MI->getOperand(Op+3);

  if (BaseReg.isFrameIndex()) {
    O << "[frame slot #" << BaseReg.getFrameIndex();
    if (DispSpec.getImmedValue())
      O << " + " << DispSpec.getImmedValue();
    O << "]";
    return;
  } else if (BaseReg.isConstantPoolIndex()) {
    O << "[.CPI" << CurrentFnName << "_"
      << BaseReg.getConstantPoolIndex();

    if (IndexReg.getReg()) {
      O << " + ";
      if (ScaleVal != 1)
        O << ScaleVal << "*";
      printOp(IndexReg);
    }

    if (DispSpec.getImmedValue())
      O << " + " << DispSpec.getImmedValue();
    O << "]";
    return;
  }

  O << "[";
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOp(BaseReg, true);
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << "*";
    printOp(IndexReg);
    NeedPlus = true;
  }

  if (DispSpec.isGlobalAddress()) {
    if (NeedPlus)
      O << " + ";
    printOp(DispSpec, true);
  } else {
    int DispVal = DispSpec.getImmedValue();
    if (DispVal || (!BaseReg.getReg() && !IndexReg.getReg())) {
      if (NeedPlus)
        if (DispVal > 0)
          O << " + ";
        else {
          O << " - ";
          DispVal = -DispVal;
        }
      O << DispVal;
    }
  }
  O << "]";
}


/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in Intel syntax to the current output stream.
///
void X86IntelAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

bool X86IntelAsmPrinter::doInitialization(Module &M) {
  AsmPrinter::doInitialization(M);
  // Tell gas we are outputting Intel syntax (not AT&T syntax) assembly.
  //
  // Bug: gas in `intel_syntax noprefix' mode interprets the symbol `Sp' in an
  // instruction as a reference to the register named sp, and if you try to
  // reference a symbol `Sp' (e.g. `mov ECX, OFFSET Sp') then it gets lowercased
  // before being looked up in the symbol table. This creates spurious
  // `undefined symbol' errors when linking. Workaround: Do not use `noprefix'
  // mode, and decorate all register names with percent signs.
  O << "\t.intel_syntax\n";
  return false;
}



namespace {
  struct X86ATTAsmPrinter : public X86SharedAsmPrinter {
    X86ATTAsmPrinter(std::ostream &O, TargetMachine &TM)
      : X86SharedAsmPrinter(O, TM) { }

    virtual const char *getPassName() const {
      return "X86 AT&T-Style Assembly Printer";
    }

    /// printInstruction - This method is automatically generated by tablegen
    /// from the instruction set description.  This method returns true if the
    /// machine instruction was sufficiently described to print it, otherwise it
    /// returns false.
    bool printInstruction(const MachineInstr *MI);

    // This method is used by the tablegen'erated instruction printer.
    void printOperand(const MachineInstr *MI, unsigned OpNo, MVT::ValueType VT){
      printOp(MI->getOperand(OpNo));
    }

    void printCallOperand(const MachineInstr *MI, unsigned OpNo,
                          MVT::ValueType VT) {
      printOp(MI->getOperand(OpNo), true); // Don't print '$' prefix.
    }

    void printMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      printMemReference(MI, OpNo);
    }

    void printMachineInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool isCallOperand = false);
    void printMemReference(const MachineInstr *MI, unsigned Op);
    bool runOnMachineFunction(MachineFunction &F);
  };
} // end of anonymous namespace


// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter.inc"


/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86ATTAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  setupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\t.text\n";
  emitAlignment(4);
  O << "\t.globl\t" << CurrentFnName << "\n";
  if (!forCygwin)
    O << "\t.type\t" << CurrentFnName << ", @function\n";
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I->pred_begin() != I->pred_end())
      O << ".LBB" << CurrentFnName << "_" << I->getNumber() << ":\t"
        << CommentString << " " << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  // We didn't modify anything.
  return false;
}

void X86ATTAsmPrinter::printOp(const MachineOperand &MO, bool isCallOp) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
  case MachineOperand::MO_MachineRegister:
    assert(MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           "Virtual registers should not make it this far!");
    O << '%';
    for (const char *Name = RI.get(MO.getReg()).Name; *Name; ++Name)
      O << (char)tolower(*Name);
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << '$' << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber () << "\t# "
      << MBBOp->getBasicBlock ()->getName ();
    return;
  }
  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building X86 MachineInstrs";
    abort ();
    return;
  case MachineOperand::MO_GlobalAddress: {
    if (!isCallOp) O << '$';
    O << Mang->getValueName(MO.getGlobal());
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << "+" << Offset;
    else if (Offset < 0)
      O << Offset;
    return;
  }
  case MachineOperand::MO_ExternalSymbol:
    if (!isCallOp) O << '$';
    O << GlobalPrefix << MO.getSymbolName();
    return;
  default:
    O << "<unknown operand type>"; return;
  }
}

void X86ATTAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op){
  assert(isMem(MI, Op) && "Invalid memory reference!");

  const MachineOperand &BaseReg  = MI->getOperand(Op);
  int ScaleVal                   = MI->getOperand(Op+1).getImmedValue();
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &DispSpec = MI->getOperand(Op+3);

  if (BaseReg.isFrameIndex()) {
    O << "[frame slot #" << BaseReg.getFrameIndex();
    if (DispSpec.getImmedValue())
      O << " + " << DispSpec.getImmedValue();
    O << "]";
    return;
  } else if (BaseReg.isConstantPoolIndex()) {
    O << ".CPI" << CurrentFnName << "_"
      << BaseReg.getConstantPoolIndex();
    if (DispSpec.getImmedValue())
      O << "+" << DispSpec.getImmedValue();
    if (IndexReg.getReg()) {
      O << "(,";
      printOp(IndexReg);
      if (ScaleVal != 1)
        O << "," << ScaleVal;
      O << ")";
    }
    return;
  }

  if (DispSpec.isGlobalAddress()) {
    printOp(DispSpec, true);
  } else {
    int DispVal = DispSpec.getImmedValue();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  }

  if (IndexReg.getReg() || BaseReg.getReg()) {
    O << "(";
    if (BaseReg.getReg())
      printOp(BaseReg);

    if (IndexReg.getReg()) {
      O << ",";
      printOp(IndexReg);
      if (ScaleVal != 1)
        O << "," << ScaleVal;
    }

    O << ")";
  }
}


/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in Intel syntax to the current output stream.
///
void X86ATTAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;
  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}


/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
FunctionPass *llvm::createX86CodePrinterPass(std::ostream &o,TargetMachine &tm){
  switch (AsmWriterFlavor) {
  default:
    assert(0 && "Unknown asm flavor!");
  case intel:
    return new X86IntelAsmPrinter(o, tm);
  case att:
    return new X86ATTAsmPrinter(o, tm);
  }
}
