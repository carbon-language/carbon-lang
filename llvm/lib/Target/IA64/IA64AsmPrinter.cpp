//===-- IA64AsmPrinter.cpp - Print out IA64 LLVM as assembly --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to assembly accepted by the GNU binutils 'gas'
// assembler. The Intel 'ias' and HP-UX 'as' assemblers *may* choke on this
// output, but if so that's a bug I'd like to hear about: please file a bug
// report in bugzilla. FYI, the excellent 'ias' assembler is bundled with
// the Intel C/C++ compiler for Itanium Linux.
//
//===----------------------------------------------------------------------===//

#include "IA64.h"
#include "IA64TargetMachine.h"
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

  struct IA64SharedAsmPrinter : public AsmPrinter {

    std::set<std::string> ExternalFunctionNames, ExternalObjectNames;

    IA64SharedAsmPrinter(std::ostream &O, TargetMachine &TM)
      : AsmPrinter(O, TM) { }

    void printConstantPool(MachineConstantPool *MCP);
    bool doFinalization(Module &M);
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

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void IA64SharedAsmPrinter::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();

  if (CP.empty()) return;

  O << "\n\t.section .data, \"aw\", \"progbits\"\n";
      // FIXME: would be nice to have rodata (no 'w') when appropriate?
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    emitAlignment(TD.getTypeAlignmentShift(CP[i]->getType()));
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t" << CommentString
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

bool IA64SharedAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
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
        if (I->hasInternalLinkage()) {
          O << "\t.lcomm " << name << "," << TD.getTypeSize(C->getType())
            << "," << (1 << Align);
          O << "\t\t// ";
        } else {
          O << "\t.common " << name << "," << TD.getTypeSize(C->getType())
            << "," << (1 << Align);
          O << "\t\t// ";
        }
        WriteAsOperand(O, I, true, true, &M);
        O << "\n";
      } else {
        switch (I->getLinkage()) {
        case GlobalValue::LinkOnceLinkage:
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          SwitchSection(O, CurSection, "");
          O << "\t.section\t.llvm.linkonce.d." << name
            << ", \"aw\", \"progbits\"\n";
          break;
        case GlobalValue::AppendingLinkage:
          // FIXME: appending linkage variables should go into a section of
          // their name or something.  For now, just emit them as external.
        case GlobalValue::ExternalLinkage:
          // If external or appending, declare as a global symbol
          O << "\t.global " << name << "\n";
          // FALL THROUGH
        case GlobalValue::InternalLinkage:
          if (C->isNullValue())
            SwitchSection(O, CurSection, ".bss");
          else
            SwitchSection(O, CurSection, ".data");
          break;
        case GlobalValue::GhostLinkage:
          std::cerr << "GhostLinkage cannot appear in IA64AsmPrinter!\n";
          abort();
        }

        emitAlignment(Align);
        O << "\t.type " << name << ",@object\n";
        O << "\t.size " << name << "," << Size << "\n";
        O << name << ":\t\t\t\t// ";
        WriteAsOperand(O, I, true, true, &M);
        O << " = ";
        WriteAsOperand(O, C, false, false, &M);
        O << "\n";
        emitGlobalConstant(C);
      }
    }

  // we print out ".global X \n .type X, @function" for each external function
  O << "\n\n// br.call targets referenced (and not defined) above: \n";
  for (std::set<std::string>::iterator i = ExternalFunctionNames.begin(),
       e = ExternalFunctionNames.end(); i!=e; ++i) {
    O << "\t.global " << *i << "\n\t.type " << *i << ", @function\n";
  }
  O << "\n\n";

  // we print out ".global X \n .type X, @object" for each external object
  O << "\n\n// (external) symbols referenced (and not defined) above: \n";
  for (std::set<std::string>::iterator i = ExternalObjectNames.begin(),
       e = ExternalObjectNames.end(); i!=e; ++i) {
    O << "\t.global " << *i << "\n\t.type " << *i << ", @object\n";
  }
  O << "\n\n";

  AsmPrinter::doFinalization(M);
  return false; // success
}

namespace {
  struct IA64AsmPrinter : public IA64SharedAsmPrinter {
    IA64AsmPrinter(std::ostream &O, TargetMachine &TM)
      : IA64SharedAsmPrinter(O, TM) {

      CommentString = "//";
      Data8bitsDirective = "\tdata1\t";     // FIXME: check that we are
      Data16bitsDirective = "\tdata2.ua\t"; // disabling auto-alignment
      Data32bitsDirective = "\tdata4.ua\t"; // properly
      Data64bitsDirective = "\tdata8.ua\t";
      ZeroDirective = "\t.skip\t";
      AsciiDirective = "\tstring\t";

      GlobalVarAddrPrefix="";
      GlobalVarAddrSuffix="";
      FunctionAddrPrefix="@fptr(";
      FunctionAddrSuffix=")";

    }

    virtual const char *getPassName() const {
      return "IA64 Assembly Printer";
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
        //XXX Bug Workaround: See note in Printer::doInitialization about %.
        O << TM.getRegisterInfo()->get(MO.getReg()).Name;
      } else {
        printOp(MO);
      }
    }

    void printS8ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      int val=(unsigned int)MI->getOperand(OpNo).getImmedValue();
      if(val>=128) val=val-256; // if negative, flip sign
      O << val;
    }
    void printS14ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      int val=(unsigned int)MI->getOperand(OpNo).getImmedValue();
      if(val>=8192) val=val-16384; // if negative, flip sign
      O << val;
    }
    void printS22ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      int val=(unsigned int)MI->getOperand(OpNo).getImmedValue();
      if(val>=2097152) val=val-4194304; // if negative, flip sign
      O << val;
    }
    void printU64ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      O << (uint64_t)MI->getOperand(OpNo).getImmedValue();
    }
    void printS64ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      O << (int64_t)MI->getOperand(OpNo).getImmedValue();
    }

    void printCallOperand(const MachineInstr *MI, unsigned OpNo,
                          MVT::ValueType VT) {
      printOp(MI->getOperand(OpNo), true); // this is a br.call instruction
    }

    void printMachineInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool isBRCALLinsn= false);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
  };
} // end of anonymous namespace


// Include the auto-generated portion of the assembly writer.
#include "IA64GenAsmWriter.inc"


/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool IA64AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  setupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\n\t.section .text, \"ax\", \"progbits\"\n";
              // ^^  means "Allocated instruXions in mem, initialized"
  emitAlignment(5);
  O << "\t.global\t" << CurrentFnName << "\n";
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

void IA64AsmPrinter::printOp(const MachineOperand &MO,
                                 bool isBRCALLinsn /* = false */) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
  case MachineOperand::MO_CCRegister: {
    O << RI.get(MO.getReg()).Name;
    return;
  }

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << /*(unsigned int)*/MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber () << "\t// "
      << MBBOp->getBasicBlock ()->getName ();
    return;
  }
  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building IA64 MachineInstrs";
    abort ();
    return;

  case MachineOperand::MO_ConstantPoolIndex: {
    O << "@gprel(.CPI" << CurrentFnName << "_"
      << MO.getConstantPoolIndex() << ")";
    return;
  }

  case MachineOperand::MO_GlobalAddress: {

    // functions need @ltoff(@fptr(fn_name)) form
    GlobalValue *GV = MO.getGlobal();
    Function *F = dyn_cast<Function>(GV);

    bool Needfptr=false; // if we're computing an address @ltoff(X), do
                         // we need to decorate it so it becomes
                         // @ltoff(@fptr(X)) ?
    if (F && !isBRCALLinsn /*&& F->isExternal()*/)
      Needfptr=true;

    // if this is the target of a call instruction, we should define
    // the function somewhere (GNU gas has no problem without this, but
    // Intel ias rightly complains of an 'undefined symbol')

    if (F /*&& isBRCALLinsn*/ && F->isExternal())
      ExternalFunctionNames.insert(Mang->getValueName(MO.getGlobal()));
    else
      if (GV->isExternal()) // e.g. stuff like 'stdin'
        ExternalObjectNames.insert(Mang->getValueName(MO.getGlobal()));

    if (!isBRCALLinsn)
      O << "@ltoff(";
    if (Needfptr)
      O << "@fptr(";
    O << Mang->getValueName(MO.getGlobal());
    if (Needfptr)
      O << ")"; // close fptr(
    if (!isBRCALLinsn)
      O << ")"; // close ltoff(
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << " + " << Offset;
    else if (Offset < 0)
      O << " - " << -Offset;
    return;
  }
  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    ExternalFunctionNames.insert(MO.getSymbolName());
    return;
  default:
    O << "<AsmPrinter: unknown operand type: " << MO.getType() << " >"; return;
  }
}

/// printMachineInstruction -- Print out a single IA64 LLVM instruction
/// MI to the current output stream.
///
void IA64AsmPrinter::printMachineInstruction(const MachineInstr *MI) {

  ++EmittedInsts;

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

bool IA64AsmPrinter::doInitialization(Module &M) {
  AsmPrinter::doInitialization(M);

  O << "\n.ident \"LLVM-ia64\"\n\n"
    << "\t.psr    lsb\n"  // should be "msb" on HP-UX, for starters
    << "\t.radix  C\n"
    << "\t.psr    abi64\n"; // we only support 64 bits for now
  return false;
}

/// createIA64CodePrinterPass - Returns a pass that prints the IA64
/// assembly code for a MachineFunction to the given output stream, using
/// the given target machine description.
///
FunctionPass *llvm::createIA64CodePrinterPass(std::ostream &o,TargetMachine &tm){
  return new IA64AsmPrinter(o, tm);
}


