//===-- EmitAssembly.cpp - Emit Sparc Specific .s File ---------------------==//
//
// This file implements all of the stuff neccesary to output a .s file from
// LLVM.  The code in this file assumes that the specified module has already
// been compiled into the internal data structures of the Module.
//
// The entry point of this file is the UltraSparc::emitAssembly method.
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "llvm/Analysis/SlotCalculator.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "llvm/Support/StringExtras.h"

namespace {

class SparcAsmPrinter {
  ostream &Out;
  SlotCalculator Table;
  const UltraSparc &Target;

  enum Sections {
    Unknown,
    Text,
    Data,
    ReadOnly,
  } CurSection;
public:
  inline SparcAsmPrinter(ostream &o, const Module *M, const UltraSparc &t)
    : Out(o), Table(SlotCalculator(M, true)), Target(t), CurSection(Unknown) {
    emitModule(M);
  }

private :
  void emitModule(const Module *M);
  /*
  void processSymbolTable(const SymbolTable &ST);
  void processConstant(const ConstPoolVal *CPV);
  void processGlobal(const GlobalVariable *GV);
  */
  void emitMethod(const Method *M);
  //void processMethodArgument(const MethodArgument *MA);
  void emitBasicBlock(const BasicBlock *BB);
  void emitMachineInst(const MachineInstr *MI);
  void printOperand(const MachineOperand &Op);
  

  // enterSection - Use this method to enter a different section of the output
  // executable.  This is used to only output neccesary section transitions.
  //
  void enterSection(enum Sections S) {
    if (S == CurSection) return;        // Only switch section if neccesary
    CurSection = S;

    Out << ".section \".";
    switch (S) {
    default: assert(0 && "Bad section name!");
    case Text:     Out << "text"; break;
    case Data:     Out << "data"; break;
    case ReadOnly: Out << "rodata"; break;
    }
    Out << "\"\n";
  }

  string getEscapedString(const string &S) {
    string Result;

    for (unsigned i = 0; i < S.size(); ++i) {
      char C = S[i];
      if ((C >= 'a' && C <= 'z') || (C >= 'A' && C <= 'Z') ||
          (C >= '0' && C <= '9')) {
        Result += C;
      } else {
        Result += '$';
        Result += char('0' + ((unsigned char)C >> 4));
        Result += char('0' + (C & 0xF));
      }
    }
    return Result;
  }

  // getID - Return a valid identifier for the specified value.  Base it on
  // the name of the identifier if possible, use a numbered value based on
  // prefix otherwise.  FPrefix is always prepended to the output identifier.
  //
  string getID(const Value *V, const char *Prefix, const char *FPrefix = 0) {
    string FP(FPrefix ? FPrefix : "");  // "Forced prefix"
    if (V->hasName()) {
      return FP + getEscapedString(V->getName());
    } else {
      assert(Table.getValSlot(V) != -1 && "Value not in value table!");
      return FP + string(Prefix) + itostr(Table.getValSlot(V));
    }
  }

  // getID Wrappers - Ensure consistent usage...
  string getID(const Method *M) { return getID(M, "anon_method$"); }
  string getID(const BasicBlock *BB) {
    return getID(BB, "LL", (".L$"+getID(BB->getParent())+"$").c_str());
  }

  unsigned getOperandMask(unsigned Opcode) {
    switch (Opcode) {
    case SUBcc:   return 1 << 3;  // Remove CC argument
    case BA:    case BRZ:         // Remove Arg #0, which is always null or xcc
    case BRLEZ: case BRLZ:
    case BRNZ:  case BRGZ:
    case BRGEZ:   return 1 << 0;
      // case RETURN:  return 1 << 1;  // Remove Arg #2 which is zero

    default:      return 0;       // By default, don't hack operands...
    }
  }
};


void SparcAsmPrinter::printOperand(const MachineOperand &Op) {
  switch (Op.getOperandType()) {
  case MachineOperand::MO_VirtualRegister:
  case MachineOperand::MO_CCRegister:
  case MachineOperand::MO_MachineRegister: {
    int RegNum = (int)Op.getAllocatedRegNum();
    
    // ****this code is temporary till NULL Values are fixed
    if (RegNum == 10000) {
      Out << "<NULL VALUE>";
    } else {
      Out << "%" << Target.getRegInfo().getUnifiedRegName(RegNum);
    }
    break;
  }
      
  case MachineOperand::MO_PCRelativeDisp: {
    const Value *Val = Op.getVRegValue();
    if (!Val) {
      Out << "\t<*NULL Value*>";
    } else if (const BasicBlock *BB = dyn_cast<const BasicBlock>(Val)) {
      Out << getID(BB);
    } else if (const Method *M = dyn_cast<const Method>(Val)) {
      Out << getID(M);
    } else {
      Out << "<unknown value=" << Val << ">";
    }
    break;
  }

  default:
    Out << Op;      // use dump field
    break;
  }
}


void SparcAsmPrinter::emitMachineInst(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpCode();

  if (TargetInstrDescriptors[Opcode].iclass & M_DUMMY_PHI_FLAG)
    return;  // IGNORE PHI NODES

  Out << "\t" << TargetInstrDescriptors[Opcode].opCodeString << "\t";

  switch (Opcode) {   // Some opcodes have special syntax...
  case JMPLCALL:
  case JMPLRET:
    assert(MI->getNumOperands() == 3 && "Unexpected JMPL instr!");
    printOperand(MI->getOperand(0));
    Out << "+";
    printOperand(MI->getOperand(1));
    Out << ", ";
    printOperand(MI->getOperand(2));
    Out << endl;
    return;
    
  case RETURN:
    assert(MI->getNumOperands() == 2 && "Unexpected RETURN instr!");
    printOperand(MI->getOperand(0));
    Out << "+";
    printOperand(MI->getOperand(1));
    Out << endl;
    return;
    
  default: break;
  }

  unsigned Mask = getOperandMask(Opcode);

  bool NeedComma = false;
  for(unsigned OpNum = 0; OpNum < MI->getNumOperands(); ++OpNum) {
    if ((1 << OpNum) & Mask) continue;  // Ignore this operand?
    
    const MachineOperand &Op = MI->getOperand(OpNum);
    if (NeedComma) Out << ", ";    // Handle comma outputing
    NeedComma = true;

    printOperand(Op);
  }
  Out << endl;
}

void SparcAsmPrinter::emitBasicBlock(const BasicBlock *BB) {
  // Emit a label for the basic block
  Out << getID(BB) << ":\n";

  // Get the vector of machine instructions corresponding to this bb.
  const MachineCodeForBasicBlock &MIs = BB->getMachineInstrVec();
  MachineCodeForBasicBlock::const_iterator MII = MIs.begin(), MIE = MIs.end();

  // Loop over all of the instructions in the basic block...
  for (; MII != MIE; ++MII)
    emitMachineInst(*MII);
  Out << "\n";  // Seperate BB's with newlines
}

void SparcAsmPrinter::emitMethod(const Method *M) {
  if (M->isExternal()) return;

  // Make sure the slot table has information about this method...
  Table.incorporateMethod(M);

  string MethName = getID(M);
  Out << "!****** Outputing Method: " << MethName << " ******\n";
  enterSection(Text);
  Out << "\t.align 4\n\t.global\t" << MethName << "\n";
  //Out << "\t.type\t" << MethName << ",#function\n";
  Out << "\t.type\t" << MethName << ", 2\n";
  Out << MethName << ":\n";

  // Output code for all of the basic blocks in the method...
  for (Method::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    emitBasicBlock(*I);

  // Output a .size directive so the debugger knows the extents of the function
  Out << ".EndOf$" << MethName << ":\n\t.size " << MethName << ", .EndOf$"
      << MethName << "-" << MethName << endl;

  // Put some spaces between the methods
  Out << "\n\n";

  // Forget all about M.
  Table.purgeMethod();
}


void SparcAsmPrinter::emitModule(const Module *M) {
  // TODO: Look for a filename annotation on M to emit a .file directive
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    emitMethod(*I);
}

}  // End anonymous namespace

//
// emitAssembly - Output assembly language code (a .s file) for the specified
// method. The specified method must have been compiled before this may be
// used.
//
void UltraSparc::emitAssembly(const Module *M, ostream &Out) const {
  SparcAsmPrinter Print(Out, M, *this);
}
