//===-- EmitAssembly.cpp - Emit SparcV9 Specific .s File -------------------==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements all of the stuff necessary to output a .s file from
// LLVM.  The code in this file assumes that the specified module has already
// been compiled into the internal data structures of the Module.
//
// This code largely consists of two LLVM Pass's: a FunctionPass and a Pass.
// The FunctionPass is pipelined together with all of the rest of the code
// generation stages, and the Pass runs at the end to emit code for global
// variables and such.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Mangler.h"
#include "Support/StringExtras.h"
#include "Support/Statistic.h"
#include "SparcV9Internals.h"
#include "MachineFunctionInfo.h"
#include <string>
using namespace llvm;

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");
}

namespace {
  struct SparcV9AsmPrinter : public AsmPrinter {
  public:
    enum Sections {
      Unknown,
      Text,
      ReadOnlyData,
      InitRWData,
      ZeroInitRWData,
    } CurSection;

    SparcV9AsmPrinter(std::ostream &OS, TargetMachine &TM)
      : AsmPrinter(OS, TM), CurSection(Unknown) {
      ZeroDirective       = 0;  // No way to get zeros.
      Data16bitsDirective = "\t.half\t";
      Data32bitsDirective = "\t.word\t";
      Data64bitsDirective = "\t.xword\t";
      CommentString = "!";
    }

    const char *getPassName() const {
      return "SparcV9 Assembly Printer";
    }

    // Print a constant (which may be an aggregate) prefixed by all the
    // appropriate directives.  Uses printConstantValueOnly() to print the
    // value or values.
    void printConstant(const Constant* CV, const std::string &valID) {
      emitAlignment(TM.getTargetData().getTypeAlignmentShift(CV->getType()));
      O << "\t.type" << "\t" << valID << ",#object\n";

      unsigned constSize = TM.getTargetData().getTypeSize(CV->getType());
      O << "\t.size" << "\t" << valID << "," << constSize << "\n";
  
      O << valID << ":\n";
  
      emitGlobalConstant(CV);
    }

    // enterSection - Use this method to enter a different section of the output
    // executable.  This is used to only output necessary section transitions.
    //
    void enterSection(enum Sections S) {
      if (S == CurSection) return;        // Only switch section if necessary
      CurSection = S;

      O << "\n\t.section ";
      switch (S)
      {
      default: assert(0 && "Bad section name!");
      case Text:         O << "\".text\""; break;
      case ReadOnlyData: O << "\".rodata\",#alloc"; break;
      case InitRWData:   O << "\".data\",#alloc,#write"; break;
      case ZeroInitRWData: O << "\".bss\",#alloc,#write"; break;
      }
      O << "\n";
    }

    // getID Wrappers - Ensure consistent usage
    // Symbol names in SparcV9 assembly language have these rules:
    // (a) Must match { letter | _ | . | $ } { letter | _ | . | $ | digit }*
    // (b) A name beginning in "." is treated as a local name.
    std::string getID(const Function *F) {
      return Mang->getValueName(F);
    }
    std::string getID(const BasicBlock *BB) {
      return ".L_" + getID(BB->getParent()) + "_" + Mang->getValueName(BB);
    }
    std::string getID(const GlobalVariable *GV) {
      return Mang->getValueName(GV);
    }
    std::string getID(const Constant *CV) {
      return ".C_" + Mang->getValueName(CV);
    }
    std::string getID(const GlobalValue *GV) {
      if (const GlobalVariable *V = dyn_cast<GlobalVariable>(GV))
        return getID(V);
      else if (const Function *F = dyn_cast<Function>(GV))
        return getID(F);
      assert(0 && "Unexpected type of GlobalValue!");
      return "";
    }

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      setupMachineFunction(MF);
      emitFunction(MF);
      return false;
    }

    virtual bool doFinalization(Module &M) {
      emitGlobals(M);
      AsmPrinter::doFinalization(M);
      return false;
    }

    void emitFunction(MachineFunction &F);
  private :
    void emitBasicBlock(const MachineBasicBlock &MBB);
    void emitMachineInst(const MachineInstr *MI);
  
    unsigned int printOperands(const MachineInstr *MI, unsigned int opNum);
    void printOneOperand(const MachineOperand &Op, MachineOpCode opCode);

    bool OpIsBranchTargetLabel(const MachineInstr *MI, unsigned int opNum);
    bool OpIsMemoryAddressBase(const MachineInstr *MI, unsigned int opNum);
  
    unsigned getOperandMask(unsigned Opcode) {
      switch (Opcode) {
      case V9::SUBccr:
      case V9::SUBcci:   return 1 << 3;  // Remove CC argument
      default:      return 0;       // By default, don't hack operands...
      }
    }

    void emitGlobals(const Module &M);
    void printGlobalVariable(const GlobalVariable *GV);
  };

} // End anonymous namespace

inline bool
SparcV9AsmPrinter::OpIsBranchTargetLabel(const MachineInstr *MI,
                                       unsigned int opNum) {
  switch (MI->getOpcode()) {
  case V9::JMPLCALLr:
  case V9::JMPLCALLi:
  case V9::JMPLRETr:
  case V9::JMPLRETi:
    return (opNum == 0);
  default:
    return false;
  }
}

inline bool
SparcV9AsmPrinter::OpIsMemoryAddressBase(const MachineInstr *MI,
                                       unsigned int opNum) {
  if (TM.getInstrInfo()->isLoad(MI->getOpcode()))
    return (opNum == 0);
  else if (TM.getInstrInfo()->isStore(MI->getOpcode()))
    return (opNum == 1);
  else
    return false;
}

unsigned int
SparcV9AsmPrinter::printOperands(const MachineInstr *MI, unsigned opNum) {
  const MachineOperand& mop = MI->getOperand(opNum);
  if (OpIsBranchTargetLabel(MI, opNum)) {
    printOneOperand(mop, MI->getOpcode());
    O << "+";
    printOneOperand(MI->getOperand(opNum+1), MI->getOpcode());
    return 2;
  } else if (OpIsMemoryAddressBase(MI, opNum)) {
    O << "[";
    printOneOperand(mop, MI->getOpcode());
    O << "+";
    printOneOperand(MI->getOperand(opNum+1), MI->getOpcode());
    O << "]";
    return 2;
  } else {
    printOneOperand(mop, MI->getOpcode());
    return 1;
  }
}

void
SparcV9AsmPrinter::printOneOperand(const MachineOperand &mop,
                                   MachineOpCode opCode)
{
  bool needBitsFlag = true;
  
  if (mop.isHiBits32())
    O << "%lm(";
  else if (mop.isLoBits32())
    O << "%lo(";
  else if (mop.isHiBits64())
    O << "%hh(";
  else if (mop.isLoBits64())
    O << "%hm(";
  else
    needBitsFlag = false;
  
  switch (mop.getType())
    {
    case MachineOperand::MO_VirtualRegister:
    case MachineOperand::MO_CCRegister:
    case MachineOperand::MO_MachineRegister:
      {
        int regNum = (int)mop.getReg();
        
        if (regNum == TM.getRegInfo()->getInvalidRegNum()) {
          // better to print code with NULL registers than to die
          O << "<NULL VALUE>";
        } else {
          O << "%" << TM.getRegInfo()->getUnifiedRegName(regNum);
        }
        break;
      }
    
    case MachineOperand::MO_ConstantPoolIndex:
      {
        O << ".CPI_" << CurrentFnName << "_" << mop.getConstantPoolIndex();
        break;
      }

    case MachineOperand::MO_PCRelativeDisp:
      {
        const Value *Val = mop.getVRegValue();
        assert(Val && "\tNULL Value in SparcV9AsmPrinter");
        
        if (const BasicBlock *BB = dyn_cast<BasicBlock>(Val))
          O << getID(BB);
        else if (const Function *F = dyn_cast<Function>(Val))
          O << getID(F);
        else if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Val))
          O << getID(GV);
        else if (const Constant *CV = dyn_cast<Constant>(Val))
          O << getID(CV);
        else
          assert(0 && "Unrecognized value in SparcV9AsmPrinter");
        break;
      }
    
    case MachineOperand::MO_SignExtendedImmed:
      O << mop.getImmedValue();
      break;

    case MachineOperand::MO_UnextendedImmed:
      O << (uint64_t) mop.getImmedValue();
      break;
    
    default:
      O << mop;      // use dump field
      break;
    }
  
  if (needBitsFlag)
    O << ")";
}

void SparcV9AsmPrinter::emitMachineInst(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();

  if (Opcode == V9::PHI)
    return;  // Ignore Machine-PHI nodes.

  O << "\t" << TM.getInstrInfo()->getName(Opcode) << "\t";

  unsigned Mask = getOperandMask(Opcode);
  
  bool NeedComma = false;
  unsigned N = 1;
  for (unsigned OpNum = 0; OpNum < MI->getNumOperands(); OpNum += N)
    if (! ((1 << OpNum) & Mask)) {        // Ignore this operand?
      if (NeedComma) O << ", ";         // Handle comma outputting
      NeedComma = true;
      N = printOperands(MI, OpNum);
    } else
      N = 1;
  
  O << "\n";
  ++EmittedInsts;
}

void SparcV9AsmPrinter::emitBasicBlock(const MachineBasicBlock &MBB) {
  // Emit a label for the basic block
  O << getID(MBB.getBasicBlock()) << ":\n";

  // Loop over all of the instructions in the basic block...
  for (MachineBasicBlock::const_iterator MII = MBB.begin(), MIE = MBB.end();
       MII != MIE; ++MII)
    emitMachineInst(MII);
  O << "\n";  // Separate BB's with newlines
}

void SparcV9AsmPrinter::emitFunction(MachineFunction &MF) {
  O << "!****** Outputing Function: " << CurrentFnName << " ******\n";

  // Emit constant pool for this function
  const MachineConstantPool *MCP = MF.getConstantPool();
  const std::vector<Constant*> &CP = MCP->getConstants();

  enterSection(ReadOnlyData);
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    std::string cpiName = ".CPI_" + CurrentFnName + "_" + utostr(i);
    printConstant(CP[i], cpiName);
  }

  enterSection(Text);
  O << "\t.align\t4\n\t.global\t" << CurrentFnName << "\n";
  //O << "\t.type\t" << CurrentFnName << ",#function\n";
  O << "\t.type\t" << CurrentFnName << ", 2\n";
  O << CurrentFnName << ":\n";

  // Output code for all of the basic blocks in the function...
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end(); I != E;++I)
    emitBasicBlock(*I);

  // Output a .size directive so the debugger knows the extents of the function
  O << ".EndOf_" << CurrentFnName << ":\n\t.size "
           << CurrentFnName << ", .EndOf_"
           << CurrentFnName << "-" << CurrentFnName << "\n";

  // Put some spaces between the functions
  O << "\n\n";
}

void SparcV9AsmPrinter::printGlobalVariable(const GlobalVariable* GV) {
  if (GV->hasExternalLinkage())
    O << "\t.global\t" << getID(GV) << "\n";
  
  if (GV->hasInitializer() && ! GV->getInitializer()->isNullValue()) {
    printConstant(GV->getInitializer(), getID(GV));
  } else {
    const Type *ValTy = GV->getType()->getElementType();
    emitAlignment(TM.getTargetData().getTypeAlignmentShift(ValTy));
    O << "\t.type\t" << getID(GV) << ",#object\n";
    O << "\t.reserve\t" << getID(GV) << ","
      << TM.getTargetData().getTypeSize(GV->getType()->getElementType())
      << "\n";
  }
}

void SparcV9AsmPrinter::emitGlobals(const Module &M) {
  // Output global variables...
  for (Module::const_giterator GI = M.gbegin(), GE = M.gend(); GI != GE; ++GI)
    if (! GI->isExternal()) {
      assert(GI->hasInitializer());
      if (GI->isConstant())
        enterSection(ReadOnlyData);   // read-only, initialized data
      else if (GI->getInitializer()->isNullValue())
        enterSection(ZeroInitRWData); // read-write zero data
      else
        enterSection(InitRWData);     // read-write non-zero data

      printGlobalVariable(GI);
    }

  O << "\n";
}

FunctionPass *llvm::createAsmPrinterPass(std::ostream &Out, TargetMachine &TM) {
  return new SparcV9AsmPrinter(Out, TM);
}
