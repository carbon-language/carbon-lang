//===-- PPC64AsmPrinter.cpp - Print machine instrs to PowerPC assembly ----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PowerPC assembly language. This printer is
// the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asmprinter"
#include "PowerPC.h"
#include "PowerPCInstrInfo.h"
#include "PPC64TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/MathExtras.h"
#include "Support/Statistic.h"
#include "Support/StringExtras.h"
#include <set>

namespace llvm {

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

  struct Printer : public MachineFunctionPass {
    /// Output stream on which we're printing assembly code.
    ///
    std::ostream &O;

    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    PPC64TargetMachine &TM;

    /// Name-mangler for global names.
    ///
    Mangler *Mang;

    /// Map for labels corresponding to global variables
    ///
    std::map<const GlobalVariable*,std::string> GVToLabelMap;

    Printer(std::ostream &o, TargetMachine &tm) : O(o),
      TM(reinterpret_cast<PPC64TargetMachine&>(tm)), LabelNumber(0) {}

    /// Cache of mangled name for current function. This is
    /// recalculated at the beginning of each call to
    /// runOnMachineFunction().
    ///
    std::string CurrentFnName;

    /// Unique incrementer for label values for referencing Global values.
    ///
    unsigned LabelNumber;
  
    virtual const char *getPassName() const {
      return "PPC64 Assembly Printer";
    }

    void printMachineInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool elideOffsetKeyword = false);
    void printImmOp(const MachineOperand &MO, unsigned ArgType);
    void printConstantPool(MachineConstantPool *MCP);
    bool runOnMachineFunction(MachineFunction &F);    
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
    void emitGlobalConstant(const Constant* CV);
    void emitConstantValueOnly(const Constant *CV);
  };
} // end of anonymous namespace

/// createPPC64AsmPrinterPass - Returns a pass that prints the PPC
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form or not.
///
FunctionPass *createPPC64AsmPrinter(std::ostream &o,TargetMachine &tm) {
  return new Printer(o, tm);
}

/// isStringCompatible - Can we treat the specified array as a string?
/// Only if it is an array of ubytes or non-negative sbytes.
///
static bool isStringCompatible(const ConstantArray *CVA) {
  const Type *ETy = cast<ArrayType>(CVA->getType())->getElementType();
  if (ETy == Type::UByteTy) return true;
  if (ETy != Type::SByteTy) return false;

  for (unsigned i = 0; i < CVA->getNumOperands(); ++i)
    if (cast<ConstantSInt>(CVA->getOperand(i))->getValue() < 0)
      return false;

  return true;
}

/// toOctal - Convert the low order bits of X into an octal digit.
///
static inline char toOctal(int X) {
  return (X&7)+'0';
}

// Possible states while outputting ASCII strings
namespace {
  enum StringSection {
    None,
    Alpha,
    Numeric
  };
}

/// SwitchStringSection - manage the changes required to output bytes as
/// characters in a string vs. numeric decimal values
/// 
static inline void SwitchStringSection(std::ostream &O, StringSection NewSect,
                                       StringSection &Current) {
  if (Current == None) {
    if (NewSect == Alpha)
      O << "\t.byte \"";
    else if (NewSect == Numeric)
      O << "\t.byte ";
  } else if (Current == Alpha) {
    if (NewSect == None)
      O << "\"";
    else if (NewSect == Numeric) 
      O << "\"\n"
        << "\t.byte ";
  } else if (Current == Numeric) {
    if (NewSect == Alpha)
      O << '\n'
        << "\t.byte \"";
    else if (NewSect == Numeric)
      O << ", ";
  }

  Current = NewSect;
}

/// getAsCString - Return the specified array as a C compatible
/// string, only if the predicate isStringCompatible is true.
///
static void printAsCString(std::ostream &O, const ConstantArray *CVA) {
  assert(isStringCompatible(CVA) && "Array is not string compatible!");

  if (CVA->getNumOperands() == 0)
    return;

  StringSection Current = None;
  for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i) {
    unsigned char C = cast<ConstantInt>(CVA->getOperand(i))->getRawValue();
    if (C == '"') {
      SwitchStringSection(O, Alpha, Current);
      O << "\"\"";
    } else if (isprint(C)) {
      SwitchStringSection(O, Alpha, Current);
      O << C;
    } else {
      SwitchStringSection(O, Numeric, Current);
      O << utostr((unsigned)C);
    }
  }
  SwitchStringSection(O, None, Current);
  O << '\n';
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void Printer::emitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue())
    O << "0";
  else if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    assert(CB == ConstantBool::True);
    O << "1";
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
    O << CI->getValue();
  else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
    O << CI->getValue();
  else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV))
    // This is a constant address for a global variable or function.  Use the
    // name of the variable or function as the address value.
    O << Mang->getValueName(GV);
  else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData &TD = TM.getTargetData();
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      if (unsigned Offset = TD.getIndexedOffset(ptrVal->getType(), idxVec)) {
        O << "(";
        emitConstantValueOnly(ptrVal);
        O << ") + " << Offset;
      } else {
        emitConstantValueOnly(ptrVal);
      }
      break;
    }
    case Instruction::Cast: {
      // Support only non-converting or widening casts for now, that is, ones
      // that do not involve a change in value.  This assertion is really gross,
      // and may not even be a complete check.
      Constant *Op = CE->getOperand(0);
      const Type *OpTy = Op->getType(), *Ty = CE->getType();

      // Remember, kids, pointers on x86 can be losslessly converted back and
      // forth into 32-bit or wider integers, regardless of signedness. :-P
      assert(((isa<PointerType>(OpTy)
               && (Ty == Type::LongTy || Ty == Type::ULongTy
                   || Ty == Type::IntTy || Ty == Type::UIntTy))
              || (isa<PointerType>(Ty)
                  && (OpTy == Type::LongTy || OpTy == Type::ULongTy
                      || OpTy == Type::IntTy || OpTy == Type::UIntTy))
              || (((TD.getTypeSize(Ty) >= TD.getTypeSize(OpTy))
                   && OpTy->isLosslesslyConvertibleTo(Ty))))
             && "FIXME: Don't yet support this kind of constant cast expr");
      O << "(";
      emitConstantValueOnly(Op);
      O << ")";
      break;
    }
    case Instruction::Add:
      O << "(";
      emitConstantValueOnly(CE->getOperand(0));
      O << ") + (";
      emitConstantValueOnly(CE->getOperand(1));
      O << ")";
      break;
    default:
      assert(0 && "Unsupported operator!");
    }
  } else {
    assert(0 && "Unknown constant value!");
  }
}

// Print a constant value or values, with the appropriate storage class as a
// prefix.
void Printer::emitGlobalConstant(const Constant *CV) {  
  const TargetData &TD = TM.getTargetData();

  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (isStringCompatible(CVA)) {
      printAsCString(O, CVA);
    } else { // Not a string.  Print the values in successive locations
      for (unsigned i=0, e = CVA->getNumOperands(); i != e; i++)
        emitGlobalConstant(CVA->getOperand(i));
    }
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    // Print the fields in successive locations. Pad to align if needed!
    const StructLayout *cvsLayout = TD.getStructLayout(CVS->getType());
    unsigned sizeSoFar = 0;
    for (unsigned i = 0, e = CVS->getNumOperands(); i != e; i++) {
      const Constant* field = CVS->getOperand(i);

      // Check if padding is needed and insert one or more 0s.
      unsigned fieldSize = TD.getTypeSize(field->getType());
      unsigned padSize = ((i == e-1? cvsLayout->StructSize
                           : cvsLayout->MemberOffsets[i+1])
                          - cvsLayout->MemberOffsets[i]) - fieldSize;
      sizeSoFar += fieldSize + padSize;

      // Now print the actual field value
      emitGlobalConstant(field);

      // Insert the field padding unless it's zero bytes...
      if (padSize)
        O << "\t.space\t " << padSize << "\n";      
    }
    assert(sizeSoFar == cvsLayout->StructSize &&
           "Layout of constant struct may be incorrect!");
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    // FP Constants are printed as integer constants to avoid losing
    // precision...
    double Val = CFP->getValue();
    switch (CFP->getType()->getTypeID()) {
    default: assert(0 && "Unknown floating point type!");
    case Type::FloatTyID: {
      union FU {                            // Abide by C TBAA rules
        float FVal;
        unsigned UVal;
      } U;
      U.FVal = Val;
      O << "\t.long " << U.UVal << "\t# float " << Val << "\n";
      return;
    }
    case Type::DoubleTyID: {
      union DU {                            // Abide by C TBAA rules
        double FVal;
        uint64_t UVal;
        struct {
          uint32_t MSWord;
          uint32_t LSWord;
        } T;
      } U;
      U.FVal = Val;
      
      O << ".long " << U.T.MSWord << "\t# double most significant word " 
        << Val << "\n";
      O << ".long " << U.T.LSWord << "\t# double least significant word " 
        << Val << "\n";
      return;
    }
    }
  } else if (CV->getType() == Type::ULongTy || CV->getType() == Type::LongTy) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
      union DU {                            // Abide by C TBAA rules
        int64_t UVal;
        struct {
          uint32_t MSWord;
          uint32_t LSWord;
        } T;
      } U;
      U.UVal = CI->getRawValue();
        
      O << ".long " << U.T.MSWord << "\t# Double-word most significant word " 
        << U.UVal << "\n";
      O << ".long " << U.T.LSWord << "\t# Double-word least significant word " 
        << U.UVal << "\n";
      return;
    }
  }

  const Type *type = CV->getType();
  O << "\t";
  switch (type->getTypeID()) {
  case Type::UByteTyID: case Type::SByteTyID:
    O << "\t.byte";
    break;
  case Type::UShortTyID: case Type::ShortTyID:
    O << "\t.short";
    break;
  case Type::BoolTyID: 
  case Type::PointerTyID:
  case Type::UIntTyID: case Type::IntTyID:
    O << "\t.long";
    break;
  case Type::ULongTyID: case Type::LongTyID:    
    assert (0 && "Should have already output double-word constant.");
  case Type::FloatTyID: case Type::DoubleTyID:
    assert (0 && "Should have already output floating point constant.");
  default:
    if (CV == Constant::getNullValue(type)) {  // Zero initializer?
      O << "\t.space " << TD.getTypeSize(type) << "\n";      
      return;
    }
    std::cerr << "Can't handle printing: " << *CV;
    abort();
    break;
  }
  O << ' ';
  emitConstantValueOnly(CV);
  O << '\n';
}

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void Printer::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();
 
  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.const\n";
    O << "\t.align " << (unsigned)TD.getTypeAlignment(CP[i]->getType())
      << "\n";
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t;"
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool Printer::runOnMachineFunction(MachineFunction &MF) {
  CurrentFnName = MF.getFunction()->getName();

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out header for the function.
  O << "\t.csect .text[PR]\n"
    << "\t.align 2\n"
    << "\t.globl "  << CurrentFnName << '\n'
    << "\t.globl ." << CurrentFnName << '\n'
    << "\t.csect "  << CurrentFnName << "[DS],3\n"
    << CurrentFnName << ":\n"
    << "\t.llong ." << CurrentFnName << ", TOC[tc0], 0\n"
    << "\t.csect .text[PR]\n"
    << '.' << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << "LBB" << CurrentFnName << "_" << I->getNumber() << ":\t# "
      << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
      II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }
  ++LabelNumber;

  O << "LT.." << CurrentFnName << ":\n"
    << "\t.long 0\n"
    << "\t.byte 0,0,32,65,128,0,0,0\n"
    << "\t.long LT.." << CurrentFnName << "-." << CurrentFnName << '\n'
    << "\t.short 3\n"
    << "\t.byte \"" << CurrentFnName << "\"\n"
    << "\t.align 2\n";

  // We didn't modify anything.
  return false;
}

void Printer::printOp(const MachineOperand &MO,
                      bool elideOffsetKeyword /* = false */) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  int new_symbol;
  
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
  case MachineOperand::MO_CCRegister: {
    // On AIX, do not print out the 'r' in register names
    const char *regName = RI.get(MO.getReg()).Name;
    O << &regName[1];
    return;
  }

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    std::cerr << "printOp() does not handle immediate values\n";
    abort();
    return;

  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building PPC MachineInstrs";
    abort();
    return;
    
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber() << "\t# "
      << MBBOp->getBasicBlock()->getName();
    return;
  }

  case MachineOperand::MO_ConstantPoolIndex:
    O << ".CPI" << CurrentFnName << "_" << MO.getConstantPoolIndex();
    return;

  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    return;

  case MachineOperand::MO_GlobalAddress:
    if (!elideOffsetKeyword) {
      GlobalValue *GV = MO.getGlobal();

      if (Function *F = dyn_cast<Function>(GV)) {
        O << "." << F->getName();
      } else if (GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
        // output the label name
        O << GVToLabelMap[GVar];
      }
    }
    return;
    
  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

void Printer::printImmOp(const MachineOperand &MO, unsigned ArgType) {
  int Imm = MO.getImmedValue();
  if (ArgType == PPCII::Simm16 || ArgType == PPCII::Disimm16) {
    O << (short)Imm;
  } else if (ArgType == PPCII::Zimm16) {
    O << (unsigned short)Imm;
  } else {
    O << Imm;
  }
}

/// printMachineInstruction -- Print out a single PPC LLVM instruction
/// MI in Darwin syntax to the current output stream.
///
void Printer::printMachineInstruction(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const TargetInstrDescriptor &Desc = TII.get(Opcode);
  unsigned i;

  unsigned ArgCount = MI->getNumOperands();
  unsigned ArgType[] = {
    (Desc.TSFlags >> PPCII::Arg0TypeShift) & PPCII::ArgTypeMask,
    (Desc.TSFlags >> PPCII::Arg1TypeShift) & PPCII::ArgTypeMask,
    (Desc.TSFlags >> PPCII::Arg2TypeShift) & PPCII::ArgTypeMask,
    (Desc.TSFlags >> PPCII::Arg3TypeShift) & PPCII::ArgTypeMask,
    (Desc.TSFlags >> PPCII::Arg4TypeShift) & PPCII::ArgTypeMask
  };
  assert(((Desc.TSFlags & PPCII::VMX) == 0) &&
         "Instruction requires VMX support");
  ++EmittedInsts;

  // CALLpcrel and CALLindirect are handled specially here to print only the
  // appropriate number of args that the assembler expects.  This is because
  // may have many arguments appended to record the uses of registers that are
  // holding arguments to the called function.
  if (Opcode == PPC::COND_BRANCH) {
    std::cerr << "Error: untranslated conditional branch psuedo instruction!\n";
    abort();
  } else if (Opcode == PPC::IMPLICIT_DEF) {
    O << "# IMPLICIT DEF ";
    printOp(MI->getOperand(0));
    O << "\n";
    return;
  } else if (Opcode == PPC::CALLpcrel) {
    O << TII.getName(Opcode) << " ";
    printOp(MI->getOperand(0));
    O << "\n";
    return;
  } else if (Opcode == PPC::CALLindirect) {
    O << TII.getName(Opcode) << " ";
    printImmOp(MI->getOperand(0), ArgType[0]);
    O << ", ";
    printImmOp(MI->getOperand(1), ArgType[0]);
    O << "\n";
    return;
  } else if (Opcode == PPC::MovePCtoLR) {
    // FIXME: should probably be converted to cout.width and cout.fill
    O << "bl \"L0000" << LabelNumber << "$pb\"\n";
    O << "\"L0000" << LabelNumber << "$pb\":\n";
    O << "\tmflr ";
    printOp(MI->getOperand(0));
    O << "\n";
    return;
  }

  O << TII.getName(Opcode) << " ";
  if (Opcode == PPC::LD || Opcode == PPC::LWA || 
      Opcode == PPC::STDU || Opcode == PPC::STDUX) {
    printOp(MI->getOperand(0));
    O << ", ";
    MachineOperand MO = MI->getOperand(1);
    if (MO.isImmediate())
      printImmOp(MO, ArgType[1]);
    else
      printOp(MO);
    O << "(";
    printOp(MI->getOperand(2));
    O << ")\n";
  } else if (Opcode == PPC::BLR || Opcode == PPC::NOP) {
    // FIXME: BuildMI() should handle 0 params
    O << "\n";
  } else if (ArgCount == 3 && ArgType[1] == PPCII::Disimm16) {
    printOp(MI->getOperand(0));
    O << ", ";
    printImmOp(MI->getOperand(1), ArgType[1]);
    O << "(";
    if (MI->getOperand(2).hasAllocatedReg() &&
        MI->getOperand(2).getReg() == PPC::R0)
      O << "0";
    else
      printOp(MI->getOperand(2));
    O << ")\n";
  } else {
    for (i = 0; i < ArgCount; ++i) {
      // addi and friends
      if (i == 1 && ArgCount == 3 && ArgType[2] == PPCII::Simm16 &&
          MI->getOperand(1).hasAllocatedReg() && 
          MI->getOperand(1).getReg() == PPC::R0) {
        O << "0";
      // for long branch support, bc $+8
      } else if (i == 1 && ArgCount == 2 && MI->getOperand(1).isImmediate() &&
                 TII.isBranch(MI->getOpcode())) {
        O << "$+8";
        assert(8 == MI->getOperand(i).getImmedValue()
          && "branch off PC not to pc+8?");
        //printOp(MI->getOperand(i));
      } else if (MI->getOperand(i).isImmediate()) {
        printImmOp(MI->getOperand(i), ArgType[i]);
      } else {
        printOp(MI->getOperand(i));
      }
      if (ArgCount - 1 == i)
        O << "\n";
      else
        O << ", ";
    }
  }
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

bool Printer::doInitialization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  O << "\t.machine \"ppc64\"\n" 
    << "\t.toc\n"
    << "\t.csect .text[PR]\n";

  // Print out module-level global variables
  for (Module::const_giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    if (!I->hasInitializer())
      continue;
 
    std::string Name = I->getName();
    Constant *C = I->getInitializer();
    // N.B.: We are defaulting to writable strings
    if (I->hasExternalLinkage()) { 
      O << "\t.globl " << Name << '\n'
        << "\t.csect .data[RW],3\n";
    } else {
      O << "\t.csect _global.rw_c[RW],3\n";
    }
    O << Name << ":\n";
    emitGlobalConstant(C);
  }

  // Output labels for globals
  if (M.gbegin() != M.gend()) O << "\t.toc\n";
  for (Module::const_giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    const GlobalVariable *GV = I;
    // Do not output labels for unused variables
    if (GV->isExternal() && GV->use_begin() == GV->use_end())
      continue;

    std::string Name = GV->getName();
    std::string Label = "LC.." + utostr(LabelNumber++);
    GVToLabelMap[GV] = Label;
    O << Label << ":\n"
      << "\t.tc " << Name << "[TC]," << Name;
    if (GV->isExternal()) O << "[RW]";
    O << '\n';
  }

  Mang = new Mangler(M, true);
  return false; // success
}

bool Printer::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  // Print out module-level global variables
  for (Module::const_giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    if (I->hasInitializer() || I->hasExternalLinkage())
      continue;

    std::string Name = I->getName();
    if (I->hasInternalLinkage()) {
      O << "\t.lcomm " << Name << ",16,_global.bss_c";
    } else {
      O << "\t.comm " << Name << "," << TD.getTypeSize(I->getType())
        << "," << log2((unsigned)TD.getTypeAlignment(I->getType()));
    }
    O << "\t\t# ";
    WriteAsOperand(O, I, true, true, &M);
    O << "\n";
  }

  O << "_section_.text:\n"
    << "\t.csect .data[RW],3\n"
    << "\t.llong _section_.text\n";

  delete Mang;
  return false; // success
}

} // End llvm namespace
