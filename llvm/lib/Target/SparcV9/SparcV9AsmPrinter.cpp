//===-- SparcV9AsmPrinter.cpp - Emit SparcV9 Specific .s File --------------==//
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
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Statistic.h"
#include "SparcV9Internals.h"
#include "MachineFunctionInfo.h"
#include <string>
using namespace llvm;

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

  //===--------------------------------------------------------------------===//
  // Utility functions

  /// getAsCString - Return the specified array as a C compatible string, only
  /// if the predicate isString() is true.
  ///
  std::string getAsCString(const ConstantArray *CVA) {
    assert(CVA->isString() && "Array is not string compatible!");

    std::string Result = "\"";
    for (unsigned i = 0; i != CVA->getNumOperands(); ++i) {
      unsigned char C = cast<ConstantInt>(CVA->getOperand(i))->getRawValue();

      if (C == '"') {
        Result += "\\\"";
      } else if (C == '\\') {
        Result += "\\\\";
      } else if (isprint(C)) {
        Result += C;
      } else {
        Result += '\\';    // print all other chars as octal value
        // Convert C to octal representation
        Result += ((C >> 6) & 7) + '0';
        Result += ((C >> 3) & 7) + '0';
        Result += ((C >> 0) & 7) + '0';
      }
    }
    Result += "\"";

    return Result;
  }

  inline bool ArrayTypeIsString(const ArrayType* arrayType) {
    return (arrayType->getElementType() == Type::UByteTy ||
            arrayType->getElementType() == Type::SByteTy);
  }

  unsigned findOptimalStorageSize(const TargetMachine &TM, const Type *Ty) {
    // All integer types smaller than ints promote to 4 byte integers.
    if (Ty->isIntegral() && Ty->getPrimitiveSize() < 4)
      return 4;

    return TM.getTargetData().getTypeSize(Ty);
  }


  inline const std::string
  TypeToDataDirective(const Type* type) {
    switch(type->getTypeID()) {
    case Type::BoolTyID: case Type::UByteTyID: case Type::SByteTyID:
      return ".byte";
    case Type::UShortTyID: case Type::ShortTyID:
      return ".half";
    case Type::UIntTyID: case Type::IntTyID:
      return ".word";
    case Type::ULongTyID: case Type::LongTyID: case Type::PointerTyID:
      return ".xword";
    case Type::FloatTyID:
      return ".word";
    case Type::DoubleTyID:
      return ".xword";
    case Type::ArrayTyID:
      if (ArrayTypeIsString((ArrayType*) type))
        return ".ascii";
      else
        return "<InvaliDataTypeForPrinting>";
    default:
      return "<InvaliDataTypeForPrinting>";
    }
  }

  /// Get the size of the constant for the given target.
  /// If this is an unsized array, return 0.
  /// 
  inline unsigned int
  ConstantToSize(const Constant* CV, const TargetMachine& target) {
    if (const ConstantArray* CVA = dyn_cast<ConstantArray>(CV)) {
      const ArrayType *aty = cast<ArrayType>(CVA->getType());
      if (ArrayTypeIsString(aty))
        return 1 + CVA->getNumOperands();
    }
  
    return findOptimalStorageSize(target, CV->getType());
  }

  /// Align data larger than one L1 cache line on L1 cache line boundaries.
  /// Align all smaller data on the next higher 2^x boundary (4, 8, ...).
  /// 
  inline unsigned int
  SizeToAlignment(unsigned int size, const TargetMachine& target) {
    const unsigned short cacheLineSize = 16;
    if (size > (unsigned) cacheLineSize / 2)
      return cacheLineSize;
    else
      for (unsigned sz=1; /*no condition*/; sz *= 2)
        if (sz >= size)
          return sz;
  }

  /// Get the size of the type and then use SizeToAlignment.
  /// 
  inline unsigned int
  TypeToAlignment(const Type* type, const TargetMachine& target) {
    return SizeToAlignment(findOptimalStorageSize(target, type), target);
  }

  /// Get the size of the constant and then use SizeToAlignment.
  /// Handles strings as a special case;
  inline unsigned int
  ConstantToAlignment(const Constant* CV, const TargetMachine& target) {
    if (const ConstantArray* CVA = dyn_cast<ConstantArray>(CV))
      if (ArrayTypeIsString(cast<ArrayType>(CVA->getType())))
        return SizeToAlignment(1 + CVA->getNumOperands(), target);
  
    return TypeToAlignment(CV->getType(), target);
  }

} // End anonymous namespace

namespace {
  enum Sections {
    Unknown,
    Text,
    ReadOnlyData,
    InitRWData,
    ZeroInitRWData,
  };

  class AsmPrinter {
    // Mangle symbol names appropriately
    Mangler *Mang;

  public:
    std::ostream &O;
    const TargetMachine &TM;

    enum Sections CurSection;

    AsmPrinter(std::ostream &os, const TargetMachine &T)
      : /* idTable(0), */ O(os), TM(T), CurSection(Unknown) {}
  
    ~AsmPrinter() {
      delete Mang;
    }

    // (start|end)(Module|Function) - Callback methods invoked by subclasses
    void startModule(Module &M) {
      Mang = new Mangler(M);
    }

    void PrintZeroBytesToPad(int numBytes) {
      //
      // Always use single unsigned bytes for padding.  We don't know upon
      // what data size the beginning address is aligned, so using anything
      // other than a byte may cause alignment errors in the assembler.
      //
      while (numBytes--)
        printSingleConstantValue(Constant::getNullValue(Type::UByteTy));
    }

    /// Print a single constant value.
    ///
    void printSingleConstantValue(const Constant* CV);

    /// Print a constant value or values (it may be an aggregate).
    /// Uses printSingleConstantValue() to print each individual value.
    ///
    void printConstantValueOnly(const Constant* CV, int numPadBytesAfter = 0);

    // Print a constant (which may be an aggregate) prefixed by all the
    // appropriate directives.  Uses printConstantValueOnly() to print the
    // value or values.
    void printConstant(const Constant* CV, std::string valID = "") {
      if (valID.length() == 0)
        valID = getID(CV);
  
      O << "\t.align\t" << ConstantToAlignment(CV, TM) << "\n";
  
      // Print .size and .type only if it is not a string.
      if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV))
        if (CVA->isString()) {
          // print it as a string and return
          O << valID << ":\n";
          O << "\t" << ".ascii" << "\t" << getAsCString(CVA) << "\n";
          return;
        }
  
      O << "\t.type" << "\t" << valID << ",#object\n";

      unsigned int constSize = ConstantToSize(CV, TM);
      if (constSize)
        O << "\t.size" << "\t" << valID << "," << constSize << "\n";
  
      O << valID << ":\n";
  
      printConstantValueOnly(CV);
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

    // Combines expressions 
    inline std::string ConstantArithExprToString(const ConstantExpr* CE,
                                                 const TargetMachine &TM,
                                                 const std::string &op) {
      return "(" + valToExprString(CE->getOperand(0), TM) + op
        + valToExprString(CE->getOperand(1), TM) + ")";
    }

    /// ConstantExprToString() - Convert a ConstantExpr to an asm expression
    /// and return this as a string.
    ///
    std::string ConstantExprToString(const ConstantExpr* CE,
                                     const TargetMachine& target);

    /// valToExprString - Helper function for ConstantExprToString().
    /// Appends result to argument string S.
    /// 
    std::string valToExprString(const Value* V, const TargetMachine& target);
  };
} // End anonymous namespace


/// Print a single constant value.
///
void AsmPrinter::printSingleConstantValue(const Constant* CV) {
  assert(CV->getType() != Type::VoidTy &&
         CV->getType() != Type::LabelTy &&
         "Unexpected type for Constant");
  
  assert((!isa<ConstantArray>(CV) && ! isa<ConstantStruct>(CV))
         && "Aggregate types should be handled outside this function");
  
  O << "\t" << TypeToDataDirective(CV->getType()) << "\t";
  
  if (const GlobalValue* GV = dyn_cast<GlobalValue>(CV)) {
    O << getID(GV) << "\n";
  } else if (isa<ConstantPointerNull>(CV) || isa<UndefValue>(CV)) {
    // Null pointer value
    O << "0\n";
  } else if (const ConstantExpr* CE = dyn_cast<ConstantExpr>(CV)) { 
    // Constant expression built from operators, constants, and symbolic addrs
    O << ConstantExprToString(CE, TM) << "\n";
  } else if (CV->getType()->isPrimitiveType()) {
    // Check primitive types last
    if (isa<UndefValue>(CV)) {
      O << "0\n";
    } else if (CV->getType()->isFloatingPoint()) {
      // FP Constants are printed as integer constants to avoid losing
      // precision...
      double Val = cast<ConstantFP>(CV)->getValue();
      if (CV->getType() == Type::FloatTy) {
        float FVal = (float)Val;
        char *ProxyPtr = (char*)&FVal;        // Abide by C TBAA rules
        O << *(unsigned int*)ProxyPtr;            
      } else if (CV->getType() == Type::DoubleTy) {
        char *ProxyPtr = (char*)&Val;         // Abide by C TBAA rules
        O << *(uint64_t*)ProxyPtr;            
      } else {
        assert(0 && "Unknown floating point type!");
      }
        
      O << "\t! " << CV->getType()->getDescription()
            << " value: " << Val << "\n";
    } else if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
      O << (int)CB->getValue() << "\n";
    } else {
      WriteAsOperand(O, CV, false, false) << "\n";
    }
  } else {
    assert(0 && "Unknown elementary type for constant");
  }
}

/// Print a constant value or values (it may be an aggregate).
/// Uses printSingleConstantValue() to print each individual value.
///
void AsmPrinter::printConstantValueOnly(const Constant* CV,
                                        int numPadBytesAfter) {
  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (CVA->isString()) {
      // print the string alone and return
      O << "\t" << ".ascii" << "\t" << getAsCString(CVA) << "\n";
    } else {
      // Not a string.  Print the values in successive locations
      for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
        printConstantValueOnly(CVA->getOperand(i));
    }
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    // Print the fields in successive locations. Pad to align if needed!
    const StructLayout *cvsLayout =
      TM.getTargetData().getStructLayout(CVS->getType());
    unsigned sizeSoFar = 0;
    for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
      const Constant* field = CVS->getOperand(i);

      // Check if padding is needed and insert one or more 0s.
      unsigned fieldSize =
        TM.getTargetData().getTypeSize(field->getType());
      int padSize = ((i == e-1? cvsLayout->StructSize
                      : cvsLayout->MemberOffsets[i+1])
                     - cvsLayout->MemberOffsets[i]) - fieldSize;
      sizeSoFar += (fieldSize + padSize);

      // Now print the actual field value
      printConstantValueOnly(field, padSize);
    }
    assert(sizeSoFar == cvsLayout->StructSize &&
           "Layout of constant struct may be incorrect!");
  } else if (isa<ConstantAggregateZero>(CV) || isa<UndefValue>(CV)) {
    PrintZeroBytesToPad(TM.getTargetData().getTypeSize(CV->getType()));
  } else
    printSingleConstantValue(CV);

  if (numPadBytesAfter)
    PrintZeroBytesToPad(numPadBytesAfter);
}

/// ConstantExprToString() - Convert a ConstantExpr to an asm expression
/// and return this as a string.
///
std::string AsmPrinter::ConstantExprToString(const ConstantExpr* CE,
                                             const TargetMachine& target) {
  std::string S;
  switch(CE->getOpcode()) {
  case Instruction::GetElementPtr:
    { // generate a symbolic expression for the byte address
      const Value* ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      const TargetData &TD = target.getTargetData();
      S += "(" + valToExprString(ptrVal, target) + ") + ("
        + utostr(TD.getIndexedOffset(ptrVal->getType(),idxVec)) + ")";
      break;
    }

  case Instruction::Cast:
    // Support only non-converting casts for now, i.e., a no-op.
    // This assertion is not a complete check.
    assert(target.getTargetData().getTypeSize(CE->getType()) ==
           target.getTargetData().getTypeSize(CE->getOperand(0)->getType()));
    S += "(" + valToExprString(CE->getOperand(0), target) + ")";
    break;

  case Instruction::Add:
    S += ConstantArithExprToString(CE, target, ") + (");
    break;

  case Instruction::Sub:
    S += ConstantArithExprToString(CE, target, ") - (");
    break;

  case Instruction::Mul:
    S += ConstantArithExprToString(CE, target, ") * (");
    break;

  case Instruction::Div:
    S += ConstantArithExprToString(CE, target, ") / (");
    break;

  case Instruction::Rem:
    S += ConstantArithExprToString(CE, target, ") % (");
    break;

  case Instruction::And:
    // Logical && for booleans; bitwise & otherwise
    S += ConstantArithExprToString(CE, target,
                                   ((CE->getType() == Type::BoolTy)? ") && (" : ") & ("));
    break;

  case Instruction::Or:
    // Logical || for booleans; bitwise | otherwise
    S += ConstantArithExprToString(CE, target,
                                   ((CE->getType() == Type::BoolTy)? ") || (" : ") | ("));
    break;

  case Instruction::Xor:
    // Bitwise ^ for all types
    S += ConstantArithExprToString(CE, target, ") ^ (");
    break;

  default:
    assert(0 && "Unsupported operator in ConstantExprToString()");
    break;
  }

  return S;
}

/// valToExprString - Helper function for ConstantExprToString().
/// Appends result to argument string S.
/// 
std::string AsmPrinter::valToExprString(const Value* V,
                                        const TargetMachine& target) {
  std::string S;
  bool failed = false;
  if (const GlobalValue* GV = dyn_cast<GlobalValue>(V)) {
    S += getID(GV);
  } else if (const Constant* CV = dyn_cast<Constant>(V)) { // symbolic or known
    if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV))
      S += std::string(CB == ConstantBool::True ? "1" : "0");
    else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
      S += itostr(CI->getValue());
    else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
      S += utostr(CI->getValue());
    else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV))
      S += ftostr(CFP->getValue());
    else if (isa<ConstantPointerNull>(CV) || isa<UndefValue>(CV))
      S += "0";
    else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV))
      S += ConstantExprToString(CE, target);
    else
      failed = true;
  } else
    failed = true;

  if (failed) {
    assert(0 && "Cannot convert value to string");
    S += "<illegal-value>";
  }
  return S;
}

namespace {

  struct SparcV9AsmPrinter : public FunctionPass, public AsmPrinter {
    inline SparcV9AsmPrinter(std::ostream &os, const TargetMachine &t)
      : AsmPrinter(os, t) {}

    const Function *currFunction;

    const char *getPassName() const {
      return "Output SparcV9 Assembly for Functions";
    }

    virtual bool doInitialization(Module &M) {
      startModule(M);
      return false;
    }

    virtual bool runOnFunction(Function &F) {
      currFunction = &F;
      emitFunction(F);
      return false;
    }

    virtual bool doFinalization(Module &M) {
      emitGlobals(M);
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    void emitFunction(const Function &F);
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
        O << ".CPI_" << getID(currFunction)
              << "_" << mop.getConstantPoolIndex();
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

void SparcV9AsmPrinter::emitFunction(const Function &F) {
  std::string CurrentFnName = getID(&F);
  MachineFunction &MF = MachineFunction::get(&F);
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
  
  if (GV->hasInitializer() &&
      !(GV->getInitializer()->isNullValue() ||
        isa<UndefValue>(GV->getInitializer()))) {
    printConstant(GV->getInitializer(), getID(GV));
  } else {
    O << "\t.align\t" << TypeToAlignment(GV->getType()->getElementType(),
                                                TM) << "\n";
    O << "\t.type\t" << getID(GV) << ",#object\n";
    O << "\t.reserve\t" << getID(GV) << ","
      << findOptimalStorageSize(TM, GV->getType()->getElementType())
      << "\n";
  }
}

void SparcV9AsmPrinter::emitGlobals(const Module &M) {
  // Output global variables...
  for (Module::const_global_iterator GI = M.global_begin(), GE = M.global_end(); GI != GE; ++GI)
    if (! GI->isExternal()) {
      assert(GI->hasInitializer());
      if (GI->isConstant())
        enterSection(ReadOnlyData);   // read-only, initialized data
      else if (GI->getInitializer()->isNullValue() ||
               isa<UndefValue>(GI->getInitializer()))
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
