//===-- EmitAssembly.cpp - Emit Sparc Specific .s File ---------------------==//
//
// This file implements all of the stuff neccesary to output a .s file from
// LLVM.  The code in this file assumes that the specified module has already
// been compiled into the internal data structures of the Module.
//
// This code largely consists of two LLVM Pass's: a FunctionPass and a Pass.
// The FunctionPass is pipelined together with all of the rest of the code
// generation stages, and the Pass runs at the end to emit code for global
// variables and such.
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/SlotCalculator.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "Support/StringExtras.h"
using std::string;

namespace {

class GlobalIdTable: public Annotation {
  static AnnotationID AnnotId;
  friend class AsmPrinter;              // give access to AnnotId
  
  typedef hash_map<const Value*, int> ValIdMap;
  typedef ValIdMap::const_iterator ValIdMapConstIterator;
  typedef ValIdMap::      iterator ValIdMapIterator;
public:
  SlotCalculator Table;    // map anonymous values to unique integer IDs
  ValIdMap valToIdMap;     // used for values not handled by SlotCalculator 
  
  GlobalIdTable(Module* M) : Annotation(AnnotId), Table(M, true) {}
};

AnnotationID GlobalIdTable::AnnotId =
  AnnotationManager::getID("ASM PRINTER GLOBAL TABLE ANNOT");
  
//===---------------------------------------------------------------------===//
//   Code Shared By the two printer passes, as a mixin
//===---------------------------------------------------------------------===//

class AsmPrinter {
  GlobalIdTable* idTable;
public:
  std::ostream &toAsm;
  const TargetMachine &Target;
  
  enum Sections {
    Unknown,
    Text,
    ReadOnlyData,
    InitRWData,
    ZeroInitRWData,
  } CurSection;

  AsmPrinter(std::ostream &os, const TargetMachine &T)
    : idTable(0), toAsm(os), Target(T), CurSection(Unknown) {}
  
  // (start|end)(Module|Function) - Callback methods to be invoked by subclasses
  void startModule(Module &M) {
    // Create the global id table if it does not already exist
    idTable = (GlobalIdTable*)M.getAnnotation(GlobalIdTable::AnnotId);
    if (idTable == NULL) {
      idTable = new GlobalIdTable(&M);
      M.addAnnotation(idTable);
    }
  }
  void startFunction(Function &F) {
    // Make sure the slot table has information about this function...
    idTable->Table.incorporateFunction(&F);
  }
  void endFunction(Function &) {
    idTable->Table.purgeFunction();  // Forget all about F
  }
  void endModule() {
  }

  // Check if a value is external or accessible from external code.
  bool isExternal(const Value* V) {
    const GlobalValue *GV = dyn_cast<GlobalValue>(V);
    return GV && GV->hasExternalLinkage();
  }
  
  // enterSection - Use this method to enter a different section of the output
  // executable.  This is used to only output neccesary section transitions.
  //
  void enterSection(enum Sections S) {
    if (S == CurSection) return;        // Only switch section if neccesary
    CurSection = S;

    toAsm << "\n\t.section ";
    switch (S)
      {
      default: assert(0 && "Bad section name!");
      case Text:         toAsm << "\".text\""; break;
      case ReadOnlyData: toAsm << "\".rodata\",#alloc"; break;
      case InitRWData:   toAsm << "\".data\",#alloc,#write"; break;
      case ZeroInitRWData: toAsm << "\".bss\",#alloc,#write"; break;
      }
    toAsm << "\n";
  }

  static string getValidSymbolName(const string &S) {
    string Result;
    
    // Symbol names in Sparc assembly language have these rules:
    // (a) Must match { letter | _ | . | $ } { letter | _ | . | $ | digit }*
    // (b) A name beginning in "." is treated as a local name.
    // 
    if (isdigit(S[0]))
      Result = "ll";
    
    for (unsigned i = 0; i < S.size(); ++i)
      {
        char C = S[i];
        if (C == '_' || C == '.' || C == '$' || isalpha(C) || isdigit(C))
          Result += C;
        else
          {
            Result += '_';
            Result += char('0' + ((unsigned char)C >> 4));
            Result += char('0' + (C & 0xF));
          }
      }
    return Result;
  }

  // getID - Return a valid identifier for the specified value.  Base it on
  // the name of the identifier if possible (qualified by the type), and
  // use a numbered value based on prefix otherwise.
  // FPrefix is always prepended to the output identifier.
  //
  string getID(const Value *V, const char *Prefix, const char *FPrefix = 0) {
    string Result = FPrefix ? FPrefix : "";  // "Forced prefix"

    Result +=  V->hasName() ? V->getName() : string(Prefix);

    // Qualify all internal names with a unique id.
    if (!isExternal(V)) {
      int valId = idTable->Table.getValSlot(V);
      if (valId == -1) {
        GlobalIdTable::ValIdMapConstIterator I = idTable->valToIdMap.find(V);
        if (I == idTable->valToIdMap.end())
          valId = idTable->valToIdMap[V] = idTable->valToIdMap.size();
        else
          valId = I->second;
      }
      Result = Result + "_" + itostr(valId);

      // Replace or prefix problem characters in the name
      Result = getValidSymbolName(Result);
    }

    return Result;
  }
  
  // getID Wrappers - Ensure consistent usage...
  string getID(const Function *F) {
    return getID(F, "LLVMFunction_");
  }
  string getID(const BasicBlock *BB) {
    return getID(BB, "LL", (".L_"+getID(BB->getParent())+"_").c_str());
  }
  string getID(const GlobalVariable *GV) {
    return getID(GV, "LLVMGlobal_");
  }
  string getID(const Constant *CV) {
    return getID(CV, "LLVMConst_", ".C_");
  }
  string getID(const GlobalValue *GV) {
    if (const GlobalVariable *V = dyn_cast<GlobalVariable>(GV))
      return getID(V);
    else if (const Function *F = dyn_cast<Function>(GV))
      return getID(F);
    assert(0 && "Unexpected type of GlobalValue!");
    return "";
  }

  // ConstantExprToString() - Convert a ConstantExpr to an asm expression
  // and return this as a string.
  string ConstantExprToString(const ConstantExpr* CE,
                              const TargetMachine& target) {
    string S;
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
      S += "(" + valToExprString(CE->getOperand(0), target) + ") + ("
               + valToExprString(CE->getOperand(1), target) + ")";
      break;

    default:
      assert(0 && "Unsupported operator in ConstantExprToString()");
      break;
    }

    return S;
  }

  // valToExprString - Helper function for ConstantExprToString().
  // Appends result to argument string S.
  // 
  string valToExprString(const Value* V, const TargetMachine& target) {
    string S;
    bool failed = false;
    if (const Constant* CV = dyn_cast<Constant>(V)) { // symbolic or known

      if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV))
        S += string(CB == ConstantBool::True ? "1" : "0");
      else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
        S += itostr(CI->getValue());
      else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
        S += utostr(CI->getValue());
      else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV))
        S += ftostr(CFP->getValue());
      else if (isa<ConstantPointerNull>(CV))
        S += "0";
      else if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(CV))
        S += valToExprString(CPR->getValue(), target);
      else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV))
        S += ConstantExprToString(CE, target);
      else
        failed = true;

    } else if (const GlobalValue* GV = dyn_cast<GlobalValue>(V)) {
      S += getID(GV);
    }
    else
      failed = true;

    if (failed) {
      assert(0 && "Cannot convert value to string");
      S += "<illegal-value>";
    }
    return S;
  }

};



//===----------------------------------------------------------------------===//
//   SparcFunctionAsmPrinter Code
//===----------------------------------------------------------------------===//

struct SparcFunctionAsmPrinter : public FunctionPass, public AsmPrinter {
  inline SparcFunctionAsmPrinter(std::ostream &os, const TargetMachine &t)
    : AsmPrinter(os, t) {}

  const char *getPassName() const {
    return "Output Sparc Assembly for Functions";
  }

  virtual bool doInitialization(Module &M) {
    startModule(M);
    return false;
  }

  virtual bool runOnFunction(Function &F) {
    startFunction(F);
    emitFunction(F);
    endFunction(F);
    return false;
  }

  virtual bool doFinalization(Module &M) {
    endModule();
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
  //case BA:      return 1 << 0;  // Remove Arg #0, which is always null or xcc
    default:      return 0;       // By default, don't hack operands...
    }
  }
};

inline bool
SparcFunctionAsmPrinter::OpIsBranchTargetLabel(const MachineInstr *MI,
                                               unsigned int opNum) {
  switch (MI->getOpCode()) {
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
SparcFunctionAsmPrinter::OpIsMemoryAddressBase(const MachineInstr *MI,
                                               unsigned int opNum) {
  if (Target.getInstrInfo().isLoad(MI->getOpCode()))
    return (opNum == 0);
  else if (Target.getInstrInfo().isStore(MI->getOpCode()))
    return (opNum == 1);
  else
    return false;
}


#define PrintOp1PlusOp2(mop1, mop2, opCode) \
  printOneOperand(mop1, opCode); \
  toAsm << "+"; \
  printOneOperand(mop2, opCode);

unsigned int
SparcFunctionAsmPrinter::printOperands(const MachineInstr *MI,
                               unsigned int opNum)
{
  const MachineOperand& mop = MI->getOperand(opNum);
  
  if (OpIsBranchTargetLabel(MI, opNum))
    {
      PrintOp1PlusOp2(mop, MI->getOperand(opNum+1), MI->getOpCode());
      return 2;
    }
  else if (OpIsMemoryAddressBase(MI, opNum))
    {
      toAsm << "[";
      PrintOp1PlusOp2(mop, MI->getOperand(opNum+1), MI->getOpCode());
      toAsm << "]";
      return 2;
    }
  else
    {
      printOneOperand(mop, MI->getOpCode());
      return 1;
    }
}

void
SparcFunctionAsmPrinter::printOneOperand(const MachineOperand &mop,
                                         MachineOpCode opCode)
{
  bool needBitsFlag = true;
  
  if (mop.opHiBits32())
    toAsm << "%lm(";
  else if (mop.opLoBits32())
    toAsm << "%lo(";
  else if (mop.opHiBits64())
    toAsm << "%hh(";
  else if (mop.opLoBits64())
    toAsm << "%hm(";
  else
    needBitsFlag = false;
  
  switch (mop.getType())
    {
    case MachineOperand::MO_VirtualRegister:
    case MachineOperand::MO_CCRegister:
    case MachineOperand::MO_MachineRegister:
      {
        int regNum = (int)mop.getAllocatedRegNum();
        
        if (regNum == Target.getRegInfo().getInvalidRegNum()) {
          // better to print code with NULL registers than to die
          toAsm << "<NULL VALUE>";
        } else {
          toAsm << "%" << Target.getRegInfo().getUnifiedRegName(regNum);
        }
        break;
      }
    
    case MachineOperand::MO_PCRelativeDisp:
      {
        const Value *Val = mop.getVRegValue();
        assert(Val && "\tNULL Value in SparcFunctionAsmPrinter");
        
        if (const BasicBlock *BB = dyn_cast<BasicBlock>(Val))
          toAsm << getID(BB);
        else if (const Function *M = dyn_cast<Function>(Val))
          toAsm << getID(M);
        else if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Val))
          toAsm << getID(GV);
        else if (const Constant *CV = dyn_cast<Constant>(Val))
          toAsm << getID(CV);
        else
          assert(0 && "Unrecognized value in SparcFunctionAsmPrinter");
        break;
      }
    
    case MachineOperand::MO_SignExtendedImmed:
      toAsm << mop.getImmedValue();
      break;

    case MachineOperand::MO_UnextendedImmed:
      toAsm << (uint64_t) mop.getImmedValue();
      break;
    
    default:
      toAsm << mop;      // use dump field
      break;
    }
  
  if (needBitsFlag)
    toAsm << ")";
}


void
SparcFunctionAsmPrinter::emitMachineInst(const MachineInstr *MI)
{
  unsigned Opcode = MI->getOpCode();

  if (Target.getInstrInfo().isDummyPhiInstr(Opcode))
    return;  // IGNORE PHI NODES

  toAsm << "\t" << Target.getInstrInfo().getName(Opcode) << "\t";

  unsigned Mask = getOperandMask(Opcode);
  
  bool NeedComma = false;
  unsigned N = 1;
  for (unsigned OpNum = 0; OpNum < MI->getNumOperands(); OpNum += N)
    if (! ((1 << OpNum) & Mask)) {        // Ignore this operand?
      if (NeedComma) toAsm << ", ";         // Handle comma outputing
      NeedComma = true;
      N = printOperands(MI, OpNum);
    } else
      N = 1;
  
  toAsm << "\n";
}

void
SparcFunctionAsmPrinter::emitBasicBlock(const MachineBasicBlock &MBB)
{
  // Emit a label for the basic block
  toAsm << getID(MBB.getBasicBlock()) << ":\n";

  // Loop over all of the instructions in the basic block...
  for (MachineBasicBlock::const_iterator MII = MBB.begin(), MIE = MBB.end();
       MII != MIE; ++MII)
    emitMachineInst(*MII);
  toAsm << "\n";  // Separate BB's with newlines
}

void
SparcFunctionAsmPrinter::emitFunction(const Function &F)
{
  string methName = getID(&F);
  toAsm << "!****** Outputing Function: " << methName << " ******\n";
  enterSection(AsmPrinter::Text);
  toAsm << "\t.align\t4\n\t.global\t" << methName << "\n";
  //toAsm << "\t.type\t" << methName << ",#function\n";
  toAsm << "\t.type\t" << methName << ", 2\n";
  toAsm << methName << ":\n";

  // Output code for all of the basic blocks in the function...
  MachineFunction &MF = MachineFunction::get(&F);
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end(); I != E;++I)
    emitBasicBlock(*I);

  // Output a .size directive so the debugger knows the extents of the function
  toAsm << ".EndOf_" << methName << ":\n\t.size "
           << methName << ", .EndOf_"
           << methName << "-" << methName << "\n";

  // Put some spaces between the functions
  toAsm << "\n\n";
}

}  // End anonymous namespace

Pass *UltraSparc::getFunctionAsmPrinterPass(std::ostream &Out) {
  return new SparcFunctionAsmPrinter(Out, *this);
}





//===----------------------------------------------------------------------===//
//   SparcFunctionAsmPrinter Code
//===----------------------------------------------------------------------===//

namespace {

class SparcModuleAsmPrinter : public Pass, public AsmPrinter {
public:
  SparcModuleAsmPrinter(std::ostream &os, TargetMachine &t)
    : AsmPrinter(os, t) {}

  const char *getPassName() const { return "Output Sparc Assembly for Module"; }

  virtual bool run(Module &M) {
    startModule(M);
    emitGlobalsAndConstants(M);
    endModule();
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

private:
  void emitGlobalsAndConstants  (const Module &M);

  void printGlobalVariable      (const GlobalVariable *GV);
  void PrintZeroBytesToPad      (int numBytes);
  void printSingleConstantValue (const Constant* CV);
  void printConstantValueOnly   (const Constant* CV, int numPadBytesAfter = 0);
  void printConstant            (const Constant* CV, string valID = "");

  static void FoldConstants     (const Module &M,
                                 hash_set<const Constant*> &moduleConstants);
};


// Can we treat the specified array as a string?  Only if it is an array of
// ubytes or non-negative sbytes.
//
static bool isStringCompatible(const ConstantArray *CVA) {
  const Type *ETy = cast<ArrayType>(CVA->getType())->getElementType();
  if (ETy == Type::UByteTy) return true;
  if (ETy != Type::SByteTy) return false;

  for (unsigned i = 0; i < CVA->getNumOperands(); ++i)
    if (cast<ConstantSInt>(CVA->getOperand(i))->getValue() < 0)
      return false;

  return true;
}

// toOctal - Convert the low order bits of X into an octal letter
static inline char toOctal(int X) {
  return (X&7)+'0';
}

// getAsCString - Return the specified array as a C compatible string, only if
// the predicate isStringCompatible is true.
//
static string getAsCString(const ConstantArray *CVA) {
  assert(isStringCompatible(CVA) && "Array is not string compatible!");

  string Result;
  const Type *ETy = cast<ArrayType>(CVA->getType())->getElementType();
  Result = "\"";
  for (unsigned i = 0; i < CVA->getNumOperands(); ++i) {
    unsigned char C = cast<ConstantInt>(CVA->getOperand(i))->getRawValue();

    if (C == '"') {
      Result += "\\\"";
    } else if (C == '\\') {
      Result += "\\\\";
    } else if (isprint(C)) {
      Result += C;
    } else {
      Result += '\\';                   // print all other chars as octal value
      Result += toOctal(C >> 6);
      Result += toOctal(C >> 3);
      Result += toOctal(C >> 0);
    }
  }
  Result += "\"";

  return Result;
}

inline bool
ArrayTypeIsString(const ArrayType* arrayType)
{
  return (arrayType->getElementType() == Type::UByteTy ||
          arrayType->getElementType() == Type::SByteTy);
}


inline const string
TypeToDataDirective(const Type* type)
{
  switch(type->getPrimitiveID())
    {
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

// Get the size of the type
// 
inline unsigned int
TypeToSize(const Type* type, const TargetMachine& target)
{
  return target.findOptimalStorageSize(type);
}

// Get the size of the constant for the given target.
// If this is an unsized array, return 0.
// 
inline unsigned int
ConstantToSize(const Constant* CV, const TargetMachine& target)
{
  if (const ConstantArray* CVA = dyn_cast<ConstantArray>(CV))
    {
      const ArrayType *aty = cast<ArrayType>(CVA->getType());
      if (ArrayTypeIsString(aty))
        return 1 + CVA->getNumOperands();
    }
  
  return TypeToSize(CV->getType(), target);
}

// Align data larger than one L1 cache line on L1 cache line boundaries.
// Align all smaller data on the next higher 2^x boundary (4, 8, ...).
// 
inline unsigned int
SizeToAlignment(unsigned int size, const TargetMachine& target)
{
  unsigned short cacheLineSize = target.getCacheInfo().getCacheLineSize(1); 
  if (size > (unsigned) cacheLineSize / 2)
    return cacheLineSize;
  else
    for (unsigned sz=1; /*no condition*/; sz *= 2)
      if (sz >= size)
        return sz;
}

// Get the size of the type and then use SizeToAlignment.
// 
inline unsigned int
TypeToAlignment(const Type* type, const TargetMachine& target)
{
  return SizeToAlignment(TypeToSize(type, target), target);
}

// Get the size of the constant and then use SizeToAlignment.
// Handles strings as a special case;
inline unsigned int
ConstantToAlignment(const Constant* CV, const TargetMachine& target)
{
  if (const ConstantArray* CVA = dyn_cast<ConstantArray>(CV))
    if (ArrayTypeIsString(cast<ArrayType>(CVA->getType())))
      return SizeToAlignment(1 + CVA->getNumOperands(), target);
  
  return TypeToAlignment(CV->getType(), target);
}


// Print a single constant value.
void
SparcModuleAsmPrinter::printSingleConstantValue(const Constant* CV)
{
  assert(CV->getType() != Type::VoidTy &&
         CV->getType() != Type::TypeTy &&
         CV->getType() != Type::LabelTy &&
         "Unexpected type for Constant");
  
  assert((!isa<ConstantArray>(CV) && ! isa<ConstantStruct>(CV))
         && "Aggregate types should be handled outside this function");
  
  toAsm << "\t" << TypeToDataDirective(CV->getType()) << "\t";
  
  if (CV->getType()->isPrimitiveType())
    {
      if (CV->getType()->isFloatingPoint()) {
        // FP Constants are printed as integer constants to avoid losing
        // precision...
        double Val = cast<ConstantFP>(CV)->getValue();
        if (CV->getType() == Type::FloatTy) {
          float FVal = (float)Val;
          char *ProxyPtr = (char*)&FVal;        // Abide by C TBAA rules
          toAsm << *(unsigned int*)ProxyPtr;            
        } else if (CV->getType() == Type::DoubleTy) {
          char *ProxyPtr = (char*)&Val;         // Abide by C TBAA rules
          toAsm << *(uint64_t*)ProxyPtr;            
        } else {
          assert(0 && "Unknown floating point type!");
        }
        
        toAsm << "\t! " << CV->getType()->getDescription()
              << " value: " << Val << "\n";
      } else {
        WriteAsOperand(toAsm, CV, false, false) << "\n";
      }
    }
  else if (const ConstantPointerRef* CPR = dyn_cast<ConstantPointerRef>(CV))
    { // This is a constant address for a global variable or method.
      // Use the name of the variable or method as the address value.
      toAsm << getID(CPR->getValue()) << "\n";
    }
  else if (isa<ConstantPointerNull>(CV))
    { // Null pointer value
      toAsm << "0\n";
    }
  else if (const ConstantExpr* CE = dyn_cast<ConstantExpr>(CV))
    { // Constant expression built from operators, constants, and symbolic addrs
      toAsm << ConstantExprToString(CE, Target) << "\n";
    }
  else
    {
      assert(0 && "Unknown elementary type for constant");
    }
}

void
SparcModuleAsmPrinter::PrintZeroBytesToPad(int numBytes)
{
  for ( ; numBytes >= 8; numBytes -= 8)
    printSingleConstantValue(Constant::getNullValue(Type::ULongTy));

  if (numBytes >= 4)
    {
      printSingleConstantValue(Constant::getNullValue(Type::UIntTy));
      numBytes -= 4;
    }

  while (numBytes--)
    printSingleConstantValue(Constant::getNullValue(Type::UByteTy));
}

// Print a constant value or values (it may be an aggregate).
// Uses printSingleConstantValue() to print each individual value.
void
SparcModuleAsmPrinter::printConstantValueOnly(const Constant* CV,
                                              int numPadBytesAfter /* = 0*/)
{
  const ConstantArray *CVA = dyn_cast<ConstantArray>(CV);

  if (CVA && isStringCompatible(CVA))
    { // print the string alone and return
      toAsm << "\t" << ".ascii" << "\t" << getAsCString(CVA) << "\n";
    }
  else if (CVA)
    { // Not a string.  Print the values in successive locations
      const std::vector<Use> &constValues = CVA->getValues();
      for (unsigned i=0; i < constValues.size(); i++)
        printConstantValueOnly(cast<Constant>(constValues[i].get()));
    }
  else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV))
    { // Print the fields in successive locations. Pad to align if needed!
      const StructLayout *cvsLayout =
        Target.getTargetData().getStructLayout(CVS->getType());
      const std::vector<Use>& constValues = CVS->getValues();
      unsigned sizeSoFar = 0;
      for (unsigned i=0, N = constValues.size(); i < N; i++)
        {
          const Constant* field = cast<Constant>(constValues[i].get());

          // Check if padding is needed and insert one or more 0s.
          unsigned fieldSize =
	    Target.getTargetData().getTypeSize(field->getType());
          int padSize = ((i == N-1? cvsLayout->StructSize
                                  : cvsLayout->MemberOffsets[i+1])
                         - cvsLayout->MemberOffsets[i]) - fieldSize;
          sizeSoFar += (fieldSize + padSize);

          // Now print the actual field value
          printConstantValueOnly(field, padSize);
        }
      assert(sizeSoFar == cvsLayout->StructSize &&
             "Layout of constant struct may be incorrect!");
    }
  else
    printSingleConstantValue(CV);

  if (numPadBytesAfter)
    PrintZeroBytesToPad(numPadBytesAfter);
}

// Print a constant (which may be an aggregate) prefixed by all the
// appropriate directives.  Uses printConstantValueOnly() to print the
// value or values.
void
SparcModuleAsmPrinter::printConstant(const Constant* CV, string valID)
{
  if (valID.length() == 0)
    valID = getID(CV);
  
  toAsm << "\t.align\t" << ConstantToAlignment(CV, Target) << "\n";
  
  // Print .size and .type only if it is not a string.
  const ConstantArray *CVA = dyn_cast<ConstantArray>(CV);
  if (CVA && isStringCompatible(CVA))
    { // print it as a string and return
      toAsm << valID << ":\n";
      toAsm << "\t" << ".ascii" << "\t" << getAsCString(CVA) << "\n";
      return;
    }
  
  toAsm << "\t.type" << "\t" << valID << ",#object\n";

  unsigned int constSize = ConstantToSize(CV, Target);
  if (constSize)
    toAsm << "\t.size" << "\t" << valID << "," << constSize << "\n";
  
  toAsm << valID << ":\n";
  
  printConstantValueOnly(CV);
}


void SparcModuleAsmPrinter::FoldConstants(const Module &M,
                                          hash_set<const Constant*> &MC) {
  for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal()) {
      const hash_set<const Constant*> &pool =
        MachineFunction::get(I).getInfo()->getConstantPoolValues();
      MC.insert(pool.begin(), pool.end());
    }
}

void SparcModuleAsmPrinter::printGlobalVariable(const GlobalVariable* GV)
{
  if (GV->hasExternalLinkage())
    toAsm << "\t.global\t" << getID(GV) << "\n";
  
  if (GV->hasInitializer() && ! GV->getInitializer()->isNullValue())
    printConstant(GV->getInitializer(), getID(GV));
  else {
    toAsm << "\t.align\t" << TypeToAlignment(GV->getType()->getElementType(),
                                                Target) << "\n";
    toAsm << "\t.type\t" << getID(GV) << ",#object\n";
    toAsm << "\t.reserve\t" << getID(GV) << ","
          << TypeToSize(GV->getType()->getElementType(), Target)
          << "\n";
  }
}


void SparcModuleAsmPrinter::emitGlobalsAndConstants(const Module &M) {
  // First, get the constants there were marked by the code generator for
  // inclusion in the assembly code data area and fold them all into a
  // single constant pool since there may be lots of duplicates.  Also,
  // lets force these constants into the slot table so that we can get
  // unique names for unnamed constants also.
  // 
  hash_set<const Constant*> moduleConstants;
  FoldConstants(M, moduleConstants);
    
  // Output constants spilled to memory
  enterSection(AsmPrinter::ReadOnlyData);
  for (hash_set<const Constant*>::const_iterator I = moduleConstants.begin(),
         E = moduleConstants.end();  I != E; ++I)
    printConstant(*I);

  // Output global variables...
  for (Module::const_giterator GI = M.gbegin(), GE = M.gend(); GI != GE; ++GI)
    if (! GI->isExternal()) {
      assert(GI->hasInitializer());
      if (GI->isConstant())
        enterSection(AsmPrinter::ReadOnlyData);   // read-only, initialized data
      else if (GI->getInitializer()->isNullValue())
        enterSection(AsmPrinter::ZeroInitRWData); // read-write zero data
      else
        enterSection(AsmPrinter::InitRWData);     // read-write non-zero data

      printGlobalVariable(GI);
    }

  toAsm << "\n";
}

}  // End anonymous namespace

Pass *UltraSparc::getModuleAsmPrinterPass(std::ostream &Out) {
  return new SparcModuleAsmPrinter(Out, *this);
}
