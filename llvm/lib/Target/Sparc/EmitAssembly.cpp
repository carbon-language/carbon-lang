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
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/SlotCalculator.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "Support/StringExtras.h"
#include <iostream>
using std::string;

namespace {

class GlobalIdTable: public Annotation {
  static AnnotationID AnnotId;
  friend class AsmPrinter;              // give access to AnnotId
  
  typedef std::hash_map<const Value*, int> ValIdMap;
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
    UninitRWData,
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

  // Check if a name is external or accessible from external code.
  // Only functions can currently be external.  "main" is the only name
  // that is visible externally.
  bool isExternal(const Value* V) {
    const Function *F = dyn_cast<Function>(V);
    return F && (F->isExternal() || F->getName() == "main");
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
      case UninitRWData: toAsm << "\".bss\",#alloc,#write\nBbss.bss:"; break;
      }
    toAsm << "\n";
  }

  static std::string getValidSymbolName(const string &S) {
    string Result;
    
    // Symbol names in Sparc assembly language have these rules:
    // (a) Must match { letter | _ | . | $ } { letter | _ | . | $ | digit }*
    // (b) A name beginning in "." is treated as a local name.
    // (c) Names beginning with "_" are reserved by ANSI C and shd not be used.
    // 
    if (S[0] == '_' || isdigit(S[0]))
      Result += "ll";
    
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
    
    Result = Result + (V->hasName()? V->getName() : string(Prefix));
    
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
    }
    
    return getValidSymbolName(Result);
  }
  
  // getID Wrappers - Ensure consistent usage...
  string getID(const Function *F) {
    return getID(F, "LLVMFunction_");
  }
  string getID(const BasicBlock *BB) {
    return getID(BB, "LL", (".L_"+getID(BB->getParent())+"_").c_str());
  }
  string getID(const GlobalVariable *GV) {
    return getID(GV, "LLVMGlobal_", ".G_");
  }
  string getID(const Constant *CV) {
    return getID(CV, "LLVMConst_", ".C_");
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
  void emitBasicBlock(const BasicBlock *BB);
  void emitMachineInst(const MachineInstr *MI);
  
  unsigned int printOperands(const MachineInstr *MI, unsigned int opNum);
  void printOneOperand(const MachineOperand &Op);

  bool OpIsBranchTargetLabel(const MachineInstr *MI, unsigned int opNum);
  bool OpIsMemoryAddressBase(const MachineInstr *MI, unsigned int opNum);
  
  unsigned getOperandMask(unsigned Opcode) {
    switch (Opcode) {
    case SUBcc:   return 1 << 3;  // Remove CC argument
  //case BA:      return 1 << 0;  // Remove Arg #0, which is always null or xcc
    default:      return 0;       // By default, don't hack operands...
    }
  }
};

inline bool
SparcFunctionAsmPrinter::OpIsBranchTargetLabel(const MachineInstr *MI,
                                               unsigned int opNum) {
  switch (MI->getOpCode()) {
  case JMPLCALL:
  case JMPLRET: return (opNum == 0);
  default:      return false;
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


#define PrintOp1PlusOp2(Op1, Op2) \
  printOneOperand(Op1); \
  toAsm << "+"; \
  printOneOperand(Op2);

unsigned int
SparcFunctionAsmPrinter::printOperands(const MachineInstr *MI,
                               unsigned int opNum)
{
  const MachineOperand& Op = MI->getOperand(opNum);
  
  if (OpIsBranchTargetLabel(MI, opNum))
    {
      PrintOp1PlusOp2(Op, MI->getOperand(opNum+1));
      return 2;
    }
  else if (OpIsMemoryAddressBase(MI, opNum))
    {
      toAsm << "[";
      PrintOp1PlusOp2(Op, MI->getOperand(opNum+1));
      toAsm << "]";
      return 2;
    }
  else
    {
      printOneOperand(Op);
      return 1;
    }
}


void
SparcFunctionAsmPrinter::printOneOperand(const MachineOperand &op)
{
  switch (op.getOperandType())
    {
    case MachineOperand::MO_VirtualRegister:
    case MachineOperand::MO_CCRegister:
    case MachineOperand::MO_MachineRegister:
      {
        int RegNum = (int)op.getAllocatedRegNum();
        
        // better to print code with NULL registers than to die
        if (RegNum == Target.getRegInfo().getInvalidRegNum()) {
          toAsm << "<NULL VALUE>";
        } else {
          toAsm << "%" << Target.getRegInfo().getUnifiedRegName(RegNum);
        }
        break;
      }
    
    case MachineOperand::MO_PCRelativeDisp:
      {
        const Value *Val = op.getVRegValue();
        assert(Val && "\tNULL Value in SparcFunctionAsmPrinter");
        
        if (const BasicBlock *BB = dyn_cast<const BasicBlock>(Val))
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
      toAsm << op.getImmedValue();
      break;

    case MachineOperand::MO_UnextendedImmed:
      toAsm << (uint64_t) op.getImmedValue();
      break;
    
    default:
      toAsm << op;      // use dump field
      break;
    }
}


void
SparcFunctionAsmPrinter::emitMachineInst(const MachineInstr *MI)
{
  unsigned Opcode = MI->getOpCode();

  if (TargetInstrDescriptors[Opcode].iclass & M_DUMMY_PHI_FLAG)
    return;  // IGNORE PHI NODES

  toAsm << "\t" << TargetInstrDescriptors[Opcode].opCodeString << "\t";

  unsigned Mask = getOperandMask(Opcode);
  
  bool NeedComma = false;
  unsigned N = 1;
  for (unsigned OpNum = 0; OpNum < MI->getNumOperands(); OpNum += N)
    if (! ((1 << OpNum) & Mask)) {        // Ignore this operand?
      if (NeedComma) toAsm << ", ";         // Handle comma outputing
      NeedComma = true;
      N = printOperands(MI, OpNum);
    }
  else
    N = 1;
  
  toAsm << "\n";
}

void
SparcFunctionAsmPrinter::emitBasicBlock(const BasicBlock *BB)
{
  // Emit a label for the basic block
  toAsm << getID(BB) << ":\n";

  // Get the vector of machine instructions corresponding to this bb.
  const MachineCodeForBasicBlock &MIs = MachineCodeForBasicBlock::get(BB);
  MachineCodeForBasicBlock::const_iterator MII = MIs.begin(), MIE = MIs.end();

  // Loop over all of the instructions in the basic block...
  for (; MII != MIE; ++MII)
    emitMachineInst(*MII);
  toAsm << "\n";  // Seperate BB's with newlines
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
  for (Function::const_iterator I = F.begin(), E = F.end(); I != E; ++I)
    emitBasicBlock(I);

  // Output a .size directive so the debugger knows the extents of the function
  toAsm << ".EndOf_" << methName << ":\n\t.size "
           << methName << ", .EndOf_"
           << methName << "-" << methName << "\n";

  // Put some spaces between the functions
  toAsm << "\n\n";
}

}  // End anonymous namespace

Pass *UltraSparc::getFunctionAsmPrinterPass(PassManager &PM, std::ostream &Out){
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
  void emitGlobalsAndConstants(const Module &M);

  void printGlobalVariable(const GlobalVariable *GV);
  void printSingleConstant(   const Constant* CV);
  void printConstantValueOnly(const Constant* CV);
  void printConstant(         const Constant* CV, std::string valID = "");

  static void FoldConstants(const Module &M,
                            std::hash_set<const Constant*> &moduleConstants);
};


// Can we treat the specified array as a string?  Only if it is an array of
// ubytes or non-negative sbytes.
//
static bool isStringCompatible(const ConstantArray *CPA) {
  const Type *ETy = cast<ArrayType>(CPA->getType())->getElementType();
  if (ETy == Type::UByteTy) return true;
  if (ETy != Type::SByteTy) return false;

  for (unsigned i = 0; i < CPA->getNumOperands(); ++i)
    if (cast<ConstantSInt>(CPA->getOperand(i))->getValue() < 0)
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
static string getAsCString(const ConstantArray *CPA) {
  assert(isStringCompatible(CPA) && "Array is not string compatible!");

  string Result;
  const Type *ETy = cast<ArrayType>(CPA->getType())->getElementType();
  Result = "\"";
  for (unsigned i = 0; i < CPA->getNumOperands(); ++i) {
    unsigned char C = (ETy == Type::SByteTy) ?
      (unsigned char)cast<ConstantSInt>(CPA->getOperand(i))->getValue() :
      (unsigned char)cast<ConstantUInt>(CPA->getOperand(i))->getValue();

    if (C == '"') {
      Result += "\\\"";
    } else if (isprint(C)) {
      Result += C;
    } else {
      switch(C) {
      case '\a': Result += "\\a"; break;
      case '\b': Result += "\\b"; break;
      case '\f': Result += "\\f"; break;
      case '\n': Result += "\\n"; break;
      case '\r': Result += "\\r"; break;
      case '\t': Result += "\\t"; break;
      case '\v': Result += "\\v"; break;
      default:
        Result += '\\';
        Result += toOctal(C >> 6);
        Result += toOctal(C >> 3);
        Result += toOctal(C >> 0);
        break;
      }
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

// Get the size of the constant for the given target.
// If this is an unsized array, return 0.
// 
inline unsigned int
ConstantToSize(const Constant* CV, const TargetMachine& target)
{
  if (const ConstantArray* CPA = dyn_cast<ConstantArray>(CV))
    {
      const ArrayType *aty = cast<ArrayType>(CPA->getType());
      if (ArrayTypeIsString(aty))
        return 1 + CPA->getNumOperands();
    }
  
  return target.findOptimalStorageSize(CV->getType());
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
  return SizeToAlignment(target.findOptimalStorageSize(type), target);
}

// Get the size of the constant and then use SizeToAlignment.
// Handles strings as a special case;
inline unsigned int
ConstantToAlignment(const Constant* CV, const TargetMachine& target)
{
  if (const ConstantArray* CPA = dyn_cast<ConstantArray>(CV))
    if (ArrayTypeIsString(cast<ArrayType>(CPA->getType())))
      return SizeToAlignment(1 + CPA->getNumOperands(), target);
  
  return TypeToAlignment(CV->getType(), target);
}


// Print a single constant value.
void
SparcModuleAsmPrinter::printSingleConstant(const Constant* CV)
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
      if (const GlobalVariable* GV = dyn_cast<GlobalVariable>(CPR->getValue()))
        toAsm << getID(GV);
      else if (const Function* F = dyn_cast<Function>(CPR->getValue()))
        toAsm << getID(F);
      else
        assert(0 && "Unexpected constant reference type");
    }
  else if (const ConstantPointer* CPP = dyn_cast<ConstantPointer>(CV))
    {
      assert(CPP->isNullValue() &&
             "Cannot yet print non-null pointer constants to assembly");
      toAsm << "0\n";
    }
  else
    {
      assert(0 && "Unknown elementary type for constant");
    }
}

// Print a constant value or values (it may be an aggregate).
// Uses printSingleConstant() to print each individual value.
void
SparcModuleAsmPrinter::printConstantValueOnly(const Constant* CV)
{
  const ConstantArray *CPA = dyn_cast<ConstantArray>(CV);
  
  if (CPA && isStringCompatible(CPA))
    { // print the string alone and return
      toAsm << "\t" << ".ascii" << "\t" << getAsCString(CPA) << "\n";
    }
  else if (CPA)
    { // Not a string.  Print the values in successive locations
      const std::vector<Use> &constValues = CPA->getValues();
      for (unsigned i=0; i < constValues.size(); i++)
        printConstantValueOnly(cast<Constant>(constValues[i].get()));
    }
  else if (const ConstantStruct *CPS = dyn_cast<ConstantStruct>(CV))
    { // Print the fields in successive locations
      const std::vector<Use>& constValues = CPS->getValues();
      for (unsigned i=0; i < constValues.size(); i++)
        printConstantValueOnly(cast<Constant>(constValues[i].get()));
    }
  else
    printSingleConstant(CV);
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
  const ConstantArray *CPA = dyn_cast<ConstantArray>(CV);
  if (CPA && isStringCompatible(CPA))
    { // print it as a string and return
      toAsm << valID << ":\n";
      toAsm << "\t" << ".ascii" << "\t" << getAsCString(CPA) << "\n";
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
                                          std::hash_set<const Constant*> &MC) {
  for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal()) {
      const std::hash_set<const Constant*> &pool =
        MachineCodeForMethod::get(I).getConstantPoolValues();
      MC.insert(pool.begin(), pool.end());
    }
}

void SparcModuleAsmPrinter::printGlobalVariable(const GlobalVariable* GV)
{
  toAsm << "\t.global\t" << getID(GV) << "\n";
  
  if (GV->hasInitializer())
    printConstant(GV->getInitializer(), getID(GV));
  else {
    toAsm << "\t.align\t" << TypeToAlignment(GV->getType()->getElementType(),
                                                Target) << "\n";
    toAsm << "\t.type\t" << getID(GV) << ",#object\n";
    toAsm << "\t.reserve\t" << getID(GV) << ","
          << Target.findOptimalStorageSize(GV->getType()->getElementType())
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
  std::hash_set<const Constant*> moduleConstants;
  FoldConstants(M, moduleConstants);
    
  // Now, emit the three data sections separately; the cost of I/O should
  // make up for the cost of extra passes over the globals list!
  
  // Section 1 : Read-only data section (implies initialized)
  enterSection(AsmPrinter::ReadOnlyData);
  for (Module::const_giterator GI = M.gbegin(), GE = M.gend(); GI != GE; ++GI)
    if (GI->hasInitializer() && GI->isConstant())
      printGlobalVariable(GI);
  
  for (std::hash_set<const Constant*>::const_iterator
         I = moduleConstants.begin(),
         E = moduleConstants.end();  I != E; ++I)
    printConstant(*I);
  
  // Section 2 : Initialized read-write data section
  enterSection(AsmPrinter::InitRWData);
  for (Module::const_giterator GI = M.gbegin(), GE = M.gend(); GI != GE; ++GI)
    if (GI->hasInitializer() && !GI->isConstant())
      printGlobalVariable(GI);
  
  // Section 3 : Uninitialized read-write data section
  enterSection(AsmPrinter::UninitRWData);
  for (Module::const_giterator GI = M.gbegin(), GE = M.gend(); GI != GE; ++GI)
    if (!GI->hasInitializer())
      printGlobalVariable(GI);
  
  toAsm << "\n";
}

}  // End anonymous namespace

Pass *UltraSparc::getModuleAsmPrinterPass(PassManager &PM, std::ostream &Out) {
  return new SparcModuleAsmPrinter(Out, *this);
}
