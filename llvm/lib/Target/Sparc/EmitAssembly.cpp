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
#include "llvm/Transforms/Linker.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalValue.h"
#include "llvm/ConstantVals.h"
#include "llvm/DerivedTypes.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "Support/StringExtras.h"
#include "Support/HashExtras.h"
#include <locale.h>

namespace {


class SparcAsmPrinter {
  typedef hash_map<const Value*, int> ValIdMap;
  typedef ValIdMap::      iterator ValIdMapIterator;
  typedef ValIdMap::const_iterator ValIdMapConstIterator;
  
  ostream &toAsm;
  SlotCalculator Table;   // map anonymous values to unique integer IDs
  ValIdMap valToIdMap;    // used for values not handled by SlotCalculator 
  const UltraSparc &Target;
  
  enum Sections {
    Unknown,
    Text,
    ReadOnlyData,
    InitRWData,
    UninitRWData,
  } CurSection;
  
public:
  inline SparcAsmPrinter(ostream &o, const Module *M, const UltraSparc &t)
    : toAsm(o), Table(SlotCalculator(M, true)), Target(t), CurSection(Unknown) {
    emitModule(M);
  }

private :
  void emitModule(const Module *M);
  void emitMethod(const Method *M);
  void emitGlobalsAndConstants(const Module* module);
  //void processMethodArgument(const MethodArgument *MA);
  void emitBasicBlock(const BasicBlock *BB);
  void emitMachineInst(const MachineInstr *MI);
  
  void printGlobalVariable(   const GlobalVariable* GV);
  void printSingleConstant(   const Constant* CV);
  void printConstantValueOnly(const Constant* CV);
  void printConstant(         const Constant* CV, string valID=string(""));
  
  unsigned int printOperands(const MachineInstr *MI, unsigned int opNum);
  void printOneOperand(const MachineOperand &Op);

  bool OpIsBranchTargetLabel(const MachineInstr *MI, unsigned int opNum);
  bool OpIsMemoryAddressBase(const MachineInstr *MI, unsigned int opNum);
  
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

  string getValidSymbolName(const string &S) {
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
  // the name of the identifier if possible, use a numbered value based on
  // prefix otherwise.  FPrefix is always prepended to the output identifier.
  //
  string getID(const Value *V, const char *Prefix, const char *FPrefix = 0) {
    string Result;
    string FP(FPrefix ? FPrefix : "");  // "Forced prefix"
    if (V->hasName()) {
      Result = FP + V->getName();
    } else {
      int valId = Table.getValSlot(V);
      if (valId == -1) {
        ValIdMapConstIterator I = valToIdMap.find(V);
        valId = (I == valToIdMap.end())? (valToIdMap[V] = valToIdMap.size())
                                       : (*I).second;
      }
      Result = FP + string(Prefix) + itostr(valId);
    }
    return getValidSymbolName(Result);
  }
  
  // getID Wrappers - Ensure consistent usage...
  string getID(const Module *M) {
    return getID(M, "LLVMModule_");
  }
  string getID(const Method *M) {
    return getID(M, "LLVMMethod_");
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
  
  unsigned getOperandMask(unsigned Opcode) {
    switch (Opcode) {
    case SUBcc:   return 1 << 3;  // Remove CC argument
    case BA:      return 1 << 0;  // Remove Arg #0, which is always null or xcc
    default:      return 0;       // By default, don't hack operands...
    }
  }
};


// Can we treat the specified array as a string?  Only if it is an array of
// ubytes or non-negative sbytes.
//
static bool isStringCompatible(ConstantArray *CPA) {
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
static string getAsCString(ConstantArray *CPA) {
  if (isStringCompatible(CPA)) {
    string Result;
    const Type *ETy = cast<ArrayType>(CPA->getType())->getElementType();
    Result = "\"";
    for (unsigned i = 0; i < CPA->getNumOperands(); ++i) {
      unsigned char C = (ETy == Type::SByteTy) ?
        (unsigned char)cast<ConstantSInt>(CPA->getOperand(i))->getValue() :
        (unsigned char)cast<ConstantUInt>(CPA->getOperand(i))->getValue();

      if (isprint(C)) {
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
  } else {
    return CPA->getStrValue();
  }
}


inline bool
SparcAsmPrinter::OpIsBranchTargetLabel(const MachineInstr *MI,
                                       unsigned int opNum) {
  switch (MI->getOpCode()) {
  case JMPLCALL:
  case JMPLRET: return (opNum == 0);
  default:      return false;
  }
}


inline bool
SparcAsmPrinter::OpIsMemoryAddressBase(const MachineInstr *MI,
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
SparcAsmPrinter::printOperands(const MachineInstr *MI,
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
SparcAsmPrinter::printOneOperand(const MachineOperand &op)
{
  switch (op.getOperandType())
    {
    case MachineOperand::MO_VirtualRegister:
    case MachineOperand::MO_CCRegister:
    case MachineOperand::MO_MachineRegister:
      {
        int RegNum = (int)op.getAllocatedRegNum();
        
        // ****this code is temporary till NULL Values are fixed
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
        if (!Val)
          toAsm << "\t<*NULL Value*>";
        else if (const BasicBlock *BB = dyn_cast<const BasicBlock>(Val))
          toAsm << getID(BB);
        else if (const Method *M = dyn_cast<const Method>(Val))
          toAsm << getID(M);
        else if (const GlobalVariable *GV=dyn_cast<const GlobalVariable>(Val))
          toAsm << getID(GV);
        else if (const Constant *CV = dyn_cast<const Constant>(Val))
          toAsm << getID(CV);
        else
          toAsm << "<unknown value=" << Val << ">";
        break;
      }
    
    case MachineOperand::MO_SignExtendedImmed:
    case MachineOperand::MO_UnextendedImmed:
      toAsm << op.getImmedValue();
      break;
    
    default:
      toAsm << op;      // use dump field
      break;
    }
}


void
SparcAsmPrinter::emitMachineInst(const MachineInstr *MI)
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
  
  toAsm << endl;
}

void
SparcAsmPrinter::emitBasicBlock(const BasicBlock *BB)
{
  // Emit a label for the basic block
  toAsm << getID(BB) << ":\n";

  // Get the vector of machine instructions corresponding to this bb.
  const MachineCodeForBasicBlock &MIs = BB->getMachineInstrVec();
  MachineCodeForBasicBlock::const_iterator MII = MIs.begin(), MIE = MIs.end();

  // Loop over all of the instructions in the basic block...
  for (; MII != MIE; ++MII)
    emitMachineInst(*MII);
  toAsm << "\n";  // Seperate BB's with newlines
}

void
SparcAsmPrinter::emitMethod(const Method *M)
{
  if (M->isExternal()) return;

  // Make sure the slot table has information about this method...
  Table.incorporateMethod(M);

  string methName = getID(M);
  toAsm << "!****** Outputing Method: " << methName << " ******\n";
  enterSection(Text);
  toAsm << "\t.align\t4\n\t.global\t" << methName << "\n";
  //toAsm << "\t.type\t" << methName << ",#function\n";
  toAsm << "\t.type\t" << methName << ", 2\n";
  toAsm << methName << ":\n";

  // Output code for all of the basic blocks in the method...
  for (Method::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    emitBasicBlock(*I);

  // Output a .size directive so the debugger knows the extents of the function
  toAsm << ".EndOf_" << methName << ":\n\t.size "
        << methName << ", .EndOf_"
        << methName << "-" << methName << endl;

  // Put some spaces between the methods
  toAsm << "\n\n";

  // Forget all about M.
  Table.purgeMethod();
}

inline bool
ArrayTypeIsString(ArrayType* arrayType)
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
      return ".single";
    case Type::DoubleTyID:
      return ".double";
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
  if (ConstantArray* CPA = dyn_cast<ConstantArray>(CV))
    {
      ArrayType *aty = cast<ArrayType>(CPA->getType());
      if (ArrayTypeIsString(aty))
        return 1 + CPA->getNumOperands();
      else if (! aty->isSized())
        return 0;
    }
  
  return target.findOptimalStorageSize(CV->getType());
}

inline
unsigned int TypeToSize(const Type* type, const TargetMachine& target)
{
  return target.findOptimalStorageSize(type);
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
// If this is an unsized array, just return the L1 cache line size
// (viz., the default behavior for large global objects).
// 
inline unsigned int
TypeToAlignment(const Type* type, const TargetMachine& target)
{
  if (ArrayType* aty = dyn_cast<ArrayType>(type))
    if (! aty->isSized())
      return target.getCacheInfo().getCacheLineSize(1);
  
  return SizeToAlignment(target.findOptimalStorageSize(type), target);
}

// Get the size of the constant and then use SizeToAlignment.
// Handles strings as a special case;
inline unsigned int
ConstantToAlignment(const Constant* CV, const TargetMachine& target)
{
  unsigned int constantSize;
  if (ConstantArray* CPA = dyn_cast<ConstantArray>(CV))
    if (ArrayTypeIsString(cast<ArrayType>(CPA->getType())))
      return SizeToAlignment(1 + CPA->getNumOperands(), target);
  
  return TypeToAlignment(CV->getType(), target);
}


// Print a single constant value.
void
SparcAsmPrinter::printSingleConstant(const Constant* CV)
{
  assert(CV->getType() != Type::VoidTy &&
         CV->getType() != Type::TypeTy &&
         CV->getType() != Type::LabelTy &&
         "Unexpected type for Constant");
  
  assert((! isa<ConstantArray>( CV) && ! isa<ConstantStruct>(CV))
         && "Collective types should be handled outside this function");
  
  toAsm << "\t"
        << TypeToDataDirective(CV->getType()) << "\t";
  
  if (CV->getType()->isPrimitiveType())
    {
      if (CV->getType() == Type::FloatTy || CV->getType() == Type::DoubleTy)
        toAsm << "0r";                  // FP constants must have this prefix
      toAsm << CV->getStrValue() << endl;
    }
  else if (ConstantPointer* CPP = dyn_cast<ConstantPointer>(CV))
    {
      if (! CPP->isNullValue())
        assert(0 && "Cannot yet print non-null pointer constants to assembly");
      else
        toAsm << (void*) NULL << endl;
    }
  else if (ConstantPointerRef* CPRef = dyn_cast<ConstantPointerRef>(CV))
    {
      assert(0 && "Cannot yet initialize pointer refs in assembly");
    }
  else
    {
      assert(0 && "Unknown elementary type for constant");
    }
}

// Print a constant value or values (it may be an aggregate).
// Uses printSingleConstant() to print each individual value.
void
SparcAsmPrinter::printConstantValueOnly(const Constant* CV)
{
  ConstantArray *CPA = dyn_cast<ConstantArray>(CV);
  
  if (CPA && isStringCompatible(CPA))
    { // print the string alone and return
      toAsm << "\t" << ".ascii" << "\t" << getAsCString(CPA) << endl;
    }
  else if (CPA)
    { // Not a string.  Print the values in successive locations
      const vector<Use>& constValues = CPA->getValues();
      for (unsigned i=1; i < constValues.size(); i++)
        this->printConstantValueOnly(cast<Constant>(constValues[i].get()));
    }
  else if (ConstantStruct *CPS = dyn_cast<ConstantStruct>(CV))
    { // Print the fields in successive locations
      const vector<Use>& constValues = CPS->getValues();
      for (unsigned i=1; i < constValues.size(); i++)
        this->printConstantValueOnly(cast<Constant>(constValues[i].get()));
    }
  else
    this->printSingleConstant(CV);
}

// Print a constant (which may be an aggregate) prefixed by all the
// appropriate directives.  Uses printConstantValueOnly() to print the
// value or values.
void
SparcAsmPrinter::printConstant(const Constant* CV, string valID)
{
  if (valID.length() == 0)
    valID = getID(CV);
  
  toAsm << "\t.align\t" << ConstantToAlignment(CV, Target)
        << endl;
  
  // Print .size and .type only if it is not a string.
  ConstantArray *CPA = dyn_cast<ConstantArray>(CV);
  if (CPA && isStringCompatible(CPA))
    { // print it as a string and return
      toAsm << valID << ":" << endl;
      toAsm << "\t" << ".ascii" << "\t" << getAsCString(CPA) << endl;
      return;
    }
  
  toAsm << "\t.type" << "\t" << valID << ",#object" << endl;

  unsigned int constSize = ConstantToSize(CV, Target);
  if (constSize)
    toAsm << "\t.size" << "\t" << valID << ","
          << constSize << endl;
  
  toAsm << valID << ":" << endl;
  
  this->printConstantValueOnly(CV);
}


void
SparcAsmPrinter::printGlobalVariable(const GlobalVariable* GV)
{
  toAsm << "\t.global\t" << getID(GV) << endl;
  
  if (GV->hasInitializer())
    printConstant(GV->getInitializer(), getID(GV));
  else {
    toAsm << "\t.align\t"
          << TypeToAlignment(GV->getType()->getValueType(), Target) << endl;
    toAsm << "\t.type\t" << getID(GV) << ",#object" << endl;
    toAsm << "\t.reserve\t" << getID(GV) << ","
          << TypeToSize(GV->getType()->getValueType(), Target)
          << endl;
  }
}


static void
FoldConstants(const Module *M,
               hash_set<const Constant*>& moduleConstants)
{
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (! (*I)->isExternal())
      {
        const hash_set<const Constant*>& pool =
          MachineCodeForMethod::get(*I).getConstantPoolValues();
        moduleConstants.insert(pool.begin(), pool.end());
      }
}


void
SparcAsmPrinter::emitGlobalsAndConstants(const Module *M)
{
  // First, get the constants there were marked by the code generator for
  // inclusion in the assembly code data area and fold them all into a
  // single constant pool since there may be lots of duplicates.  Also,
  // lets force these constants into the slot table so that we can get
  // unique names for unnamed constants also.
  // 
  hash_set<const Constant*> moduleConstants;
  FoldConstants(M, moduleConstants);
  
  // Now, emit the three data sections separately; the cost of I/O should
  // make up for the cost of extra passes over the globals list!
  // 
  // Read-only data section (implies initialized)
  for (Module::const_giterator GI=M->gbegin(), GE=M->gend(); GI != GE; ++GI)
    {
      const GlobalVariable* GV = *GI;
      if (GV->hasInitializer() && GV->isConstant())
        {
          if (GI == M->gbegin())
            enterSection(ReadOnlyData);
          printGlobalVariable(GV);
        }
  }
  
  for (hash_set<const Constant*>::const_iterator I = moduleConstants.begin(),
         E = moduleConstants.end();  I != E; ++I)
    printConstant(*I);
  
  // Initialized read-write data section
  for (Module::const_giterator GI=M->gbegin(), GE=M->gend(); GI != GE; ++GI)
    {
      const GlobalVariable* GV = *GI;
      if (GV->hasInitializer() && ! GV->isConstant())
        {
          if (GI == M->gbegin())
            enterSection(InitRWData);
          printGlobalVariable(GV);
        }
  }

  // Uninitialized read-write data section
  for (Module::const_giterator GI=M->gbegin(), GE=M->gend(); GI != GE; ++GI)
    {
      const GlobalVariable* GV = *GI;
      if (! GV->hasInitializer())
        {
          if (GI == M->gbegin())
            enterSection(UninitRWData);
          printGlobalVariable(GV);
        }
  }

  toAsm << endl;
}


void
SparcAsmPrinter::emitModule(const Module *M)
{
  // TODO: Look for a filename annotation on M to emit a .file directive
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    emitMethod(*I);
  
  emitGlobalsAndConstants(M);
}

}  // End anonymous namespace


//
// emitAssembly - Output assembly language code (a .s file) for the specified
// method. The specified method must have been compiled before this may be
// used.
//
void
UltraSparc::emitAssembly(const Module *M, ostream &toAsm) const
{
  SparcAsmPrinter Print(toAsm, M, *this);
}
