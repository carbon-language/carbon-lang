//===-- Writer.cpp - Library for converting LLVM code to C ----------------===//
//
// This library implements the functionality defined in llvm/Assembly/CWriter.h
//
// TODO : Recursive types.
//
//===-----------------------------------------------------------------------==//

#include "llvm/Assembly/CWriter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/Pass.h"
#include "llvm/SymbolTable.h"
#include "llvm/SlotCalculator.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/InstIterator.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <set>
using std::string;
using std::map;
using std::ostream;

namespace {
  class CWriter : public Pass, public InstVisitor<CWriter> {
    ostream &Out; 
    SlotCalculator *Table;
    const Module *TheModule;
    map<const Type *, string> TypeNames;
    std::set<const Value*> MangledGlobals;
    std::set<const StructType *> StructPrinted;

  public:
    CWriter(ostream &o) : Out(o) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<FindUsedTypes>();
    }

    virtual bool run(Module &M) {
      // Initialize
      Table = new SlotCalculator(&M, false);
      TheModule = &M;

      // Ensure that all structure types have names...
      bool Changed = nameAllUsedStructureTypes(M);

      // Run...
      printModule(&M);

      // Free memory...
      delete Table;
      TypeNames.clear();
      MangledGlobals.clear();
      return false;
    }

    ostream &printType(const Type *Ty, const string &VariableName = "",
                       bool IgnoreName = false, bool namedContext = true);

    void writeOperand(Value *Operand);
    void writeOperandInternal(Value *Operand);

    string getValueName(const Value *V);

  private :
    bool nameAllUsedStructureTypes(Module &M);
    void parseStruct(const Type *Ty);
    void printModule(Module *M);
    void printSymbolTable(const SymbolTable &ST);
    void printGlobal(const GlobalVariable *GV);
    void printFunctionSignature(const Function *F, bool Prototype);

    void printFunction(Function *);

    void printConstant(Constant *CPV);
    void printConstantArray(ConstantArray *CPA);

    // isInlinableInst - Attempt to inline instructions into their uses to build
    // trees as much as possible.  To do this, we have to consistently decide
    // what is acceptable to inline, so that variable declarations don't get
    // printed and an extra copy of the expr is not emitted.
    //
    static bool isInlinableInst(const Instruction &I) {
      // Must be an expression, must be used exactly once.  If it is dead, we
      // emit it inline where it would go.
      if (I.getType() == Type::VoidTy || I.use_size() != 1 ||
          isa<TerminatorInst>(I) || isa<CallInst>(I) || isa<PHINode>(I))
        return false;

      // Only inline instruction it it's use is in the same BB as the inst.
      return I.getParent() == cast<Instruction>(I.use_back())->getParent();
    }

    // Instruction visitation functions
    friend class InstVisitor<CWriter>;

    void visitReturnInst(ReturnInst &I);
    void visitBranchInst(BranchInst &I);

    void visitPHINode(PHINode &I) {}
    void visitBinaryOperator(Instruction &I);

    void visitCastInst (CastInst &I);
    void visitCallInst (CallInst &I);
    void visitShiftInst(ShiftInst &I) { visitBinaryOperator(I); }

    void visitMallocInst(MallocInst &I);
    void visitAllocaInst(AllocaInst &I);
    void visitFreeInst  (FreeInst   &I);
    void visitLoadInst  (LoadInst   &I);
    void visitStoreInst (StoreInst  &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "C Writer does not know about " << I;
      abort();
    }

    void outputLValue(Instruction *I) {
      Out << "  " << getValueName(I) << " = ";
    }
    void printBranchToBlock(BasicBlock *CurBlock, BasicBlock *SuccBlock,
                            unsigned Indent);
    void printIndexingExpression(Value *Ptr, User::op_iterator I,
                                 User::op_iterator E);
  };
}

// We dont want identifier names with ., space, -  in them. 
// So we replace them with _
static string makeNameProper(string x) {
  string tmp;
  for (string::iterator sI = x.begin(), sEnd = x.end(); sI != sEnd; sI++)
    switch (*sI) {
    case '.': tmp += "d_"; break;
    case ' ': tmp += "s_"; break;
    case '-': tmp += "D_"; break;
    default:  tmp += *sI;
    }

  return tmp;
}

string CWriter::getValueName(const Value *V) {
  if (V->hasName()) {              // Print out the label if it exists...
    if (isa<GlobalValue>(V) &&     // Do not mangle globals...
        cast<GlobalValue>(V)->hasExternalLinkage() && // Unless it's internal or
        !MangledGlobals.count(V))  // Unless the name would collide if we don't
      return makeNameProper(V->getName());

    return "l" + utostr(V->getType()->getUniqueID()) + "_" +
           makeNameProper(V->getName());      
  }

  int Slot = Table->getValSlot(V);
  assert(Slot >= 0 && "Invalid value!");
  return "ltmp_" + itostr(Slot) + "_" + utostr(V->getType()->getUniqueID());
}

// A pointer type should not use parens around *'s alone, e.g., (**)
inline bool ptrTypeNameNeedsParens(const string &NameSoFar) {
  return (NameSoFar.find_last_not_of('*') != std::string::npos);
}

// Pass the Type* and the variable name and this prints out the variable
// declaration.
//
ostream &CWriter::printType(const Type *Ty, const string &NameSoFar,
                            bool IgnoreName, bool namedContext) {
  if (Ty->isPrimitiveType())
    switch (Ty->getPrimitiveID()) {
    case Type::VoidTyID:   return Out << "void "               << NameSoFar;
    case Type::BoolTyID:   return Out << "bool "               << NameSoFar;
    case Type::UByteTyID:  return Out << "unsigned char "      << NameSoFar;
    case Type::SByteTyID:  return Out << "signed char "        << NameSoFar;
    case Type::UShortTyID: return Out << "unsigned short "     << NameSoFar;
    case Type::ShortTyID:  return Out << "short "              << NameSoFar;
    case Type::UIntTyID:   return Out << "unsigned "           << NameSoFar;
    case Type::IntTyID:    return Out << "int "                << NameSoFar;
    case Type::ULongTyID:  return Out << "unsigned long long " << NameSoFar;
    case Type::LongTyID:   return Out << "signed long long "   << NameSoFar;
    case Type::FloatTyID:  return Out << "float "              << NameSoFar;
    case Type::DoubleTyID: return Out << "double "             << NameSoFar;
    default :
      std::cerr << "Unknown primitive type: " << Ty << "\n";
      abort();
    }
  
  // Check to see if the type is named.
  if (!IgnoreName) {
    map<const Type *, string>::iterator I = TypeNames.find(Ty);
    if (I != TypeNames.end()) {
      return Out << I->second << " " << NameSoFar;
    }
  }  

  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *MTy = cast<FunctionType>(Ty);
    printType(MTy->getReturnType(), "");
    Out << " " << NameSoFar << " (";

    for (FunctionType::ParamTypes::const_iterator
           I = MTy->getParamTypes().begin(),
           E = MTy->getParamTypes().end(); I != E; ++I) {
      if (I != MTy->getParamTypes().begin())
        Out << ", ";
      printType(*I, "");
    }
    if (MTy->isVarArg()) {
      if (!MTy->getParamTypes().empty()) 
	Out << ", ";
      Out << "...";
    }
    return Out << ")";
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    Out << NameSoFar + " {\n";
    unsigned Idx = 0;
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I) {
      Out << "  ";
      printType(*I, "field" + utostr(Idx++));
      Out << ";\n";
    }
    return Out << "}";
  }  

  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    std::string ptrName = "*" + NameSoFar;

    // Do not need parens around "* NameSoFar" if NameSoFar consists only
    // of zero or more '*' chars *and* this is not an unnamed pointer type
    // such as the result type in a cast statement.  Otherwise, enclose in ( ).
    if (ptrTypeNameNeedsParens(NameSoFar) || !namedContext)
      ptrName = "(" + ptrName + ")";    // 

    return printType(PTy->getElementType(), ptrName);
  }

  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    unsigned NumElements = ATy->getNumElements();
    return printType(ATy->getElementType(),
                     NameSoFar + "[" + utostr(NumElements) + "]");
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    abort();
  }

  return Out;
}

void CWriter::printConstantArray(ConstantArray *CPA) {

  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  // 
  const Type *ETy = CPA->getType()->getElementType();
  bool isString = (ETy == Type::SByteTy || ETy == Type::UByteTy);

  // Make sure the last character is a null char, as automatically added by C
  if (CPA->getNumOperands() == 0 ||
      !cast<Constant>(*(CPA->op_end()-1))->isNullValue())
    isString = false;
  
  if (isString) {
    Out << "\"";
    // Do not include the last character, which we know is null
    for (unsigned i = 0, e = CPA->getNumOperands()-1; i != e; ++i) {
      unsigned char C = (ETy == Type::SByteTy) ?
        (unsigned char)cast<ConstantSInt>(CPA->getOperand(i))->getValue() :
        (unsigned char)cast<ConstantUInt>(CPA->getOperand(i))->getValue();
      
      if (isprint(C)) {
        Out << C;
      } else {
        switch (C) {
        case '\n': Out << "\\n"; break;
        case '\t': Out << "\\t"; break;
        case '\r': Out << "\\r"; break;
        case '\v': Out << "\\v"; break;
        case '\a': Out << "\\a"; break;
        default:
          Out << "\\x";
          Out << ( C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A');
          Out << ((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A');
          break;
        }
      }
    }
    Out << "\"";
  } else {
    Out << "{";
    if (CPA->getNumOperands()) {
      Out << " ";
      printConstant(cast<Constant>(CPA->getOperand(0)));
      for (unsigned i = 1, e = CPA->getNumOperands(); i != e; ++i) {
        Out << ", ";
        printConstant(cast<Constant>(CPA->getOperand(i)));
      }
    }
    Out << " }";
  }
}


// printConstant - The LLVM Constant to C Constant converter.
void CWriter::printConstant(Constant *CPV) {
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    switch (CE->getOpcode()) {
    case Instruction::Cast:
      Out << "((";
      printType(CPV->getType());
      Out << ")";
      printConstant(cast<Constant>(CPV->getOperand(0)));
      Out << ")";
      return;

    case Instruction::GetElementPtr:
      Out << "&(";
      printIndexingExpression(CPV->getOperand(0),
                              CPV->op_begin()+1, CPV->op_end());
      Out << ")";
      return;
    case Instruction::Add:
      Out << "(";
      printConstant(cast<Constant>(CPV->getOperand(0)));
      Out << " + ";
      printConstant(cast<Constant>(CPV->getOperand(1)));
      Out << ")";
      return;
    case Instruction::Sub:
      Out << "(";
      printConstant(cast<Constant>(CPV->getOperand(0)));
      Out << " - ";
      printConstant(cast<Constant>(CPV->getOperand(1)));
      Out << ")";
      return;

    default:
      std::cerr << "CWriter Error: Unhandled constant expression: "
                << CE << "\n";
      abort();
    }
  }

  switch (CPV->getType()->getPrimitiveID()) {
  case Type::BoolTyID:
    Out << (CPV == ConstantBool::False ? "0" : "1"); break;
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
    Out << cast<ConstantSInt>(CPV)->getValue(); break;
  case Type::LongTyID:
    Out << cast<ConstantSInt>(CPV)->getValue() << "ll"; break;

  case Type::UByteTyID:
  case Type::UShortTyID:
    Out << cast<ConstantUInt>(CPV)->getValue(); break;
  case Type::UIntTyID:
    Out << cast<ConstantUInt>(CPV)->getValue() << "u"; break;
  case Type::ULongTyID:
    Out << cast<ConstantUInt>(CPV)->getValue() << "ull"; break;

  case Type::FloatTyID:
  case Type::DoubleTyID:
    Out << cast<ConstantFP>(CPV)->getValue(); break;

  case Type::ArrayTyID:
    printConstantArray(cast<ConstantArray>(CPV));
    break;

  case Type::StructTyID: {
    Out << "{";
    if (CPV->getNumOperands()) {
      Out << " ";
      printConstant(cast<Constant>(CPV->getOperand(0)));
      for (unsigned i = 1, e = CPV->getNumOperands(); i != e; ++i) {
        Out << ", ";
        printConstant(cast<Constant>(CPV->getOperand(i)));
      }
    }
    Out << " }";
    break;
  }

  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(CPV)) {
      Out << "((";
      printType(CPV->getType(), "");
      Out << ")NULL)";
      break;
    } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(CPV)) {
      writeOperand(CPR->getValue());
      break;
    }
    // FALL THROUGH
  default:
    std::cerr << "Unknown constant type: " << CPV << "\n";
    abort();
  }
}

void CWriter::writeOperandInternal(Value *Operand) {
  if (Instruction *I = dyn_cast<Instruction>(Operand))
    if (isInlinableInst(*I)) {
      // Should we inline this instruction to build a tree?
      Out << "(";
      visit(*I);
      Out << ")";    
      return;
    }
  
  if (Operand->hasName()) {   
    Out << getValueName(Operand);
  } else if (Constant *CPV = dyn_cast<Constant>(Operand)) {
    printConstant(CPV); 
  } else {
    int Slot = Table->getValSlot(Operand);
    assert(Slot >= 0 && "Malformed LLVM!");
    Out << "ltmp_" << Slot << "_" << Operand->getType()->getUniqueID();
  }
}

void CWriter::writeOperand(Value *Operand) {
  if (isa<GlobalVariable>(Operand))
    Out << "(&";  // Global variables are references as their addresses by llvm

  writeOperandInternal(Operand);

  if (isa<GlobalVariable>(Operand))
    Out << ")";
}

// nameAllUsedStructureTypes - If there are structure types in the module that
// are used but do not have names assigned to them in the symbol table yet then
// we assign them names now.
//
bool CWriter::nameAllUsedStructureTypes(Module &M) {
  // Get a set of types that are used by the program...
  std::set<const Type *> UT = getAnalysis<FindUsedTypes>().getTypes();

  // Loop over the module symbol table, removing types from UT that are already
  // named.
  //
  SymbolTable *MST = M.getSymbolTableSure();
  if (MST->find(Type::TypeTy) != MST->end())
    for (SymbolTable::type_iterator I = MST->type_begin(Type::TypeTy),
           E = MST->type_end(Type::TypeTy); I != E; ++I)
      UT.erase(cast<Type>(I->second));

  // UT now contains types that are not named.  Loop over it, naming structure
  // types.
  //
  bool Changed = false;
  for (std::set<const Type *>::const_iterator I = UT.begin(), E = UT.end();
       I != E; ++I)
    if (const StructType *ST = dyn_cast<StructType>(*I)) {
      ((Value*)ST)->setName("unnamed", MST);
      Changed = true;
    }
  return Changed;
}

void CWriter::printModule(Module *M) {
  // Calculate which global values have names that will collide when we throw
  // away type information.
  {  // Scope to delete the FoundNames set when we are done with it...
    std::set<string> FoundNames;
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      if (I->hasName())                      // If the global has a name...
        if (FoundNames.count(I->getName()))  // And the name is already used
          MangledGlobals.insert(I);          // Mangle the name
        else
          FoundNames.insert(I->getName());   // Otherwise, keep track of name

    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
      if (I->hasName())                      // If the global has a name...
        if (FoundNames.count(I->getName()))  // And the name is already used
          MangledGlobals.insert(I);          // Mangle the name
        else
          FoundNames.insert(I->getName());   // Otherwise, keep track of name
  }

  // printing stdlib inclusion
  // Out << "#include <stdlib.h>\n";

  // get declaration for alloca
  Out << "/* Provide Declarations */\n"
      << "#include <malloc.h>\n"
      << "#include <alloca.h>\n\n"

    // Provide a definition for null if one does not already exist,
    // and for `bool' if not compiling with a C++ compiler.
      << "#ifndef NULL\n#define NULL 0\n#endif\n\n"
      << "#ifndef __cplusplus\ntypedef unsigned char bool;\n#endif\n"

      << "\n\n/* Global Declarations */\n";

  // First output all the declarations for the program, because C requires
  // Functions & globals to be declared before they are used.
  //

  // Loop over the symbol table, emitting all named constants...
  if (M->hasSymbolTable())
    printSymbolTable(*M->getSymbolTable());

  // Global variable declarations...
  if (!M->gempty()) {
    Out << "\n/* External Global Variable Declarations */\n";
    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I) {
      if (I->hasExternalLinkage()) {
        Out << "extern ";
        printType(I->getType()->getElementType(), getValueName(I));
        Out << ";\n";
      }
    }
  }

  // Function declarations
  if (!M->empty()) {
    Out << "\n/* Function Declarations */\n";
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
      printFunctionSignature(I, true);
      Out << ";\n";
    }
  }

  // Output the global variable definitions and contents...
  if (!M->gempty()) {
    Out << "\n\n/* Global Variable Definitions and Initialization */\n";
    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I) {
      if (I->hasExternalLinkage())
        continue;                       // printed above!
      Out << "static ";
      printType(I->getType()->getElementType(), getValueName(I));
      
      if (I->hasInitializer()) {
        Out << " = " ;
        writeOperand(I->getInitializer());
      }
      Out << ";\n";
    }
  }

  // Output all of the functions...
  if (!M->empty()) {
    Out << "\n\n/* Function Bodies */\n";
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      printFunction(I);
  }
}


/// printSymbolTable - Run through symbol table looking for named constants
/// if a named constant is found, emit it's declaration...
/// Assuming that symbol table has only types and constants.
///
void CWriter::printSymbolTable(const SymbolTable &ST) {
  for (SymbolTable::const_iterator TI = ST.begin(); TI != ST.end(); ++TI) {
    SymbolTable::type_const_iterator I   = ST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = ST.type_end(TI->first);
    
    for (; I != End; ++I) {
      const Value *V = I->second;
      if (const Type *Ty = dyn_cast<Type>(V))
        if (const Type *STy = dyn_cast<StructType>(Ty)) {
          string Name = "struct l_" + makeNameProper(I->first);
          Out << Name << ";\n";
          TypeNames.insert(std::make_pair(STy, Name));
        } else {
          string Name = "l_" + makeNameProper(I->first);
          Out << "typedef ";
          printType(Ty, Name, true);
          Out << ";\n";
        }
    }
  }

  Out << "\n";

  // Loop over all structures then push them into the stack so they are
  // printed in the correct order.
  for (SymbolTable::const_iterator TI = ST.begin(); TI != ST.end(); ++TI) {
    SymbolTable::type_const_iterator I = ST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = ST.type_end(TI->first);
    
    for (; I != End; ++I)
      if (const StructType *STy = dyn_cast<StructType>(I->second))
        parseStruct(STy);
  }
}

// Push the struct onto the stack and recursively push all structs
// this one depends on.
void CWriter::parseStruct(const Type *Ty) {
  if (const StructType *STy = dyn_cast<StructType>(Ty)){
    //Check to see if we have already printed this struct
    if (StructPrinted.find(STy) == StructPrinted.end()){   
      for (StructType::ElementTypes::const_iterator
             I = STy->getElementTypes().begin(),
             E = STy->getElementTypes().end(); I != E; ++I) {
        const Type *Ty1 = dyn_cast<Type>(I->get());
        if (isa<StructType>(Ty1) || isa<ArrayType>(Ty1))
          parseStruct(Ty1);
      }
      
      //Print struct
      StructPrinted.insert(STy);
      string Name = TypeNames[STy];  
      printType(STy, Name, true);
      Out << ";\n";
    }
  }

  // If it is an array, check it's type and continue
  else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)){
    const Type *Ty1 = ATy->getElementType();
    if (isa<StructType>(Ty1) || isa<ArrayType>(Ty1))
      parseStruct(Ty1);
  }
}


void CWriter::printFunctionSignature(const Function *F, bool Prototype) {
  if (F->hasInternalLinkage()) Out << "static ";
  
  // Loop over the arguments, printing them...
  const FunctionType *FT = cast<FunctionType>(F->getFunctionType());
  
  // Print out the return type and name...
  printType(F->getReturnType());
  Out << getValueName(F) << "(";
    
  if (!F->isExternal()) {
    if (!F->aempty()) {
      string ArgName;
      if (F->abegin()->hasName() || !Prototype)
        ArgName = getValueName(F->abegin());

      printType(F->afront().getType(), ArgName);

      for (Function::const_aiterator I = ++F->abegin(), E = F->aend();
           I != E; ++I) {
        Out << ", ";
        if (I->hasName() || !Prototype)
          ArgName = getValueName(I);
        else 
          ArgName = "";
        printType(I->getType(), ArgName);
      }
    }
  } else {
    // Loop over the arguments, printing them...
    for (FunctionType::ParamTypes::const_iterator I = 
	   FT->getParamTypes().begin(),
	   E = FT->getParamTypes().end(); I != E; ++I) {
      if (I != FT->getParamTypes().begin()) Out << ", ";
      printType(*I);
    }
  }

  // Finish printing arguments...
  if (FT->isVarArg()) {
    if (FT->getParamTypes().size()) Out << ", ";
    Out << "...";  // Output varargs portion of signature!
  }
  Out << ")";
}


void CWriter::printFunction(Function *F) {
  if (F->isExternal()) return;

  Table->incorporateFunction(F);

  printFunctionSignature(F, false);
  Out << " {\n";

  // print local variable information for the function
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if ((*I)->getType() != Type::VoidTy && !isInlinableInst(**I)) {
      Out << "  ";
      printType((*I)->getType(), getValueName(*I));
      Out << ";\n";
    }
 
  // print the basic blocks
  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    BasicBlock *Prev = BB->getPrev();

    // Don't print the label for the basic block if there are no uses, or if the
    // only terminator use is the precessor basic block's terminator.  We have
    // to scan the use list because PHI nodes use basic blocks too but do not
    // require a label to be generated.
    //
    bool NeedsLabel = false;
    for (Value::use_iterator UI = BB->use_begin(), UE = BB->use_end();
         UI != UE; ++UI)
      if (TerminatorInst *TI = dyn_cast<TerminatorInst>(*UI))
        if (TI != Prev->getTerminator()) {
          NeedsLabel = true;
          break;        
        }

    if (NeedsLabel) Out << getValueName(BB) << ":\n";

    // Output all of the instructions in the basic block...
    for (BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E; ++II){
      if (!isInlinableInst(*II) && !isa<PHINode>(*II)) {
        if (II->getType() != Type::VoidTy)
          outputLValue(II);
        else
          Out << "  ";
        visit(*II);
        Out << ";\n";
      }
    }

    // Don't emit prefix or suffix for the terminator...
    visit(*BB->getTerminator());
  }
  
  Out << "}\n\n";
  Table->purgeFunction();
}

// Specific Instruction type classes... note that all of the casts are
// neccesary because we use the instruction classes as opaque types...
//
void CWriter::visitReturnInst(ReturnInst &I) {
  // Don't output a void return if this is the last basic block in the function
  if (I.getNumOperands() == 0 && 
      &*--I.getParent()->getParent()->end() == I.getParent() &&
      !I.getParent()->size() == 1) {
    return;
  }

  Out << "  return";
  if (I.getNumOperands()) {
    Out << " ";
    writeOperand(I.getOperand(0));
  }
  Out << ";\n";
}

static bool isGotoCodeNeccessary(BasicBlock *From, BasicBlock *To) {
  // If PHI nodes need copies, we need the copy code...
  if (isa<PHINode>(To->front()) ||
      From->getNext() != To)      // Not directly successor, need goto
    return true;

  // Otherwise we don't need the code.
  return false;
}

void CWriter::printBranchToBlock(BasicBlock *CurBB, BasicBlock *Succ,
                                           unsigned Indent) {
  for (BasicBlock::iterator I = Succ->begin();
       PHINode *PN = dyn_cast<PHINode>(&*I); ++I) {
    //  now we have to do the printing
    Out << string(Indent, ' ');
    outputLValue(PN);
    writeOperand(PN->getIncomingValue(PN->getBasicBlockIndex(CurBB)));
    Out << ";   /* for PHI node */\n";
  }

  if (CurBB->getNext() != Succ) {
    Out << string(Indent, ' ') << "  goto ";
    writeOperand(Succ);
    Out << ";\n";
  }
}

// Brach instruction printing - Avoid printing out a brach to a basic block that
// immediately succeeds the current one.
//
void CWriter::visitBranchInst(BranchInst &I) {
  if (I.isConditional()) {
    if (isGotoCodeNeccessary(I.getParent(), I.getSuccessor(0))) {
      Out << "  if (";
      writeOperand(I.getCondition());
      Out << ") {\n";
      
      printBranchToBlock(I.getParent(), I.getSuccessor(0), 2);
      
      if (isGotoCodeNeccessary(I.getParent(), I.getSuccessor(1))) {
        Out << "  } else {\n";
        printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
      }
    } else {
      // First goto not neccesary, assume second one is...
      Out << "  if (!";
      writeOperand(I.getCondition());
      Out << ") {\n";

      printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
    }

    Out << "  }\n";
  } else {
    printBranchToBlock(I.getParent(), I.getSuccessor(0), 0);
  }
  Out << "\n";
}


void CWriter::visitBinaryOperator(Instruction &I) {
  // binary instructions, shift instructions, setCond instructions.
  if (isa<PointerType>(I.getType())) {
    Out << "(";
    printType(I.getType());
    Out << ")";
  }
      
  if (isa<PointerType>(I.getType())) Out << "(long long)";
  writeOperand(I.getOperand(0));

  switch (I.getOpcode()) {
  case Instruction::Add: Out << " + "; break;
  case Instruction::Sub: Out << " - "; break;
  case Instruction::Mul: Out << "*"; break;
  case Instruction::Div: Out << "/"; break;
  case Instruction::Rem: Out << "%"; break;
  case Instruction::And: Out << " & "; break;
  case Instruction::Or: Out << " | "; break;
  case Instruction::Xor: Out << " ^ "; break;
  case Instruction::SetEQ: Out << " == "; break;
  case Instruction::SetNE: Out << " != "; break;
  case Instruction::SetLE: Out << " <= "; break;
  case Instruction::SetGE: Out << " >= "; break;
  case Instruction::SetLT: Out << " < "; break;
  case Instruction::SetGT: Out << " > "; break;
  case Instruction::Shl : Out << " << "; break;
  case Instruction::Shr : Out << " >> "; break;
  default: std::cerr << "Invalid operator type!" << I; abort();
  }

  if (isa<PointerType>(I.getType())) Out << "(long long)";
  writeOperand(I.getOperand(1));
}

void CWriter::visitCastInst(CastInst &I) {
  Out << "(";
  printType(I.getType(), string(""),/*ignoreName*/false, /*namedContext*/false);
  Out << ")";
  writeOperand(I.getOperand(0));
}

void CWriter::visitCallInst(CallInst &I) {
  const PointerType  *PTy   = cast<PointerType>(I.getCalledValue()->getType());
  const FunctionType *FTy   = cast<FunctionType>(PTy->getElementType());
  const Type         *RetTy = FTy->getReturnType();
  
  writeOperand(I.getOperand(0));
  Out << "(";

  if (I.getNumOperands() > 1) {
    writeOperand(I.getOperand(1));

    for (unsigned op = 2, Eop = I.getNumOperands(); op != Eop; ++op) {
      Out << ", ";
      writeOperand(I.getOperand(op));
    }
  }
  Out << ")";
}  

void CWriter::visitMallocInst(MallocInst &I) {
  Out << "(";
  printType(I.getType());
  Out << ")malloc(sizeof(";
  printType(I.getType()->getElementType());
  Out << ")";

  if (I.isArrayAllocation()) {
    Out << " * " ;
    writeOperand(I.getOperand(0));
  }
  Out << ")";
}

void CWriter::visitAllocaInst(AllocaInst &I) {
  Out << "(";
  printType(I.getType());
  Out << ") alloca(sizeof(";
  printType(I.getType()->getElementType());
  Out << ")";
  if (I.isArrayAllocation()) {
    Out << " * " ;
    writeOperand(I.getOperand(0));
  }
  Out << ")";
}

void CWriter::visitFreeInst(FreeInst &I) {
  Out << "free(";
  writeOperand(I.getOperand(0));
  Out << ")";
}

void CWriter::printIndexingExpression(Value *Ptr, User::op_iterator I,
                                      User::op_iterator E) {
  bool HasImplicitAddress = false;
  // If accessing a global value with no indexing, avoid *(&GV) syndrome
  if (GlobalValue *V = dyn_cast<GlobalValue>(Ptr)) {
    HasImplicitAddress = true;
  } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Ptr)) {
    HasImplicitAddress = true;
    Ptr = CPR->getValue();         // Get to the global...
  }

  if (I == E) {
    if (!HasImplicitAddress)
      Out << "*";  // Implicit zero first argument: '*x' is equivalent to 'x[0]'

    writeOperandInternal(Ptr);
    return;
  }

  const Constant *CI = dyn_cast<Constant>(I->get());
  if (HasImplicitAddress && (!CI || !CI->isNullValue()))
    Out << "(&";

  writeOperandInternal(Ptr);

  if (HasImplicitAddress && (!CI || !CI->isNullValue()))
    Out << ")";

  // Print out the -> operator if possible...
  if (CI && CI->isNullValue() && I+1 != E) {
    if ((*(I+1))->getType() == Type::UByteTy) {
      Out << (HasImplicitAddress ? "." : "->");
      Out << "field" << cast<ConstantUInt>(*(I+1))->getValue();
      I += 2;
    } else {  // First array index of 0: Just skip it
      ++I;
    }
  }

  for (; I != E; ++I)
    if ((*I)->getType() == Type::LongTy) {
      Out << "[";
      writeOperand(*I);
      Out << "]";
    } else {
      Out << ".field" << cast<ConstantUInt>(*I)->getValue();
    }
}

void CWriter::visitLoadInst(LoadInst &I) {
  Out << "*";
  writeOperand(I.getOperand(0));
}

void CWriter::visitStoreInst(StoreInst &I) {
  Out << "*";
  writeOperand(I.getPointerOperand());
  Out << " = ";
  writeOperand(I.getOperand(0));
}

void CWriter::visitGetElementPtrInst(GetElementPtrInst &I) {
  Out << "&";
  printIndexingExpression(I.getPointerOperand(), I.idx_begin(), I.idx_end());
}

//===----------------------------------------------------------------------===//
//                       External Interface declaration
//===----------------------------------------------------------------------===//

Pass *createWriteToCPass(std::ostream &o) { return new CWriter(o); }
