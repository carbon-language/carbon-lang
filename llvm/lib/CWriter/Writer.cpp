//===-- Writer.cpp - Library for writing C files --------------------------===//
//
// This library implements the functionality defined in llvm/Assembly/CWriter.h
// and CLocalVars.h
//
// TODO : Recursive types.
//
//===-----------------------------------------------------------------------==//

#include "llvm/Assembly/CWriter.h"
#include "CLocalVars.h"
#include "llvm/SlotCalculator.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Argument.h"
#include "llvm/BasicBlock.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/SymbolTable.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"

#include <algorithm>
#include <strstream>
using std::string;
using std::map;
using std::vector;
using std::ostream;

//===-----------------------------------------------------------------------==//
//
// Implementation of the CLocalVars methods

// Appends a variable to the LocalVars map if it does not already exist
// Also check that the type exists on the map.
void CLocalVars::addLocalVar(const Type *t, const string & var) {
  if (!LocalVars.count(t) || 
      find(LocalVars[t].begin(), LocalVars[t].end(), var) 
      == LocalVars[t].end()) {
      LocalVars[t].push_back(var);
  } 
}

static std::string getConstStrValue(const Constant* CPV);


static std::string getConstArrayStrValue(const Constant* CPV) {
  std::string Result;
  
  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  // 
  const Type *ETy = cast<ArrayType>(CPV->getType())->getElementType();
  bool isString = (ETy == Type::SByteTy || ETy == Type::UByteTy);

  if (ETy == Type::SByteTy) {
    for (unsigned i = 0; i < CPV->getNumOperands(); ++i)
      if (ETy == Type::SByteTy &&
          cast<ConstantSInt>(CPV->getOperand(i))->getValue() < 0) {
        isString = false;
        break;
      }
  }
  if (isString) {
    // Make sure the last character is a null char, as automatically added by C
    if (CPV->getNumOperands() == 0 ||
        !cast<Constant>(*(CPV->op_end()-1))->isNullValue())
      isString = false;
  }
  
  if (isString) {
    Result = "\"";
    // Do not include the last character, which we know is null
    for (unsigned i = 0, e = CPV->getNumOperands()-1; i != e; ++i) {
      unsigned char C = (ETy == Type::SByteTy) ?
        (unsigned char)cast<ConstantSInt>(CPV->getOperand(i))->getValue() :
        (unsigned char)cast<ConstantUInt>(CPV->getOperand(i))->getValue();
      
      if (isprint(C)) {
        Result += C;
      } else {
        switch (C) {
        case '\n': Result += "\\n"; break;
        case '\t': Result += "\\t"; break;
        case '\r': Result += "\\r"; break;
        case '\v': Result += "\\v"; break;
        case '\a': Result += "\\a"; break;
        default:
          Result += "\\x";
          Result += ( C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A');
          Result += ((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A');
          break;
        }
      }
    }
    Result += "\"";
    
  } else {
    Result = "{";
    if (CPV->getNumOperands()) {
      Result += " " +  getConstStrValue(cast<Constant>(CPV->getOperand(0)));
      for (unsigned i = 1; i < CPV->getNumOperands(); i++)
        Result += ", " + getConstStrValue(cast<Constant>(CPV->getOperand(i)));
    }
    Result += " }";
  }
  
  return Result;
}

static std::string getConstStrValue(const Constant* CPV) {
  switch (CPV->getType()->getPrimitiveID()) {
  case Type::BoolTyID:
    return CPV == ConstantBool::False ? "0" : "1";
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
    return itostr(cast<ConstantSInt>(CPV)->getValue());
  case Type::LongTyID:
    return itostr(cast<ConstantSInt>(CPV)->getValue()) + "ll";

  case Type::UByteTyID:
    return utostr(cast<ConstantUInt>(CPV)->getValue());
  case Type::UShortTyID:
    return utostr(cast<ConstantUInt>(CPV)->getValue());
  case Type::UIntTyID:
    return utostr(cast<ConstantUInt>(CPV)->getValue())+"u";
  case Type::ULongTyID:
    return utostr(cast<ConstantUInt>(CPV)->getValue())+"ull";

  case Type::FloatTyID:
  case Type::DoubleTyID:
    return ftostr(cast<ConstantFP>(CPV)->getValue());

  case Type::ArrayTyID:
    return getConstArrayStrValue(CPV);

  case Type::StructTyID: {
    std::string Result = "{";
    if (CPV->getNumOperands()) {
      Result += " " + getConstStrValue(cast<Constant>(CPV->getOperand(0)));
      for (unsigned i = 1; i < CPV->getNumOperands(); i++)
        Result += ", " + getConstStrValue(cast<Constant>(CPV->getOperand(i)));
    }
    return Result + " }";
  }

  default:
    cerr << "Unknown constant type: " << CPV << "\n";
    abort();
  }
}

// Internal function
// Pass the Type* variable and and the variable name and this prints out the 
// variable declaration.
// This is different from calcTypeName because if you need to declare an array
// the size of the array would appear after the variable name itself
// For eg. int a[10];
static string calcTypeNameVar(const Type *Ty,
                              map<const Type *, string> &TypeNames, 
                              const string &NameSoFar, bool ignoreName = false){
  if (Ty->isPrimitiveType())
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID: 
      return "bool " + NameSoFar;
    case Type::UByteTyID: 
      return "unsigned char " + NameSoFar;
    case Type::SByteTyID:
      return "signed char " + NameSoFar;
    case Type::UShortTyID:
      return "unsigned long long " + NameSoFar;
    case Type::ULongTyID:
      return "unsigned long long " + NameSoFar;
    case Type::LongTyID:
      return "signed long long " + NameSoFar;
    case Type::UIntTyID:
      return "unsigned " + NameSoFar;
    default :
      return Ty->getDescription() + " " + NameSoFar;
    }
  
  // Check to see if the type is named.
  if (!ignoreName) {
    map<const Type *, string>::iterator I = TypeNames.find(Ty);
    if (I != TypeNames.end())
      return I->second + " " + NameSoFar;
  }  

  string Result;
  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *MTy = cast<const FunctionType>(Ty);
    Result += calcTypeNameVar(MTy->getReturnType(), TypeNames, "");
    Result += " " + NameSoFar;
    Result += " (";
    for (FunctionType::ParamTypes::const_iterator
           I = MTy->getParamTypes().begin(),
           E = MTy->getParamTypes().end(); I != E; ++I) {
      if (I != MTy->getParamTypes().begin())
        Result += ", ";
      Result += calcTypeNameVar(*I, TypeNames, "");
    }
    if (MTy->isVarArg()) {
      if (!MTy->getParamTypes().empty()) 
	Result += ", ";
      Result += "...";
    }
    Result += ")";
    break;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<const StructType>(Ty);
    Result = " struct {\n ";
    int indx = 0;
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I) {
      Result += calcTypeNameVar(*I, TypeNames, "field" + itostr(indx++));
      Result += ";\n ";
    }
    Result += " }";
    Result += " " + NameSoFar;
    break;
  }  

  case Type::PointerTyID: {
    Result = calcTypeNameVar(cast<const PointerType>(Ty)->getElementType(), 
			     TypeNames, "(*" + NameSoFar + ")");
    break;
  }
  
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<const ArrayType>(Ty);
    int NumElements = ATy->getNumElements();
    Result = calcTypeNameVar(ATy->getElementType(),  TypeNames, 
			     NameSoFar + "[" + itostr(NumElements) + "]");
    break;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    Result = "<error>";
  }

  return Result;
}

namespace {
  class CWriter {
    ostream& Out; 
    SlotCalculator &Table;
    const Module *TheModule;
    map<const Type *, string> TypeNames;
  public:
    inline CWriter(ostream &o, SlotCalculator &Tab, const Module *M)
      : Out(o), Table(Tab), TheModule(M) {
    }
    
    inline void write(const Module *M) { printModule(M); }

    ostream& printTypeVar(const Type *Ty, const string &VariableName) {
      return Out << calcTypeNameVar(Ty, TypeNames, VariableName);
    }

    ostream& printType(const Type *Ty) {
      return Out << calcTypeNameVar(Ty, TypeNames, "");
    }

    void writeOperand(const Value *Operand);

    string getValueName(const Value *V);
  private :

    void printModule(const Module *M);
    void printSymbolTable(const SymbolTable &ST);
    void printGlobal(const GlobalVariable *GV);
    void printFunctionSignature(const Function *F);
    void printFunctionDecl(const Function *F); // Print just the forward decl
    void printFunctionArgument(const Argument *FA);
    
    void printFunction(const Function *);
    
    void outputBasicBlock(const BasicBlock *);
  };
  /* END class CWriter */


  /* CLASS InstLocalVarsVisitor */
  class InstLocalVarsVisitor : public InstVisitor<InstLocalVarsVisitor> {
    CWriter& CW;
    void handleTerminator(TerminatorInst *tI, int indx);
  public:
    CLocalVars CLV;
    
    InstLocalVarsVisitor(CWriter &cw) : CW(cw) {}
    
    void visitInstruction(Instruction *I) {
      if (I->getType() != Type::VoidTy)
        CLV.addLocalVar(I->getType(), CW.getValueName(I));
    }

    void visitBranchInst(BranchInst *I) {
      handleTerminator(I, 0);
      if (I->isConditional())
	handleTerminator(I, 1);
    }
  };
}

void InstLocalVarsVisitor::handleTerminator(TerminatorInst *tI,int indx) {
  BasicBlock *bb = tI->getSuccessor(indx);

  BasicBlock::const_iterator insIt = bb->begin();
  while (insIt != bb->end()) {
    if (const PHINode *pI = dyn_cast<PHINode>(*insIt)) {
      // Its a phinode!
      // Calculate the incoming index for this
      assert(pI->getBasicBlockIndex(tI->getParent()) != -1);

      CLV.addLocalVar(pI->getType(), CW.getValueName(pI));
    } else
      break;
    insIt++;
  }
}

namespace {
  /* CLASS CInstPrintVisitor */

  class CInstPrintVisitor: public InstVisitor<CInstPrintVisitor> {
    CWriter& CW;
    SlotCalculator& Table;
    ostream &Out;

    void outputLValue(Instruction *);
    void printPhiFromNextBlock(TerminatorInst *tI, int indx);
    void printIndexingExpr(MemAccessInst *MAI);

  public:
    CInstPrintVisitor (CWriter &cw, SlotCalculator& table, ostream& o) 
      : CW(cw), Table(table), Out(o) {}
    
    void visitCastInst(CastInst *I);
    void visitCallInst(CallInst *I);
    void visitShiftInst(ShiftInst *I) { visitBinaryOperator(I); }
    void visitReturnInst(ReturnInst *I);
    void visitBranchInst(BranchInst *I);
    void visitSwitchInst(SwitchInst *I);
    void visitInvokeInst(InvokeInst *I) ;
    void visitMallocInst(MallocInst *I);
    void visitAllocaInst(AllocaInst *I);
    void visitFreeInst(FreeInst   *I);
    void visitLoadInst(LoadInst   *I);
    void visitStoreInst(StoreInst  *I);
    void visitGetElementPtrInst(GetElementPtrInst *I);
    void visitPHINode(PHINode *I) {}

    void visitNot(GenericUnaryInst *I);
    void visitBinaryOperator(Instruction *I);
  };
}

void CInstPrintVisitor::outputLValue(Instruction *I) {
  Out << "  " << CW.getValueName(I) << " = ";
}

void CInstPrintVisitor::printPhiFromNextBlock(TerminatorInst *tI, int indx) {
  BasicBlock *bb = tI->getSuccessor(indx);
  BasicBlock::const_iterator insIt = bb->begin();
  while (insIt != bb->end()) {
    if (PHINode *pI = dyn_cast<PHINode>(*insIt)) {
      //Its a phinode!
      //Calculate the incoming index for this
      int incindex = pI->getBasicBlockIndex(tI->getParent());
      if (incindex != -1) {
        //now we have to do the printing
        outputLValue(pI);
        CW.writeOperand(pI->getIncomingValue(incindex));
        Out << ";\n";
      }
    }
    else break;
    insIt++;
  }
}

// Implement all "other" instructions, except for PHINode
void CInstPrintVisitor::visitCastInst(CastInst *I) {
  outputLValue(I);
  Out << "(";
  CW.printType(I->getType());
  Out << ")";
  CW.writeOperand(I->getOperand(0));
  Out << ";\n";
}

void CInstPrintVisitor::visitCallInst(CallInst *I) {
  if (I->getType() != Type::VoidTy)
    outputLValue(I);
  else
    Out << "  ";

  const PointerType  *PTy   = cast<PointerType>(I->getCalledValue()->getType());
  const FunctionType *FTy   = cast<FunctionType>(PTy->getElementType());
  const Type         *RetTy = FTy->getReturnType();
  
  Out << CW.getValueName(I->getOperand(0)) << "(";

  if (I->getNumOperands() > 1) {
    CW.writeOperand(I->getOperand(1));

    for (unsigned op = 2, Eop = I->getNumOperands(); op != Eop; ++op) {
      Out << ", ";
      CW.writeOperand(I->getOperand(op));
    }
  }
  Out << ");\n";
} 
 
// Specific Instruction type classes... note that all of the casts are
// neccesary because we use the instruction classes as opaque types...
//
void CInstPrintVisitor::visitReturnInst(ReturnInst *I) {
  Out << "  return";
  if (I->getNumOperands()) {
    Out << " ";
    CW.writeOperand(I->getOperand(0));
  }
  Out << ";\n";
}

void CInstPrintVisitor::visitBranchInst(BranchInst *I) {
  TerminatorInst *tI = cast<TerminatorInst>(I);
  if (I->isConditional()) {
    Out << "  if (";
    CW.writeOperand(I->getCondition());
    Out << ") {\n";
    printPhiFromNextBlock(tI,0);
    Out << "    goto ";
    CW.writeOperand(I->getOperand(0));
    Out << ";\n";
    Out << "  } else {\n";
    printPhiFromNextBlock(tI,1);
    Out << "    goto ";
    CW.writeOperand(I->getOperand(1));
    Out << ";\n  }\n";
  } else {
    printPhiFromNextBlock(tI,0);
    Out << "  goto ";
    CW.writeOperand(I->getOperand(0));
    Out << ";\n";
  }
  Out << "\n";
}

void CInstPrintVisitor::visitSwitchInst(SwitchInst *I) {
  assert(0 && "Switch not implemented!");
}

void CInstPrintVisitor::visitInvokeInst(InvokeInst *I) {
  assert(0 && "Invoke not implemented!");
}

void CInstPrintVisitor::visitMallocInst(MallocInst *I) {
  outputLValue(I);
  Out << "(";
  CW.printType(I->getType()->getElementType());
  Out << "*)malloc(sizeof(";
  CW.printTypeVar(I->getType()->getElementType(), "");
  Out << ")";

  if (I->isArrayAllocation()) {
    Out << " * " ;
    CW.writeOperand(I->getOperand(0));
  }
  Out << ");";
}

void CInstPrintVisitor::visitAllocaInst(AllocaInst *I) {
  outputLValue(I);
  Out << "(";
  CW.printTypeVar(I->getType(), "");
  Out << ") alloca(sizeof(";
  CW.printTypeVar(I->getType()->getElementType(), "");
  Out << ")";
  if (I->isArrayAllocation()) {
    Out << " * " ;
    CW.writeOperand(I->getOperand(0));
  }
  Out << ");\n";
}

void CInstPrintVisitor::visitFreeInst(FreeInst   *I) {
  Out << "free(";
  CW.writeOperand(I->getOperand(0));
  Out << ");\n";
}

void CInstPrintVisitor::printIndexingExpr(MemAccessInst *MAI) {
  CW.writeOperand(MAI->getPointerOperand());

  for (MemAccessInst::op_iterator I = MAI->idx_begin(), E = MAI->idx_end();
       I != E; ++I)
    if ((*I)->getType() == Type::UIntTy) {
      Out << "[";
      CW.writeOperand(*I);
      Out << "]";
    } else {
      Out << ".field" << cast<ConstantUInt>(*I)->getValue();
    }
}

void CInstPrintVisitor::visitLoadInst(LoadInst *I) {
  outputLValue(I);
  printIndexingExpr(I);
  Out << ";\n";
}

void CInstPrintVisitor::visitStoreInst(StoreInst *I) {
  Out << "  ";
  printIndexingExpr(I);
  Out << " = ";
  CW.writeOperand(I->getOperand(0));
  Out << ";\n";
}

void CInstPrintVisitor::visitGetElementPtrInst(GetElementPtrInst *I) {
  outputLValue(I);
  Out << "&";
  printIndexingExpr(I);
  Out << ";\n";
}

void CInstPrintVisitor::visitNot(GenericUnaryInst *I) {
  outputLValue(I);
  Out << "~";
  CW.writeOperand(I->getOperand(0));
  Out << ";\n";
}

void CInstPrintVisitor::visitBinaryOperator(Instruction *I) {
  // binary instructions, shift instructions, setCond instructions.
  outputLValue(I);
  if (isa<PointerType>(I->getType())) {
    Out << "(";
    CW.printType(I->getType());
    Out << ")";
  }
      
  if (isa<PointerType>(I->getType())) Out << "(long long)";
  CW.writeOperand(I->getOperand(0));

  switch (I->getOpcode()) {
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
  default: cerr << "Invalid operator type!" << I; abort();
  }

  if (isa<PointerType>(I->getType())) Out << "(long long)";
  CW.writeOperand(I->getOperand(1));
  Out << ";\n";
}

/* END : CInstPrintVisitor implementation */

// We dont want identifier names with ., space, -  in them. 
// So we replace them with _
static string makeNameProper(string x) {
  string tmp;
  for (string::iterator sI = x.begin(), sEnd = x.end(); sI != sEnd; sI++)
    switch (*sI) {
    case '.': tmp += "_d"; break;
    case ' ': tmp += "_s"; break;
    case '-': tmp += "_D"; break;
    case '_': tmp += "__"; break;
    default:  tmp += *sI;
    }

  return tmp;
}

string CWriter::getValueName(const Value *V) {
  if (V->hasName()) {             // Print out the label if it exists...
    if (isa<GlobalValue>(V))  // Do not mangle globals...
      return makeNameProper(V->getName());

    return "l" + utostr(V->getType()->getUniqueID()) + "_" +
           makeNameProper(V->getName());      
  }

  int Slot = Table.getValSlot(V);
  assert(Slot >= 0 && "Invalid value!");
  return "ltmp_" + itostr(Slot) + "_" + utostr(V->getType()->getUniqueID());
}

void CWriter::printModule(const Module *M) {
  // printing stdlib inclusion
  // Out << "#include <stdlib.h>\n";

  // get declaration for alloca
  Out << "/* Provide Declarations */\n"
      << "#include <alloca.h>\n\n"

    // Provide a definition for null if one does not already exist.
      << "#ifndef NULL\n#define NULL 0\n#endif\n\n"
      << "typedef unsigned char bool;\n"

      << "\n\n/* Global Symbols */\n";

  // Loop over the symbol table, emitting all named constants...
  if (M->hasSymbolTable())
    printSymbolTable(*M->getSymbolTable());

  Out << "\n\n/* Global Data */\n";
  for_each(M->gbegin(), M->gend(), 
	   bind_obj(this, &CWriter::printGlobal));

  // First output all the declarations of the functions as C requires Functions 
  // be declared before they are used.
  //
  Out << "\n\n/* Function Declarations */\n";
  for_each(M->begin(), M->end(), bind_obj(this, &CWriter::printFunctionDecl));
  
  // Output all of the functions...
  Out << "\n\n/* Function Bodies */\n";
  for_each(M->begin(), M->end(), bind_obj(this, &CWriter::printFunction));
}

// prints the global constants
void CWriter::printGlobal(const GlobalVariable *GV) {
  if (GV->hasInternalLinkage()) Out << "static ";

  printTypeVar(GV->getType()->getElementType(), getValueName(GV));

  if (GV->hasInitializer()) {
    Out << " = " ;
    writeOperand(GV->getInitializer());
  }

  Out << ";\n";
}

// printSymbolTable - Run through symbol table looking for named constants
// if a named constant is found, emit it's declaration...
// Assuming that symbol table has only types and constants.
void CWriter::printSymbolTable(const SymbolTable &ST) {
  for (SymbolTable::const_iterator TI = ST.begin(); TI != ST.end(); ++TI) {
    SymbolTable::type_const_iterator I = ST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = ST.type_end(TI->first);
    
    for (; I != End; ++I)
      if (const Type *Ty = dyn_cast<const StructType>(I->second)) {
	string Name = "struct l_" + I->first;
        Out << Name << ";\n";

        TypeNames.insert(std::make_pair(Ty, Name));
      }
  }

  Out << "\n";

  for (SymbolTable::const_iterator TI = ST.begin(); TI != ST.end(); ++TI) {
    SymbolTable::type_const_iterator I = ST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = ST.type_end(TI->first);
    
    for (; I != End; ++I) {
      const Value *V = I->second;
      if (const Type *Ty = dyn_cast<const Type>(V)) {
	Out << "typedef ";
	string Name = "l_" + I->first;
        if (isa<StructType>(Ty)) Name = "struct " + Name;
	Out << calcTypeNameVar(Ty, TypeNames, Name, true) << ";\n";
      }
    }
  }
}


// printFunctionDecl - Print function declaration
//
void CWriter::printFunctionDecl(const Function *F) {
  printFunctionSignature(F);
  Out << ";\n";
}

void CWriter::printFunctionSignature(const Function *F) {
  if (F->hasInternalLinkage()) Out << "static ";
  
  // Loop over the arguments, printing them...
  const FunctionType *FT = cast<FunctionType>(F->getFunctionType());
  
  // Print out the return type and name...
  printType(F->getReturnType());
  Out << " " << getValueName(F) << "(";
    
  if (!F->isExternal()) {
    for_each(F->getArgumentList().begin(), F->getArgumentList().end(),
	     bind_obj(this, &CWriter::printFunctionArgument));
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


// printFunctionArgument - This member is called for every argument that 
// is passed into the method.  Simply print it out
//
void CWriter::printFunctionArgument(const Argument *Arg) {
  // Insert commas as we go... the first arg doesn't get a comma
  if (Arg != Arg->getParent()->getArgumentList().front()) Out << ", ";
  
  // Output type...
  printTypeVar(Arg->getType(), getValueName(Arg));
}

void CWriter::printFunction(const Function *F) {
  if (F->isExternal()) return;

  Table.incorporateFunction(F);

  // Process each of the basic blocks, gather information and call the  
  // output methods on the CLocalVars and Function* objects.
    
  // gather local variable information for each basic block
  InstLocalVarsVisitor ILV(*this);
  ILV.visit((Function *)F);

  printFunctionSignature(F);
  Out << " {\n";

  // Loop over the symbol table, emitting all named constants...
  if (F->hasSymbolTable())
    printSymbolTable(*F->getSymbolTable()); 
  
  // print the local variables
  // we assume that every local variable is alloca'ed in the C code.
  std::map<const Type*, VarListType> &locals = ILV.CLV.LocalVars;
  
  map<const Type*, VarListType>::iterator iter;
  for (iter = locals.begin(); iter != locals.end(); ++iter) {
    VarListType::iterator listiter;
    for (listiter = iter->second.begin(); listiter != iter->second.end(); 
         ++listiter) {
      Out << "  ";
      printTypeVar(iter->first, *listiter);
      Out << ";\n";
    }
  }
 
  // print the basic blocks
  for_each(F->begin(), F->end(), bind_obj(this, &CWriter::outputBasicBlock));
  
  Out << "}\n";
  Table.purgeFunction();
}

void CWriter::outputBasicBlock(const BasicBlock* BB) {
  Out << getValueName(BB) << ":\n";

  // Output all of the instructions in the basic block...
  // print the basic blocks
  CInstPrintVisitor CIPV(*this, Table, Out);
  CIPV.visit((BasicBlock *) BB);
}

void CWriter::writeOperand(const Value *Operand) {
  if (isa<GlobalVariable>(Operand))
    Out << "(&";  // Global variables are references as their addresses by llvm

  if (Operand->hasName()) {   
    Out << getValueName(Operand);
  } else if (const Constant *CPV = dyn_cast<const Constant>(Operand)) {
    if (isa<ConstantPointerNull>(CPV))
      Out << "NULL";
    else
      Out << getConstStrValue(CPV); 
  } else {
    int Slot = Table.getValSlot(Operand);
    assert(Slot >= 0 && "Malformed LLVM!");
    Out << "ltmp_" << Slot << "_" << Operand->getType()->getUniqueID();
  }

  if (isa<GlobalVariable>(Operand))
    Out << ")";
}


//===----------------------------------------------------------------------===//
//                       External Interface declaration
//===----------------------------------------------------------------------===//

void WriteToC(const Module *M, ostream &Out) {
  assert(M && "You can't write a null module!!");
  SlotCalculator SlotTable(M, false);
  CWriter W(Out, SlotTable, M);
  W.write(M);
  Out.flush();
}
