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

static string calcTypeNameVar(const Type *Ty,
			      map<const Type *, string> &TypeNames, 
			      string VariableName, string NameSoFar);

static std::string getConstStrValue(const Constant* CPV);


// We dont want identifier names with ., space, -  in them. 
// So we replace them with _
static string makeNameProper(string x) {
  string tmp;
  for (string::iterator sI = x.begin(), sEnd = x.end(); sI != sEnd; sI++) {
    if (*sI == '.')
      tmp += '_';
    else if (*sI == ' ')
      tmp += '_';
    else if (*sI == '-')
      tmp += "__";
    else
      tmp += *sI;
  }
  return tmp;
}

static string getConstantName(const Constant *CPV) {
  return CPV->getName();
}


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
    Result = "\"";
    for (unsigned i = 0; i < CPV->getNumOperands(); ++i) {
      unsigned char C = (ETy == Type::SByteTy) ?
        (unsigned char)cast<ConstantSInt>(CPV->getOperand(i))->getValue() :
        (unsigned char)cast<ConstantUInt>(CPV->getOperand(i))->getValue();
      
      if (isprint(C)) {
        Result += C;
      } else {
        Result += "\\x";
        Result += ( C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A');
        Result += ((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A');
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

static std::string getConstStructStrValue(const Constant* CPV) {
  std::string Result = "{";
  if (CPV->getNumOperands()) {
    Result += " " + getConstStrValue(cast<Constant>(CPV->getOperand(0)));
    for (unsigned i = 1; i < CPV->getNumOperands(); i++)
      Result += ", " + getConstStrValue(cast<Constant>(CPV->getOperand(i)));
  }

  return Result + " }";
}

// our own getStrValue function for constant initializers
static std::string getConstStrValue(const Constant* CPV) {
  // Does not handle null pointers, that needs to be checked explicitly
  string tempstr;
  if (CPV == ConstantBool::False)
    return "0";
  else if (CPV == ConstantBool::True)
    return "1";
  
  else if (isa<ConstantArray>(CPV)) {
    tempstr = getConstArrayStrValue(CPV);
  }
  else  if (isa<ConstantStruct>(CPV)) {
    tempstr = getConstStructStrValue(CPV);
  }
  else if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(CPV)) {
    tempstr = utostr(CUI->getValue());
  } 
  else if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(CPV)) {
    tempstr = itostr(CSI->getValue());
  }
  else if (ConstantFP *CFP = dyn_cast<ConstantFP>(CPV)) {
    tempstr = ftostr(CFP->getValue());
  }
  
  if (CPV->getType() == Type::ULongTy)
    tempstr += "ull";
  else if (CPV->getType() == Type::LongTy)
    tempstr += "ll";
  else if (CPV->getType() == Type::UIntTy ||
	   CPV->getType() == Type::UShortTy)
    tempstr += "u";
  
  return tempstr;

}

// Internal function
// Essentially pass the Type* variable, an empty typestack and this prints 
// out the C type
static string calcTypeName(const Type *Ty, map<const Type *, string> &TypeNames,
			   string &FunctionInfo) {
  
  // Takin' care of the fact that boolean would be int in C
  // and that ushort would be unsigned short etc.
  
  // Base Case
  if (Ty->isPrimitiveType())
    switch (Ty->getPrimitiveID()) {
    case Type::VoidTyID:   return "void";
    case Type::BoolTyID:   return "bool";
    case Type::UByteTyID:  return "unsigned char";
    case Type::SByteTyID:  return "signed char";
    case Type::UShortTyID: return "unsigned short";
    case Type::ShortTyID:  return "short";
    case Type::UIntTyID:   return "unsigned";
    case Type::IntTyID:    return "int";
    case Type::ULongTyID:  return "unsigned long long";
    case Type::LongTyID:   return "signed long long";
    case Type::FloatTyID:  return "float";
    case Type::DoubleTyID: return "double";
    default : assert(0 && "Unknown primitive type!");
    }
  
  // Check to see if the type is named.
  map<const Type *, string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end())
    return I->second;
  
  string Result;
  string MInfo = "";
  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *MTy = cast<const FunctionType>(Ty);
    Result = calcTypeName(MTy->getReturnType(), TypeNames, MInfo);
    if (MInfo != "")
      Result += ") " + MInfo;
    Result += "(";
    FunctionInfo += " (";
    for (FunctionType::ParamTypes::const_iterator
           I = MTy->getParamTypes().begin(),
           E = MTy->getParamTypes().end(); I != E; ++I) {
      if (I != MTy->getParamTypes().begin())
        FunctionInfo += ", ";
      MInfo = "";
      FunctionInfo += calcTypeName(*I, TypeNames, MInfo);
      if (MInfo != "")
	Result += ") " + MInfo;
    }
    if (MTy->isVarArg()) {
      if (!MTy->getParamTypes().empty()) 
	FunctionInfo += ", ";
      FunctionInfo += "...";
    }
    FunctionInfo += ")";
    break;
  }
  case Type::StructTyID: {
    string tempstr = "";
    const StructType *STy = cast<const StructType>(Ty);
    Result = " struct {\n ";
    int indx = 0;
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I) {
      Result += calcTypeNameVar(*I, TypeNames, 
				"field" + itostr(indx++), tempstr);
      Result += ";\n ";
    }
    Result += " } ";
    break;
  }
  case Type::PointerTyID:
    Result = calcTypeName(cast<const PointerType>(Ty)->getElementType(), 
                          TypeNames, MInfo);
    Result += "*";
    break;
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<const ArrayType>(Ty);
    int NumElements = ATy->getNumElements();
    Result = calcTypeName(ATy->getElementType(), TypeNames, MInfo);
    Result += "*";
    break;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    Result = "<error>";
  }

  return Result;
}

// Internal function
// Pass the Type* variable and and the variable name and this prints out the 
// variable declaration.
// This is different from calcTypeName because if you need to declare an array
// the size of the array would appear after the variable name itself
// For eg. int a[10];
static string calcTypeNameVar(const Type *Ty,
			      map<const Type *, string> &TypeNames, 
			      string VariableName, string NameSoFar) {
  if (Ty->isPrimitiveType())
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID: 
      return "bool " + NameSoFar + VariableName;
    case Type::UByteTyID: 
      return "unsigned char " + NameSoFar + VariableName;
    case Type::SByteTyID:
      return "signed char " + NameSoFar + VariableName;
    case Type::UShortTyID:
      return "unsigned long long " + NameSoFar + VariableName;
    case Type::ULongTyID:
      return "unsigned long long " + NameSoFar + VariableName;
    case Type::LongTyID:
      return "signed long long " + NameSoFar + VariableName;
    case Type::UIntTyID:
      return "unsigned " + NameSoFar + VariableName;
    default :
      return Ty->getDescription() + " " + NameSoFar + VariableName; 
    }
  
  // Check to see if the type is named.
  map<const Type *, string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end())
    return I->second + " " + NameSoFar + VariableName;
  
  string Result;
  string tempstr = "";

  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    string MInfo = "";
    const FunctionType *MTy = cast<const FunctionType>(Ty);
    Result += calcTypeName(MTy->getReturnType(), TypeNames, MInfo);
    if (MInfo != "")
      Result += ") " + MInfo;
    Result += " " + NameSoFar + VariableName;
    Result += " (";
    for (FunctionType::ParamTypes::const_iterator
           I = MTy->getParamTypes().begin(),
           E = MTy->getParamTypes().end(); I != E; ++I) {
      if (I != MTy->getParamTypes().begin())
        Result += ", ";
      MInfo = "";
      Result += calcTypeName(*I, TypeNames, MInfo);
      if (MInfo != "")
	Result += ") " + MInfo;
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
      Result += calcTypeNameVar(*I, TypeNames, 
				"field" + itostr(indx++), "");
      Result += ";\n ";
    }
    Result += " }";
    Result += " " + NameSoFar + VariableName;
    break;
  }  

  case Type::PointerTyID: {
    Result = calcTypeNameVar(cast<const PointerType>(Ty)->getElementType(), 
			     TypeNames, tempstr, 
			     "(*" + NameSoFar + VariableName + ")");
    break;
  }
  
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<const ArrayType>(Ty);
    int NumElements = ATy->getNumElements();
    Result = calcTypeNameVar(ATy->getElementType(),  TypeNames, 
			     tempstr, NameSoFar + VariableName + "[" + 
			     itostr(NumElements) + "]");
    break;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    Result = "<error>";
  }

  return Result;
}

// printTypeVarInt - The internal guts of printing out a type that has a
// potentially named portion and the variable associated with the type.
static ostream &printTypeVarInt(ostream &Out, const Type *Ty,
                             map<const Type *, string> &TypeNames,
			     const string &VariableName) {
  // Primitive types always print out their description, regardless of whether
  // they have been named or not.
  
  if (Ty->isPrimitiveType())
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID: 
      return Out << "bool " << VariableName;
    case Type::UByteTyID:
      return Out << "unsigned char " << VariableName;
    case Type::SByteTyID:
      return Out << "signed char " << VariableName;
    case Type::UShortTyID:
      return Out << "unsigned long long " << VariableName;
    case Type::ULongTyID:
      return Out << "unsigned long long " << VariableName;
    case Type::LongTyID:
      return Out << "signed long long " << VariableName;
    case Type::UIntTyID:
      return Out << "unsigned " << VariableName;
    default :
      return Out << Ty->getDescription() << " " << VariableName; 
    }
  
  // Check to see if the type is named.
  map<const Type *, string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return Out << I->second << " " << VariableName;
  
  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  //
  string TypeNameVar, tempstr = "";
  TypeNameVar = calcTypeNameVar(Ty, TypeNames, VariableName, tempstr);
  return Out << TypeNameVar;
}

// Internal guts of printing a type name
static ostream &printTypeInt(ostream &Out, const Type *Ty,
                             map<const Type *, string> &TypeNames) {
  // Primitive types always print out their description, regardless of whether
  // they have been named or not.
  
  if (Ty->isPrimitiveType())
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID:
      return Out << "bool";
    case Type::UByteTyID:
      return Out << "unsigned char";
    case Type::SByteTyID:
      return Out << "signed char";
    case Type::UShortTyID:
      return Out << "unsigned short";
    case Type::ULongTyID:
      return Out << "unsigned long long";
    case Type::LongTyID:
      return Out << "signed long long";
    case Type::UIntTyID:
      return Out << "unsigned";
    default :
      return Out << Ty->getDescription(); 
    }
  
  // Check to see if the type is named.
  map<const Type *, string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return Out << I->second;
  
  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  //
  string MInfo;
  string TypeName = calcTypeName(Ty, TypeNames, MInfo);
  // TypeNames.insert(std::make_pair(Ty, TypeName));
  //Cache type name for later use
  if (MInfo != "")
    return Out << TypeName << ")" << MInfo;
  else
    return Out << TypeName;
}

namespace {

  //Internal CWriter class mimics AssemblyWriter.
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
      return printTypeVarInt(Out, Ty, TypeNames, VariableName);
    }



    ostream& printType(const Type *Ty, ostream &Out);
    void writeOperand(const Value *Operand, ostream &Out,bool PrintName = true);

    string getValueName(const Value *V);
  private :

    void printModule(const Module *M);
    void printSymbolTable(const SymbolTable &ST);
    void printConstant(const Constant *CPV);
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
      if (I->getType() != Type::VoidTy) {
        string tempostr = CW.getValueName(I);
        CLV.addLocalVar(I->getType(), tempostr);
      }
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
    const Value *Operand;

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
        CW.writeOperand(pI->getIncomingValue(incindex), Out);
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
  Operand = I->getNumOperands() ? I->getOperand(0) : 0;
  Out << "(";
  CW.printType(I->getType(), Out);
  Out << ")";
  CW.writeOperand(Operand, Out);
  Out << ";\n";
}

void CInstPrintVisitor::visitCallInst(CallInst *I) {
  if (I->getType() != Type::VoidTy)
    outputLValue(I);
  else
    Out << "  ";

  Operand = I->getNumOperands() ? I->getOperand(0) : 0;
  const PointerType *PTy = dyn_cast<PointerType>(Operand->getType());
  const FunctionType  *MTy = PTy 
    ? dyn_cast<FunctionType>(PTy->getElementType()):0;
  const Type      *RetTy = MTy ? MTy->getReturnType() : 0;
  
  // If possible, print out the short form of the call instruction, but we can
  // only do this if the first argument is a pointer to a nonvararg method,
  // and if the value returned is not a pointer to a method.
  //
  if (RetTy && !MTy->isVarArg() &&
      (!isa<PointerType>(RetTy)||
       !isa<FunctionType>(cast<PointerType>(RetTy)))){
    Out << " ";
    Out << makeNameProper(Operand->getName());
  } else {
    Out << makeNameProper(Operand->getName());      
  }
  Out << "(";
  if (I->getNumOperands() > 1) 
    CW.writeOperand(I->getOperand(1), Out);
  for (unsigned op = 2, Eop = I->getNumOperands(); op < Eop; ++op) {
    Out << ",";
    CW.writeOperand(I->getOperand(op), Out);
  }
  
  Out << " );\n";
} 
 
// Specific Instruction type classes... note that all of the casts are
// neccesary because we use the instruction classes as opaque types...
//
void CInstPrintVisitor::visitReturnInst(ReturnInst *I) {
  Out << "  return ";
  if (I->getNumOperands())
    CW.writeOperand(I->getOperand(0), Out);
  Out << ";\n";
}

void CInstPrintVisitor::visitBranchInst(BranchInst *I) {
  TerminatorInst *tI = cast<TerminatorInst>(I);
  if (I->isConditional()) {
    Out << "  if (";
    CW.writeOperand(I->getCondition(), Out);
    Out << ") {\n";
    printPhiFromNextBlock(tI,0);
    Out << "    goto ";
    CW.writeOperand(I->getOperand(0), Out);
    Out << ";\n";
    Out << "  } else {\n";
    printPhiFromNextBlock(tI,1);
    Out << "    goto ";
    CW.writeOperand(I->getOperand(1), Out);
    Out << ";\n  }\n";
  } else {
    printPhiFromNextBlock(tI,0);
    Out << "  goto ";
    CW.writeOperand(I->getOperand(0), Out);
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
  CW.printType(I->getType()->getElementType(), Out);
  Out << "*)malloc(sizeof(";
  CW.printTypeVar(I->getType()->getElementType(), "");
  Out << ")";

  if (I->isArrayAllocation()) {
    Out << " * " ;
    CW.writeOperand(I->getOperand(0), Out);
  }
  Out << ");";
}

void CInstPrintVisitor::visitAllocaInst(AllocaInst *I) {
  outputLValue(I);
  Operand = I->getNumOperands() ? I->getOperand(0) : 0;
  string tempstr = "";
  Out << "(";
  CW.printTypeVar(I->getType(), tempstr);
  Out << ") alloca(sizeof(";
  CW.printTypeVar(cast<PointerType>(I->getType())->getElementType(), 
                  tempstr);
  Out << ")";
  if (I->getNumOperands()) {
    Out << " * " ;
    CW.writeOperand(Operand, Out);
  }
  Out << ");\n";
}

void CInstPrintVisitor::visitFreeInst(FreeInst   *I) {
  Operand = I->getNumOperands() ? I->getOperand(0) : 0;
  Out << "free(";
  CW.writeOperand(Operand, Out);
  Out << ");\n";
}

void CInstPrintVisitor::printIndexingExpr(MemAccessInst *MAI) {
  CW.writeOperand(MAI->getPointerOperand(), Out);

  for (MemAccessInst::op_iterator I = MAI->idx_begin(), E = MAI->idx_end();
       I != E; ++I)
    if ((*I)->getType() == Type::UIntTy) {
      Out << "[";
      CW.writeOperand(*I, Out);
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
  CW.writeOperand(I->getOperand(0), Out);
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
  CW.writeOperand(I->getOperand(0), Out);
  Out << ";\n";
}

void CInstPrintVisitor::visitBinaryOperator(Instruction *I) {
  // binary instructions, shift instructions, setCond instructions.
  outputLValue(I);
  if (isa<PointerType>(I->getType())) {
    Out << "(";
    CW.printType(I->getType(), Out);
    Out << ")";
  }
      
  if (isa<PointerType>(I->getType())) Out << "(long long)";
  CW.writeOperand(I->getOperand(0), Out);

  switch (I->getOpcode()) {
  case Instruction::Add: Out << "+"; break;
  case Instruction::Sub: Out << "-"; break;
  case Instruction::Mul: Out << "*"; break;
  case Instruction::Div: Out << "/"; break;
  case Instruction::Rem: Out << "%"; break;
  case Instruction::And: Out << "&"; break;
  case Instruction::Or: Out << "|"; break;
  case Instruction::Xor: Out << "^"; break;
  case Instruction::SetEQ: Out << "=="; break;
  case Instruction::SetNE: Out << "!="; break;
  case Instruction::SetLE: Out << "<="; break;
  case Instruction::SetGE: Out << ">="; break;
  case Instruction::SetLT: Out << "<"; break;
  case Instruction::SetGT: Out << ">"; break;
  case Instruction::Shl : Out << "<<"; break;
  case Instruction::Shr : Out << ">>"; break;
  default: cerr << "Invalid operator type!" << I; abort();
  }

  if (isa<PointerType>(I->getType())) Out << "(long long)";
  CW.writeOperand(I->getOperand(1), Out);
  Out << ";\n";
}

/* END : CInstPrintVisitor implementation */

string CWriter::getValueName(const Value *V) {
  if (V->hasName())              // Print out the label if it exists...
    return "l_" + makeNameProper(V->getName()) + "_" +
           utostr(V->getType()->getUniqueID());

  int Slot = Table.getValSlot(V);
  assert(Slot >= 0 && "Invalid value!");
  return "ltmp_" + itostr(Slot) + "_" +
         utostr(V->getType()->getUniqueID());
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
  string tempostr = getValueName(GV);
  if (GV->hasInternalLinkage()) Out << "static ";

  printTypeVar(GV->getType()->getElementType(), tempostr);

  if (GV->hasInitializer()) {
    Out << " = " ;
    writeOperand(GV->getInitializer(), Out, false);
  }

  Out << ";\n";
}

// printSymbolTable - Run through symbol table looking for named constants
// if a named constant is found, emit it's declaration...
// Assuming that symbol table has only types and constants.
void CWriter::printSymbolTable(const SymbolTable &ST) {
  // GraphT G;
  for (SymbolTable::const_iterator TI = ST.begin(); TI != ST.end(); ++TI) {
    SymbolTable::type_const_iterator I = ST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = ST.type_end(TI->first);
    
    // TODO
    // Need to run through all the used types in the program
    // FindUsedTypes &FUT = new FindUsedTypes();
    // const std::set<const Type *> &UsedTypes = FUT.getTypes();
    // Filter out the structures printing forward definitions for each of them
    // and creating the dependency graph.
    // Print forward definitions to all of them
    // print the typedefs topologically sorted

    // But for now we have
    for (; I != End; ++I) {
      const Value *V = I->second;
      if (const Constant *CPV = dyn_cast<const Constant>(V)) {
	printConstant(CPV);
      } else if (const Type *Ty = dyn_cast<const Type>(V)) {
	string tempostr;
	string tempstr = "";
	Out << "typedef ";
	tempostr = "llvm__" + I->first;
	string TypeNameVar = calcTypeNameVar(Ty, TypeNames, 
					     tempostr, tempstr);
	Out << TypeNameVar << ";\n";
	if (!isa<PointerType>(Ty) ||
	    !cast<PointerType>(Ty)->getElementType()->isPrimitiveType())
	  TypeNames.insert(std::make_pair(Ty, "llvm__"+I->first));
      }
    }
  }
}


// printConstant - Print out a constant pool entry...
//
void CWriter::printConstant(const Constant *CPV) {
  // TODO
  // Dinakar : Don't know what to do with unnamed constants
  // should do something about it later.

  string tempostr = getValueName(CPV);
  
  // Print out the constant type...
  printTypeVar(CPV->getType(), tempostr);
  
  Out << " = ";
  // Write the value out now...
  writeOperand(CPV, Out, false);
  
  Out << "\n";
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
  printType(F->getReturnType(), Out);
  Out << " " << makeNameProper(F->getName()) << "(";
    
  if (!F->isExternal()) {
    for_each(F->getArgumentList().begin(), F->getArgumentList().end(),
	     bind_obj(this, &CWriter::printFunctionArgument));
  } else {
    // Loop over the arguments, printing them...
    for (FunctionType::ParamTypes::const_iterator I = 
	   FT->getParamTypes().begin(),
	   E = FT->getParamTypes().end(); I != E; ++I) {
      if (I != FT->getParamTypes().begin()) Out << ", ";
      printType(*I, Out);
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

// printType - Go to extreme measures to attempt to print out a short, symbolic
// version of a type name.
ostream& CWriter::printType(const Type *Ty, ostream &Out) {
  return printTypeInt(Out, Ty, TypeNames);
}


void CWriter::writeOperand(const Value *Operand,
                           ostream &Out, bool PrintName = true) {
  if (isa<GlobalValue>(Operand))
    Out << "(&";  // Global values are references as their addresses by llvm

  if (PrintName && Operand->hasName()) {   
    // If Operand has a name.
    Out << "l_" << makeNameProper(Operand->getName()) << "_" << 
      Operand->getType()->getUniqueID();
  } else if (const Constant *CPV = dyn_cast<const Constant>(Operand)) {
    if (isa<ConstantPointerNull>(CPV))
      Out << "NULL";
    else
      Out << getConstStrValue(CPV); 
  } else {
    int Slot = Table.getValSlot(Operand);
    if (Slot >= 0)  
      Out << "ltmp_" << Slot << "_" << Operand->getType()->getUniqueID();
    else if (PrintName)
      Out << "<badref>";
  }

  if (isa<GlobalValue>(Operand))
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
