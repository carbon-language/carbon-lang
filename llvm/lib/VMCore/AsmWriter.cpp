//===-- AsmWriter.cpp - Printing LLVM as an assembly file -----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Assembly/Writer.h
//
// Note that these routines must be extremely tolerant of various errors in the
// LLVM code, because it can be used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/CachedWriter.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Assembly/AsmAnnotationWriter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Analysis/SlotCalculator.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"
#include <algorithm>
using namespace llvm;

static RegisterPass<PrintModulePass>
X("printm", "Print module to stderr",PassInfo::Analysis|PassInfo::Optimization);
static RegisterPass<PrintFunctionPass>
Y("print","Print function to stderr",PassInfo::Analysis|PassInfo::Optimization);

static void WriteAsOperandInternal(std::ostream &Out, const Value *V, 
                                   bool PrintName,
                                 std::map<const Type *, std::string> &TypeTable,
                                   SlotCalculator *Table);

static const Module *getModuleFromVal(const Value *V) {
  if (const Argument *MA = dyn_cast<Argument>(V))
    return MA->getParent() ? MA->getParent()->getParent() : 0;
  else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent() ? BB->getParent()->getParent() : 0;
  else if (const Instruction *I = dyn_cast<Instruction>(V)) {
    const Function *M = I->getParent() ? I->getParent()->getParent() : 0;
    return M ? M->getParent() : 0;
  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return GV->getParent();
  return 0;
}

static SlotCalculator *createSlotCalculator(const Value *V) {
  assert(!isa<Type>(V) && "Can't create an SC for a type!");
  if (const Argument *FA = dyn_cast<Argument>(V)) {
    return new SlotCalculator(FA->getParent(), false);
  } else if (const Instruction *I = dyn_cast<Instruction>(V)) {
    return new SlotCalculator(I->getParent()->getParent(), false);
  } else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    return new SlotCalculator(BB->getParent(), false);
  } else if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V)){
    return new SlotCalculator(GV->getParent(), false);
  } else if (const Function *Func = dyn_cast<Function>(V)) {
    return new SlotCalculator(Func, false);
  }
  return 0;
}

// getLLVMName - Turn the specified string into an 'LLVM name', which is either
// prefixed with % (if the string only contains simple characters) or is
// surrounded with ""'s (if it has special chars in it).
static std::string getLLVMName(const std::string &Name) {
  assert(!Name.empty() && "Cannot get empty name!");

  // First character cannot start with a number...
  if (Name[0] >= '0' && Name[0] <= '9')
    return "\"" + Name + "\"";

  // Scan to see if we have any characters that are not on the "white list"
  for (unsigned i = 0, e = Name.size(); i != e; ++i) {
    char C = Name[i];
    assert(C != '"' && "Illegal character in LLVM value name!");
    if ((C < 'a' || C > 'z') && (C < 'A' || C > 'Z') && (C < '0' || C > '9') &&
        C != '-' && C != '.' && C != '_')
      return "\"" + Name + "\"";
  }
  
  // If we get here, then the identifier is legal to use as a "VarID".
  return "%"+Name;
}


/// fillTypeNameTable - If the module has a symbol table, take all global types
/// and stuff their names into the TypeNames map.
///
static void fillTypeNameTable(const Module *M,
                              std::map<const Type *, std::string> &TypeNames) {
  if (!M) return;
  const SymbolTable &ST = M->getSymbolTable();
  SymbolTable::const_iterator PI = ST.find(Type::TypeTy);
  if (PI != ST.end()) {
    SymbolTable::type_const_iterator I = PI->second.begin();
    for (; I != PI->second.end(); ++I) {
      // As a heuristic, don't insert pointer to primitive types, because
      // they are used too often to have a single useful name.
      //
      const Type *Ty = cast<Type>(I->second);
      if (!isa<PointerType>(Ty) ||
          !cast<PointerType>(Ty)->getElementType()->isPrimitiveType() ||
          isa<OpaqueType>(cast<PointerType>(Ty)->getElementType()))
        TypeNames.insert(std::make_pair(Ty, getLLVMName(I->first)));
    }
  }
}



static std::string calcTypeName(const Type *Ty, 
                                std::vector<const Type *> &TypeStack,
                                std::map<const Type *, std::string> &TypeNames){
  if (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty))
    return Ty->getDescription();  // Base case

  // Check to see if the type is named.
  std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return I->second;

  if (isa<OpaqueType>(Ty))
    return "opaque";

  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type

  // This is another base case for the recursion.  In this case, we know 
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  if (Slot < CurSize)
    return "\\" + utostr(CurSize-Slot);       // Here's the upreference

  TypeStack.push_back(Ty);    // Recursive case: Add us to the stack..
  
  std::string Result;
  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    Result = calcTypeName(FTy->getReturnType(), TypeStack, TypeNames) + " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Result += ", ";
      Result += calcTypeName(*I, TypeStack, TypeNames);
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) Result += ", ";
      Result += "...";
    }
    Result += ")";
    break;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    Result = "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        Result += ", ";
      Result += calcTypeName(*I, TypeStack, TypeNames);
    }
    Result += " }";
    break;
  }
  case Type::PointerTyID:
    Result = calcTypeName(cast<PointerType>(Ty)->getElementType(), 
                          TypeStack, TypeNames) + "*";
    break;
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    Result = "[" + utostr(ATy->getNumElements()) + " x ";
    Result += calcTypeName(ATy->getElementType(), TypeStack, TypeNames) + "]";
    break;
  }
  case Type::OpaqueTyID:
    Result = "opaque";
    break;
  default:
    Result = "<unrecognized-type>";
  }

  TypeStack.pop_back();       // Remove self from stack...
  return Result;
}


/// printTypeInt - The internal guts of printing out a type that has a
/// potentially named portion.
///
static std::ostream &printTypeInt(std::ostream &Out, const Type *Ty,
                              std::map<const Type *, std::string> &TypeNames) {
  // Primitive types always print out their description, regardless of whether
  // they have been named or not.
  //
  if (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty))
    return Out << Ty->getDescription();

  // Check to see if the type is named.
  std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return Out << I->second;

  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  //
  std::vector<const Type *> TypeStack;
  std::string TypeName = calcTypeName(Ty, TypeStack, TypeNames);
  TypeNames.insert(std::make_pair(Ty, TypeName));//Cache type name for later use
  return Out << TypeName;
}


/// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
/// type, iff there is an entry in the modules symbol table for the specified
/// type or one of it's component types. This is slower than a simple x << Type
///
std::ostream &llvm::WriteTypeSymbolic(std::ostream &Out, const Type *Ty,
                                      const Module *M) {
  Out << " "; 

  // If they want us to print out a type, attempt to make it symbolic if there
  // is a symbol table in the module...
  if (M) {
    std::map<const Type *, std::string> TypeNames;
    fillTypeNameTable(M, TypeNames);
    
    return printTypeInt(Out, Ty, TypeNames);
  } else {
    return Out << Ty->getDescription();
  }
}

static void WriteConstantInt(std::ostream &Out, const Constant *CV, 
                             bool PrintName,
                             std::map<const Type *, std::string> &TypeTable,
                             SlotCalculator *Table) {
  if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    Out << (CB == ConstantBool::True ? "true" : "false");
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV)) {
    Out << CI->getValue();
  } else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV)) {
    Out << CI->getValue();
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    // We would like to output the FP constant value in exponential notation,
    // but we cannot do this if doing so will lose precision.  Check here to
    // make sure that we only output it in exponential format if we can parse
    // the value back and get the same value.
    //
    std::string StrVal = ftostr(CFP->getValue());

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check that
    // the string matches the "[-+]?[0-9]" regex.
    //
    if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
        ((StrVal[0] == '-' || StrVal[0] == '+') &&
         (StrVal[1] >= '0' && StrVal[1] <= '9')))
      // Reparse stringized version!
      if (atof(StrVal.c_str()) == CFP->getValue()) {
        Out << StrVal; return;
      }
    
    // Otherwise we could not reparse it to exactly the same value, so we must
    // output the string in hexadecimal format!
    //
    // Behave nicely in the face of C TBAA rules... see:
    // http://www.nullstone.com/htmls/category/aliastyp.htm
    //
    double Val = CFP->getValue();
    char *Ptr = (char*)&Val;
    assert(sizeof(double) == sizeof(uint64_t) && sizeof(double) == 8 &&
           "assuming that double is 64 bits!");
    Out << "0x" << utohexstr(*(uint64_t*)Ptr);

  } else if (isa<ConstantAggregateZero>(CV)) {
    Out << "zeroinitializer";
  } else if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
    // As a special case, print the array as a string if it is an array of
    // ubytes or an array of sbytes with positive values.
    // 
    const Type *ETy = CA->getType()->getElementType();
    bool isString = (ETy == Type::SByteTy || ETy == Type::UByteTy);

    if (ETy == Type::SByteTy)
      for (unsigned i = 0; i < CA->getNumOperands(); ++i)
        if (cast<ConstantSInt>(CA->getOperand(i))->getValue() < 0) {
          isString = false;
          break;
        }

    if (isString) {
      Out << "c\"";
      for (unsigned i = 0; i < CA->getNumOperands(); ++i) {
        unsigned char C = cast<ConstantInt>(CA->getOperand(i))->getRawValue();
        
        if (isprint(C) && C != '"' && C != '\\') {
          Out << C;
        } else {
          Out << '\\'
              << (char) ((C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A'))
              << (char)(((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A'));
        }
      }
      Out << "\"";

    } else {                // Cannot output in string format...
      Out << "[";
      if (CA->getNumOperands()) {
        Out << " ";
        printTypeInt(Out, ETy, TypeTable);
        WriteAsOperandInternal(Out, CA->getOperand(0),
                               PrintName, TypeTable, Table);
        for (unsigned i = 1, e = CA->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printTypeInt(Out, ETy, TypeTable);
          WriteAsOperandInternal(Out, CA->getOperand(i), PrintName,
                                 TypeTable, Table);
        }
      }
      Out << " ]";
    }
  } else if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(CV)) {
    Out << "{";
    if (CS->getNumOperands()) {
      Out << " ";
      printTypeInt(Out, CS->getOperand(0)->getType(), TypeTable);

      WriteAsOperandInternal(Out, CS->getOperand(0),
                             PrintName, TypeTable, Table);

      for (unsigned i = 1; i < CS->getNumOperands(); i++) {
        Out << ", ";
        printTypeInt(Out, CS->getOperand(i)->getType(), TypeTable);

        WriteAsOperandInternal(Out, CS->getOperand(i),
                               PrintName, TypeTable, Table);
      }
    }

    Out << " }";
  } else if (isa<ConstantPointerNull>(CV)) {
    Out << "null";

  } else if (const ConstantPointerRef *PR = dyn_cast<ConstantPointerRef>(CV)) {
    WriteAsOperandInternal(Out, PR->getValue(), true, TypeTable, Table);

  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    Out << CE->getOpcodeName() << " (";
    
    for (User::const_op_iterator OI=CE->op_begin(); OI != CE->op_end(); ++OI) {
      printTypeInt(Out, (*OI)->getType(), TypeTable);
      WriteAsOperandInternal(Out, *OI, PrintName, TypeTable, Table);
      if (OI+1 != CE->op_end())
        Out << ", ";
    }
    
    if (CE->getOpcode() == Instruction::Cast) {
      Out << " to ";
      printTypeInt(Out, CE->getType(), TypeTable);
    }
    Out << ")";

  } else {
    Out << "<placeholder or erroneous Constant>";
  }
}


/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
static void WriteAsOperandInternal(std::ostream &Out, const Value *V, 
                                   bool PrintName,
                                  std::map<const Type*, std::string> &TypeTable,
                                   SlotCalculator *Table) {
  Out << " ";
  if (PrintName && V->hasName()) {
    Out << getLLVMName(V->getName());
  } else {
    if (const Constant *CV = dyn_cast<Constant>(V)) {
      WriteConstantInt(Out, CV, PrintName, TypeTable, Table);
    } else {
      int Slot;
      if (Table) {
	Slot = Table->getSlot(V);
      } else {
        if (const Type *Ty = dyn_cast<Type>(V)) {
          Out << Ty->getDescription();
          return;
        }

        Table = createSlotCalculator(V);
        if (Table == 0) { Out << "BAD VALUE TYPE!"; return; }

	Slot = Table->getSlot(V);
	delete Table;
      }
      if (Slot >= 0)  Out << "%" << Slot;
      else if (PrintName)
        if (V->hasName())
          Out << "<badref: " << getLLVMName(V->getName()) << ">";
        else
          Out << "<badref>";     // Not embedded into a location?
    }
  }
}


/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
std::ostream &llvm::WriteAsOperand(std::ostream &Out, const Value *V,
                                   bool PrintType, bool PrintName, 
                                   const Module *Context) {
  std::map<const Type *, std::string> TypeNames;
  if (Context == 0) Context = getModuleFromVal(V);

  if (Context)
    fillTypeNameTable(Context, TypeNames);

  if (PrintType)
    printTypeInt(Out, V->getType(), TypeNames);
  
  if (const Type *Ty = dyn_cast<Type> (V))
    printTypeInt(Out, Ty, TypeNames);

  WriteAsOperandInternal(Out, V, PrintName, TypeNames, 0);
  return Out;
}

namespace llvm {

class AssemblyWriter {
  std::ostream *Out;
  SlotCalculator &Table;
  const Module *TheModule;
  std::map<const Type *, std::string> TypeNames;
  AssemblyAnnotationWriter *AnnotationWriter;
public:
  inline AssemblyWriter(std::ostream &o, SlotCalculator &Tab, const Module *M,
                        AssemblyAnnotationWriter *AAW)
    : Out(&o), Table(Tab), TheModule(M), AnnotationWriter(AAW) {

    // If the module has a symbol table, take all global types and stuff their
    // names into the TypeNames map.
    //
    fillTypeNameTable(M, TypeNames);
  }

  inline void write(const Module *M)         { printModule(M);      }
  inline void write(const GlobalVariable *G) { printGlobal(G);      }
  inline void write(const Function *F)       { printFunction(F);    }
  inline void write(const BasicBlock *BB)    { printBasicBlock(BB); }
  inline void write(const Instruction *I)    { printInstruction(*I); }
  inline void write(const Constant *CPV)     { printConstant(CPV);  }
  inline void write(const Type *Ty)          { printType(Ty);       }

  void writeOperand(const Value *Op, bool PrintType, bool PrintName = true);

  const Module* getModule() { return TheModule; }
  void setStream(std::ostream &os) { Out = &os; }

private :
  void printModule(const Module *M);
  void printSymbolTable(const SymbolTable &ST);
  void printConstant(const Constant *CPV);
  void printGlobal(const GlobalVariable *GV);
  void printFunction(const Function *F);
  void printArgument(const Argument *FA);
  void printBasicBlock(const BasicBlock *BB);
  void printInstruction(const Instruction &I);

  // printType - Go to extreme measures to attempt to print out a short,
  // symbolic version of a type name.
  //
  std::ostream &printType(const Type *Ty) {
    return printTypeInt(*Out, Ty, TypeNames);
  }

  // printTypeAtLeastOneLevel - Print out one level of the possibly complex type
  // without considering any symbolic types that we may have equal to it.
  //
  std::ostream &printTypeAtLeastOneLevel(const Type *Ty);

  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value &V);
};
}  // end of anonymous namespace

/// printTypeAtLeastOneLevel - Print out one level of the possibly complex type
/// without considering any symbolic types that we may have equal to it.
///
std::ostream &AssemblyWriter::printTypeAtLeastOneLevel(const Type *Ty) {
  if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    printType(FTy->getReturnType()) << " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        *Out << ", ";
      printType(*I);
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) *Out << ", ";
      *Out << "...";
    }
    *Out << ")";
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    *Out << "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        *Out << ", ";
      printType(*I);
    }
    *Out << " }";
  } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    printType(PTy->getElementType()) << "*";
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    *Out << "[" << ATy->getNumElements() << " x ";
    printType(ATy->getElementType()) << "]";
  } else if (const OpaqueType *OTy = dyn_cast<OpaqueType>(Ty)) {
    *Out << "opaque";
  } else {
    if (!Ty->isPrimitiveType())
      *Out << "<unknown derived type>";
    printType(Ty);
  }
  return *Out;
}


void AssemblyWriter::writeOperand(const Value *Operand, bool PrintType, 
				  bool PrintName) {
  if (PrintType) { *Out << " "; printType(Operand->getType()); }
  WriteAsOperandInternal(*Out, Operand, PrintName, TypeNames, &Table);
}


void AssemblyWriter::printModule(const Module *M) {
  switch (M->getEndianness()) {
  case Module::LittleEndian: *Out << "target endian = little\n"; break;
  case Module::BigEndian:    *Out << "target endian = big\n";    break;
  case Module::AnyEndianness: break;
  }
  switch (M->getPointerSize()) {
  case Module::Pointer32:    *Out << "target pointersize = 32\n"; break;
  case Module::Pointer64:    *Out << "target pointersize = 64\n"; break;
  case Module::AnyPointerSize: break;
  }
  
  // Loop over the symbol table, emitting all named constants...
  printSymbolTable(M->getSymbolTable());
  
  for (Module::const_giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    printGlobal(I);

  *Out << "\nimplementation   ; Functions:\n";
  
  // Output all of the functions...
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    printFunction(I);
}

void AssemblyWriter::printGlobal(const GlobalVariable *GV) {
  if (GV->hasName()) *Out << getLLVMName(GV->getName()) << " = ";

  if (!GV->hasInitializer()) 
    *Out << "external ";
  else
    switch (GV->getLinkage()) {
    case GlobalValue::InternalLinkage:  *Out << "internal "; break;
    case GlobalValue::LinkOnceLinkage:  *Out << "linkonce "; break;
    case GlobalValue::WeakLinkage:      *Out << "weak "; break;
    case GlobalValue::AppendingLinkage: *Out << "appending "; break;
    case GlobalValue::ExternalLinkage: break;
    }

  *Out << (GV->isConstant() ? "constant " : "global ");
  printType(GV->getType()->getElementType());

  if (GV->hasInitializer())
    writeOperand(GV->getInitializer(), false, false);

  printInfoComment(*GV);
  *Out << "\n";
}


/// printSymbolTable - Run through symbol table looking for named constants
/// if a named constant is found, emit it's declaration...
///
void AssemblyWriter::printSymbolTable(const SymbolTable &ST) {
  for (SymbolTable::const_iterator TI = ST.begin(); TI != ST.end(); ++TI) {
    SymbolTable::type_const_iterator I = ST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = ST.type_end(TI->first);
    
    for (; I != End; ++I) {
      const Value *V = I->second;
      if (const Constant *CPV = dyn_cast<Constant>(V)) {
	printConstant(CPV);
      } else if (const Type *Ty = dyn_cast<Type>(V)) {
        assert(Ty->getType() == Type::TypeTy && TI->first == Type::TypeTy);
	*Out << "\t" << getLLVMName(I->first) << " = type ";

        // Make sure we print out at least one level of the type structure, so
        // that we do not get %FILE = type %FILE
        //
        printTypeAtLeastOneLevel(Ty) << "\n";
      }
    }
  }
}


/// printConstant - Print out a constant pool entry...
///
void AssemblyWriter::printConstant(const Constant *CPV) {
  // Don't print out unnamed constants, they will be inlined
  if (!CPV->hasName()) return;

  // Print out name...
  *Out << "\t" << getLLVMName(CPV->getName()) << " =";

  // Write the value out now...
  writeOperand(CPV, true, false);

  printInfoComment(*CPV);
  *Out << "\n";
}

/// printFunction - Print all aspects of a function.
///
void AssemblyWriter::printFunction(const Function *F) {
  // Print out the return type and name...
  *Out << "\n";

  if (AnnotationWriter) AnnotationWriter->emitFunctionAnnot(F, *Out);

  if (F->isExternal())
    *Out << "declare ";
  else
    switch (F->getLinkage()) {
    case GlobalValue::InternalLinkage:  *Out << "internal "; break;
    case GlobalValue::LinkOnceLinkage:  *Out << "linkonce "; break;
    case GlobalValue::WeakLinkage:      *Out << "weak "; break;
    case GlobalValue::AppendingLinkage: *Out << "appending "; break;
    case GlobalValue::ExternalLinkage: break;
    }

  printType(F->getReturnType()) << " ";
  if (!F->getName().empty())
    *Out << getLLVMName(F->getName());
  else
    *Out << "\"\"";
  *Out << "(";
  Table.incorporateFunction(F);

  // Loop over the arguments, printing them...
  const FunctionType *FT = F->getFunctionType();

  for(Function::const_aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    printArgument(I);

  // Finish printing arguments...
  if (FT->isVarArg()) {
    if (FT->getNumParams()) *Out << ", ";
    *Out << "...";  // Output varargs portion of signature!
  }
  *Out << ")";

  if (F->isExternal()) {
    *Out << "\n";
  } else {
    *Out << " {";
  
    // Output all of its basic blocks... for the function
    for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
      printBasicBlock(I);

    *Out << "}\n";
  }

  Table.purgeFunction();
}

/// printArgument - This member is called for every argument that is passed into
/// the function.  Simply print it out
///
void AssemblyWriter::printArgument(const Argument *Arg) {
  // Insert commas as we go... the first arg doesn't get a comma
  if (Arg != &Arg->getParent()->afront()) *Out << ", ";

  // Output type...
  printType(Arg->getType());
  
  // Output name, if available...
  if (Arg->hasName())
    *Out << " " << getLLVMName(Arg->getName());
  else if (Table.getSlot(Arg) < 0)
    *Out << "<badref>";
}

/// printBasicBlock - This member is called for each basic block in a method.
///
void AssemblyWriter::printBasicBlock(const BasicBlock *BB) {
  if (BB->hasName()) {              // Print out the label if it exists...
    *Out << "\n" << BB->getName() << ":";
  } else if (!BB->use_empty()) {      // Don't print block # of no uses...
    int Slot = Table.getSlot(BB);
    *Out << "\n; <label>:";
    if (Slot >= 0) 
      *Out << Slot;         // Extra newline separates out label's
    else 
      *Out << "<badref>"; 
  }

  if (BB->getParent() == 0)
    *Out << "\t\t; Error: Block without parent!";
  else {
    if (BB != &BB->getParent()->front()) {  // Not the entry block?
      // Output predecessors for the block...
      *Out << "\t\t;";
      pred_const_iterator PI = pred_begin(BB), PE = pred_end(BB);
      
      if (PI == PE) {
        *Out << " No predecessors!";
      } else {
        *Out << " preds =";
        writeOperand(*PI, false, true);
        for (++PI; PI != PE; ++PI) {
          *Out << ",";
          writeOperand(*PI, false, true);
        }
      }
    }
  }
  
  *Out << "\n";

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockStartAnnot(BB, *Out);

  // Output all of the instructions in the basic block...
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    printInstruction(*I);

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockEndAnnot(BB, *Out);
}


/// printInfoComment - Print a little comment after the instruction indicating
/// which slot it occupies.
///
void AssemblyWriter::printInfoComment(const Value &V) {
  if (V.getType() != Type::VoidTy) {
    *Out << "\t\t; <";
    printType(V.getType()) << ">";

    if (!V.hasName()) {
      int Slot = Table.getSlot(&V); // Print out the def slot taken...
      if (Slot >= 0) *Out << ":" << Slot;
      else *Out << ":<badref>";
    }
    *Out << " [#uses=" << V.use_size() << "]";  // Output # uses
  }
}

/// printInstruction - This member is called for each Instruction in a method.
///
void AssemblyWriter::printInstruction(const Instruction &I) {
  if (AnnotationWriter) AnnotationWriter->emitInstructionAnnot(&I, *Out);

  *Out << "\t";

  // Print out name if it exists...
  if (I.hasName())
    *Out << getLLVMName(I.getName()) << " = ";

  // If this is a volatile load or store, print out the volatile marker
  if ((isa<LoadInst>(I)  && cast<LoadInst>(I).isVolatile()) ||
      (isa<StoreInst>(I) && cast<StoreInst>(I).isVolatile()))
      *Out << "volatile ";

  // Print out the opcode...
  *Out << I.getOpcodeName();

  // Print out the type of the operands...
  const Value *Operand = I.getNumOperands() ? I.getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (isa<BranchInst>(I) && I.getNumOperands() > 1) {
    writeOperand(I.getOperand(2), true);
    *Out << ",";
    writeOperand(Operand, true);
    *Out << ",";
    writeOperand(I.getOperand(1), true);

  } else if (isa<SwitchInst>(I)) {
    // Special case switch statement to get formatting nice and correct...
    writeOperand(Operand        , true); *Out << ",";
    writeOperand(I.getOperand(1), true); *Out << " [";

    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; op += 2) {
      *Out << "\n\t\t";
      writeOperand(I.getOperand(op  ), true); *Out << ",";
      writeOperand(I.getOperand(op+1), true);
    }
    *Out << "\n\t]";
  } else if (isa<PHINode>(I)) {
    *Out << " ";
    printType(I.getType());
    *Out << " ";

    for (unsigned op = 0, Eop = I.getNumOperands(); op < Eop; op += 2) {
      if (op) *Out << ", ";
      *Out << "[";  
      writeOperand(I.getOperand(op  ), false); *Out << ",";
      writeOperand(I.getOperand(op+1), false); *Out << " ]";
    }
  } else if (isa<ReturnInst>(I) && !Operand) {
    *Out << " void";
  } else if (isa<CallInst>(I)) {
    const PointerType  *PTy = cast<PointerType>(Operand->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
    const Type       *RetTy = FTy->getReturnType();

    // If possible, print out the short form of the call instruction.  We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    if (!FTy->isVarArg() &&
        (!isa<PointerType>(RetTy) || 
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      *Out << " "; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    *Out << "(";
    if (I.getNumOperands() > 1) writeOperand(I.getOperand(1), true);
    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; ++op) {
      *Out << ",";
      writeOperand(I.getOperand(op), true);
    }

    *Out << " )";
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    const PointerType  *PTy = cast<PointerType>(Operand->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
    const Type       *RetTy = FTy->getReturnType();

    // If possible, print out the short form of the invoke instruction. We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    if (!FTy->isVarArg() &&
        (!isa<PointerType>(RetTy) || 
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      *Out << " "; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }

    *Out << "(";
    if (I.getNumOperands() > 3) writeOperand(I.getOperand(3), true);
    for (unsigned op = 4, Eop = I.getNumOperands(); op < Eop; ++op) {
      *Out << ",";
      writeOperand(I.getOperand(op), true);
    }

    *Out << " )\n\t\t\tto";
    writeOperand(II->getNormalDest(), true);
    *Out << " unwind";
    writeOperand(II->getUnwindDest(), true);

  } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(&I)) {
    *Out << " ";
    printType(AI->getType()->getElementType());
    if (AI->isArrayAllocation()) {
      *Out << ",";
      writeOperand(AI->getArraySize(), true);
    }
  } else if (isa<CastInst>(I)) {
    if (Operand) writeOperand(Operand, true);   // Work with broken code
    *Out << " to ";
    printType(I.getType());
  } else if (isa<VAArgInst>(I)) {
    if (Operand) writeOperand(Operand, true);   // Work with broken code
    *Out << ", ";
    printType(I.getType());
  } else if (const VANextInst *VAN = dyn_cast<VANextInst>(&I)) {
    if (Operand) writeOperand(Operand, true);   // Work with broken code
    *Out << ", ";
    printType(VAN->getArgType());
  } else if (Operand) {   // Print the normal way...

    // PrintAllTypes - Instructions who have operands of all the same type 
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    // Shift Left & Right print both types even for Ubyte LHS, and select prints
    // types even if all operands are bools.
    if (isa<ShiftInst>(I) || isa<SelectInst>(I)) {
      PrintAllTypes = true;
    } else {
      for (unsigned i = 1, E = I.getNumOperands(); i != E; ++i) {
        Operand = I.getOperand(i);
        if (Operand->getType() != TheType) {
          PrintAllTypes = true;    // We have differing types!  Print them all!
          break;
        }
      }
    }
    
    if (!PrintAllTypes) {
      *Out << " ";
      printType(TheType);
    }

    for (unsigned i = 0, E = I.getNumOperands(); i != E; ++i) {
      if (i) *Out << ",";
      writeOperand(I.getOperand(i), PrintAllTypes);
    }
  }

  printInfoComment(I);
  *Out << "\n";
}


//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//

void Module::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  SlotCalculator SlotTable(this, false);
  AssemblyWriter W(o, SlotTable, this, AAW);
  W.write(this);
}

void GlobalVariable::print(std::ostream &o) const {
  SlotCalculator SlotTable(getParent(), false);
  AssemblyWriter W(o, SlotTable, getParent(), 0);
  W.write(this);
}

void Function::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  SlotCalculator SlotTable(getParent(), false);
  AssemblyWriter W(o, SlotTable, getParent(), AAW);

  W.write(this);
}

void BasicBlock::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  SlotCalculator SlotTable(getParent(), false);
  AssemblyWriter W(o, SlotTable, 
                   getParent() ? getParent()->getParent() : 0, AAW);
  W.write(this);
}

void Instruction::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  const Function *F = getParent() ? getParent()->getParent() : 0;
  SlotCalculator SlotTable(F, false);
  AssemblyWriter W(o, SlotTable, F ? F->getParent() : 0, AAW);

  W.write(this);
}

void Constant::print(std::ostream &o) const {
  if (this == 0) { o << "<null> constant value\n"; return; }

  // Handle CPR's special, because they have context information...
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(this)) {
    CPR->getValue()->print(o);  // Print as a global value, with context info.
    return;
  }

  o << " " << getType()->getDescription() << " ";

  std::map<const Type *, std::string> TypeTable;
  WriteConstantInt(o, this, false, TypeTable, 0);
}

void Type::print(std::ostream &o) const { 
  if (this == 0)
    o << "<null Type>";
  else
    o << getDescription();
}

void Argument::print(std::ostream &o) const {
  o << getType() << " " << getName();
}

void Value::dump() const { print(std::cerr); }

//===----------------------------------------------------------------------===//
//  CachedWriter Class Implementation
//===----------------------------------------------------------------------===//

void CachedWriter::setModule(const Module *M) {
  delete SC; delete AW;
  if (M) {
    SC = new SlotCalculator(M, false);
    AW = new AssemblyWriter(*Out, *SC, M, 0);
  } else {
    SC = 0; AW = 0;
  }
}

CachedWriter::~CachedWriter() {
  delete AW;
  delete SC;
}

CachedWriter &CachedWriter::operator<<(const Value *V) {
  assert(AW && SC && "CachedWriter does not have a current module!");
  switch (V->getValueType()) {
  case Value::ConstantVal:
  case Value::ArgumentVal:       AW->writeOperand(V, true, true); break;
  case Value::TypeVal:           AW->write(cast<Type>(V)); break;
  case Value::InstructionVal:    AW->write(cast<Instruction>(V)); break;
  case Value::BasicBlockVal:     AW->write(cast<BasicBlock>(V)); break;
  case Value::FunctionVal:       AW->write(cast<Function>(V)); break;
  case Value::GlobalVariableVal: AW->write(cast<GlobalVariable>(V)); break;
  default: *Out << "<unknown value type: " << V->getValueType() << ">"; break;
  }
  return *this;
}

CachedWriter& CachedWriter::operator<<(const Type *X) {
  if (SymbolicTypes) {
    const Module *M = AW->getModule();
    if (M) WriteTypeSymbolic(*Out, X, M);
    return *this;
  } else
    return *this << (const Value*)X;
}

void CachedWriter::setStream(std::ostream &os) {
  Out = &os;
  if (AW) AW->setStream(os);
}
