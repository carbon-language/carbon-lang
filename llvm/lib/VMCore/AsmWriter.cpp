//===-- AsmWriter.cpp - Printing LLVM as an assembly file -----------------===//
//
// This library implements the functionality defined in llvm/Assembly/Writer.h
//
// Note that these routines must be extremely tolerant of various errors in the
// LLVM code, because of of the primary uses of it is for debugging
// transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/CachedWriter.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/SlotCalculator.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/SymbolTable.h"
#include "llvm/Support/CFG.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"
#include <algorithm>
using std::string;
using std::map;
using std::vector;
using std::ostream;

static RegisterPass<PrintModulePass>
X("printm", "Print module to stderr",PassInfo::Analysis|PassInfo::Optimization);
static RegisterPass<PrintFunctionPass>
Y("print","Print function to stderr",PassInfo::Analysis|PassInfo::Optimization);

static void WriteAsOperandInternal(ostream &Out, const Value *V, bool PrintName,
                                   map<const Type *, string> &TypeTable,
                                   SlotCalculator *Table);

static const Module *getModuleFromVal(const Value *V) {
  if (const Argument *MA = dyn_cast<const Argument>(V))
    return MA->getParent() ? MA->getParent()->getParent() : 0;
  else if (const BasicBlock *BB = dyn_cast<const BasicBlock>(V))
    return BB->getParent() ? BB->getParent()->getParent() : 0;
  else if (const Instruction *I = dyn_cast<const Instruction>(V)) {
    const Function *M = I->getParent() ? I->getParent()->getParent() : 0;
    return M ? M->getParent() : 0;
  } else if (const GlobalValue *GV = dyn_cast<const GlobalValue>(V))
    return GV->getParent();
  return 0;
}

static SlotCalculator *createSlotCalculator(const Value *V) {
  assert(!isa<Type>(V) && "Can't create an SC for a type!");
  if (const Argument *FA = dyn_cast<const Argument>(V)) {
    return new SlotCalculator(FA->getParent(), true);
  } else if (const Instruction *I = dyn_cast<const Instruction>(V)) {
    return new SlotCalculator(I->getParent()->getParent(), true);
  } else if (const BasicBlock *BB = dyn_cast<const BasicBlock>(V)) {
    return new SlotCalculator(BB->getParent(), true);
  } else if (const GlobalVariable *GV = dyn_cast<const GlobalVariable>(V)){
    return new SlotCalculator(GV->getParent(), true);
  } else if (const Function *Func = dyn_cast<const Function>(V)) {
    return new SlotCalculator(Func, true);
  }
  return 0;
}


// If the module has a symbol table, take all global types and stuff their
// names into the TypeNames map.
//
static void fillTypeNameTable(const Module *M,
                              map<const Type *, string> &TypeNames) {
  if (!M) return;
  const SymbolTable &ST = M->getSymbolTable();
  SymbolTable::const_iterator PI = ST.find(Type::TypeTy);
  if (PI != ST.end()) {
    SymbolTable::type_const_iterator I = PI->second.begin();
    for (; I != PI->second.end(); ++I) {
      // As a heuristic, don't insert pointer to primitive types, because
      // they are used too often to have a single useful name.
      //
      const Type *Ty = cast<const Type>(I->second);
      if (!isa<PointerType>(Ty) ||
          !cast<PointerType>(Ty)->getElementType()->isPrimitiveType())
        TypeNames.insert(std::make_pair(Ty, "%"+I->first));
    }
  }
}



static string calcTypeName(const Type *Ty, vector<const Type *> &TypeStack,
                           map<const Type *, string> &TypeNames) {
  if (Ty->isPrimitiveType()) return Ty->getDescription();  // Base case

  // Check to see if the type is named.
  map<const Type *, string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return I->second;

  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type

  // This is another base case for the recursion.  In this case, we know 
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  // 
  if (Slot < CurSize)
    return "\\" + utostr(CurSize-Slot);       // Here's the upreference

  TypeStack.push_back(Ty);    // Recursive case: Add us to the stack..
  
  string Result;
  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<const FunctionType>(Ty);
    Result = calcTypeName(FTy->getReturnType(), TypeStack, TypeNames) + " (";
    for (FunctionType::ParamTypes::const_iterator
           I = FTy->getParamTypes().begin(),
           E = FTy->getParamTypes().end(); I != E; ++I) {
      if (I != FTy->getParamTypes().begin())
        Result += ", ";
      Result += calcTypeName(*I, TypeStack, TypeNames);
    }
    if (FTy->isVarArg()) {
      if (!FTy->getParamTypes().empty()) Result += ", ";
      Result += "...";
    }
    Result += ")";
    break;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<const StructType>(Ty);
    Result = "{ ";
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I) {
      if (I != STy->getElementTypes().begin())
        Result += ", ";
      Result += calcTypeName(*I, TypeStack, TypeNames);
    }
    Result += " }";
    break;
  }
  case Type::PointerTyID:
    Result = calcTypeName(cast<const PointerType>(Ty)->getElementType(), 
                          TypeStack, TypeNames) + "*";
    break;
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<const ArrayType>(Ty);
    Result = "[" + utostr(ATy->getNumElements()) + " x ";
    Result += calcTypeName(ATy->getElementType(), TypeStack, TypeNames) + "]";
    break;
  }
  default:
    Result = "<unrecognized-type>";
  }

  TypeStack.pop_back();       // Remove self from stack...
  return Result;
}


// printTypeInt - The internal guts of printing out a type that has a
// potentially named portion.
//
static ostream &printTypeInt(ostream &Out, const Type *Ty,
                             map<const Type *, string> &TypeNames) {
  // Primitive types always print out their description, regardless of whether
  // they have been named or not.
  //
  if (Ty->isPrimitiveType()) return Out << Ty->getDescription();

  // Check to see if the type is named.
  map<const Type *, string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return Out << I->second;

  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  //
  vector<const Type *> TypeStack;
  string TypeName = calcTypeName(Ty, TypeStack, TypeNames);
  TypeNames.insert(std::make_pair(Ty, TypeName));//Cache type name for later use
  return Out << TypeName;
}


// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
// type, iff there is an entry in the modules symbol table for the specified
// type or one of it's component types.  This is slower than a simple x << Type;
//
ostream &WriteTypeSymbolic(ostream &Out, const Type *Ty, const Module *M) {
  Out << " "; 

  // If they want us to print out a type, attempt to make it symbolic if there
  // is a symbol table in the module...
  if (M) {
    map<const Type *, string> TypeNames;
    fillTypeNameTable(M, TypeNames);
    
    return printTypeInt(Out, Ty, TypeNames);
  } else {
    return Out << Ty->getDescription();
  }
}

static void WriteConstantInt(ostream &Out, const Constant *CV, bool PrintName,
                             map<const Type *, string> &TypeTable,
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
         (StrVal[0] >= '0' && StrVal[0] <= '9')))
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
        unsigned char C = (ETy == Type::SByteTy) ?
          (unsigned char)cast<ConstantSInt>(CA->getOperand(i))->getValue() :
          (unsigned char)cast<ConstantUInt>(CA->getOperand(i))->getValue();
        
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
    const GlobalValue *V = PR->getValue();
    if (V->hasName()) {
      Out << "%" << V->getName();
    } else if (Table) {
      int Slot = Table->getValSlot(V);
      if (Slot >= 0)
        Out << "%" << Slot;
      else
        Out << "<pointer reference badref>";
    } else {
      Out << "<pointer reference without context info>";
    }

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


// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.
//
static void WriteAsOperandInternal(ostream &Out, const Value *V, bool PrintName,
                                   map<const Type *, string> &TypeTable,
                                   SlotCalculator *Table) {
  Out << " ";
  if (PrintName && V->hasName()) {
    Out << "%" << V->getName();
  } else {
    if (const Constant *CV = dyn_cast<const Constant>(V)) {
      WriteConstantInt(Out, CV, PrintName, TypeTable, Table);
    } else {
      int Slot;
      if (Table) {
	Slot = Table->getValSlot(V);
      } else {
        if (const Type *Ty = dyn_cast<const Type>(V)) {
          Out << Ty->getDescription();
          return;
        }

        Table = createSlotCalculator(V);
        if (Table == 0) { Out << "BAD VALUE TYPE!"; return; }

	Slot = Table->getValSlot(V);
	delete Table;
      }
      if (Slot >= 0)  Out << "%" << Slot;
      else if (PrintName)
        Out << "<badref>";     // Not embeded into a location?
    }
  }
}



// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.
//
ostream &WriteAsOperand(ostream &Out, const Value *V, bool PrintType, 
			bool PrintName, const Module *Context) {
  map<const Type *, string> TypeNames;
  if (Context == 0) Context = getModuleFromVal(V);

  if (Context)
    fillTypeNameTable(Context, TypeNames);

  if (PrintType)
    printTypeInt(Out, V->getType(), TypeNames);
  
  WriteAsOperandInternal(Out, V, PrintName, TypeNames, 0);
  return Out;
}



class AssemblyWriter {
  ostream &Out;
  SlotCalculator &Table;
  const Module *TheModule;
  map<const Type *, string> TypeNames;
public:
  inline AssemblyWriter(ostream &o, SlotCalculator &Tab, const Module *M)
    : Out(o), Table(Tab), TheModule(M) {

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
  ostream &printType(const Type *Ty) {
    return printTypeInt(Out, Ty, TypeNames);
  }

  // printTypeAtLeastOneLevel - Print out one level of the possibly complex type
  // without considering any symbolic types that we may have equal to it.
  //
  ostream &printTypeAtLeastOneLevel(const Type *Ty);

  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value &V);
};


// printTypeAtLeastOneLevel - Print out one level of the possibly complex type
// without considering any symbolic types that we may have equal to it.
//
ostream &AssemblyWriter::printTypeAtLeastOneLevel(const Type *Ty) {
  if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    printType(FTy->getReturnType()) << " (";
    for (FunctionType::ParamTypes::const_iterator
           I = FTy->getParamTypes().begin(),
           E = FTy->getParamTypes().end(); I != E; ++I) {
      if (I != FTy->getParamTypes().begin())
        Out << ", ";
      printType(*I);
    }
    if (FTy->isVarArg()) {
      if (!FTy->getParamTypes().empty()) Out << ", ";
      Out << "...";
    }
    Out << ")";
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    Out << "{ ";
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I) {
      if (I != STy->getElementTypes().begin())
        Out << ", ";
      printType(*I);
    }
    Out << " }";
  } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    printType(PTy->getElementType()) << "*";
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Out << "[" << ATy->getNumElements() << " x ";
    printType(ATy->getElementType()) << "]";
  } else if (const OpaqueType *OTy = dyn_cast<OpaqueType>(Ty)) {
    Out << OTy->getDescription();
  } else {
    if (!Ty->isPrimitiveType())
      Out << "<unknown derived type>";
    printType(Ty);
  }
  return Out;
}


void AssemblyWriter::writeOperand(const Value *Operand, bool PrintType, 
				  bool PrintName) {
  if (PrintType) { Out << " "; printType(Operand->getType()); }
  WriteAsOperandInternal(Out, Operand, PrintName, TypeNames, &Table);
}


void AssemblyWriter::printModule(const Module *M) {
  // Loop over the symbol table, emitting all named constants...
  printSymbolTable(M->getSymbolTable());
  
  for (Module::const_giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    printGlobal(I);

  Out << "\nimplementation   ; Functions:\n";
  
  // Output all of the functions...
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    printFunction(I);
}

void AssemblyWriter::printGlobal(const GlobalVariable *GV) {
  if (GV->hasName()) Out << "%" << GV->getName() << " = ";

  if (!GV->hasInitializer()) 
    Out << "external ";
  else
    switch (GV->getLinkage()) {
    case GlobalValue::InternalLinkage: Out << "internal "; break;
    case GlobalValue::LinkOnceLinkage: Out << "linkonce "; break;
    case GlobalValue::AppendingLinkage: Out << "appending "; break;
    case GlobalValue::ExternalLinkage: break;
    }

  Out << (GV->isConstant() ? "constant " : "global ");
  printType(GV->getType()->getElementType());

  if (GV->hasInitializer())
    writeOperand(GV->getInitializer(), false, false);

  printInfoComment(*GV);
  Out << "\n";
}


// printSymbolTable - Run through symbol table looking for named constants
// if a named constant is found, emit it's declaration...
//
void AssemblyWriter::printSymbolTable(const SymbolTable &ST) {
  for (SymbolTable::const_iterator TI = ST.begin(); TI != ST.end(); ++TI) {
    SymbolTable::type_const_iterator I = ST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = ST.type_end(TI->first);
    
    for (; I != End; ++I) {
      const Value *V = I->second;
      if (const Constant *CPV = dyn_cast<const Constant>(V)) {
	printConstant(CPV);
      } else if (const Type *Ty = dyn_cast<const Type>(V)) {
	Out << "\t%" << I->first << " = type ";

        // Make sure we print out at least one level of the type structure, so
        // that we do not get %FILE = type %FILE
        //
        printTypeAtLeastOneLevel(Ty) << "\n";
      }
    }
  }
}


// printConstant - Print out a constant pool entry...
//
void AssemblyWriter::printConstant(const Constant *CPV) {
  // Don't print out unnamed constants, they will be inlined
  if (!CPV->hasName()) return;

  // Print out name...
  Out << "\t%" << CPV->getName() << " =";

  // Write the value out now...
  writeOperand(CPV, true, false);

  printInfoComment(*CPV);
  Out << "\n";
}

// printFunction - Print all aspects of a function.
//
void AssemblyWriter::printFunction(const Function *F) {
  // Print out the return type and name...
  Out << "\n";

  if (F->isExternal())
    Out << "declare ";
  else
    switch (F->getLinkage()) {
    case GlobalValue::InternalLinkage: Out << "internal "; break;
    case GlobalValue::LinkOnceLinkage: Out << "linkonce "; break;
    case GlobalValue::AppendingLinkage: Out << "appending "; break;
    case GlobalValue::ExternalLinkage: break;
    }

  printType(F->getReturnType()) << " %" << F->getName() << "(";
  Table.incorporateFunction(F);

  // Loop over the arguments, printing them...
  const FunctionType *FT = F->getFunctionType();

  for(Function::const_aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    printArgument(I);

  // Finish printing arguments...
  if (FT->isVarArg()) {
    if (FT->getParamTypes().size()) Out << ", ";
    Out << "...";  // Output varargs portion of signature!
  }
  Out << ")";

  if (F->isExternal()) {
    Out << "\n";
  } else {
    Out << " {";
  
    // Output all of its basic blocks... for the function
    for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
      printBasicBlock(I);

    Out << "}\n";
  }

  Table.purgeFunction();
}

// printArgument - This member is called for every argument that 
// is passed into the function.  Simply print it out
//
void AssemblyWriter::printArgument(const Argument *Arg) {
  // Insert commas as we go... the first arg doesn't get a comma
  if (Arg != &Arg->getParent()->afront()) Out << ", ";

  // Output type...
  printType(Arg->getType());
  
  // Output name, if available...
  if (Arg->hasName())
    Out << " %" << Arg->getName();
  else if (Table.getValSlot(Arg) < 0)
    Out << "<badref>";
}

// printBasicBlock - This member is called for each basic block in a methd.
//
void AssemblyWriter::printBasicBlock(const BasicBlock *BB) {
  if (BB->hasName()) {              // Print out the label if it exists...
    Out << "\n" << BB->getName() << ":";
  } else if (!BB->use_empty()) {      // Don't print block # of no uses...
    int Slot = Table.getValSlot(BB);
    Out << "\n; <label>:";
    if (Slot >= 0) 
      Out << Slot;         // Extra newline seperates out label's
    else 
      Out << "<badref>"; 
  }
  
  // Output predecessors for the block...
  Out << "\t\t;";
  pred_const_iterator PI = pred_begin(BB), PE = pred_end(BB);

  if (PI == PE) {
    Out << " No predecessors!";
  } else {
    Out << " preds =";
    writeOperand(*PI, false, true);
    for (++PI; PI != PE; ++PI) {
      Out << ",";
      writeOperand(*PI, false, true);
    }
  }
  
  Out << "\n";

  // Output all of the instructions in the basic block...
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    printInstruction(*I);
}


// printInfoComment - Print a little comment after the instruction indicating
// which slot it occupies.
//
void AssemblyWriter::printInfoComment(const Value &V) {
  if (V.getType() != Type::VoidTy) {
    Out << "\t\t; <";
    printType(V.getType()) << ">";

    if (!V.hasName()) {
      int Slot = Table.getValSlot(&V); // Print out the def slot taken...
      if (Slot >= 0) Out << ":" << Slot;
      else Out << ":<badref>";
    }
    Out << " [#uses=" << V.use_size() << "]";  // Output # uses
  }
}

// printInstruction - This member is called for each Instruction in a methd.
//
void AssemblyWriter::printInstruction(const Instruction &I) {
  Out << "\t";

  // Print out name if it exists...
  if (I.hasName())
    Out << "%" << I.getName() << " = ";

  // Print out the opcode...
  Out << I.getOpcodeName();

  // Print out the type of the operands...
  const Value *Operand = I.getNumOperands() ? I.getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (isa<BranchInst>(I) && I.getNumOperands() > 1) {
    writeOperand(I.getOperand(2), true);
    Out << ",";
    writeOperand(Operand, true);
    Out << ",";
    writeOperand(I.getOperand(1), true);

  } else if (isa<SwitchInst>(I)) {
    // Special case switch statement to get formatting nice and correct...
    writeOperand(Operand        , true); Out << ",";
    writeOperand(I.getOperand(1), true); Out << " [";

    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; op += 2) {
      Out << "\n\t\t";
      writeOperand(I.getOperand(op  ), true); Out << ",";
      writeOperand(I.getOperand(op+1), true);
    }
    Out << "\n\t]";
  } else if (isa<PHINode>(I)) {
    Out << " ";
    printType(I.getType());
    Out << " ";

    for (unsigned op = 0, Eop = I.getNumOperands(); op < Eop; op += 2) {
      if (op) Out << ", ";
      Out << "[";  
      writeOperand(I.getOperand(op  ), false); Out << ",";
      writeOperand(I.getOperand(op+1), false); Out << " ]";
    }
  } else if (isa<ReturnInst>(I) && !Operand) {
    Out << " void";
  } else if (isa<CallInst>(I)) {
    const PointerType *PTy = dyn_cast<PointerType>(Operand->getType());
    const FunctionType*MTy = PTy ? dyn_cast<FunctionType>(PTy->getElementType()):0;
    const Type      *RetTy = MTy ? MTy->getReturnType() : 0;

    // If possible, print out the short form of the call instruction, but we can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the value returned is not a pointer to a function.
    //
    if (RetTy && MTy && !MTy->isVarArg() &&
        (!isa<PointerType>(RetTy) || 
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      Out << " "; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << "(";
    if (I.getNumOperands() > 1) writeOperand(I.getOperand(1), true);
    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; ++op) {
      Out << ",";
      writeOperand(I.getOperand(op), true);
    }

    Out << " )";
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    // TODO: Should try to print out short form of the Invoke instruction
    writeOperand(Operand, true);
    Out << "(";
    if (I.getNumOperands() > 3) writeOperand(I.getOperand(3), true);
    for (unsigned op = 4, Eop = I.getNumOperands(); op < Eop; ++op) {
      Out << ",";
      writeOperand(I.getOperand(op), true);
    }

    Out << " )\n\t\t\tto";
    writeOperand(II->getNormalDest(), true);
    Out << " except";
    writeOperand(II->getExceptionalDest(), true);

  } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(&I)) {
    Out << " ";
    printType(AI->getType()->getElementType());
    if (AI->isArrayAllocation()) {
      Out << ",";
      writeOperand(AI->getArraySize(), true);
    }
  } else if (isa<CastInst>(I)) {
    if (Operand) writeOperand(Operand, true);
    Out << " to ";
    printType(I.getType());
  } else if (Operand) {   // Print the normal way...

    // PrintAllTypes - Instructions who have operands of all the same type 
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    // Shift Left & Right print both types even for Ubyte LHS
    if (isa<ShiftInst>(I)) {
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
      Out << " ";
      printType(TheType);
    }

    for (unsigned i = 0, E = I.getNumOperands(); i != E; ++i) {
      if (i) Out << ",";
      writeOperand(I.getOperand(i), PrintAllTypes);
    }
  }

  printInfoComment(I);
  Out << "\n";
}


//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//


void Module::print(std::ostream &o) const {
  SlotCalculator SlotTable(this, true);
  AssemblyWriter W(o, SlotTable, this);
  W.write(this);
}

void GlobalVariable::print(std::ostream &o) const {
  SlotCalculator SlotTable(getParent(), true);
  AssemblyWriter W(o, SlotTable, getParent());
  W.write(this);
}

void Function::print(std::ostream &o) const {
  SlotCalculator SlotTable(getParent(), true);
  AssemblyWriter W(o, SlotTable, getParent());

  W.write(this);
}

void BasicBlock::print(std::ostream &o) const {
  SlotCalculator SlotTable(getParent(), true);
  AssemblyWriter W(o, SlotTable, 
                   getParent() ? getParent()->getParent() : 0);
  W.write(this);
}

void Instruction::print(std::ostream &o) const {
  const Function *F = getParent() ? getParent()->getParent() : 0;
  SlotCalculator SlotTable(F, true);
  AssemblyWriter W(o, SlotTable, F ? F->getParent() : 0);

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

  map<const Type *, string> TypeTable;
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
    SC = new SlotCalculator(M, true);
    AW = new AssemblyWriter(Out, *SC, M);
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
  case Value::TypeVal:           AW->write(cast<const Type>(V)); break;
  case Value::InstructionVal:    AW->write(cast<Instruction>(V)); break;
  case Value::BasicBlockVal:     AW->write(cast<BasicBlock>(V)); break;
  case Value::FunctionVal:       AW->write(cast<Function>(V)); break;
  case Value::GlobalVariableVal: AW->write(cast<GlobalVariable>(V)); break;
  default: Out << "<unknown value type: " << V->getValueType() << ">"; break;
  }
  return *this;
}
