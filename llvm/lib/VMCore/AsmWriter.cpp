//===-- Writer.cpp - Library for Printing VM assembly files ------*- C++ -*--=//
//
// This library implements the functionality defined in llvm/Assembly/Writer.h
//
// This library uses the Analysis library to figure out offsets for
// variables in the method tables...
//
// TODO: print out the type name instead of the full type if a particular type
//       is in the symbol table...
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/CachedWriter.h"
#include "llvm/Analysis/SlotCalculator.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/GlobalVariable.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/SymbolTable.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <map>

static const Module *getModuleFromVal(const Value *V) {
  if (const MethodArgument *MA =dyn_cast<const MethodArgument>(V))
    return MA->getParent() ? MA->getParent()->getParent() : 0;
  else if (const BasicBlock *BB = dyn_cast<const BasicBlock>(V))
    return BB->getParent() ? BB->getParent()->getParent() : 0;
  else if (const Instruction *I = dyn_cast<const Instruction>(V)) {
    const Method *M = I->getParent() ? I->getParent()->getParent() : 0;
    return M ? M->getParent() : 0;
  } else if (const GlobalValue *GV =dyn_cast<const GlobalValue>(V))
    return GV->getParent();
  else if (const Module *Mod  = dyn_cast<const Module>(V))
    return Mod;
  return 0;
}

static SlotCalculator *createSlotCalculator(const Value *V) {
  assert(!isa<Type>(V) && "Can't create an SC for a type!");
  if (const MethodArgument *MA =dyn_cast<const MethodArgument>(V)){
    return new SlotCalculator(MA->getParent(), true);
  } else if (const Instruction *I = dyn_cast<const Instruction>(V)) {
    return new SlotCalculator(I->getParent()->getParent(), true);
  } else if (const BasicBlock *BB = dyn_cast<const BasicBlock>(V)) {
    return new SlotCalculator(BB->getParent(), true);
  } else if (const GlobalVariable *GV =dyn_cast<const GlobalVariable>(V)){
    return new SlotCalculator(GV->getParent(), true);
  } else if (const Method *Meth = dyn_cast<const Method>(V)) {
    return new SlotCalculator(Meth, true);
  } else if (const Module *Mod  = dyn_cast<const Module>(V)) {
    return new SlotCalculator(Mod, true);
  }
  return 0;
}

// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.
//
static void WriteAsOperandInternal(ostream &Out, const Value *V, bool PrintName,
                                   SlotCalculator *Table) {
  if (PrintName && V->hasName()) {
    Out << " %" << V->getName();
  } else {
    if (const ConstPoolVal *CPV = dyn_cast<const ConstPoolVal>(V)) {
      Out << " " << CPV->getStrValue();
    } else {
      int Slot;
      if (Table) {
	Slot = Table->getValSlot(V);
      } else {
        if (const Type *Ty = dyn_cast<const Type>(V)) {
          Out << " " << Ty->getDescription();
          return;
        }

        Table = createSlotCalculator(V);
        if (Table == 0) { Out << "BAD VALUE TYPE!"; return; }

	Slot = Table->getValSlot(V);
	delete Table;
      }
      if (Slot >= 0)  Out << " %" << Slot;
      else if (PrintName)
        Out << "<badref>";     // Not embeded into a location?
    }
  }
}


// If the module has a symbol table, take all global types and stuff their
// names into the TypeNames map.
//
static void fillTypeNameTable(const Module *M,
                              map<const Type *, string> &TypeNames) {
  if (M && M->hasSymbolTable()) {
    const SymbolTable *ST = M->getSymbolTable();
    SymbolTable::const_iterator PI = ST->find(Type::TypeTy);
    if (PI != ST->end()) {
      SymbolTable::type_const_iterator I = PI->second.begin();
      for (; I != PI->second.end(); ++I) {
        // As a heuristic, don't insert pointer to primitive types, because
        // they are used too often to have a single useful name.
        //
        const Type *Ty = cast<const Type>(I->second);
        if (!isa<PointerType>(Ty) ||
            !cast<PointerType>(Ty)->getValueType()->isPrimitiveType())
          TypeNames.insert(make_pair(Ty, "%"+I->first));
      }
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
  case Type::MethodTyID: {
    const MethodType *MTy = cast<const MethodType>(Ty);
    Result = calcTypeName(MTy->getReturnType(), TypeStack, TypeNames) + " (";
    for (MethodType::ParamTypes::const_iterator
           I = MTy->getParamTypes().begin(),
           E = MTy->getParamTypes().end(); I != E; ++I) {
      if (I != MTy->getParamTypes().begin())
        Result += ", ";
      Result += calcTypeName(*I, TypeStack, TypeNames);
    }
    if (MTy->isVarArg()) {
      if (!MTy->getParamTypes().empty()) Result += ", ";
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
    Result = calcTypeName(cast<const PointerType>(Ty)->getValueType(), 
                          TypeStack, TypeNames) + " *";
    break;
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<const ArrayType>(Ty);
    int NumElements = ATy->getNumElements();
    Result = "[";
    if (NumElements != -1) Result += itostr(NumElements) + " x ";
    Result += calcTypeName(ATy->getElementType(), TypeStack, TypeNames) + "]";
    break;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    Result = "<error>";
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
  TypeNames.insert(make_pair(Ty, TypeName));   // Cache type name for later use
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
  if (M && M->hasSymbolTable()) {
    map<const Type *, string> TypeNames;
    fillTypeNameTable(M, TypeNames);
    
    return printTypeInt(Out, Ty, TypeNames);
  } else {
    return Out << Ty->getDescription();
  }
}


// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.
//
ostream &WriteAsOperand(ostream &Out, const Value *V, bool PrintType, 
			bool PrintName, SlotCalculator *Table) {
  if (PrintType)
    WriteTypeSymbolic(Out, V->getType(), getModuleFromVal(V));

  WriteAsOperandInternal(Out, V, PrintName, Table);
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
  inline void write(const Method *M)         { printMethod(M);      }
  inline void write(const BasicBlock *BB)    { printBasicBlock(BB); }
  inline void write(const Instruction *I)    { printInstruction(I); }
  inline void write(const ConstPoolVal *CPV) { printConstant(CPV);  }
  inline void write(const Type *Ty)          { printType(Ty);       }

private :
  void printModule(const Module *M);
  void printSymbolTable(const SymbolTable &ST);
  void printConstant(const ConstPoolVal *CPV);
  void printGlobal(const GlobalVariable *GV);
  void printMethod(const Method *M);
  void printMethodArgument(const MethodArgument *MA);
  void printBasicBlock(const BasicBlock *BB);
  void printInstruction(const Instruction *I);
  ostream &printType(const Type *Ty);

  void writeOperand(const Value *Op, bool PrintType, bool PrintName = true);

  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value *V);
};


void AssemblyWriter::writeOperand(const Value *Operand, bool PrintType, 
				  bool PrintName) {
  if (PrintType) { Out << " "; printType(Operand->getType()); }
  WriteAsOperandInternal(Out, Operand, PrintName, &Table);
}


void AssemblyWriter::printModule(const Module *M) {
  // Loop over the symbol table, emitting all named constants...
  if (M->hasSymbolTable())
    printSymbolTable(*M->getSymbolTable());
  
  for_each(M->gbegin(), M->gend(), 
	   bind_obj(this, &AssemblyWriter::printGlobal));

  Out << "implementation\n";
  
  // Output all of the methods...
  for_each(M->begin(), M->end(), bind_obj(this,&AssemblyWriter::printMethod));
}

void AssemblyWriter::printGlobal(const GlobalVariable *GV) {
  if (GV->hasName()) Out << "%" << GV->getName() << " = ";

  if (GV->hasInternalLinkage()) Out << "internal ";
  if (!GV->hasInitializer()) Out << "uninitialized ";

  Out << (GV->isConstant() ? "constant " : "global ");
  printType(GV->getType()->getValueType());

  if (GV->hasInitializer())
    writeOperand(GV->getInitializer(), false, false);

  printInfoComment(GV);
  Out << endl;
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
      if (const ConstPoolVal *CPV = dyn_cast<const ConstPoolVal>(V)) {
	printConstant(CPV);
      } else if (const Type *Ty = dyn_cast<const Type>(V)) {
	Out << "\t%" << I->first << " = type " << Ty->getDescription() << endl;
      }
    }
  }
}


// printConstant - Print out a constant pool entry...
//
void AssemblyWriter::printConstant(const ConstPoolVal *CPV) {
  // Don't print out unnamed constants, they will be inlined
  if (!CPV->hasName()) return;

  // Print out name...
  Out << "\t%" << CPV->getName() << " = ";

  // Print out the constant type...
  printType(CPV->getType());

  // Write the value out now...
  writeOperand(CPV, false, false);

  if (!CPV->hasName() && CPV->getType() != Type::VoidTy) {
    int Slot = Table.getValSlot(CPV); // Print out the def slot taken...
    Out << "\t\t; <";
    printType(CPV->getType()) << ">:";
    if (Slot >= 0) Out << Slot;
    else Out << "<badref>";
  } 

  Out << endl;
}

// printMethod - Print all aspects of a method.
//
void AssemblyWriter::printMethod(const Method *M) {
  // Print out the return type and name...
  Out << "\n" << (M->isExternal() ? "declare " : "")
      << (M->hasInternalLinkage() ? "internal " : "");
  printType(M->getReturnType()) << " \"" << M->getName() << "\"(";
  Table.incorporateMethod(M);

  // Loop over the arguments, printing them...
  const MethodType *MT = cast<const MethodType>(M->getMethodType());

  if (!M->isExternal()) {
    for_each(M->getArgumentList().begin(), M->getArgumentList().end(),
	     bind_obj(this, &AssemblyWriter::printMethodArgument));
  } else {
    // Loop over the arguments, printing them...
    const MethodType *MT = cast<const MethodType>(M->getMethodType());
    for (MethodType::ParamTypes::const_iterator I = MT->getParamTypes().begin(),
	   E = MT->getParamTypes().end(); I != E; ++I) {
      if (I != MT->getParamTypes().begin()) Out << ", ";
      printType(*I);
    }
  }

  // Finish printing arguments...
  if (MT->isVarArg()) {
    if (MT->getParamTypes().size()) Out << ", ";
    Out << "...";  // Output varargs portion of signature!
  }
  Out << ")\n";

  if (!M->isExternal()) {
    // Loop over the symbol table, emitting all named constants...
    if (M->hasSymbolTable())
      printSymbolTable(*M->getSymbolTable());

    Out << "begin";
  
    // Output all of its basic blocks... for the method
    for_each(M->begin(), M->end(),
	     bind_obj(this, &AssemblyWriter::printBasicBlock));

    Out << "end\n";
  }

  Table.purgeMethod();
}

// printMethodArgument - This member is called for every argument that 
// is passed into the method.  Simply print it out
//
void AssemblyWriter::printMethodArgument(const MethodArgument *Arg) {
  // Insert commas as we go... the first arg doesn't get a comma
  if (Arg != Arg->getParent()->getArgumentList().front()) Out << ", ";

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
  } else {
    int Slot = Table.getValSlot(BB);
    Out << "\n; <label>:";
    if (Slot >= 0) 
      Out << Slot;         // Extra newline seperates out label's
    else 
      Out << "<badref>"; 
  }
  Out << "\t\t\t\t\t;[#uses=" << BB->use_size() << "]\n";  // Output # uses

  // Output all of the instructions in the basic block...
  for_each(BB->begin(), BB->end(),
	   bind_obj(this, &AssemblyWriter::printInstruction));
}


// printInfoComment - Print a little comment after the instruction indicating
// which slot it occupies.
//
void AssemblyWriter::printInfoComment(const Value *V) {
  if (V->getType() != Type::VoidTy) {
    Out << "\t\t; <";
    printType(V->getType()) << ">";

    if (!V->hasName()) {
      int Slot = Table.getValSlot(V); // Print out the def slot taken...
      if (Slot >= 0) Out << ":" << Slot;
      else Out << ":<badref>";
    }
    Out << "\t[#uses=" << V->use_size() << "]";  // Output # uses
  }
}

// printInstruction - This member is called for each Instruction in a methd.
//
void AssemblyWriter::printInstruction(const Instruction *I) {
  Out << "\t";

  // Print out name if it exists...
  if (I && I->hasName())
    Out << "%" << I->getName() << " = ";

  // Print out the opcode...
  Out << I->getOpcodeName();

  // Print out the type of the operands...
  const Value *Operand = I->getNumOperands() ? I->getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (I->getOpcode() == Instruction::Br && I->getNumOperands() > 1) {
    writeOperand(I->getOperand(2), true);
    Out << ",";
    writeOperand(Operand, true);
    Out << ",";
    writeOperand(I->getOperand(1), true);

  } else if (I->getOpcode() == Instruction::Switch) {
    // Special case switch statement to get formatting nice and correct...
    writeOperand(Operand         , true); Out << ",";
    writeOperand(I->getOperand(1), true); Out << " [";

    for (unsigned op = 2, Eop = I->getNumOperands(); op < Eop; op += 2) {
      Out << "\n\t\t";
      writeOperand(I->getOperand(op  ), true); Out << ",";
      writeOperand(I->getOperand(op+1), true);
    }
    Out << "\n\t]";
  } else if (isa<PHINode>(I)) {
    Out << " ";
    printType(I->getType());
    Out << " ";

    for (unsigned op = 0, Eop = I->getNumOperands(); op < Eop; op += 2) {
      if (op) Out << ", ";
      Out << "[";  
      writeOperand(I->getOperand(op  ), false); Out << ",";
      writeOperand(I->getOperand(op+1), false); Out << " ]";
    }
  } else if (isa<ReturnInst>(I) && !Operand) {
    Out << " void";
  } else if (isa<CallInst>(I)) {
    const PointerType *PTy = dyn_cast<PointerType>(Operand->getType());
    const MethodType  *MTy = PTy ? dyn_cast<MethodType>(PTy->getValueType()) :0;
    const Type      *RetTy = MTy ? MTy->getReturnType() : 0;

    // If possible, print out the short form of the call instruction, but we can
    // only do this if the first argument is a pointer to a nonvararg method,
    // and if the value returned is not a pointer to a method.
    //
    if (RetTy && !MTy->isVarArg() &&
        (!isa<PointerType>(RetTy)||!isa<MethodType>(cast<PointerType>(RetTy)))){
      Out << " "; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << "(";
    if (I->getNumOperands() > 1) writeOperand(I->getOperand(1), true);
    for (unsigned op = 2, Eop = I->getNumOperands(); op < Eop; ++op) {
      Out << ",";
      writeOperand(I->getOperand(op), true);
    }

    Out << " )";
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(I)) {
    // TODO: Should try to print out short form of the Invoke instruction
    writeOperand(Operand, true);
    Out << "(";
    if (I->getNumOperands() > 3) writeOperand(I->getOperand(3), true);
    for (unsigned op = 4, Eop = I->getNumOperands(); op < Eop; ++op) {
      Out << ",";
      writeOperand(I->getOperand(op), true);
    }

    Out << " )\n\t\t\tto";
    writeOperand(II->getNormalDest(), true);
    Out << " except";
    writeOperand(II->getExceptionalDest(), true);

  } else if (I->getOpcode() == Instruction::Malloc || 
	     I->getOpcode() == Instruction::Alloca) {
    Out << " ";
    printType(cast<const PointerType>(I->getType())->getValueType());
    if (I->getNumOperands()) {
      Out << ",";
      writeOperand(I->getOperand(0), true);
    }
  } else if (isa<CastInst>(I)) {
    writeOperand(Operand, true);
    Out << " to ";
    printType(I->getType());
  } else if (Operand) {   // Print the normal way...

    // PrintAllTypes - Instructions who have operands of all the same type 
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    for (unsigned i = 1, E = I->getNumOperands(); i != E; ++i) {
      Operand = I->getOperand(i);
      if (Operand->getType() != TheType) {
	PrintAllTypes = true;       // We have differing types!  Print them all!
	break;
      }
    }

    // Shift Left & Right print both types even for Ubyte LHS
    if (isa<ShiftInst>(I)) PrintAllTypes = true;

    if (!PrintAllTypes) {
      Out << " ";
      printType(I->getOperand(0)->getType());
    }

    for (unsigned i = 0, E = I->getNumOperands(); i != E; ++i) {
      if (i) Out << ",";
      writeOperand(I->getOperand(i), PrintAllTypes);
    }
  }

  printInfoComment(I);
  Out << endl;
}


// printType - Go to extreme measures to attempt to print out a short, symbolic
// version of a type name.
//
ostream &AssemblyWriter::printType(const Type *Ty) {
  return printTypeInt(Out, Ty, TypeNames);
}


//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//



void WriteToAssembly(const Module *M, ostream &o) {
  if (M == 0) { o << "<null> module\n"; return; }
  SlotCalculator SlotTable(M, true);
  AssemblyWriter W(o, SlotTable, M);

  W.write(M);
}

void WriteToAssembly(const GlobalVariable *G, ostream &o) {
  if (G == 0) { o << "<null> global variable\n"; return; }
  SlotCalculator SlotTable(G->getParent(), true);
  AssemblyWriter W(o, SlotTable, G->getParent());
  W.write(G);
}

void WriteToAssembly(const Method *M, ostream &o) {
  if (M == 0) { o << "<null> method\n"; return; }
  SlotCalculator SlotTable(M->getParent(), true);
  AssemblyWriter W(o, SlotTable, M->getParent());

  W.write(M);
}


void WriteToAssembly(const BasicBlock *BB, ostream &o) {
  if (BB == 0) { o << "<null> basic block\n"; return; }

  SlotCalculator SlotTable(BB->getParent(), true);
  AssemblyWriter W(o, SlotTable, 
                   BB->getParent() ? BB->getParent()->getParent() : 0);

  W.write(BB);
}

void WriteToAssembly(const ConstPoolVal *CPV, ostream &o) {
  if (CPV == 0) { o << "<null> constant pool value\n"; return; }
  o << " " << CPV->getType()->getDescription() << " " << CPV->getStrValue();
}

void WriteToAssembly(const Instruction *I, ostream &o) {
  if (I == 0) { o << "<null> instruction\n"; return; }

  const Method *M = I->getParent() ? I->getParent()->getParent() : 0;
  SlotCalculator SlotTable(M, true);
  AssemblyWriter W(o, SlotTable, M ? M->getParent() : 0);

  W.write(I);
}

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
    Out << " "; AW->write(V->getType());
    Out << " " << cast<ConstPoolVal>(V)->getStrValue(); break;
  case Value::MethodArgumentVal: 
    AW->write(V->getType()); Out << " " << V->getName(); break;
  case Value::TypeVal:           AW->write(cast<const Type>(V)); break;
  case Value::InstructionVal:    AW->write(cast<Instruction>(V)); break;
  case Value::BasicBlockVal:     AW->write(cast<BasicBlock>(V)); break;
  case Value::MethodVal:         AW->write(cast<Method>(V)); break;
  case Value::GlobalVariableVal: AW->write(cast<GlobalVariable>(V)); break;
  case Value::ModuleVal:         AW->write(cast<Module>(V)); break;
  default: Out << "<unknown value type: " << V->getValueType() << ">"; break;
  }
  return *this;
}
