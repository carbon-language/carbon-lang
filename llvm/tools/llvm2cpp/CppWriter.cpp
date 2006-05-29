//===-- CppWriter.cpp - Printing LLVM IR as a C++ Source File -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the writing of the LLVM IR as a set of C++ calls to the
// LLVM IR interface. The input module is assumed to be verified.
//
//===----------------------------------------------------------------------===//

#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instruction.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <iostream>

using namespace llvm;

namespace {
/// This class provides computation of slot numbers for LLVM Assembly writing.
/// @brief LLVM Assembly Writing Slot Computation.
class SlotMachine {

/// @name Types
/// @{
public:

  /// @brief A mapping of Values to slot numbers
  typedef std::map<const Value*, unsigned> ValueMap;
  typedef std::map<const Type*, unsigned> TypeMap;

  /// @brief A plane with next slot number and ValueMap
  struct ValuePlane {
    unsigned next_slot;        ///< The next slot number to use
    ValueMap map;              ///< The map of Value* -> unsigned
    ValuePlane() { next_slot = 0; } ///< Make sure we start at 0
  };

  struct TypePlane {
    unsigned next_slot;
    TypeMap map;
    TypePlane() { next_slot = 0; }
    void clear() { map.clear(); next_slot = 0; }
  };

  /// @brief The map of planes by Type
  typedef std::map<const Type*, ValuePlane> TypedPlanes;

/// @}
/// @name Constructors
/// @{
public:
  /// @brief Construct from a module
  SlotMachine(const Module *M );

/// @}
/// @name Accessors
/// @{
public:
  /// Return the slot number of the specified value in it's type
  /// plane.  Its an error to ask for something not in the SlotMachine.
  /// Its an error to ask for a Type*
  int getSlot(const Value *V);
  int getSlot(const Type*Ty);

  /// Determine if a Value has a slot or not
  bool hasSlot(const Value* V);
  bool hasSlot(const Type* Ty);

/// @}
/// @name Mutators
/// @{
public:
  /// If you'd like to deal with a function instead of just a module, use
  /// this method to get its data into the SlotMachine.
  void incorporateFunction(const Function *F) {
    TheFunction = F;
    FunctionProcessed = false;
  }

  /// After calling incorporateFunction, use this method to remove the
  /// most recently incorporated function from the SlotMachine. This
  /// will reset the state of the machine back to just the module contents.
  void purgeFunction();

/// @}
/// @name Implementation Details
/// @{
private:
  /// Values can be crammed into here at will. If they haven't
  /// been inserted already, they get inserted, otherwise they are ignored.
  /// Either way, the slot number for the Value* is returned.
  unsigned createSlot(const Value *V);
  unsigned createSlot(const Type* Ty);

  /// Insert a value into the value table. Return the slot number
  /// that it now occupies.  BadThings(TM) will happen if you insert a
  /// Value that's already been inserted.
  unsigned insertValue( const Value *V );
  unsigned insertValue( const Type* Ty);

  /// Add all of the module level global variables (and their initializers)
  /// and function declarations, but not the contents of those functions.
  void processModule();

  /// Add all of the functions arguments, basic blocks, and instructions
  void processFunction();

  SlotMachine(const SlotMachine &);  // DO NOT IMPLEMENT
  void operator=(const SlotMachine &);  // DO NOT IMPLEMENT

/// @}
/// @name Data
/// @{
public:

  /// @brief The module for which we are holding slot numbers
  const Module* TheModule;

  /// @brief The function for which we are holding slot numbers
  const Function* TheFunction;
  bool FunctionProcessed;

  /// @brief The TypePlanes map for the module level data
  TypedPlanes mMap;
  TypePlane mTypes;

  /// @brief The TypePlanes map for the function level data
  TypedPlanes fMap;
  TypePlane fTypes;

/// @}

};

typedef std::vector<const Type*> TypeList;
typedef std::map<const Type*,std::string> TypeMap;
typedef std::map<const Value*,std::string> ValueMap;

void WriteAsOperandInternal(std::ostream &Out, const Value *V,
                                   bool PrintName, TypeMap &TypeTable,
                                   SlotMachine *Machine);

void WriteAsOperandInternal(std::ostream &Out, const Type *T,
                                   bool PrintName, TypeMap& TypeTable,
                                   SlotMachine *Machine);

const Module *getModuleFromVal(const Value *V) {
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

// getLLVMName - Turn the specified string into an 'LLVM name', which is either
// prefixed with % (if the string only contains simple characters) or is
// surrounded with ""'s (if it has special chars in it).
std::string getLLVMName(const std::string &Name,
                               bool prefixName = true) {
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
  if (prefixName)
    return "%"+Name;
  else
    return Name;
}


/// fillTypeNameTable - If the module has a symbol table, take all global types
/// and stuff their names into the TypeNames map.
///
void fillTypeNameTable(const Module *M, TypeMap& TypeNames) {
  if (!M) return;
  const SymbolTable &ST = M->getSymbolTable();
  SymbolTable::type_const_iterator TI = ST.type_begin();
  for (; TI != ST.type_end(); ++TI ) {
    // As a heuristic, don't insert pointer to primitive types, because
    // they are used too often to have a single useful name.
    //
    const Type *Ty = cast<Type>(TI->second);
    if (!isa<PointerType>(Ty) ||
        !cast<PointerType>(Ty)->getElementType()->isPrimitiveType() ||
        isa<OpaqueType>(cast<PointerType>(Ty)->getElementType()))
      TypeNames.insert(std::make_pair(Ty, getLLVMName(TI->first)));
  }
}

void calcTypeName(const Type *Ty,
                         std::vector<const Type *> &TypeStack,
                         TypeMap& TypeNames,
                         std::string & Result){
  if (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty)) {
    Result += Ty->getDescription();  // Base case
    return;
  }

  // Check to see if the type is named.
  TypeMap::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) {
    Result += I->second;
    return;
  }

  if (isa<OpaqueType>(Ty)) {
    Result += "opaque";
    return;
  }

  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type

  // This is another base case for the recursion.  In this case, we know
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  if (Slot < CurSize) {
    Result += "\\" + utostr(CurSize-Slot);     // Here's the upreference
    return;
  }

  TypeStack.push_back(Ty);    // Recursive case: Add us to the stack..

  switch (Ty->getTypeID()) {
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    calcTypeName(FTy->getReturnType(), TypeStack, TypeNames, Result);
    Result += " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Result += ", ";
      calcTypeName(*I, TypeStack, TypeNames, Result);
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
    Result += "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        Result += ", ";
      calcTypeName(*I, TypeStack, TypeNames, Result);
    }
    Result += " }";
    break;
  }
  case Type::PointerTyID:
    calcTypeName(cast<PointerType>(Ty)->getElementType(),
                          TypeStack, TypeNames, Result);
    Result += "*";
    break;
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    Result += "[" + utostr(ATy->getNumElements()) + " x ";
    calcTypeName(ATy->getElementType(), TypeStack, TypeNames, Result);
    Result += "]";
    break;
  }
  case Type::PackedTyID: {
    const PackedType *PTy = cast<PackedType>(Ty);
    Result += "<" + utostr(PTy->getNumElements()) + " x ";
    calcTypeName(PTy->getElementType(), TypeStack, TypeNames, Result);
    Result += ">";
    break;
  }
  case Type::OpaqueTyID:
    Result += "opaque";
    break;
  default:
    Result += "<unrecognized-type>";
  }

  TypeStack.pop_back();       // Remove self from stack...
  return;
}


/// printTypeInt - The internal guts of printing out a type that has a
/// potentially named portion.
///
std::ostream &printTypeInt(std::ostream &Out, const Type *Ty,TypeMap&TypeNames){
  // Primitive types always print out their description, regardless of whether
  // they have been named or not.
  //
  if (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty))
    return Out << Ty->getDescription();

  // Check to see if the type is named.
  TypeMap::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return Out << I->second;

  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  //
  std::vector<const Type *> TypeStack;
  std::string TypeName;
  calcTypeName(Ty, TypeStack, TypeNames, TypeName);
  TypeNames.insert(std::make_pair(Ty, TypeName));//Cache type name for later use
  return (Out << TypeName);
}


/// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
/// type, iff there is an entry in the modules symbol table for the specified
/// type or one of it's component types. This is slower than a simple x << Type
///
std::ostream &WriteTypeSymbolic(std::ostream &Out, const Type *Ty,
                                      const Module *M) {
  Out << ' ';

  // If they want us to print out a type, attempt to make it symbolic if there
  // is a symbol table in the module...
  if (M) {
    TypeMap TypeNames;
    fillTypeNameTable(M, TypeNames);

    return printTypeInt(Out, Ty, TypeNames);
  } else {
    return Out << Ty->getDescription();
  }
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
void PrintEscapedString(const std::string &Str, std::ostream &Out) {
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    unsigned char C = Str[i];
    if (isprint(C) && C != '"' && C != '\\') {
      Out << C;
    } else {
      Out << '\\'
          << (char) ((C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A'))
          << (char)(((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A'));
    }
  }
}

/// @brief Internal constant writer.
void WriteConstantInternal(std::ostream &Out, const Constant *CV,
                             bool PrintName,
                             TypeMap& TypeTable,
                             SlotMachine *Machine) {
  const int IndentSize = 4;
  static std::string Indent = "\n";
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
        Out << StrVal;
        return;
      }

    // Otherwise we could not reparse it to exactly the same value, so we must
    // output the string in hexadecimal format!
    assert(sizeof(double) == sizeof(uint64_t) &&
           "assuming that double is 64 bits!");
    Out << "0x" << utohexstr(DoubleToBits(CFP->getValue()));

  } else if (isa<ConstantAggregateZero>(CV)) {
    Out << "zeroinitializer";
  } else if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
    // As a special case, print the array as a string if it is an array of
    // ubytes or an array of sbytes with positive values.
    //
    const Type *ETy = CA->getType()->getElementType();
    if (CA->isString()) {
      Out << "c\"";
      PrintEscapedString(CA->getAsString(), Out);
      Out << "\"";

    } else {                // Cannot output in string format...
      Out << '[';
      if (CA->getNumOperands()) {
        Out << ' ';
        printTypeInt(Out, ETy, TypeTable);
        WriteAsOperandInternal(Out, CA->getOperand(0),
                               PrintName, TypeTable, Machine);
        for (unsigned i = 1, e = CA->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printTypeInt(Out, ETy, TypeTable);
          WriteAsOperandInternal(Out, CA->getOperand(i), PrintName,
                                 TypeTable, Machine);
        }
      }
      Out << " ]";
    }
  } else if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(CV)) {
    Out << '{';
    unsigned N = CS->getNumOperands();
    if (N) {
      if (N > 2) {
        Indent += std::string(IndentSize, ' ');
        Out << Indent;
      } else {
        Out << ' ';
      }
      printTypeInt(Out, CS->getOperand(0)->getType(), TypeTable);

      WriteAsOperandInternal(Out, CS->getOperand(0),
                             PrintName, TypeTable, Machine);

      for (unsigned i = 1; i < N; i++) {
        Out << ", ";
        if (N > 2) Out << Indent;
        printTypeInt(Out, CS->getOperand(i)->getType(), TypeTable);

        WriteAsOperandInternal(Out, CS->getOperand(i),
                               PrintName, TypeTable, Machine);
      }
      if (N > 2) Indent.resize(Indent.size() - IndentSize);
    }
 
    Out << " }";
  } else if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(CV)) {
      const Type *ETy = CP->getType()->getElementType();
      assert(CP->getNumOperands() > 0 &&
             "Number of operands for a PackedConst must be > 0");
      Out << '<';
      Out << ' ';
      printTypeInt(Out, ETy, TypeTable);
      WriteAsOperandInternal(Out, CP->getOperand(0),
                             PrintName, TypeTable, Machine);
      for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printTypeInt(Out, ETy, TypeTable);
          WriteAsOperandInternal(Out, CP->getOperand(i), PrintName,
                                 TypeTable, Machine);
      }
      Out << " >";
  } else if (isa<ConstantPointerNull>(CV)) {
    Out << "null";

  } else if (isa<UndefValue>(CV)) {
    Out << "undef";

  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    Out << CE->getOpcodeName() << " (";

    for (User::const_op_iterator OI=CE->op_begin(); OI != CE->op_end(); ++OI) {
      printTypeInt(Out, (*OI)->getType(), TypeTable);
      WriteAsOperandInternal(Out, *OI, PrintName, TypeTable, Machine);
      if (OI+1 != CE->op_end())
        Out << ", ";
    }

    if (CE->getOpcode() == Instruction::Cast) {
      Out << " to ";
      printTypeInt(Out, CE->getType(), TypeTable);
    }
    Out << ')';

  } else {
    Out << "<placeholder or erroneous Constant>";
  }
}


/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
void WriteAsOperandInternal(std::ostream &Out, const Value *V,
                                   bool PrintName, TypeMap& TypeTable,
                                   SlotMachine *Machine) {
  Out << ' ';
  if ((PrintName || isa<GlobalValue>(V)) && V->hasName())
    Out << getLLVMName(V->getName());
  else {
    const Constant *CV = dyn_cast<Constant>(V);
    if (CV && !isa<GlobalValue>(CV)) {
      WriteConstantInternal(Out, CV, PrintName, TypeTable, Machine);
    } else if (const InlineAsm *IA = dyn_cast<InlineAsm>(V)) {
      Out << "asm ";
      if (IA->hasSideEffects())
        Out << "sideeffect ";
      Out << '"';
      PrintEscapedString(IA->getAsmString(), Out);
      Out << "\", \"";
      PrintEscapedString(IA->getConstraintString(), Out);
      Out << '"';
    } else {
      int Slot = Machine->getSlot(V);
      if (Slot != -1)
        Out << '%' << Slot;
      else
        Out << "<badref>";
    }
  }
}

/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
std::ostream &WriteAsOperand(std::ostream &Out, const Value *V,
                                   bool PrintType, bool PrintName,
                                   const Module *Context) {
  TypeMap TypeNames;
  if (Context == 0) Context = getModuleFromVal(V);

  if (Context)
    fillTypeNameTable(Context, TypeNames);

  if (PrintType)
    printTypeInt(Out, V->getType(), TypeNames);

  WriteAsOperandInternal(Out, V, PrintName, TypeNames, 0);
  return Out;
}

/// WriteAsOperandInternal - Write the name of the specified value out to
/// the specified ostream.  This can be useful when you just want to print
/// int %reg126, not the whole instruction that generated it.
///
void WriteAsOperandInternal(std::ostream &Out, const Type *T,
                                   bool PrintName, TypeMap& TypeTable,
                                   SlotMachine *Machine) {
  Out << ' ';
  int Slot = Machine->getSlot(T);
  if (Slot != -1)
    Out << '%' << Slot;
  else
    Out << "<badref>";
}

/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
std::ostream &WriteAsOperand(std::ostream &Out, const Type *Ty,
                                   bool PrintType, bool PrintName,
                                   const Module *Context) {
  TypeMap TypeNames;
  assert(Context != 0 && "Can't write types as operand without module context");

  fillTypeNameTable(Context, TypeNames);

  // if (PrintType)
    // printTypeInt(Out, V->getType(), TypeNames);

  printTypeInt(Out, Ty, TypeNames);

  WriteAsOperandInternal(Out, Ty, PrintName, TypeNames, 0);
  return Out;
}

class CppWriter {
  std::ostream &Out;
  SlotMachine &Machine;
  const Module *TheModule;
  unsigned long uniqueNum;
  TypeMap TypeNames;
  ValueMap ValueNames;
  TypeMap UnresolvedTypes;
  TypeList TypeStack;

public:
  inline CppWriter(std::ostream &o, SlotMachine &Mac, const Module *M)
    : Out(o), Machine(Mac), TheModule(M), uniqueNum(0), TypeNames(),
      ValueNames(), UnresolvedTypes(), TypeStack() { }

  inline void write(const Module *M)         { printModule(M);      }
  inline void write(const GlobalVariable *G) { printGlobal(G);      }
  inline void write(const Function *F)       { printFunction(F);    }
  inline void write(const BasicBlock *BB)    { printBasicBlock(BB); }
  inline void write(const Instruction *I)    { printInstruction(*I); }
  inline void write(const Constant *CPV)     { printConstant(CPV);  }
  inline void write(const Type *Ty)          { printType(Ty);       }

  void writeOperand(const Value *Op, bool PrintType, bool PrintName = true);

  const Module* getModule() { return TheModule; }

private:
  void printModule(const Module *M);
  void printTypes(const Module* M);
  void printConstants(const Module* M);
  void printConstant(const Constant *CPV);
  void printGlobal(const GlobalVariable *GV);
  void printFunction(const Function *F);
  void printArgument(const Argument *FA);
  void printBasicBlock(const BasicBlock *BB);
  void printInstruction(const Instruction &I);
  void printSymbolTable(const SymbolTable &ST);
  void printLinkageType(GlobalValue::LinkageTypes LT);
  void printCallingConv(unsigned cc);


  // printType - Go to extreme measures to attempt to print out a short,
  // symbolic version of a type name.
  //
  std::ostream &printType(const Type *Ty) {
    return printTypeInt(Out, Ty, TypeNames);
  }

  // printTypeAtLeastOneLevel - Print out one level of the possibly complex type
  // without considering any symbolic types that we may have equal to it.
  //
  std::ostream &printTypeAtLeastOneLevel(const Type *Ty);

  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value &V);

  std::string getCppName(const Type* val);
  std::string getCppName(const Value* val);
  inline void printCppName(const Value* val);
  inline void printCppName(const Type* val);
  bool isOnStack(const Type*) const;
  inline void printTypeDef(const Type* Ty);
  bool printTypeDefInternal(const Type* Ty);
};

std::string
CppWriter::getCppName(const Value* val) {
  std::string name;
  ValueMap::iterator I = ValueNames.find(val);
  if (I != ValueNames.end()) {
    name = I->second;
  } else {
    const char* prefix;
    switch (val->getType()->getTypeID()) {
      case Type::VoidTyID:     prefix = "void_"; break;
      case Type::BoolTyID:     prefix = "bool_"; break; 
      case Type::UByteTyID:    prefix = "ubyte_"; break;
      case Type::SByteTyID:    prefix = "sbyte_"; break;
      case Type::UShortTyID:   prefix = "ushort_"; break;
      case Type::ShortTyID:    prefix = "short_"; break;
      case Type::UIntTyID:     prefix = "uint_"; break;
      case Type::IntTyID:      prefix = "int_"; break;
      case Type::ULongTyID:    prefix = "ulong_"; break;
      case Type::LongTyID:     prefix = "long_"; break;
      case Type::FloatTyID:    prefix = "float_"; break;
      case Type::DoubleTyID:   prefix = "double_"; break;
      case Type::LabelTyID:    prefix = "label_"; break;
      case Type::FunctionTyID: prefix = "func_"; break;
      case Type::StructTyID:   prefix = "struct_"; break;
      case Type::ArrayTyID:    prefix = "array_"; break;
      case Type::PointerTyID:  prefix = "ptr_"; break;
      case Type::PackedTyID:   prefix = "packed_"; break;
      default:                 prefix = "other_"; break;
    }
    name = ValueNames[val] = std::string(prefix) +
        (val->hasName() ? val->getName() : utostr(uniqueNum++));
  }
  return name;
}

void
CppWriter::printCppName(const Value* val) {
  PrintEscapedString(getCppName(val),Out);
}

void
CppWriter::printCppName(const Type* Ty)
{
  PrintEscapedString(getCppName(Ty),Out);
}

// Gets the C++ name for a type. Returns true if we already saw the type,
// false otherwise.
//
inline const std::string* 
findTypeName(const SymbolTable& ST, const Type* Ty)
{
  SymbolTable::type_const_iterator TI = ST.type_begin();
  SymbolTable::type_const_iterator TE = ST.type_end();
  for (;TI != TE; ++TI)
    if (TI->second == Ty)
      return &(TI->first);
  return 0;
}

std::string
CppWriter::getCppName(const Type* Ty)
{
  // First, handle the primitive types .. easy
  if (Ty->isPrimitiveType()) {
    switch (Ty->getTypeID()) {
      case Type::VoidTyID:     return "Type::VoidTy";
      case Type::BoolTyID:     return "Type::BoolTy"; 
      case Type::UByteTyID:    return "Type::UByteTy";
      case Type::SByteTyID:    return "Type::SByteTy";
      case Type::UShortTyID:   return "Type::UShortTy";
      case Type::ShortTyID:    return "Type::ShortTy";
      case Type::UIntTyID:     return "Type::UIntTy";
      case Type::IntTyID:      return "Type::IntTy";
      case Type::ULongTyID:    return "Type::ULongTy";
      case Type::LongTyID:     return "Type::LongTy";
      case Type::FloatTyID:    return "Type::FloatTy";
      case Type::DoubleTyID:   return "Type::DoubleTy";
      case Type::LabelTyID:    return "Type::LabelTy";
      default:
        assert(!"Can't get here");
        break;
    }
    return "Type::VoidTy"; // shouldn't be returned, but make it sensible
  }

  // Now, see if we've seen the type before and return that
  TypeMap::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end())
    return I->second;

  // Okay, let's build a new name for this type. Start with a prefix
  const char* prefix = 0;
  switch (Ty->getTypeID()) {
    case Type::FunctionTyID:    prefix = "FuncTy_"; break;
    case Type::StructTyID:      prefix = "StructTy_"; break;
    case Type::ArrayTyID:       prefix = "ArrayTy_"; break;
    case Type::PointerTyID:     prefix = "PointerTy_"; break;
    case Type::OpaqueTyID:      prefix = "OpaqueTy_"; break;
    case Type::PackedTyID:      prefix = "PackedTy_"; break;
    default:                    prefix = "OtherTy_"; break; // prevent breakage
  }

  // See if the type has a name in the symboltable and build accordingly
  const std::string* tName = findTypeName(TheModule->getSymbolTable(), Ty);
  std::string name;
  if (tName) 
    name = std::string(prefix) + *tName;
  else
    name = std::string(prefix) + utostr(uniqueNum++);

  // Save the name
  return TypeNames[Ty] = name;
}

/// printTypeAtLeastOneLevel - Print out one level of the possibly complex type
/// without considering any symbolic types that we may have equal to it.
///
std::ostream &CppWriter::printTypeAtLeastOneLevel(const Type *Ty) {
  if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    printType(FTy->getReturnType()) << " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Out << ", ";
      printType(*I);
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) Out << ", ";
      Out << "...";
    }
    Out << ')';
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    Out << "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        Out << ", ";
      printType(*I);
    }
    Out << " }";
  } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    printType(PTy->getElementType()) << '*';
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Out << '[' << ATy->getNumElements() << " x ";
    printType(ATy->getElementType()) << ']';
  } else if (const PackedType *PTy = dyn_cast<PackedType>(Ty)) {
    Out << '<' << PTy->getNumElements() << " x ";
    printType(PTy->getElementType()) << '>';
  }
  else if (const OpaqueType *OTy = dyn_cast<OpaqueType>(Ty)) {
    Out << "opaque";
  } else {
    if (!Ty->isPrimitiveType())
      Out << "<unknown derived type>";
    printType(Ty);
  }
  return Out;
}


void CppWriter::writeOperand(const Value *Operand, bool PrintType,
                                  bool PrintName) {
  if (Operand != 0) {
    if (PrintType) { Out << ' '; printType(Operand->getType()); }
    WriteAsOperandInternal(Out, Operand, PrintName, TypeNames, &Machine);
  } else {
    Out << "<null operand!>";
  }
}


void CppWriter::printModule(const Module *M) {
  Out << "\n// Module Construction\n";
  Out << "Module* mod = new Module(\"";
  PrintEscapedString(M->getModuleIdentifier(),Out);
  Out << "\");\n";
  Out << "mod->setEndianness(";
  switch (M->getEndianness()) {
    case Module::LittleEndian: Out << "Module::LittleEndian);\n"; break;
    case Module::BigEndian:    Out << "Module::BigEndian);\n";    break;
    case Module::AnyEndianness:Out << "Module::AnyEndianness);\n";  break;
  }
  Out << "mod->setPointerSize(";
  switch (M->getPointerSize()) {
    case Module::Pointer32:      Out << "Module::Pointer32);\n"; break;
    case Module::Pointer64:      Out << "Module::Pointer64);\n"; break;
    case Module::AnyPointerSize: Out << "Module::AnyPointerSize);\n"; break;
  }
  if (!M->getTargetTriple().empty())
    Out << "mod->setTargetTriple(\"" << M->getTargetTriple() << "\");\n";

  if (!M->getModuleInlineAsm().empty()) {
    Out << "mod->setModuleInlineAsm(\"";
    PrintEscapedString(M->getModuleInlineAsm(),Out);
    Out << "\");\n";
  }
  
  // Loop over the dependent libraries and emit them.
  Module::lib_iterator LI = M->lib_begin();
  Module::lib_iterator LE = M->lib_end();
  while (LI != LE) {
    Out << "mod->addLibrary(\"" << *LI << "\");\n";
    ++LI;
  }

  // Print out all the type definitions
  Out << "\n// Type Definitions\n";
  printTypes(M);

  // Print out all the constants declarations
  Out << "\n// Constants Construction\n";
  printConstants(M);

  // Process the global variables
  Out << "\n// Global Variable Construction\n";
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    printGlobal(I);
  }

  // Output all of the functions.
  Out << "\n// Function Construction\n";
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    printFunction(I);
}

void
CppWriter::printCallingConv(unsigned cc){
  // Print the calling convention.
  switch (cc) {
    default:
    case CallingConv::C:     Out << "CallingConv::C"; break;
    case CallingConv::CSRet: Out << "CallingConv::CSRet"; break;
    case CallingConv::Fast:  Out << "CallingConv::Fast"; break;
    case CallingConv::Cold:  Out << "CallingConv::Cold"; break;
    case CallingConv::FirstTargetCC: Out << "CallingConv::FirstTargetCC"; break;
  }
}

void 
CppWriter::printLinkageType(GlobalValue::LinkageTypes LT) {
  switch (LT) {
    case GlobalValue::InternalLinkage:  
      Out << "GlobalValue::InternalLinkage"; break;
    case GlobalValue::LinkOnceLinkage:  
      Out << "GlobalValue::LinkOnceLinkage "; break;
    case GlobalValue::WeakLinkage:      
      Out << "GlobalValue::WeakLinkage"; break;
    case GlobalValue::AppendingLinkage: 
      Out << "GlobalValue::AppendingLinkage"; break;
    case GlobalValue::ExternalLinkage: 
      Out << "GlobalValue::ExternalLinkage"; break;
    case GlobalValue::GhostLinkage:
      Out << "GlobalValue::GhostLinkage"; break;
  }
}
void CppWriter::printGlobal(const GlobalVariable *GV) {
  Out << "\n";
  Out << "GlobalVariable* ";
  printCppName(GV);
  Out << " = new GlobalVariable(\n";
  Out << "  /*Type=*/";
  printCppName(GV->getType()->getElementType());
  Out << ",\n";
  Out << "  /*isConstant=*/" << (GV->isConstant()?"true":"false") 
      << ",\n  /*Linkage=*/";
  printLinkageType(GV->getLinkage());
  Out << ",\n  /*Initializer=*/";
  if (GV->hasInitializer()) {
    printCppName(GV->getInitializer());
  } else {
    Out << "0";
  }
  Out << ",\n  /*Name=*/\"";
  PrintEscapedString(GV->getName(),Out);
  Out << "\",\n  mod);\n";

  if (GV->hasSection()) {
    printCppName(GV);
    Out << "->setSection(\"";
    PrintEscapedString(GV->getSection(),Out);
    Out << "\");\n";
  }
  if (GV->getAlignment()) {
    printCppName(GV);
    Out << "->setAlignment(" << utostr(GV->getAlignment()) << ");\n";
  };
}

bool
CppWriter::isOnStack(const Type* Ty) const {
  TypeList::const_iterator TI = 
    std::find(TypeStack.begin(),TypeStack.end(),Ty);
  return TI != TypeStack.end();
}

// Prints a type definition. Returns true if it could not resolve all the types
// in the definition but had to use a forward reference.
void
CppWriter::printTypeDef(const Type* Ty) {
  assert(TypeStack.empty());
  TypeStack.clear();
  printTypeDefInternal(Ty);
  assert(TypeStack.empty());
  // early resolve as many unresolved types as possible. Search the unresolved
  // types map for the type we just printed. Now that its definition is complete
  // we can resolve any preview references to it. This prevents a cascade of
  // unresolved types.
  TypeMap::iterator I = UnresolvedTypes.find(Ty);
  if (I != UnresolvedTypes.end()) {
    Out << "cast<OpaqueType>(" << I->second 
        << "_fwd.get())->refineAbstractTypeTo(" << I->second << ");\n";
    Out << I->second << " = cast<";
    switch (Ty->getTypeID()) {
      case Type::FunctionTyID: Out << "FunctionType"; break;
      case Type::ArrayTyID:    Out << "ArrayType"; break;
      case Type::StructTyID:   Out << "StructType"; break;
      case Type::PackedTyID:   Out << "PackedType"; break;
      case Type::PointerTyID:  Out << "PointerType"; break;
      case Type::OpaqueTyID:   Out << "OpaqueType"; break;
      default:                 Out << "NoSuchDerivedType"; break;
    }
    Out << ">(" << I->second << "_fwd.get());\n";
    UnresolvedTypes.erase(I);
  }
  Out << "\n";
}

bool
CppWriter::printTypeDefInternal(const Type* Ty) {
  // We don't print definitions for primitive types
  if (Ty->isPrimitiveType())
    return false;

  // Determine if the name is in the name list before we modify that list.
  TypeMap::const_iterator TNI = TypeNames.find(Ty);

  // Everything below needs the name for the type so get it now
  std::string typeName(getCppName(Ty));

  // Search the type stack for recursion. If we find it, then generate this
  // as an OpaqueType, but make sure not to do this multiple times because
  // the type could appear in multiple places on the stack. Once the opaque
  // definition is issues, it must not be re-issued. Consequently we have to
  // check the UnresolvedTypes list as well.
  if (isOnStack(Ty)) {
    TypeMap::const_iterator I = UnresolvedTypes.find(Ty);
    if (I == UnresolvedTypes.end()) {
      Out << "PATypeHolder " << typeName << "_fwd = OpaqueType::get();\n";
      UnresolvedTypes[Ty] = typeName;
      return true;
    }
  }

  // Avoid printing things we have already printed. Since TNI was obtained
  // before the name was inserted with getCppName and because we know the name
  // is not on the stack (currently being defined), we can surmise here that if
  // we got the name we've also already emitted its definition.
  if (TNI != TypeNames.end())
    return false;

  // We're going to print a derived type which, by definition, contains other
  // types. So, push this one we're printing onto the type stack to assist with
  // recursive definitions.
  TypeStack.push_back(Ty); // push on type stack
  bool didRecurse = false;

  // Print the type definition
  switch (Ty->getTypeID()) {
    case Type::FunctionTyID:  {
      const FunctionType* FT = cast<FunctionType>(Ty);
      Out << "std::vector<const Type*>" << typeName << "_args;\n";
      FunctionType::param_iterator PI = FT->param_begin();
      FunctionType::param_iterator PE = FT->param_end();
      for (; PI != PE; ++PI) {
        const Type* argTy = static_cast<const Type*>(*PI);
        bool isForward = printTypeDefInternal(argTy);
        std::string argName(getCppName(argTy));
        Out << typeName << "_args.push_back(" << argName;
        if (isForward)
          Out << "_fwd";
        Out << ");\n";
      }
      bool isForward = printTypeDefInternal(FT->getReturnType());
      std::string retTypeName(getCppName(FT->getReturnType()));
      Out << "FunctionType* " << typeName << " = FunctionType::get(\n"
          << "  /*Result=*/" << retTypeName;
      if (isForward)
        Out << "_fwd";
      Out << ",\n  /*Params=*/" << typeName << "_args,\n  /*isVarArg=*/"
          << (FT->isVarArg() ? "true" : "false") << ");\n";
      break;
    }
    case Type::StructTyID: {
      const StructType* ST = cast<StructType>(Ty);
      Out << "std::vector<const Type*>" << typeName << "_fields;\n";
      StructType::element_iterator EI = ST->element_begin();
      StructType::element_iterator EE = ST->element_end();
      for (; EI != EE; ++EI) {
        const Type* fieldTy = static_cast<const Type*>(*EI);
        bool isForward = printTypeDefInternal(fieldTy);
        std::string fieldName(getCppName(fieldTy));
        Out << typeName << "_fields.push_back(" << fieldName;
        if (isForward)
          Out << "_fwd";
        Out << ");\n";
      }
      Out << "StructType* " << typeName << " = StructType::get("
          << typeName << "_fields);\n";
      break;
    }
    case Type::ArrayTyID: {
      const ArrayType* AT = cast<ArrayType>(Ty);
      const Type* ET = AT->getElementType();
      bool isForward = printTypeDefInternal(ET);
      std::string elemName(getCppName(ET));
      Out << "ArrayType* " << typeName << " = ArrayType::get("
          << elemName << (isForward ? "_fwd" : "") 
          << ", " << utostr(AT->getNumElements()) << ");\n";
      break;
    }
    case Type::PointerTyID: {
      const PointerType* PT = cast<PointerType>(Ty);
      const Type* ET = PT->getElementType();
      bool isForward = printTypeDefInternal(ET);
      std::string elemName(getCppName(ET));
      Out << "PointerType* " << typeName << " = PointerType::get("
          << elemName << (isForward ? "_fwd" : "") << ");\n";
      break;
    }
    case Type::PackedTyID: {
      const PackedType* PT = cast<PackedType>(Ty);
      const Type* ET = PT->getElementType();
      bool isForward = printTypeDefInternal(ET);
      std::string elemName(getCppName(ET));
      Out << "PackedType* " << typeName << " = PackedType::get("
          << elemName << (isForward ? "_fwd" : "") 
          << ", " << utostr(PT->getNumElements()) << ");\n";
      break;
    }
    case Type::OpaqueTyID: {
      const OpaqueType* OT = cast<OpaqueType>(Ty);
      Out << "OpaqueType* " << typeName << " = OpaqueType::get();\n";
      break;
    }
    default:
      assert(!"Invalid TypeID");
  }

  // Pop us off the type stack
  TypeStack.pop_back();

  // We weren't a recursive type
  return false;
}

void
CppWriter::printTypes(const Module* M) {
  // Add all of the global variables to the value table...
  for (Module::const_global_iterator I = TheModule->global_begin(), 
       E = TheModule->global_end(); I != E; ++I) {
    if (I->hasInitializer())
      printTypeDef(I->getInitializer()->getType());
    printTypeDef(I->getType());
  }

  // Add all the functions to the table
  for (Module::const_iterator FI = TheModule->begin(), FE = TheModule->end();
       FI != FE; ++FI) {
    printTypeDef(FI->getReturnType());
    printTypeDef(FI->getFunctionType());
    // Add all the function arguments
    for(Function::const_arg_iterator AI = FI->arg_begin(),
        AE = FI->arg_end(); AI != AE; ++AI) {
      printTypeDef(AI->getType());
    }

    // Add all of the basic blocks and instructions
    for (Function::const_iterator BB = FI->begin(),
         E = FI->end(); BB != E; ++BB) {
      printTypeDef(BB->getType());
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; 
           ++I) {
        printTypeDef(I->getType());
      }
    }
  }
}

void
CppWriter::printConstants(const Module* M) {
  const SymbolTable& ST = M->getSymbolTable();

  // Print the constants, in type plane order.
  for (SymbolTable::plane_const_iterator PI = ST.plane_begin();
       PI != ST.plane_end(); ++PI ) {
    SymbolTable::value_const_iterator VI = ST.value_begin(PI->first);
    SymbolTable::value_const_iterator VE = ST.value_end(PI->first);

    for (; VI != VE; ++VI) {
      const Value* V = VI->second;
      const Constant *CPV = dyn_cast<Constant>(V) ;
      if (CPV && !isa<GlobalValue>(V)) {
        printConstant(CPV);
      }
    }
  }

  // Add all of the global variables to the value table...
  for (Module::const_global_iterator I = TheModule->global_begin(), 
       E = TheModule->global_end(); I != E; ++I)
    if (I->hasInitializer())
      printConstant(I->getInitializer());
}

// printSymbolTable - Run through symbol table looking for constants
// and types. Emit their declarations.
void CppWriter::printSymbolTable(const SymbolTable &ST) {

  // Print the types.
  for (SymbolTable::type_const_iterator TI = ST.type_begin();
       TI != ST.type_end(); ++TI ) {
    Out << "\t" << getLLVMName(TI->first) << " = type ";

    // Make sure we print out at least one level of the type structure, so
    // that we do not get %FILE = type %FILE
    //
    printTypeAtLeastOneLevel(TI->second) << "\n";
  }

}


/// printConstant - Print out a constant pool entry...
///
void CppWriter::printConstant(const Constant *CV) {
  const int IndentSize = 2;
  static std::string Indent = "\n";
  std::string constName(getCppName(CV));
  std::string typeName(getCppName(CV->getType()));
  if (CV->isNullValue()) {
    Out << "Constant* " << constName << " = Constant::getNullValue("
        << typeName << ");\n";
    return;
  }
  if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    Out << "Constant* " << constName << " = ConstantBool::get(" 
        << (CB == ConstantBool::True ? "true" : "false")
        << ");";
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV)) {
    Out << "Constant* " << constName << " = ConstantSInt::get(" 
        << typeName << ", " << CI->getValue() << ");";
  } else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV)) {
    Out << "Constant* " << constName << " = ConstantUInt::get(" 
        << typeName << ", " << CI->getValue() << ");";
  } else if (isa<ConstantAggregateZero>(CV)) {
    Out << "Constant* " << constName << " = ConstantAggregateZero::get(" 
        << typeName << ");";
  } else if (isa<ConstantPointerNull>(CV)) {
    Out << "Constant* " << constName << " = ConstanPointerNull::get(" 
        << typeName << ");";
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    Out << "ConstantFP::get(" << typeName << ", ";
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
        Out << StrVal;
        return;
      }

    // Otherwise we could not reparse it to exactly the same value, so we must
    // output the string in hexadecimal format!
    assert(sizeof(double) == sizeof(uint64_t) &&
           "assuming that double is 64 bits!");
    Out << "0x" << utohexstr(DoubleToBits(CFP->getValue())) << ");";
  } else if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
    if (CA->isString()) {
      Out << "Constant* " << constName << " = ConstantArray::get(\"";
      PrintEscapedString(CA->getAsString(),Out);
      Out << "\");";
    } else {
      Out << "std::vector<Constant*> " << constName << "_elems;\n";
      unsigned N = CA->getNumOperands();
      for (unsigned i = 0; i < N; ++i) {
        printConstant(CA->getOperand(i));
        Out << constName << "_elems.push_back("
            << getCppName(CA->getOperand(i)) << ");\n";
      }
      Out << "Constant* " << constName << " = ConstantArray::get(" 
          << typeName << ", " << constName << "_elems);";
    }
  } else if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(CV)) {
    Out << "std::vector<Constant*> " << constName << "_fields;\n";
    unsigned N = CS->getNumOperands();
    for (unsigned i = 0; i < N; i++) {
      printConstant(CS->getOperand(i));
      Out << constName << "_fields.push_back("
          << getCppName(CA->getOperand(i)) << ");\n";
    }
    Out << "Constant* " << constName << " = ConstantStruct::get(" 
        << typeName << ", " << constName << "_fields);";
  } else if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(CV)) {
    Out << "std::vector<Constant*> " << constName << "_elems;\n";
    unsigned N = CP->getNumOperands();
    for (unsigned i = 0; i < N; ++i) {
      printConstant(CP->getOperand(i));
      Out << constName << "_elems.push_back("
          << getCppName(CP->getOperand(i)) << ");\n";
    }
    Out << "Constant* " << constName << " = ConstantPacked::get(" 
        << typeName << ", " << constName << "_elems);";
  } else if (isa<UndefValue>(CV)) {
    Out << "Constant* " << constName << " = UndefValue::get(" 
        << typeName << ");\n";
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    Out << CE->getOpcodeName() << " (";

    for (User::const_op_iterator OI=CE->op_begin(); OI != CE->op_end(); ++OI) {
      //printTypeInt(Out, (*OI)->getType(), TypeTable);
      //WriteAsOperandInternal(Out, *OI, PrintName, TypeTable, Machine);
      if (OI+1 != CE->op_end())
        Out << ", ";
    }

    if (CE->getOpcode() == Instruction::Cast) {
      Out << " to ";
      // printTypeInt(Out, CE->getType(), TypeTable);
    }
    Out << ')';

  } else {
    Out << "<placeholder or erroneous Constant>";
  }
  Out << "\n";
}

/// printFunction - Print all aspects of a function.
///
void CppWriter::printFunction(const Function *F) {
  std::string funcTypeName(getCppName(F->getFunctionType()));

  Out << "Function* ";
  printCppName(F);
  Out << " = new Function(" << funcTypeName << ", " ;
  printLinkageType(F->getLinkage());
  Out << ", \"" << F->getName() << "\", mod);\n";
  printCppName(F);
  Out << "->setCallingConv(";
  printCallingConv(F->getCallingConv());
  Out << ");\n";
  if (F->hasSection()) {
    printCppName(F);
    Out << "->setSection(" << F->getSection() << ");\n";
  }
  if (F->getAlignment()) {
    printCppName(F);
    Out << "->setAlignment(" << F->getAlignment() << ");\n";
  }

  Machine.incorporateFunction(F);

  if (!F->isExternal()) {
    Out << "{";
    // Output all of its basic blocks... for the function
    for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
      printBasicBlock(I);
    Out << "}\n";
  }

  Machine.purgeFunction();
}

/// printArgument - This member is called for every argument that is passed into
/// the function.  Simply print it out
///
void CppWriter::printArgument(const Argument *Arg) {
  // Insert commas as we go... the first arg doesn't get a comma
  if (Arg != Arg->getParent()->arg_begin()) Out << ", ";

  // Output type...
  printType(Arg->getType());

  // Output name, if available...
  if (Arg->hasName())
    Out << ' ' << getLLVMName(Arg->getName());
}

/// printBasicBlock - This member is called for each basic block in a method.
///
void CppWriter::printBasicBlock(const BasicBlock *BB) {
  if (BB->hasName()) {              // Print out the label if it exists...
    Out << "\n" << getLLVMName(BB->getName(), false) << ':';
  } else if (!BB->use_empty()) {      // Don't print block # of no uses...
    Out << "\n; <label>:";
    int Slot = Machine.getSlot(BB);
    if (Slot != -1)
      Out << Slot;
    else
      Out << "<badref>";
  }

  if (BB->getParent() == 0)
    Out << "\t\t; Error: Block without parent!";
  else {
    if (BB != &BB->getParent()->front()) {  // Not the entry block?
      // Output predecessors for the block...
      Out << "\t\t;";
      pred_const_iterator PI = pred_begin(BB), PE = pred_end(BB);

      if (PI == PE) {
        Out << " No predecessors!";
      } else {
        Out << " preds =";
        writeOperand(*PI, false, true);
        for (++PI; PI != PE; ++PI) {
          Out << ',';
          writeOperand(*PI, false, true);
        }
      }
    }
  }

  Out << "\n";

  // Output all of the instructions in the basic block...
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    printInstruction(*I);
}


/// printInfoComment - Print a little comment after the instruction indicating
/// which slot it occupies.
///
void CppWriter::printInfoComment(const Value &V) {
  if (V.getType() != Type::VoidTy) {
    Out << "\t\t; <";
    printType(V.getType()) << '>';

    if (!V.hasName()) {
      int SlotNum = Machine.getSlot(&V);
      if (SlotNum == -1)
        Out << ":<badref>";
      else
        Out << ':' << SlotNum; // Print out the def slot taken.
    }
    Out << " [#uses=" << V.getNumUses() << ']';  // Output # uses
  }
}

/// printInstruction - This member is called for each Instruction in a function..
///
void CppWriter::printInstruction(const Instruction &I) {
  Out << "\t";

  // Print out name if it exists...
  if (I.hasName())
    Out << getLLVMName(I.getName()) << " = ";

  // If this is a volatile load or store, print out the volatile marker.
  if ((isa<LoadInst>(I)  && cast<LoadInst>(I).isVolatile()) ||
      (isa<StoreInst>(I) && cast<StoreInst>(I).isVolatile())) {
      Out << "volatile ";
  } else if (isa<CallInst>(I) && cast<CallInst>(I).isTailCall()) {
    // If this is a call, check if it's a tail call.
    Out << "tail ";
  }

  // Print out the opcode...
  Out << I.getOpcodeName();

  // Print out the type of the operands...
  const Value *Operand = I.getNumOperands() ? I.getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (isa<BranchInst>(I) && I.getNumOperands() > 1) {
    writeOperand(I.getOperand(2), true);
    Out << ',';
    writeOperand(Operand, true);
    Out << ',';
    writeOperand(I.getOperand(1), true);

  } else if (isa<SwitchInst>(I)) {
    // Special case switch statement to get formatting nice and correct...
    writeOperand(Operand        , true); Out << ',';
    writeOperand(I.getOperand(1), true); Out << " [";

    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; op += 2) {
      Out << "\n\t\t";
      writeOperand(I.getOperand(op  ), true); Out << ',';
      writeOperand(I.getOperand(op+1), true);
    }
    Out << "\n\t]";
  } else if (isa<PHINode>(I)) {
    Out << ' ';
    printType(I.getType());
    Out << ' ';

    for (unsigned op = 0, Eop = I.getNumOperands(); op < Eop; op += 2) {
      if (op) Out << ", ";
      Out << '[';
      writeOperand(I.getOperand(op  ), false); Out << ',';
      writeOperand(I.getOperand(op+1), false); Out << " ]";
    }
  } else if (isa<ReturnInst>(I) && !Operand) {
    Out << " void";
  } else if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
    // Print the calling convention being used.
    switch (CI->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::CSRet: Out << " csretcc"; break;
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    default: Out << " cc" << CI->getCallingConv(); break;
    }

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
      Out << ' '; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << '(';
    if (CI->getNumOperands() > 1) writeOperand(CI->getOperand(1), true);
    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; ++op) {
      Out << ',';
      writeOperand(I.getOperand(op), true);
    }

    Out << " )";
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    const PointerType  *PTy = cast<PointerType>(Operand->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
    const Type       *RetTy = FTy->getReturnType();

    // Print the calling convention being used.
    switch (II->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::CSRet: Out << " csretcc"; break;
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    default: Out << " cc" << II->getCallingConv(); break;
    }

    // If possible, print out the short form of the invoke instruction. We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    if (!FTy->isVarArg() &&
        (!isa<PointerType>(RetTy) ||
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      Out << ' '; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }

    Out << '(';
    if (I.getNumOperands() > 3) writeOperand(I.getOperand(3), true);
    for (unsigned op = 4, Eop = I.getNumOperands(); op < Eop; ++op) {
      Out << ',';
      writeOperand(I.getOperand(op), true);
    }

    Out << " )\n\t\t\tto";
    writeOperand(II->getNormalDest(), true);
    Out << " unwind";
    writeOperand(II->getUnwindDest(), true);

  } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(&I)) {
    Out << ' ';
    printType(AI->getType()->getElementType());
    if (AI->isArrayAllocation()) {
      Out << ',';
      writeOperand(AI->getArraySize(), true);
    }
    if (AI->getAlignment()) {
      Out << ", align " << AI->getAlignment();
    }
  } else if (isa<CastInst>(I)) {
    if (Operand) writeOperand(Operand, true);   // Work with broken code
    Out << " to ";
    printType(I.getType());
  } else if (isa<VAArgInst>(I)) {
    if (Operand) writeOperand(Operand, true);   // Work with broken code
    Out << ", ";
    printType(I.getType());
  } else if (Operand) {   // Print the normal way...

    // PrintAllTypes - Instructions who have operands of all the same type
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    // Shift Left & Right print both types even for Ubyte LHS, and select prints
    // types even if all operands are bools.
    if (isa<ShiftInst>(I) || isa<SelectInst>(I) || isa<StoreInst>(I) ||
        isa<ShuffleVectorInst>(I)) {
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
      Out << ' ';
      printType(TheType);
    }

    for (unsigned i = 0, E = I.getNumOperands(); i != E; ++i) {
      if (i) Out << ',';
      writeOperand(I.getOperand(i), PrintAllTypes);
    }
  }

  printInfoComment(I);
  Out << "\n";
}


//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//===--                    SlotMachine Implementation
//===----------------------------------------------------------------------===//

#if 0
#define SC_DEBUG(X) std::cerr << X
#else
#define SC_DEBUG(X)
#endif

// Module level constructor. Causes the contents of the Module (sans functions)
// to be added to the slot table.
SlotMachine::SlotMachine(const Module *M)
  : TheModule(M)    ///< Saved for lazy initialization.
  , mMap()
  , mTypes()
  , fMap()
  , fTypes()
{
  assert(M != 0 && "Invalid Module");
  processModule();
}

// Iterate through all the global variables, functions, and global
// variable initializers and create slots for them.
void SlotMachine::processModule() {
  // Add all of the global variables to the value table...
  for (Module::const_global_iterator I = TheModule->global_begin(), E = TheModule->global_end();
       I != E; ++I)
    createSlot(I);

  // Add all the functions to the table
  for (Module::const_iterator FI = TheModule->begin(), FE = TheModule->end();
       FI != FE; ++FI) {
    createSlot(FI);
    // Add all the function arguments
    for(Function::const_arg_iterator AI = FI->arg_begin(),
        AE = FI->arg_end(); AI != AE; ++AI)
      createSlot(AI);

    // Add all of the basic blocks and instructions
    for (Function::const_iterator BB = FI->begin(),
         E = FI->end(); BB != E; ++BB) {
      createSlot(BB);
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; 
           ++I) {
        createSlot(I);
      }
    }
  }
}

// Process the arguments, basic blocks, and instructions  of a function.
void SlotMachine::processFunction() {

}

// Clean up after incorporating a function. This is the only way
// to get out of the function incorporation state that affects the
// getSlot/createSlot lock. Function incorporation state is indicated
// by TheFunction != 0.
void SlotMachine::purgeFunction() {
  SC_DEBUG("begin purgeFunction!\n");
  fMap.clear(); // Simply discard the function level map
  fTypes.clear();
  TheFunction = 0;
  FunctionProcessed = false;
  SC_DEBUG("end purgeFunction!\n");
}

/// Get the slot number for a value. This function will assert if you
/// ask for a Value that hasn't previously been inserted with createSlot.
/// Types are forbidden because Type does not inherit from Value (any more).
int SlotMachine::getSlot(const Value *V) {
  assert( V && "Can't get slot for null Value" );
  assert(!isa<Constant>(V) || isa<GlobalValue>(V) &&
    "Can't insert a non-GlobalValue Constant into SlotMachine");

  // Get the type of the value
  const Type* VTy = V->getType();

  // Find the type plane in the module map
  TypedPlanes::const_iterator MI = mMap.find(VTy);

  if ( TheFunction ) {
    // Lookup the type in the function map too
    TypedPlanes::const_iterator FI = fMap.find(VTy);
    // If there is a corresponding type plane in the function map
    if ( FI != fMap.end() ) {
      // Lookup the Value in the function map
      ValueMap::const_iterator FVI = FI->second.map.find(V);
      // If the value doesn't exist in the function map
      if ( FVI == FI->second.map.end() ) {
        // Look up the value in the module map.
        if (MI == mMap.end()) return -1;
        ValueMap::const_iterator MVI = MI->second.map.find(V);
        // If we didn't find it, it wasn't inserted
        if (MVI == MI->second.map.end()) return -1;
        assert( MVI != MI->second.map.end() && "Value not found");
        // We found it only at the module level
        return MVI->second;

      // else the value exists in the function map
      } else {
        // Return the slot number as the module's contribution to
        // the type plane plus the index in the function's contribution
        // to the type plane.
        if (MI != mMap.end())
          return MI->second.next_slot + FVI->second;
        else
          return FVI->second;
      }
    }
  }

  // N.B. Can get here only if either !TheFunction or the function doesn't
  // have a corresponding type plane for the Value

  // Make sure the type plane exists
  if (MI == mMap.end()) return -1;
  // Lookup the value in the module's map
  ValueMap::const_iterator MVI = MI->second.map.find(V);
  // Make sure we found it.
  if (MVI == MI->second.map.end()) return -1;
  // Return it.
  return MVI->second;
}

/// Get the slot number for a type. This function will assert if you
/// ask for a Type that hasn't previously been inserted with createSlot.
int SlotMachine::getSlot(const Type *Ty) {
  assert( Ty && "Can't get slot for null Type" );

  if ( TheFunction ) {
    // Lookup the Type in the function map
    TypeMap::const_iterator FTI = fTypes.map.find(Ty);
    // If the Type doesn't exist in the function map
    if ( FTI == fTypes.map.end() ) {
      TypeMap::const_iterator MTI = mTypes.map.find(Ty);
      // If we didn't find it, it wasn't inserted
      if (MTI == mTypes.map.end())
        return -1;
      // We found it only at the module level
      return MTI->second;

    // else the value exists in the function map
    } else {
      // Return the slot number as the module's contribution to
      // the type plane plus the index in the function's contribution
      // to the type plane.
      return mTypes.next_slot + FTI->second;
    }
  }

  // N.B. Can get here only if !TheFunction

  // Lookup the value in the module's map
  TypeMap::const_iterator MTI = mTypes.map.find(Ty);
  // Make sure we found it.
  if (MTI == mTypes.map.end()) return -1;
  // Return it.
  return MTI->second;
}

// Create a new slot, or return the existing slot if it is already
// inserted. Note that the logic here parallels getSlot but instead
// of asserting when the Value* isn't found, it inserts the value.
unsigned SlotMachine::createSlot(const Value *V) {
  assert( V && "Can't insert a null Value to SlotMachine");
  assert(!isa<Constant>(V) || isa<GlobalValue>(V) &&
    "Can't insert a non-GlobalValue Constant into SlotMachine");

  const Type* VTy = V->getType();

  // Just ignore void typed things
  if (VTy == Type::VoidTy) return 0; // FIXME: Wrong return value!

  // Look up the type plane for the Value's type from the module map
  TypedPlanes::const_iterator MI = mMap.find(VTy);

  if ( TheFunction ) {
    // Get the type plane for the Value's type from the function map
    TypedPlanes::const_iterator FI = fMap.find(VTy);
    // If there is a corresponding type plane in the function map
    if ( FI != fMap.end() ) {
      // Lookup the Value in the function map
      ValueMap::const_iterator FVI = FI->second.map.find(V);
      // If the value doesn't exist in the function map
      if ( FVI == FI->second.map.end() ) {
        // If there is no corresponding type plane in the module map
        if ( MI == mMap.end() )
          return insertValue(V);
        // Look up the value in the module map
        ValueMap::const_iterator MVI = MI->second.map.find(V);
        // If we didn't find it, it wasn't inserted
        if ( MVI == MI->second.map.end() )
          return insertValue(V);
        else
          // We found it only at the module level
          return MVI->second;

      // else the value exists in the function map
      } else {
        if ( MI == mMap.end() )
          return FVI->second;
        else
          // Return the slot number as the module's contribution to
          // the type plane plus the index in the function's contribution
          // to the type plane.
          return MI->second.next_slot + FVI->second;
      }

    // else there is not a corresponding type plane in the function map
    } else {
      // If the type plane doesn't exists at the module level
      if ( MI == mMap.end() ) {
        return insertValue(V);
      // else type plane exists at the module level, examine it
      } else {
        // Look up the value in the module's map
        ValueMap::const_iterator MVI = MI->second.map.find(V);
        // If we didn't find it there either
        if ( MVI == MI->second.map.end() )
          // Return the slot number as the module's contribution to
          // the type plane plus the index of the function map insertion.
          return MI->second.next_slot + insertValue(V);
        else
          return MVI->second;
      }
    }
  }

  // N.B. Can only get here if !TheFunction

  // If the module map's type plane is not for the Value's type
  if ( MI != mMap.end() ) {
    // Lookup the value in the module's map
    ValueMap::const_iterator MVI = MI->second.map.find(V);
    if ( MVI != MI->second.map.end() )
      return MVI->second;
  }

  return insertValue(V);
}

// Create a new slot, or return the existing slot if it is already
// inserted. Note that the logic here parallels getSlot but instead
// of asserting when the Value* isn't found, it inserts the value.
unsigned SlotMachine::createSlot(const Type *Ty) {
  assert( Ty && "Can't insert a null Type to SlotMachine");

  if ( TheFunction ) {
    // Lookup the Type in the function map
    TypeMap::const_iterator FTI = fTypes.map.find(Ty);
    // If the type doesn't exist in the function map
    if ( FTI == fTypes.map.end() ) {
      // Look up the type in the module map
      TypeMap::const_iterator MTI = mTypes.map.find(Ty);
      // If we didn't find it, it wasn't inserted
      if ( MTI == mTypes.map.end() )
        return insertValue(Ty);
      else
        // We found it only at the module level
        return MTI->second;

    // else the value exists in the function map
    } else {
      // Return the slot number as the module's contribution to
      // the type plane plus the index in the function's contribution
      // to the type plane.
      return mTypes.next_slot + FTI->second;
    }
  }

  // N.B. Can only get here if !TheFunction

  // Lookup the type in the module's map
  TypeMap::const_iterator MTI = mTypes.map.find(Ty);
  if ( MTI != mTypes.map.end() )
    return MTI->second;

  return insertValue(Ty);
}

// Low level insert function. Minimal checking is done. This
// function is just for the convenience of createSlot (above).
unsigned SlotMachine::insertValue(const Value *V ) {
  assert(V && "Can't insert a null Value into SlotMachine!");
  assert(!isa<Constant>(V) || isa<GlobalValue>(V) &&
    "Can't insert a non-GlobalValue Constant into SlotMachine");

  // If this value does not contribute to a plane (is void)
  // or if the value already has a name then ignore it.
  if (V->getType() == Type::VoidTy || V->hasName() ) {
      SC_DEBUG("ignored value " << *V << "\n");
      return 0;   // FIXME: Wrong return value
  }

  const Type *VTy = V->getType();
  unsigned DestSlot = 0;

  if ( TheFunction ) {
    TypedPlanes::iterator I = fMap.find( VTy );
    if ( I == fMap.end() )
      I = fMap.insert(std::make_pair(VTy,ValuePlane())).first;
    DestSlot = I->second.map[V] = I->second.next_slot++;
  } else {
    TypedPlanes::iterator I = mMap.find( VTy );
    if ( I == mMap.end() )
      I = mMap.insert(std::make_pair(VTy,ValuePlane())).first;
    DestSlot = I->second.map[V] = I->second.next_slot++;
  }

  SC_DEBUG("  Inserting value [" << VTy << "] = " << V << " slot=" <<
           DestSlot << " [");
  // G = Global, C = Constant, T = Type, F = Function, o = other
  SC_DEBUG((isa<GlobalVariable>(V) ? 'G' : (isa<Function>(V) ? 'F' :
           (isa<Constant>(V) ? 'C' : 'o'))));
  SC_DEBUG("]\n");
  return DestSlot;
}

// Low level insert function. Minimal checking is done. This
// function is just for the convenience of createSlot (above).
unsigned SlotMachine::insertValue(const Type *Ty ) {
  assert(Ty && "Can't insert a null Type into SlotMachine!");

  unsigned DestSlot = fTypes.map[Ty] = fTypes.next_slot++;
  SC_DEBUG("  Inserting type [" << DestSlot << "] = " << Ty << "\n");
  return DestSlot;
}

}  // end anonymous llvm

namespace llvm {

void WriteModuleToCppFile(Module* mod, std::ostream& o) {
  o << "#include <llvm/Module.h>\n";
  o << "#include <llvm/DerivedTypes.h>\n";
  o << "#include <llvm/Constants.h>\n";
  o << "#include <llvm/GlobalVariable.h>\n";
  o << "#include <llvm/Function.h>\n";
  o << "#include <llvm/CallingConv.h>\n";
  o << "#include <llvm/BasicBlock.h>\n";
  o << "#include <llvm/Instructions.h>\n";
  o << "#include <llvm/Pass.h>\n";
  o << "#include <llvm/PassManager.h>\n";
  o << "#include <llvm/Analysis/Verifier.h>\n";
  o << "#include <llvm/Assembly/PrintModulePass.h>\n";
  o << "#include <algorithm>\n";
  o << "#include <iostream>\n\n";
  o << "using namespace llvm;\n\n";
  o << "Module* makeLLVMModule();\n\n";
  o << "int main(int argc, char**argv) {\n";
  o << "  Module* Mod = makeLLVMModule();\n";
  o << "  verifyModule(*Mod, PrintMessageAction);\n";
  o << "  PassManager PM;\n";
  o << "  PM.add(new PrintModulePass(&std::cout));\n";
  o << "  PM.run(*Mod);\n";
  o << "  return 0;\n";
  o << "}\n\n";
  o << "Module* makeLLVMModule() {\n";
  SlotMachine SlotTable(mod);
  CppWriter W(o, SlotTable, mod);
  W.write(mod);
  o << "}\n";
}

}
