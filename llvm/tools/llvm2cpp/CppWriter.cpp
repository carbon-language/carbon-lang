//===-- CppWriter.cpp - Printing LLVM IR as a C++ Source File -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Config/config.h"
#include <algorithm>
#include <iostream>
#include <set>

using namespace llvm;

static cl::opt<std::string>
FuncName("funcname", cl::desc("Specify the name of the generated function"),
         cl::value_desc("function name"));

enum WhatToGenerate {
  GenProgram,
  GenModule,
  GenContents,
  GenFunction,
  GenInline,
  GenVariable,
  GenType
};

static cl::opt<WhatToGenerate> GenerationType(cl::Optional,
  cl::desc("Choose what kind of output to generate"),
  cl::init(GenProgram),
  cl::values(
    clEnumValN(GenProgram, "gen-program",  "Generate a complete program"),
    clEnumValN(GenModule,  "gen-module",   "Generate a module definition"),
    clEnumValN(GenContents,"gen-contents", "Generate contents of a module"),
    clEnumValN(GenFunction,"gen-function", "Generate a function definition"),
    clEnumValN(GenInline,  "gen-inline",   "Generate an inline function"),
    clEnumValN(GenVariable,"gen-variable", "Generate a variable definition"),
    clEnumValN(GenType,    "gen-type",     "Generate a type definition"),
    clEnumValEnd
  )
);

static cl::opt<std::string> NameToGenerate("for", cl::Optional,
  cl::desc("Specify the name of the thing to generate"),
  cl::init("!bad!"));

namespace {
typedef std::vector<const Type*> TypeList;
typedef std::map<const Type*,std::string> TypeMap;
typedef std::map<const Value*,std::string> ValueMap;
typedef std::set<std::string> NameSet;
typedef std::set<const Type*> TypeSet;
typedef std::set<const Value*> ValueSet;
typedef std::map<const Value*,std::string> ForwardRefMap;

class CppWriter {
  const char* progname;
  std::ostream &Out;
  const Module *TheModule;
  uint64_t uniqueNum;
  TypeMap TypeNames;
  ValueMap ValueNames;
  TypeMap UnresolvedTypes;
  TypeList TypeStack;
  NameSet UsedNames;
  TypeSet DefinedTypes;
  ValueSet DefinedValues;
  ForwardRefMap ForwardRefs;
  bool is_inline;

public:
  inline CppWriter(std::ostream &o, const Module *M, const char* pn="llvm2cpp")
    : progname(pn), Out(o), TheModule(M), uniqueNum(0), TypeNames(),
      ValueNames(), UnresolvedTypes(), TypeStack(), is_inline(false) { }

  const Module* getModule() { return TheModule; }

  void printProgram(const std::string& fname, const std::string& modName );
  void printModule(const std::string& fname, const std::string& modName );
  void printContents(const std::string& fname, const std::string& modName );
  void printFunction(const std::string& fname, const std::string& funcName );
  void printInline(const std::string& fname, const std::string& funcName );
  void printVariable(const std::string& fname, const std::string& varName );
  void printType(const std::string& fname, const std::string& typeName );

  void error(const std::string& msg);

private:
  void printLinkageType(GlobalValue::LinkageTypes LT);
  void printCallingConv(unsigned cc);
  void printEscapedString(const std::string& str);
  void printCFP(const ConstantFP* CFP);

  std::string getCppName(const Type* val);
  inline void printCppName(const Type* val);

  std::string getCppName(const Value* val);
  inline void printCppName(const Value* val);

  bool printTypeInternal(const Type* Ty);
  inline void printType(const Type* Ty);
  void printTypes(const Module* M);

  void printConstant(const Constant *CPV);
  void printConstants(const Module* M);

  void printVariableUses(const GlobalVariable *GV);
  void printVariableHead(const GlobalVariable *GV);
  void printVariableBody(const GlobalVariable *GV);

  void printFunctionUses(const Function *F);
  void printFunctionHead(const Function *F);
  void printFunctionBody(const Function *F);
  void printInstruction(const Instruction *I, const std::string& bbname);
  std::string getOpName(Value*);

  void printModuleBody();

};

static unsigned indent_level = 0;
inline std::ostream& nl(std::ostream& Out, int delta = 0) {
  Out << "\n";
  if (delta >= 0 || indent_level >= unsigned(-delta))
    indent_level += delta;
  for (unsigned i = 0; i < indent_level; ++i) 
    Out << "  ";
  return Out;
}

inline void in() { indent_level++; }
inline void out() { if (indent_level >0) indent_level--; }

inline void
sanitize(std::string& str) {
  for (size_t i = 0; i < str.length(); ++i)
    if (!isalnum(str[i]) && str[i] != '_')
      str[i] = '_';
}

inline const char* 
getTypePrefix(const Type* Ty ) {
  const char* prefix;
  switch (Ty->getTypeID()) {
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
    case Type::OpaqueTyID:   prefix = "opaque_"; break;
    default:                 prefix = "other_"; break;
  }
  return prefix;
}

// Looks up the type in the symbol table and returns a pointer to its name or
// a null pointer if it wasn't found. Note that this isn't the same as the
// Mode::getTypeName function which will return an empty string, not a null
// pointer if the name is not found.
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

void
CppWriter::error(const std::string& msg) {
  std::cerr << progname << ": " << msg << "\n";
  exit(2);
}

// printCFP - Print a floating point constant .. very carefully :)
// This makes sure that conversion to/from floating yields the same binary
// result so that we don't lose precision.
void 
CppWriter::printCFP(const ConstantFP *CFP) {
  Out << "ConstantFP::get(";
  if (CFP->getType() == Type::DoubleTy)
    Out << "Type::DoubleTy, ";
  else
    Out << "Type::FloatTy, ";
#if HAVE_PRINTF_A
  char Buffer[100];
  sprintf(Buffer, "%A", CFP->getValue());
  if ((!strncmp(Buffer, "0x", 2) ||
       !strncmp(Buffer, "-0x", 3) ||
       !strncmp(Buffer, "+0x", 3)) &&
      (atof(Buffer) == CFP->getValue()))
    if (CFP->getType() == Type::DoubleTy)
      Out << "BitsToDouble(" << Buffer << ")";
    else
      Out << "BitsToFloat(" << Buffer << ")";
  else {
#endif
    std::string StrVal = ftostr(CFP->getValue());

    while (StrVal[0] == ' ')
      StrVal.erase(StrVal.begin());

    // Check to make sure that the stringized number is not some string like 
    // "Inf" or NaN.  Check that the string matches the "[-+]?[0-9]" regex.
    if (((StrVal[0] >= '0' && StrVal[0] <= '9') ||
        ((StrVal[0] == '-' || StrVal[0] == '+') &&
         (StrVal[1] >= '0' && StrVal[1] <= '9'))) &&
        (atof(StrVal.c_str()) == CFP->getValue()))
      if (CFP->getType() == Type::DoubleTy)
        Out <<  StrVal;
      else
        Out << StrVal;
    else if (CFP->getType() == Type::DoubleTy)
      Out << "BitsToDouble(0x" << std::hex << DoubleToBits(CFP->getValue()) 
          << std::dec << "ULL) /* " << StrVal << " */";
    else 
      Out << "BitsToFloat(0x" << std::hex << FloatToBits(CFP->getValue()) 
          << std::dec << "U) /* " << StrVal << " */";
#if HAVE_PRINTF_A
  }
#endif
  Out << ")";
}

void
CppWriter::printCallingConv(unsigned cc){
  // Print the calling convention.
  switch (cc) {
    case CallingConv::C:     Out << "CallingConv::C"; break;
    case CallingConv::CSRet: Out << "CallingConv::CSRet"; break;
    case CallingConv::Fast:  Out << "CallingConv::Fast"; break;
    case CallingConv::Cold:  Out << "CallingConv::Cold"; break;
    case CallingConv::FirstTargetCC: Out << "CallingConv::FirstTargetCC"; break;
    default:                 Out << cc; break;
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

// printEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
void 
CppWriter::printEscapedString(const std::string &Str) {
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    unsigned char C = Str[i];
    if (isprint(C) && C != '"' && C != '\\') {
      Out << C;
    } else {
      Out << "\\x"
          << (char) ((C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A'))
          << (char)(((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A'));
    }
  }
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
        error("Invalid primitive type");
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
  sanitize(name);

  // Save the name
  return TypeNames[Ty] = name;
}

void
CppWriter::printCppName(const Type* Ty)
{
  printEscapedString(getCppName(Ty));
}

std::string
CppWriter::getCppName(const Value* val) {
  std::string name;
  ValueMap::iterator I = ValueNames.find(val);
  if (I != ValueNames.end() && I->first == val)
    return  I->second;

  if (const GlobalVariable* GV = dyn_cast<GlobalVariable>(val)) {
    name = std::string("gvar_") + 
           getTypePrefix(GV->getType()->getElementType());
  } else if (const Function* F = dyn_cast<Function>(val)) {
    name = std::string("func_");
  } else if (const Constant* C = dyn_cast<Constant>(val)) {
    name = std::string("const_") + getTypePrefix(C->getType());
  } else if (const Argument* Arg = dyn_cast<Argument>(val)) {
    if (is_inline) {
      unsigned argNum = std::distance(Arg->getParent()->arg_begin(),
          Function::const_arg_iterator(Arg)) + 1;
      name = std::string("arg_") + utostr(argNum);
      NameSet::iterator NI = UsedNames.find(name);
      if (NI != UsedNames.end())
        name += std::string("_") + utostr(uniqueNum++);
      UsedNames.insert(name);
      return ValueNames[val] = name;
    } else {
      name = getTypePrefix(val->getType());
    }
  } else {
    name = getTypePrefix(val->getType());
  }
  name += (val->hasName() ? val->getName() : utostr(uniqueNum++));
  sanitize(name);
  NameSet::iterator NI = UsedNames.find(name);
  if (NI != UsedNames.end())
    name += std::string("_") + utostr(uniqueNum++);
  UsedNames.insert(name);
  return ValueNames[val] = name;
}

void
CppWriter::printCppName(const Value* val) {
  printEscapedString(getCppName(val));
}

bool
CppWriter::printTypeInternal(const Type* Ty) {
  // We don't print definitions for primitive types
  if (Ty->isPrimitiveType())
    return false;

  // If we already defined this type, we don't need to define it again.
  if (DefinedTypes.find(Ty) != DefinedTypes.end())
    return false;

  // Everything below needs the name for the type so get it now.
  std::string typeName(getCppName(Ty));

  // Search the type stack for recursion. If we find it, then generate this
  // as an OpaqueType, but make sure not to do this multiple times because
  // the type could appear in multiple places on the stack. Once the opaque
  // definition is issued, it must not be re-issued. Consequently we have to
  // check the UnresolvedTypes list as well.
  TypeList::const_iterator TI = std::find(TypeStack.begin(),TypeStack.end(),Ty);
  if (TI != TypeStack.end()) {
    TypeMap::const_iterator I = UnresolvedTypes.find(Ty);
    if (I == UnresolvedTypes.end()) {
      Out << "PATypeHolder " << typeName << "_fwd = OpaqueType::get();";
      nl(Out);
      UnresolvedTypes[Ty] = typeName;
    }
    return true;
  }

  // We're going to print a derived type which, by definition, contains other
  // types. So, push this one we're printing onto the type stack to assist with
  // recursive definitions.
  TypeStack.push_back(Ty);

  // Print the type definition
  switch (Ty->getTypeID()) {
    case Type::FunctionTyID:  {
      const FunctionType* FT = cast<FunctionType>(Ty);
      Out << "std::vector<const Type*>" << typeName << "_args;";
      nl(Out);
      FunctionType::param_iterator PI = FT->param_begin();
      FunctionType::param_iterator PE = FT->param_end();
      for (; PI != PE; ++PI) {
        const Type* argTy = static_cast<const Type*>(*PI);
        bool isForward = printTypeInternal(argTy);
        std::string argName(getCppName(argTy));
        Out << typeName << "_args.push_back(" << argName;
        if (isForward)
          Out << "_fwd";
        Out << ");";
        nl(Out);
      }
      bool isForward = printTypeInternal(FT->getReturnType());
      std::string retTypeName(getCppName(FT->getReturnType()));
      Out << "FunctionType* " << typeName << " = FunctionType::get(";
      in(); nl(Out) << "/*Result=*/" << retTypeName;
      if (isForward)
        Out << "_fwd";
      Out << ",";
      nl(Out) << "/*Params=*/" << typeName << "_args,";
      nl(Out) << "/*isVarArg=*/" << (FT->isVarArg() ? "true" : "false") << ");";
      out(); 
      nl(Out);
      break;
    }
    case Type::StructTyID: {
      const StructType* ST = cast<StructType>(Ty);
      Out << "std::vector<const Type*>" << typeName << "_fields;";
      nl(Out);
      StructType::element_iterator EI = ST->element_begin();
      StructType::element_iterator EE = ST->element_end();
      for (; EI != EE; ++EI) {
        const Type* fieldTy = static_cast<const Type*>(*EI);
        bool isForward = printTypeInternal(fieldTy);
        std::string fieldName(getCppName(fieldTy));
        Out << typeName << "_fields.push_back(" << fieldName;
        if (isForward)
          Out << "_fwd";
        Out << ");";
        nl(Out);
      }
      Out << "StructType* " << typeName << " = StructType::get("
          << typeName << "_fields);";
      nl(Out);
      break;
    }
    case Type::ArrayTyID: {
      const ArrayType* AT = cast<ArrayType>(Ty);
      const Type* ET = AT->getElementType();
      bool isForward = printTypeInternal(ET);
      std::string elemName(getCppName(ET));
      Out << "ArrayType* " << typeName << " = ArrayType::get("
          << elemName << (isForward ? "_fwd" : "") 
          << ", " << utostr(AT->getNumElements()) << ");";
      nl(Out);
      break;
    }
    case Type::PointerTyID: {
      const PointerType* PT = cast<PointerType>(Ty);
      const Type* ET = PT->getElementType();
      bool isForward = printTypeInternal(ET);
      std::string elemName(getCppName(ET));
      Out << "PointerType* " << typeName << " = PointerType::get("
          << elemName << (isForward ? "_fwd" : "") << ");";
      nl(Out);
      break;
    }
    case Type::PackedTyID: {
      const PackedType* PT = cast<PackedType>(Ty);
      const Type* ET = PT->getElementType();
      bool isForward = printTypeInternal(ET);
      std::string elemName(getCppName(ET));
      Out << "PackedType* " << typeName << " = PackedType::get("
          << elemName << (isForward ? "_fwd" : "") 
          << ", " << utostr(PT->getNumElements()) << ");";
      nl(Out);
      break;
    }
    case Type::OpaqueTyID: {
      const OpaqueType* OT = cast<OpaqueType>(Ty);
      Out << "OpaqueType* " << typeName << " = OpaqueType::get();";
      nl(Out);
      break;
    }
    default:
      error("Invalid TypeID");
  }

  // If the type had a name, make sure we recreate it.
  const std::string* progTypeName = 
    findTypeName(TheModule->getSymbolTable(),Ty);
  if (progTypeName)
    Out << "mod->addTypeName(\"" << *progTypeName << "\", " 
        << typeName << ");";
    nl(Out);

  // Pop us off the type stack
  TypeStack.pop_back();

  // Indicate that this type is now defined.
  DefinedTypes.insert(Ty);

  // Early resolve as many unresolved types as possible. Search the unresolved
  // types map for the type we just printed. Now that its definition is complete
  // we can resolve any previous references to it. This prevents a cascade of
  // unresolved types.
  TypeMap::iterator I = UnresolvedTypes.find(Ty);
  if (I != UnresolvedTypes.end()) {
    Out << "cast<OpaqueType>(" << I->second 
        << "_fwd.get())->refineAbstractTypeTo(" << I->second << ");";
    nl(Out);
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
    Out << ">(" << I->second << "_fwd.get());";
    nl(Out); nl(Out);
    UnresolvedTypes.erase(I);
  }

  // Finally, separate the type definition from other with a newline.
  nl(Out);

  // We weren't a recursive type
  return false;
}

// Prints a type definition. Returns true if it could not resolve all the types
// in the definition but had to use a forward reference.
void
CppWriter::printType(const Type* Ty) {
  assert(TypeStack.empty());
  TypeStack.clear();
  printTypeInternal(Ty);
  assert(TypeStack.empty());
}

void
CppWriter::printTypes(const Module* M) {

  // Walk the symbol table and print out all its types
  const SymbolTable& symtab = M->getSymbolTable();
  for (SymbolTable::type_const_iterator TI = symtab.type_begin(), 
       TE = symtab.type_end(); TI != TE; ++TI) {

    // For primitive types and types already defined, just add a name
    TypeMap::const_iterator TNI = TypeNames.find(TI->second);
    if (TI->second->isPrimitiveType() || TNI != TypeNames.end()) {
      Out << "mod->addTypeName(\"";
      printEscapedString(TI->first);
      Out << "\", " << getCppName(TI->second) << ");";
      nl(Out);
    // For everything else, define the type
    } else {
      printType(TI->second);
    }
  }

  // Add all of the global variables to the value table...
  for (Module::const_global_iterator I = TheModule->global_begin(), 
       E = TheModule->global_end(); I != E; ++I) {
    if (I->hasInitializer())
      printType(I->getInitializer()->getType());
    printType(I->getType());
  }

  // Add all the functions to the table
  for (Module::const_iterator FI = TheModule->begin(), FE = TheModule->end();
       FI != FE; ++FI) {
    printType(FI->getReturnType());
    printType(FI->getFunctionType());
    // Add all the function arguments
    for(Function::const_arg_iterator AI = FI->arg_begin(),
        AE = FI->arg_end(); AI != AE; ++AI) {
      printType(AI->getType());
    }

    // Add all of the basic blocks and instructions
    for (Function::const_iterator BB = FI->begin(),
         E = FI->end(); BB != E; ++BB) {
      printType(BB->getType());
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; 
           ++I) {
        printType(I->getType());
        for (unsigned i = 0; i < I->getNumOperands(); ++i)
          printType(I->getOperand(i)->getType());
      }
    }
  }
}


// printConstant - Print out a constant pool entry...
void CppWriter::printConstant(const Constant *CV) {
  // First, if the constant is actually a GlobalValue (variable or function) or
  // its already in the constant list then we've printed it already and we can
  // just return.
  if (isa<GlobalValue>(CV) || ValueNames.find(CV) != ValueNames.end())
    return;

  std::string constName(getCppName(CV));
  std::string typeName(getCppName(CV->getType()));
  if (CV->isNullValue()) {
    Out << "Constant* " << constName << " = Constant::getNullValue("
        << typeName << ");";
    nl(Out);
    return;
  }
  if (isa<GlobalValue>(CV)) {
    // Skip variables and functions, we emit them elsewhere
    return;
  }
  if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    Out << "ConstantBool* " << constName << " = ConstantBool::get(" 
        << (CB == ConstantBool::True ? "true" : "false")
        << ");";
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV)) {
    Out << "ConstantSInt* " << constName << " = ConstantSInt::get(" 
        << typeName << ", " << CI->getValue() << ");";
  } else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV)) {
    Out << "ConstantUInt* " << constName << " = ConstantUInt::get(" 
        << typeName << ", " << CI->getValue() << ");";
  } else if (isa<ConstantAggregateZero>(CV)) {
    Out << "ConstantAggregateZero* " << constName 
        << " = ConstantAggregateZero::get(" << typeName << ");";
  } else if (isa<ConstantPointerNull>(CV)) {
    Out << "ConstantPointerNull* " << constName 
        << " = ConstanPointerNull::get(" << typeName << ");";
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    Out << "ConstantFP* " << constName << " = ";
    printCFP(CFP);
    Out << ";";
  } else if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
    if (CA->isString() && CA->getType()->getElementType() == Type::SByteTy) {
      Out << "Constant* " << constName << " = ConstantArray::get(\"";
      printEscapedString(CA->getAsString());
      // Determine if we want null termination or not.
      if (CA->getType()->getNumElements() <= CA->getAsString().length())
        Out << "\", false";// No null terminator
      else
        Out << "\", true"; // Indicate that the null terminator should be added.
      Out << ");";
    } else { 
      Out << "std::vector<Constant*> " << constName << "_elems;";
      nl(Out);
      unsigned N = CA->getNumOperands();
      for (unsigned i = 0; i < N; ++i) {
        printConstant(CA->getOperand(i)); // recurse to print operands
        Out << constName << "_elems.push_back("
            << getCppName(CA->getOperand(i)) << ");";
        nl(Out);
      }
      Out << "Constant* " << constName << " = ConstantArray::get(" 
          << typeName << ", " << constName << "_elems);";
    }
  } else if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(CV)) {
    Out << "std::vector<Constant*> " << constName << "_fields;";
    nl(Out);
    unsigned N = CS->getNumOperands();
    for (unsigned i = 0; i < N; i++) {
      printConstant(CS->getOperand(i));
      Out << constName << "_fields.push_back("
          << getCppName(CS->getOperand(i)) << ");";
      nl(Out);
    }
    Out << "Constant* " << constName << " = ConstantStruct::get(" 
        << typeName << ", " << constName << "_fields);";
  } else if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(CV)) {
    Out << "std::vector<Constant*> " << constName << "_elems;";
    nl(Out);
    unsigned N = CP->getNumOperands();
    for (unsigned i = 0; i < N; ++i) {
      printConstant(CP->getOperand(i));
      Out << constName << "_elems.push_back("
          << getCppName(CP->getOperand(i)) << ");";
      nl(Out);
    }
    Out << "Constant* " << constName << " = ConstantPacked::get(" 
        << typeName << ", " << constName << "_elems);";
  } else if (isa<UndefValue>(CV)) {
    Out << "UndefValue* " << constName << " = UndefValue::get(" 
        << typeName << ");";
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      Out << "std::vector<Constant*> " << constName << "_indices;";
      nl(Out);
      printConstant(CE->getOperand(0));
      for (unsigned i = 1; i < CE->getNumOperands(); ++i ) {
        printConstant(CE->getOperand(i));
        Out << constName << "_indices.push_back("
            << getCppName(CE->getOperand(i)) << ");";
        nl(Out);
      }
      Out << "Constant* " << constName 
          << " = ConstantExpr::getGetElementPtr(" 
          << getCppName(CE->getOperand(0)) << ", " 
          << constName << "_indices);";
    } else if (CE->getOpcode() == Instruction::Cast) {
      printConstant(CE->getOperand(0));
      Out << "Constant* " << constName << " = ConstantExpr::getCast(";
      Out << getCppName(CE->getOperand(0)) << ", " << getCppName(CE->getType())
          << ");";
    } else {
      unsigned N = CE->getNumOperands();
      for (unsigned i = 0; i < N; ++i ) {
        printConstant(CE->getOperand(i));
      }
      Out << "Constant* " << constName << " = ConstantExpr::";
      switch (CE->getOpcode()) {
        case Instruction::Add:    Out << "getAdd";  break;
        case Instruction::Sub:    Out << "getSub"; break;
        case Instruction::Mul:    Out << "getMul"; break;
        case Instruction::Div:    Out << "getDiv"; break;
        case Instruction::Rem:    Out << "getRem"; break;
        case Instruction::And:    Out << "getAnd"; break;
        case Instruction::Or:     Out << "getOr"; break;
        case Instruction::Xor:    Out << "getXor"; break;
        case Instruction::SetEQ:  Out << "getSetEQ"; break;
        case Instruction::SetNE:  Out << "getSetNE"; break;
        case Instruction::SetLE:  Out << "getSetLE"; break;
        case Instruction::SetGE:  Out << "getSetGE"; break;
        case Instruction::SetLT:  Out << "getSetLT"; break;
        case Instruction::SetGT:  Out << "getSetGT"; break;
        case Instruction::Shl:    Out << "getShl"; break;
        case Instruction::Shr:    Out << "getShr"; break;
        case Instruction::Select: Out << "getSelect"; break;
        case Instruction::ExtractElement: Out << "getExtractElement"; break;
        case Instruction::InsertElement:  Out << "getInsertElement"; break;
        case Instruction::ShuffleVector:  Out << "getShuffleVector"; break;
        default:
          error("Invalid constant expression");
          break;
      }
      Out << getCppName(CE->getOperand(0));
      for (unsigned i = 1; i < CE->getNumOperands(); ++i) 
        Out << ", " << getCppName(CE->getOperand(i));
      Out << ");";
    }
  } else {
    error("Bad Constant");
    Out << "Constant* " << constName << " = 0; ";
  }
  nl(Out);
}

void
CppWriter::printConstants(const Module* M) {
  // Traverse all the global variables looking for constant initializers
  for (Module::const_global_iterator I = TheModule->global_begin(), 
       E = TheModule->global_end(); I != E; ++I)
    if (I->hasInitializer())
      printConstant(I->getInitializer());

  // Traverse the LLVM functions looking for constants
  for (Module::const_iterator FI = TheModule->begin(), FE = TheModule->end();
       FI != FE; ++FI) {
    // Add all of the basic blocks and instructions
    for (Function::const_iterator BB = FI->begin(),
         E = FI->end(); BB != E; ++BB) {
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; 
           ++I) {
        for (unsigned i = 0; i < I->getNumOperands(); ++i) {
          if (Constant* C = dyn_cast<Constant>(I->getOperand(i))) {
            printConstant(C);
          }
        }
      }
    }
  }
}

void CppWriter::printVariableUses(const GlobalVariable *GV) {
  nl(Out) << "// Type Definitions";
  nl(Out);
  printType(GV->getType());
  if (GV->hasInitializer()) {
    Constant* Init = GV->getInitializer();
    printType(Init->getType());
    if (Function* F = dyn_cast<Function>(Init)) {
      nl(Out)<< "/ Function Declarations"; nl(Out);
      printFunctionHead(F);
    } else if (GlobalVariable* gv = dyn_cast<GlobalVariable>(Init)) {
      nl(Out) << "// Global Variable Declarations"; nl(Out);
      printVariableHead(gv);
    } else  {
      nl(Out) << "// Constant Definitions"; nl(Out);
      printConstant(gv);
    }
    if (GlobalVariable* gv = dyn_cast<GlobalVariable>(Init)) {
      nl(Out) << "// Global Variable Definitions"; nl(Out);
      printVariableBody(gv);
    }
  }
}

void CppWriter::printVariableHead(const GlobalVariable *GV) {
  nl(Out) << "GlobalVariable* " << getCppName(GV);
  if (is_inline) {
     Out << " = mod->getGlobalVariable(";
     printEscapedString(GV->getName());
     Out << ", " << getCppName(GV->getType()->getElementType()) << ",true)";
     nl(Out) << "if (!" << getCppName(GV) << ") {";
     in(); nl(Out) << getCppName(GV);
  }
  Out << " = new GlobalVariable(";
  nl(Out) << "/*Type=*/";
  printCppName(GV->getType()->getElementType());
  Out << ",";
  nl(Out) << "/*isConstant=*/" << (GV->isConstant()?"true":"false");
  Out << ",";
  nl(Out) << "/*Linkage=*/";
  printLinkageType(GV->getLinkage());
  Out << ",";
  nl(Out) << "/*Initializer=*/0, ";
  if (GV->hasInitializer()) {
    Out << "// has initializer, specified below";
  }
  nl(Out) << "/*Name=*/\"";
  printEscapedString(GV->getName());
  Out << "\",";
  nl(Out) << "mod);";
  nl(Out);

  if (GV->hasSection()) {
    printCppName(GV);
    Out << "->setSection(\"";
    printEscapedString(GV->getSection());
    Out << "\");";
    nl(Out);
  }
  if (GV->getAlignment()) {
    printCppName(GV);
    Out << "->setAlignment(" << utostr(GV->getAlignment()) << ");";
    nl(Out);
  };
  if (is_inline) {
    out(); Out << "}"; nl(Out);
  }
}

void 
CppWriter::printVariableBody(const GlobalVariable *GV) {
  if (GV->hasInitializer()) {
    printCppName(GV);
    Out << "->setInitializer(";
    //if (!isa<GlobalValue(GV->getInitializer()))
    //else 
      Out << getCppName(GV->getInitializer()) << ");";
      nl(Out);
  }
}

std::string
CppWriter::getOpName(Value* V) {
  if (!isa<Instruction>(V) || DefinedValues.find(V) != DefinedValues.end())
    return getCppName(V);

  // See if its alread in the map of forward references, if so just return the
  // name we already set up for it
  ForwardRefMap::const_iterator I = ForwardRefs.find(V);
  if (I != ForwardRefs.end())
    return I->second;

  // This is a new forward reference. Generate a unique name for it
  std::string result(std::string("fwdref_") + utostr(uniqueNum++));

  // Yes, this is a hack. An Argument is the smallest instantiable value that
  // we can make as a placeholder for the real value. We'll replace these
  // Argument instances later.
  Out << "Argument* " << result << " = new Argument(" 
      << getCppName(V->getType()) << ");";
  nl(Out);
  ForwardRefs[V] = result;
  return result;
}

// printInstruction - This member is called for each Instruction in a function.
void 
CppWriter::printInstruction(const Instruction *I, const std::string& bbname) {
  std::string iName(getCppName(I));

  // Before we emit this instruction, we need to take care of generating any
  // forward references. So, we get the names of all the operands in advance
  std::string* opNames = new std::string[I->getNumOperands()];
  for (unsigned i = 0; i < I->getNumOperands(); i++) {
    opNames[i] = getOpName(I->getOperand(i));
  }

  switch (I->getOpcode()) {
    case Instruction::Ret: {
      const ReturnInst* ret =  cast<ReturnInst>(I);
      Out << "ReturnInst* " << iName << " = new ReturnInst("
          << (ret->getReturnValue() ? opNames[0] + ", " : "") << bbname << ");";
      break;
    }
    case Instruction::Br: {
      const BranchInst* br = cast<BranchInst>(I);
      Out << "BranchInst* " << iName << " = new BranchInst(" ;
      if (br->getNumOperands() == 3 ) {
        Out << opNames[0] << ", " 
            << opNames[1] << ", "
            << opNames[2] << ", ";

      } else if (br->getNumOperands() == 1) {
        Out << opNames[0] << ", ";
      } else {
        error("Branch with 2 operands?");
      }
      Out << bbname << ");";
      break;
    }
    case Instruction::Switch: {
      const SwitchInst* sw = cast<SwitchInst>(I);
      Out << "SwitchInst* " << iName << " = new SwitchInst("
          << opNames[0] << ", "
          << opNames[1] << ", "
          << sw->getNumCases() << ", " << bbname << ");";
      nl(Out);
      for (unsigned i = 2; i < sw->getNumOperands(); i += 2 ) {
        Out << iName << "->addCase(" 
            << opNames[i] << ", "
            << opNames[i+1] << ");";
        nl(Out);
      }
      break;
    }
    case Instruction::Invoke: {
      const InvokeInst* inv = cast<InvokeInst>(I);
      Out << "std::vector<Value*> " << iName << "_params;";
      nl(Out);
      for (unsigned i = 3; i < inv->getNumOperands(); ++i) {
        Out << iName << "_params.push_back("
            << opNames[i] << ");";
        nl(Out);
      }
      Out << "InvokeInst* " << iName << " = new InvokeInst("
          << opNames[0] << ", "
          << opNames[1] << ", "
          << opNames[2] << ", "
          << iName << "_params, \"";
      printEscapedString(inv->getName());
      Out << "\", " << bbname << ");";
      nl(Out) << iName << "->setCallingConv(";
      printCallingConv(inv->getCallingConv());
      Out << ");";
      break;
    }
    case Instruction::Unwind: {
      Out << "UnwindInst* " << iName << " = new UnwindInst("
          << bbname << ");";
      break;
    }
    case Instruction::Unreachable:{
      Out << "UnreachableInst* " << iName << " = new UnreachableInst("
          << bbname << ");";
      break;
    }
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::Div:
    case Instruction::Rem:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::Shl: 
    case Instruction::Shr:{
      Out << "BinaryOperator* " << iName << " = BinaryOperator::create(";
      switch (I->getOpcode()) {
        case Instruction::Add: Out << "Instruction::Add"; break;
        case Instruction::Sub: Out << "Instruction::Sub"; break;
        case Instruction::Mul: Out << "Instruction::Mul"; break;
        case Instruction::Div: Out << "Instruction::Div"; break;
        case Instruction::Rem: Out << "Instruction::Rem"; break;
        case Instruction::And: Out << "Instruction::And"; break;
        case Instruction::Or:  Out << "Instruction::Or";  break;
        case Instruction::Xor: Out << "Instruction::Xor"; break;
        case Instruction::Shl: Out << "Instruction::Shl"; break;
        case Instruction::Shr: Out << "Instruction::Shr"; break;
        default: Out << "Instruction::BadOpCode"; break;
      }
      Out << ", " << opNames[0] << ", " << opNames[1] << ", \"";
      printEscapedString(I->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::SetEQ:
    case Instruction::SetNE:
    case Instruction::SetLE:
    case Instruction::SetGE:
    case Instruction::SetLT:
    case Instruction::SetGT: {
      Out << "SetCondInst* " << iName << " = new SetCondInst(";
      switch (I->getOpcode()) {
        case Instruction::SetEQ: Out << "Instruction::SetEQ"; break;
        case Instruction::SetNE: Out << "Instruction::SetNE"; break;
        case Instruction::SetLE: Out << "Instruction::SetLE"; break;
        case Instruction::SetGE: Out << "Instruction::SetGE"; break;
        case Instruction::SetLT: Out << "Instruction::SetLT"; break;
        case Instruction::SetGT: Out << "Instruction::SetGT"; break;
        default: Out << "Instruction::BadOpCode"; break;
      }
      Out << ", " << opNames[0] << ", " << opNames[1] << ", \"";
      printEscapedString(I->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::Malloc: {
      const MallocInst* mallocI = cast<MallocInst>(I);
      Out << "MallocInst* " << iName << " = new MallocInst("
          << getCppName(mallocI->getAllocatedType()) << ", ";
      if (mallocI->isArrayAllocation())
        Out << opNames[0] << ", " ;
      Out << "\"";
      printEscapedString(mallocI->getName());
      Out << "\", " << bbname << ");";
      if (mallocI->getAlignment())
        nl(Out) << iName << "->setAlignment(" 
            << mallocI->getAlignment() << ");";
      break;
    }
    case Instruction::Free: {
      Out << "FreeInst* " << iName << " = new FreeInst("
          << getCppName(I->getOperand(0)) << ", " << bbname << ");";
      break;
    }
    case Instruction::Alloca: {
      const AllocaInst* allocaI = cast<AllocaInst>(I);
      Out << "AllocaInst* " << iName << " = new AllocaInst("
          << getCppName(allocaI->getAllocatedType()) << ", ";
      if (allocaI->isArrayAllocation())
        Out << opNames[0] << ", ";
      Out << "\"";
      printEscapedString(allocaI->getName());
      Out << "\", " << bbname << ");";
      if (allocaI->getAlignment())
        nl(Out) << iName << "->setAlignment(" 
            << allocaI->getAlignment() << ");";
      break;
    }
    case Instruction::Load:{
      const LoadInst* load = cast<LoadInst>(I);
      Out << "LoadInst* " << iName << " = new LoadInst(" 
          << opNames[0] << ", \"";
      printEscapedString(load->getName());
      Out << "\", " << (load->isVolatile() ? "true" : "false" )
          << ", " << bbname << ");";
      break;
    }
    case Instruction::Store: {
      const StoreInst* store = cast<StoreInst>(I);
      Out << "StoreInst* " << iName << " = new StoreInst(" 
          << opNames[0] << ", "
          << opNames[1] << ", "
          << (store->isVolatile() ? "true" : "false") 
          << ", " << bbname << ");";
      break;
    }
    case Instruction::GetElementPtr: {
      const GetElementPtrInst* gep = cast<GetElementPtrInst>(I);
      if (gep->getNumOperands() <= 2) {
        Out << "GetElementPtrInst* " << iName << " = new GetElementPtrInst("
            << opNames[0]; 
        if (gep->getNumOperands() == 2)
          Out << ", " << opNames[1];
      } else {
        Out << "std::vector<Value*> " << iName << "_indices;";
        nl(Out);
        for (unsigned i = 1; i < gep->getNumOperands(); ++i ) {
          Out << iName << "_indices.push_back("
              << opNames[i] << ");";
          nl(Out);
        }
        Out << "Instruction* " << iName << " = new GetElementPtrInst(" 
            << opNames[0] << ", " << iName << "_indices";
      }
      Out << ", \"";
      printEscapedString(gep->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::PHI: {
      const PHINode* phi = cast<PHINode>(I);

      Out << "PHINode* " << iName << " = new PHINode("
          << getCppName(phi->getType()) << ", \"";
      printEscapedString(phi->getName());
      Out << "\", " << bbname << ");";
      nl(Out) << iName << "->reserveOperandSpace(" 
        << phi->getNumIncomingValues()
          << ");";
      nl(Out);
      for (unsigned i = 0; i < phi->getNumOperands(); i+=2) {
        Out << iName << "->addIncoming("
            << opNames[i] << ", " << opNames[i+1] << ");";
        nl(Out);
      }
      break;
    }
    case Instruction::Cast: {
      const CastInst* cst = cast<CastInst>(I);
      Out << "CastInst* " << iName << " = new CastInst("
          << opNames[0] << ", "
          << getCppName(cst->getType()) << ", \"";
      printEscapedString(cst->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::Call:{
      const CallInst* call = cast<CallInst>(I);
      if (InlineAsm* ila = dyn_cast<InlineAsm>(call->getOperand(0))) {
        Out << "InlineAsm* " << getCppName(ila) << " = InlineAsm::get("
            << getCppName(ila->getFunctionType()) << ", \""
            << ila->getAsmString() << "\", \""
            << ila->getConstraintString() << "\","
            << (ila->hasSideEffects() ? "true" : "false") << ");";
        nl(Out);
      }
      if (call->getNumOperands() > 3) {
        Out << "std::vector<Value*> " << iName << "_params;";
        nl(Out);
        for (unsigned i = 1; i < call->getNumOperands(); ++i) {
          Out << iName << "_params.push_back(" << opNames[i] << ");";
          nl(Out);
        }
        Out << "CallInst* " << iName << " = new CallInst("
            << opNames[0] << ", " << iName << "_params, \"";
      } else if (call->getNumOperands() == 3) {
        Out << "CallInst* " << iName << " = new CallInst("
            << opNames[0] << ", " << opNames[1] << ", " << opNames[2] << ", \"";
      } else if (call->getNumOperands() == 2) {
        Out << "CallInst* " << iName << " = new CallInst("
            << opNames[0] << ", " << opNames[1] << ", \"";
      } else {
        Out << "CallInst* " << iName << " = new CallInst(" << opNames[0] 
            << ", \"";
      }
      printEscapedString(call->getName());
      Out << "\", " << bbname << ");";
      nl(Out) << iName << "->setCallingConv(";
      printCallingConv(call->getCallingConv());
      Out << ");";
      nl(Out) << iName << "->setTailCall(" 
          << (call->isTailCall() ? "true":"false");
      Out << ");";
      break;
    }
    case Instruction::Select: {
      const SelectInst* sel = cast<SelectInst>(I);
      Out << "SelectInst* " << getCppName(sel) << " = new SelectInst(";
      Out << opNames[0] << ", " << opNames[1] << ", " << opNames[2] << ", \"";
      printEscapedString(sel->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::UserOp1:
      /// FALL THROUGH
    case Instruction::UserOp2: {
      /// FIXME: What should be done here?
      break;
    }
    case Instruction::VAArg: {
      const VAArgInst* va = cast<VAArgInst>(I);
      Out << "VAArgInst* " << getCppName(va) << " = new VAArgInst("
          << opNames[0] << ", " << getCppName(va->getType()) << ", \"";
      printEscapedString(va->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::ExtractElement: {
      const ExtractElementInst* eei = cast<ExtractElementInst>(I);
      Out << "ExtractElementInst* " << getCppName(eei) 
          << " = new ExtractElementInst(" << opNames[0]
          << ", " << opNames[1] << ", \"";
      printEscapedString(eei->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::InsertElement: {
      const InsertElementInst* iei = cast<InsertElementInst>(I);
      Out << "InsertElementInst* " << getCppName(iei) 
          << " = new InsertElementInst(" << opNames[0]
          << ", " << opNames[1] << ", " << opNames[2] << ", \"";
      printEscapedString(iei->getName());
      Out << "\", " << bbname << ");";
      break;
    }
    case Instruction::ShuffleVector: {
      const ShuffleVectorInst* svi = cast<ShuffleVectorInst>(I);
      Out << "ShuffleVectorInst* " << getCppName(svi) 
          << " = new ShuffleVectorInst(" << opNames[0]
          << ", " << opNames[1] << ", " << opNames[2] << ", \"";
      printEscapedString(svi->getName());
      Out << "\", " << bbname << ");";
      break;
    }
  }
  DefinedValues.insert(I);
  nl(Out);
  delete [] opNames;
}

// Print out the types, constants and declarations needed by one function
void CppWriter::printFunctionUses(const Function* F) {

  nl(Out) << "// Type Definitions"; nl(Out);
  if (!is_inline) {
    // Print the function's return type
    printType(F->getReturnType());

    // Print the function's function type
    printType(F->getFunctionType());

    // Print the types of each of the function's arguments
    for(Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end(); 
        AI != AE; ++AI) {
      printType(AI->getType());
    }
  }

  // Print type definitions for every type referenced by an instruction and
  // make a note of any global values or constants that are referenced
  std::vector<GlobalValue*> gvs;
  std::vector<Constant*> consts;
  for (Function::const_iterator BB = F->begin(), BE = F->end(); BB != BE; ++BB){
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); 
         I != E; ++I) {
      // Print the type of the instruction itself
      printType(I->getType());

      // Print the type of each of the instruction's operands
      for (unsigned i = 0; i < I->getNumOperands(); ++i) {
        Value* operand = I->getOperand(i);
        printType(operand->getType());
        if (GlobalValue* GV = dyn_cast<GlobalValue>(operand))
          gvs.push_back(GV);
        else if (Constant* C = dyn_cast<Constant>(operand))
          consts.push_back(C);
      }
    }
  }

  // Print the function declarations for any functions encountered
  nl(Out) << "// Function Declarations"; nl(Out);
  for (std::vector<GlobalValue*>::iterator I = gvs.begin(), E = gvs.end();
       I != E; ++I) {
    if (Function* Fun = dyn_cast<Function>(*I)) {
      if (!is_inline || Fun != F)
        printFunctionHead(Fun);
    }
  }

  // Print the global variable declarations for any variables encountered
  nl(Out) << "// Global Variable Declarations"; nl(Out);
  for (std::vector<GlobalValue*>::iterator I = gvs.begin(), E = gvs.end();
       I != E; ++I) {
    if (GlobalVariable* F = dyn_cast<GlobalVariable>(*I))
      printVariableHead(F);
  }

  // Print the constants found
  nl(Out) << "// Constant Definitions"; nl(Out);
  for (std::vector<Constant*>::iterator I = consts.begin(), E = consts.end();
       I != E; ++I) {
      printConstant(*I);
  }

  // Process the global variables definitions now that all the constants have
  // been emitted. These definitions just couple the gvars with their constant
  // initializers.
  nl(Out) << "// Global Variable Definitions"; nl(Out);
  for (std::vector<GlobalValue*>::iterator I = gvs.begin(), E = gvs.end();
       I != E; ++I) {
    if (GlobalVariable* GV = dyn_cast<GlobalVariable>(*I))
      printVariableBody(GV);
  }
}

void CppWriter::printFunctionHead(const Function* F) {
  nl(Out) << "Function* " << getCppName(F); 
  if (is_inline) {
    Out << " = mod->getFunction(\"";
    printEscapedString(F->getName());
    Out << "\", " << getCppName(F->getFunctionType()) << ");";
    nl(Out) << "if (!" << getCppName(F) << ") {";
    nl(Out) << getCppName(F);
  }
  Out<< " = new Function(";
  nl(Out,1) << "/*Type=*/" << getCppName(F->getFunctionType()) << ",";
  nl(Out) << "/*Linkage=*/";
  printLinkageType(F->getLinkage());
  Out << ",";
  nl(Out) << "/*Name=*/\"";
  printEscapedString(F->getName());
  Out << "\", mod); " << (F->isExternal()? "// (external, no body)" : "");
  nl(Out,-1);
  printCppName(F);
  Out << "->setCallingConv(";
  printCallingConv(F->getCallingConv());
  Out << ");";
  nl(Out);
  if (F->hasSection()) {
    printCppName(F);
    Out << "->setSection(\"" << F->getSection() << "\");";
    nl(Out);
  }
  if (F->getAlignment()) {
    printCppName(F);
    Out << "->setAlignment(" << F->getAlignment() << ");";
    nl(Out);
  }
  if (is_inline) {
    Out << "}";
    nl(Out);
  }
}

void CppWriter::printFunctionBody(const Function *F) {
  if (F->isExternal())
    return; // external functions have no bodies.

  // Clear the DefinedValues and ForwardRefs maps because we can't have 
  // cross-function forward refs
  ForwardRefs.clear();
  DefinedValues.clear();

  // Create all the argument values
  if (!is_inline) {
    if (!F->arg_empty()) {
      Out << "Function::arg_iterator args = " << getCppName(F) 
          << "->arg_begin();";
      nl(Out);
    }
    for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
         AI != AE; ++AI) {
      Out << "Value* " << getCppName(AI) << " = args++;";
      nl(Out);
      if (AI->hasName()) {
        Out << getCppName(AI) << "->setName(\"" << AI->getName() << "\");";
        nl(Out);
      }
    }
  }

  // Create all the basic blocks
  nl(Out);
  for (Function::const_iterator BI = F->begin(), BE = F->end(); 
       BI != BE; ++BI) {
    std::string bbname(getCppName(BI));
    Out << "BasicBlock* " << bbname << " = new BasicBlock(\"";
    if (BI->hasName())
      printEscapedString(BI->getName());
    Out << "\"," << getCppName(BI->getParent()) << ",0);";
    nl(Out);
  }

  // Output all of its basic blocks... for the function
  for (Function::const_iterator BI = F->begin(), BE = F->end(); 
       BI != BE; ++BI) {
    std::string bbname(getCppName(BI));
    nl(Out) << "// Block " << BI->getName() << " (" << bbname << ")";
    nl(Out);

    // Output all of the instructions in the basic block...
    for (BasicBlock::const_iterator I = BI->begin(), E = BI->end(); 
         I != E; ++I) {
      printInstruction(I,bbname);
    }
  }

  // Loop over the ForwardRefs and resolve them now that all instructions
  // are generated.
  if (!ForwardRefs.empty()) {
    nl(Out) << "// Resolve Forward References";
    nl(Out);
  }
  
  while (!ForwardRefs.empty()) {
    ForwardRefMap::iterator I = ForwardRefs.begin();
    Out << I->second << "->replaceAllUsesWith(" 
        << getCppName(I->first) << "); delete " << I->second << ";";
    nl(Out);
    ForwardRefs.erase(I);
  }
}

void CppWriter::printInline(const std::string& fname, const std::string& func) {
  const Function* F = TheModule->getNamedFunction(func);
  if (!F) {
    error(std::string("Function '") + func + "' not found in input module");
    return;
  }
  if (F->isExternal()) {
    error(std::string("Function '") + func + "' is external!");
    return;
  }
  nl(Out) << "BasicBlock* " << fname << "(Module* mod, Function *" 
      << getCppName(F);
  unsigned arg_count = 1;
  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI) {
    Out << ", Value* arg_" << arg_count;
  }
  Out << ") {";
  nl(Out);
  is_inline = true;
  printFunctionUses(F);
  printFunctionBody(F);
  is_inline = false;
  Out << "return " << getCppName(F->begin()) << ";";
  nl(Out) << "}";
  nl(Out);
}

void CppWriter::printModuleBody() {
  // Print out all the type definitions
  nl(Out) << "// Type Definitions"; nl(Out);
  printTypes(TheModule);

  // Functions can call each other and global variables can reference them so 
  // define all the functions first before emitting their function bodies.
  nl(Out) << "// Function Declarations"; nl(Out);
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end(); 
       I != E; ++I)
    printFunctionHead(I);

  // Process the global variables declarations. We can't initialze them until
  // after the constants are printed so just print a header for each global
  nl(Out) << "// Global Variable Declarations\n"; nl(Out);
  for (Module::const_global_iterator I = TheModule->global_begin(), 
       E = TheModule->global_end(); I != E; ++I) {
    printVariableHead(I);
  }

  // Print out all the constants definitions. Constants don't recurse except
  // through GlobalValues. All GlobalValues have been declared at this point
  // so we can proceed to generate the constants.
  nl(Out) << "// Constant Definitions"; nl(Out);
  printConstants(TheModule);

  // Process the global variables definitions now that all the constants have
  // been emitted. These definitions just couple the gvars with their constant
  // initializers.
  nl(Out) << "// Global Variable Definitions"; nl(Out);
  for (Module::const_global_iterator I = TheModule->global_begin(), 
       E = TheModule->global_end(); I != E; ++I) {
    printVariableBody(I);
  }

  // Finally, we can safely put out all of the function bodies.
  nl(Out) << "// Function Definitions"; nl(Out);
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end(); 
       I != E; ++I) {
    if (!I->isExternal()) {
      nl(Out) << "// Function: " << I->getName() << " (" << getCppName(I) 
          << ")";
      nl(Out) << "{";
      nl(Out,1);
      printFunctionBody(I);
      nl(Out,-1) << "}";
      nl(Out);
    }
  }
}

void CppWriter::printProgram(
  const std::string& fname, 
  const std::string& mName
) {
  Out << "#include <llvm/Module.h>\n";
  Out << "#include <llvm/DerivedTypes.h>\n";
  Out << "#include <llvm/Constants.h>\n";
  Out << "#include <llvm/GlobalVariable.h>\n";
  Out << "#include <llvm/Function.h>\n";
  Out << "#include <llvm/CallingConv.h>\n";
  Out << "#include <llvm/BasicBlock.h>\n";
  Out << "#include <llvm/Instructions.h>\n";
  Out << "#include <llvm/InlineAsm.h>\n";
  Out << "#include <llvm/Support/MathExtras.h>\n";
  Out << "#include <llvm/Pass.h>\n";
  Out << "#include <llvm/PassManager.h>\n";
  Out << "#include <llvm/Analysis/Verifier.h>\n";
  Out << "#include <llvm/Assembly/PrintModulePass.h>\n";
  Out << "#include <algorithm>\n";
  Out << "#include <iostream>\n\n";
  Out << "using namespace llvm;\n\n";
  Out << "Module* " << fname << "();\n\n";
  Out << "int main(int argc, char**argv) {\n";
  Out << "  Module* Mod = makeLLVMModule();\n";
  Out << "  verifyModule(*Mod, PrintMessageAction);\n";
  Out << "  std::cerr.flush();\n";
  Out << "  std::cout.flush();\n";
  Out << "  PassManager PM;\n";
  Out << "  PM.add(new PrintModulePass(&std::cout));\n";
  Out << "  PM.run(*Mod);\n";
  Out << "  return 0;\n";
  Out << "}\n\n";
  printModule(fname,mName);
}

void CppWriter::printModule(
  const std::string& fname, 
  const std::string& mName
) {
  nl(Out) << "Module* " << fname << "() {";
  nl(Out,1) << "// Module Construction";
  nl(Out) << "Module* mod = new Module(\"" << mName << "\");"; 
  nl(Out) << "mod->setEndianness(";
  switch (TheModule->getEndianness()) {
    case Module::LittleEndian: Out << "Module::LittleEndian);"; break;
    case Module::BigEndian:    Out << "Module::BigEndian);";    break;
    case Module::AnyEndianness:Out << "Module::AnyEndianness);";  break;
  }
  nl(Out) << "mod->setPointerSize(";
  switch (TheModule->getPointerSize()) {
    case Module::Pointer32:      Out << "Module::Pointer32);"; break;
    case Module::Pointer64:      Out << "Module::Pointer64);"; break;
    case Module::AnyPointerSize: Out << "Module::AnyPointerSize);"; break;
  }
  nl(Out);
  if (!TheModule->getTargetTriple().empty()) {
    Out << "mod->setTargetTriple(\"" << TheModule->getTargetTriple() 
        << "\");";
    nl(Out);
  }

  if (!TheModule->getModuleInlineAsm().empty()) {
    Out << "mod->setModuleInlineAsm(\"";
    printEscapedString(TheModule->getModuleInlineAsm());
    Out << "\");";
    nl(Out);
  }
  
  // Loop over the dependent libraries and emit them.
  Module::lib_iterator LI = TheModule->lib_begin();
  Module::lib_iterator LE = TheModule->lib_end();
  while (LI != LE) {
    Out << "mod->addLibrary(\"" << *LI << "\");";
    nl(Out);
    ++LI;
  }
  printModuleBody();
  nl(Out) << "return mod;";
  nl(Out,-1) << "}";
  nl(Out);
}

void CppWriter::printContents(
  const std::string& fname, // Name of generated function
  const std::string& mName // Name of module generated module
) {
  Out << "\nModule* " << fname << "(Module *mod) {\n";
  Out << "\nmod->setModuleIdentifier(\"" << mName << "\");\n";
  printModuleBody();
  Out << "\nreturn mod;\n";
  Out << "\n}\n";
}

void CppWriter::printFunction(
  const std::string& fname, // Name of generated function
  const std::string& funcName // Name of function to generate
) {
  const Function* F = TheModule->getNamedFunction(funcName);
  if (!F) {
    error(std::string("Function '") + funcName + "' not found in input module");
    return;
  }
  Out << "\nFunction* " << fname << "(Module *mod) {\n";
  printFunctionUses(F);
  printFunctionHead(F);
  printFunctionBody(F);
  Out << "return " << getCppName(F) << ";\n";
  Out << "}\n";
}

void CppWriter::printVariable(
  const std::string& fname,  /// Name of generated function
  const std::string& varName // Name of variable to generate
) {
  const GlobalVariable* GV = TheModule->getNamedGlobal(varName);

  if (!GV) {
    error(std::string("Variable '") + varName + "' not found in input module");
    return;
  }
  Out << "\nGlobalVariable* " << fname << "(Module *mod) {\n";
  printVariableUses(GV);
  printVariableHead(GV);
  printVariableBody(GV);
  Out << "return " << getCppName(GV) << ";\n";
  Out << "}\n";
}

void CppWriter::printType(
  const std::string& fname,  /// Name of generated function
  const std::string& typeName // Name of type to generate
) {
  const Type* Ty = TheModule->getTypeByName(typeName);
  if (!Ty) {
    error(std::string("Type '") + typeName + "' not found in input module");
    return;
  }
  Out << "\nType* " << fname << "(Module *mod) {\n";
  printType(Ty);
  Out << "return " << getCppName(Ty) << ";\n";
  Out << "}\n";
}

}  // end anonymous llvm

namespace llvm {

void WriteModuleToCppFile(Module* mod, std::ostream& o) {
  // Initialize a CppWriter for us to use
  CppWriter W(o, mod);

  // Emit a header
  o << "// Generated by llvm2cpp - DO NOT MODIFY!\n\n";

  // Get the name of the function we're supposed to generate
  std::string fname = FuncName.getValue();

  // Get the name of the thing we are to generate
  std::string tgtname = NameToGenerate.getValue();
  if (GenerationType == GenModule || 
      GenerationType == GenContents || 
      GenerationType == GenProgram) {
    if (tgtname == "!bad!") {
      if (mod->getModuleIdentifier() == "-")
        tgtname = "<stdin>";
      else
        tgtname = mod->getModuleIdentifier();
    }
  } else if (tgtname == "!bad!") {
    W.error("You must use the -for option with -gen-{function,variable,type}");
  }

  switch (WhatToGenerate(GenerationType)) {
    case GenProgram:
      if (fname.empty())
        fname = "makeLLVMModule";
      W.printProgram(fname,tgtname);
      break;
    case GenModule:
      if (fname.empty())
        fname = "makeLLVMModule";
      W.printModule(fname,tgtname);
      break;
    case GenContents:
      if (fname.empty())
        fname = "makeLLVMModuleContents";
      W.printContents(fname,tgtname);
      break;
    case GenFunction:
      if (fname.empty())
        fname = "makeLLVMFunction";
      W.printFunction(fname,tgtname);
      break;
    case GenInline:
      if (fname.empty())
        fname = "makeLLVMInline";
      W.printInline(fname,tgtname);
      break;
    case GenVariable:
      if (fname.empty())
        fname = "makeLLVMVariable";
      W.printVariable(fname,tgtname);
      break;
    case GenType:
      if (fname.empty())
        fname = "makeLLVMType";
      W.printType(fname,tgtname);
      break;
    default:
      W.error("Invalid generation option");
  }
}

}
