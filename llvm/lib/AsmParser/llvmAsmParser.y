//===-- llvmAsmParser.y - Parser for llvm assembly files ---------*- C++ -*--=//
//
//  This file implements the bison parser for LLVM assembly languages files.
//
//===------------------------------------------------------------------------=//

//
// TODO: Parse comments and add them to an internal node... so that they may
// be saved in the bytecode format as well as everything else.  Very important
// for a general IR format.
//

%{
#include "ParserInternals.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/ConstantPool.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include <list>
#include <utility>            // Get definition of pair class
#include <algorithm>          // Get definition of find_if
#include <stdio.h>            // This embarasment is due to our flex lexer...

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit 
int yylex();                       // declaration" of xxx warnings.
int yyparse();

static Module *ParserResult;
string CurFilename;

// This contains info used when building the body of a method.  It is destroyed
// when the method is completed.
//
typedef vector<Value *> ValueList;           // Numbered defs
static void ResolveDefinitions(vector<ValueList> &LateResolvers);

static struct PerModuleInfo {
  Module *CurrentModule;
  vector<ValueList> Values;     // Module level numbered definitions
  vector<ValueList> LateResolveValues;

  void ModuleDone() {
    // If we could not resolve some blocks at parsing time (forward branches)
    // resolve the branches now...
    ResolveDefinitions(LateResolveValues);

    Values.clear();         // Clear out method local definitions
    CurrentModule = 0;
  }
} CurModule;

static struct PerMethodInfo {
  Method *CurrentMethod;         // Pointer to current method being created

  vector<ValueList> Values;      // Keep track of numbered definitions
  vector<ValueList> LateResolveValues;
  bool isDeclare;                // Is this method a forward declararation?

  inline PerMethodInfo() {
    CurrentMethod = 0;
    isDeclare = false;
  }

  inline ~PerMethodInfo() {}

  inline void MethodStart(Method *M) {
    CurrentMethod = M;
  }

  void MethodDone() {
    // If we could not resolve some blocks at parsing time (forward branches)
    // resolve the branches now...
    ResolveDefinitions(LateResolveValues);

    Values.clear();         // Clear out method local definitions
    CurrentMethod = 0;
    isDeclare = false;
  }
} CurMeth;  // Info for the current method...


//===----------------------------------------------------------------------===//
//               Code to handle definitions of all the types
//===----------------------------------------------------------------------===//

static void InsertValue(Value *D, vector<ValueList> &ValueTab = CurMeth.Values) {
  if (!D->hasName()) {             // Is this a numbered definition?
    unsigned type = D->getType()->getUniqueID();
    if (ValueTab.size() <= type)
      ValueTab.resize(type+1, ValueList());
    //printf("Values[%d][%d] = %d\n", type, ValueTab[type].size(), D);
    ValueTab[type].push_back(D);
  }
}

static Value *getVal(const Type *Type, const ValID &D, 
                     bool DoNotImprovise = false) {
  switch (D.Type) {
  case 0: {                 // Is it a numbered definition?
    unsigned type = Type->getUniqueID();
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    if (type < CurModule.Values.size()) {
      if (Num < CurModule.Values[type].size()) 
        return CurModule.Values[type][Num];

      Num -= CurModule.Values[type].size();
    }

    // Make sure that our type is within bounds
    if (CurMeth.Values.size() <= type)
      break;

    // Check that the number is within bounds...
    if (CurMeth.Values[type].size() <= Num)
      break;
  
    return CurMeth.Values[type][Num];
  }
  case 1: {                // Is it a named definition?
    string Name(D.Name);
    SymbolTable *SymTab = 0;
    if (CurMeth.CurrentMethod) 
      SymTab = CurMeth.CurrentMethod->getSymbolTable();
    Value *N = SymTab ? SymTab->lookup(Type, Name) : 0;

    if (N == 0) {
      SymTab = CurModule.CurrentModule->getSymbolTable();
      if (SymTab)
        N = SymTab->lookup(Type, Name);
      if (N == 0) break;
    }

    D.destroy();  // Free old strdup'd memory...
    return N;
  }

  case 2:                 // Is it a constant pool reference??
  case 3:                 // Is it an unsigned const pool reference?
  case 4:                 // Is it a string const pool reference?
  case 5:{                // Is it a floating point const pool reference?
    ConstPoolVal *CPV = 0;

    // Check to make sure that "Type" is an integral type, and that our 
    // value will fit into the specified type...
    switch (D.Type) {
    case 2:
      if (Type == Type::BoolTy) {  // Special handling for boolean data
        CPV = new ConstPoolBool(D.ConstPool64 != 0);
      } else {
        if (!ConstPoolSInt::isValueValidForType(Type, D.ConstPool64))
          ThrowException("Symbolic constant pool value '" +
			 itostr(D.ConstPool64) + "' is invalid for type '" + 
			 Type->getName() + "'!");
        CPV = new ConstPoolSInt(Type, D.ConstPool64);
      }
      break;
    case 3:
      if (!ConstPoolUInt::isValueValidForType(Type, D.UConstPool64)) {
        if (!ConstPoolSInt::isValueValidForType(Type, D.ConstPool64)) {
          ThrowException("Integral constant pool reference is invalid!");
        } else {     // This is really a signed reference.  Transmogrify.
          CPV = new ConstPoolSInt(Type, D.ConstPool64);
        }
      } else {
        CPV = new ConstPoolUInt(Type, D.UConstPool64);
      }
      break;
    case 4:
      cerr << "FIXME: TODO: String constants [sbyte] not implemented yet!\n";
      abort();
      //CPV = new ConstPoolString(D.Name);
      D.destroy();   // Free the string memory
      break;
    case 5:
      if (!ConstPoolFP::isValueValidForType(Type, D.ConstPoolFP))
	ThrowException("FP constant invalid for type!!");
      else
	CPV = new ConstPoolFP(Type, D.ConstPoolFP);
      break;
    }
    assert(CPV && "How did we escape creating a constant??");

    // Scan through the constant table and see if we already have loaded this
    // constant.
    //
    ConstantPool &CP = CurMeth.CurrentMethod ? 
                         CurMeth.CurrentMethod->getConstantPool() :
                           CurModule.CurrentModule->getConstantPool();
    ConstPoolVal *C = CP.find(CPV);      // Already have this constant?
    if (C) {
      delete CPV;  // Didn't need this after all, oh well.
      return C;    // Yup, we already have one, recycle it!
    }
    CP.insert(CPV);
      
    // Success, everything is kosher. Lets go!
    return CPV;
  }   // End of case 2,3,4
  }   // End of switch


  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?

  // TODO: Attempt to coallecse nodes that are the same with previous ones.
  Value *d = 0;
  switch (Type->getPrimitiveID()) {
  case Type::LabelTyID: d = new    BBPlaceHolder(Type, D); break;
  case Type::MethodTyID:
    d = new MethPlaceHolder(Type, D); 
    InsertValue(d, CurModule.LateResolveValues);
    return d;
//case Type::ClassTyID:      d = new ClassPlaceHolder(Type, D); break;
  default:                   d = new   DefPlaceHolder(Type, D); break;
  }

  assert(d != 0 && "How did we not make something?");
  InsertValue(d, CurMeth.LateResolveValues);
  return d;
}


//===----------------------------------------------------------------------===//
//              Code to handle forward references in instructions
//===----------------------------------------------------------------------===//
//
// This code handles the late binding needed with statements that reference
// values not defined yet... for example, a forward branch, or the PHI node for
// a loop body.
//
// This keeps a table (CurMeth.LateResolveValues) of all such forward references
// and back patchs after we are done.
//

// ResolveDefinitions - If we could not resolve some defs at parsing 
// time (forward branches, phi functions for loops, etc...) resolve the 
// defs now...
//
static void ResolveDefinitions(vector<ValueList> &LateResolvers) {
  // Loop over LateResolveDefs fixing up stuff that couldn't be resolved
  for (unsigned ty = 0; ty < LateResolvers.size(); ty++) {
    while (!LateResolvers[ty].empty()) {
      Value *V = LateResolvers[ty].back();
      LateResolvers[ty].pop_back();
      ValID &DID = getValIDFromPlaceHolder(V);

      Value *TheRealValue = getVal(Type::getUniqueIDType(ty), DID, true);

      if (TheRealValue == 0 && DID.Type == 1)
        ThrowException("Reference to an invalid definition: '" +DID.getName() +
                       "' of type '" + V->getType()->getName() + "'");
      else if (TheRealValue == 0)
        ThrowException("Reference to an invalid definition: #" +itostr(DID.Num)+
                       " of type '" + V->getType()->getName() + "'");

      V->replaceAllUsesWith(TheRealValue);
      assert(V->use_empty());
      delete V;
    }
  }

  LateResolvers.clear();
}

// addConstValToConstantPool - This code is used to insert a constant into the
// current constant pool.  This is designed to make maximal (but not more than
// possible) reuse (merging) of constants in the constant pool.  This means that
// multiple references to %4, for example will all get merged.
//
static ConstPoolVal *addConstValToConstantPool(ConstPoolVal *C) {
  vector<ValueList> &ValTab = CurMeth.CurrentMethod ? 
                                  CurMeth.Values : CurModule.Values;
  ConstantPool &CP = CurMeth.CurrentMethod ? 
                          CurMeth.CurrentMethod->getConstantPool() : 
                          CurModule.CurrentModule->getConstantPool();

  if (ConstPoolVal *CPV = CP.find(C)) {
    // Constant already in constant pool. Try to merge the two constants
    if (CPV->hasName() && !C->hasName()) {
      // Merge the two values, we inherit the existing CPV's name.  
      // InsertValue requires that the value have no name to insert correctly
      // (because we want to fill the slot this constant would have filled)
      //
      string Name = CPV->getName();
      CPV->setName("");
      InsertValue(CPV, ValTab);
      CPV->setName(Name);
      delete C;
      return CPV;
    } else if (!CPV->hasName() && C->hasName()) {
      // If we have a name on this value and there isn't one in the const 
      // pool val already, propogate it.
      //
      CPV->setName(C->getName());
      delete C;   // Sorry, you're toast
      return CPV;
    } else if (CPV->hasName() && C->hasName()) {
      // Both values have distinct names.  We cannot merge them.
      CP.insert(C);
      InsertValue(C, ValTab);
      return C;
    } else if (!CPV->hasName() && !C->hasName()) {
      // Neither value has a name, trivially merge them.
      InsertValue(CPV, ValTab);
      delete C;
      return CPV;
    }

    assert(0 && "Not reached!");
    return 0;
  } else {           // No duplication of value.
    CP.insert(C);
    InsertValue(C, ValTab);
    return C;
  } 
}


struct EqualsType {
  const Type *T;
  inline EqualsType(const Type *t) { T = t; }
  inline bool operator()(const ConstPoolVal *CPV) const {
    return static_cast<const ConstPoolType*>(CPV)->getValue() == T;
  }
};


// checkNewType - We have to be careful to add all types referenced by the
// program to the constant pool of the method or module.  Because of this, we
// often want to check to make sure that types used are in the constant pool,
// and add them if they aren't.  That's what this function does.
//
static const Type *checkNewType(const Type *Ty) {
  ConstantPool &CP = CurMeth.CurrentMethod ? 
                          CurMeth.CurrentMethod->getConstantPool() : 
                          CurModule.CurrentModule->getConstantPool();

  // TODO: This should use ConstantPool::ensureTypeAvailable

  // Get the type type plane...
  ConstantPool::PlaneType &P = CP.getPlane(Type::TypeTy);
  ConstantPool::PlaneType::const_iterator PI = find_if(P.begin(), P.end(), 
						       EqualsType(Ty));
  if (PI == P.end()) {
    vector<ValueList> &ValTab = CurMeth.CurrentMethod ? 
                                CurMeth.Values : CurModule.Values;
    ConstPoolVal *CPT = new ConstPoolType(Ty);
    CP.insert(CPT);
    InsertValue(CPT, ValTab);
  }
  return Ty;
}


//===----------------------------------------------------------------------===//
//            RunVMAsmParser - Define an interface to this parser
//===----------------------------------------------------------------------===//
//
Module *RunVMAsmParser(const string &Filename, FILE *F) {
  llvmAsmin = F;
  CurFilename = Filename;
  llvmAsmlineno = 1;      // Reset the current line number...

  CurModule.CurrentModule = new Module();  // Allocate a new module to read
  yyparse();       // Parse the file.
  Module *Result = ParserResult;
  llvmAsmin = stdin;    // F is about to go away, don't use it anymore...
  ParserResult = 0;

  return Result;
}

%}

%union {
  Module                  *ModuleVal;
  Method                  *MethodVal;
  MethodArgument          *MethArgVal;
  BasicBlock              *BasicBlockVal;
  TerminatorInst          *TermInstVal;
  Instruction             *InstVal;
  ConstPoolVal            *ConstVal;
  const Type              *TypeVal;
  Value                   *ValueVal;

  list<MethodArgument*>   *MethodArgList;
  list<Value*>            *ValueList;
  list<const Type*>       *TypeList;
  list<pair<Value*, BasicBlock*> > *PHIList;   // Represent the RHS of PHI node
  list<pair<ConstPoolVal*, BasicBlock*> > *JumpTable;
  vector<ConstPoolVal*>   *ConstVector;

  int64_t                  SInt64Val;
  uint64_t                 UInt64Val;
  int                      SIntVal;
  unsigned                 UIntVal;
  double                   FPVal;

  char                    *StrVal;   // This memory is allocated by strdup!
  ValID                    ValIDVal; // May contain memory allocated by strdup

  Instruction::UnaryOps    UnaryOpVal;
  Instruction::BinaryOps   BinaryOpVal;
  Instruction::TermOps     TermOpVal;
  Instruction::MemoryOps   MemOpVal;
  Instruction::OtherOps    OtherOpVal;
}

%type <ModuleVal>     Module MethodList
%type <MethodVal>     Method MethodProto MethodHeader BasicBlockList
%type <BasicBlockVal> BasicBlock InstructionList
%type <TermInstVal>   BBTerminatorInst
%type <InstVal>       Inst InstVal MemoryInst
%type <ConstVal>      ConstVal ExtendedConstVal
%type <ConstVector>   ConstVector UByteList
%type <MethodArgList> ArgList ArgListH
%type <MethArgVal>    ArgVal
%type <PHIList>       PHIList
%type <ValueList>     ValueRefList ValueRefListE  // For call param lists
%type <TypeList>      TypeList ArgTypeList
%type <JumpTable>     JumpTable

%type <ValIDVal>      ValueRef ConstValueRef // Reference to a definition or BB
%type <ValueVal>      ResolvedVal            // <type> <valref> pair
// Tokens and types for handling constant integer values
//
// ESINT64VAL - A negative number within long long range
%token <SInt64Val> ESINT64VAL

// EUINT64VAL - A positive number within uns. long long range
%token <UInt64Val> EUINT64VAL
%type  <SInt64Val> EINT64VAL

%token  <SIntVal>   SINTVAL   // Signed 32 bit ints...
%token  <UIntVal>   UINTVAL   // Unsigned 32 bit ints...
%type   <SIntVal>   INTVAL
%token  <FPVal>     FPVAL     // Float or Double constant

// Built in types...
%type  <TypeVal> Types TypesV SIntType UIntType IntType FPType
%token <TypeVal> VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token <TypeVal> FLOAT DOUBLE STRING TYPE LABEL

%token <StrVal>     VAR_ID LABELSTR STRINGCONSTANT
%type  <StrVal>  OptVAR_ID OptAssign


%token IMPLEMENTATION TRUE FALSE BEGINTOK END DECLARE TO DOTDOTDOT

// Basic Block Terminating Operators 
%token <TermOpVal> RET BR SWITCH

// Unary Operators 
%type  <UnaryOpVal> UnaryOps  // all the unary operators
%token <UnaryOpVal> NOT

// Binary Operators 
%type  <BinaryOpVal> BinaryOps  // all the binary operators
%token <BinaryOpVal> ADD SUB MUL DIV REM
%token <BinaryOpVal> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comarators

// Memory Instructions
%token <MemoryOpVal> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR

// Other Operators
%type  <OtherOpVal> ShiftOps
%token <OtherOpVal> PHI CALL CAST SHL SHR

%start Module
%%

// Handle constant integer size restriction and conversion...
//

INTVAL : SINTVAL
INTVAL : UINTVAL {
  if ($1 > (uint32_t)INT32_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  $$ = (int32_t)$1;
}


EINT64VAL : ESINT64VAL       // These have same type and can't cause problems...
EINT64VAL : EUINT64VAL {
  if ($1 > (uint64_t)INT64_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  $$ = (int64_t)$1;
}

// Types includes all predefined types... except void, because you can't do 
// anything with it except for certain specific things...
//
// User defined types are added later...
//
Types     : BOOL | SBYTE | UBYTE | SHORT | USHORT | INT | UINT 
Types     : LONG | ULONG | FLOAT | DOUBLE | STRING | TYPE | LABEL

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    : Types | VOID

// Operations that are notably excluded from this list include: 
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
//
UnaryOps  : NOT
BinaryOps : ADD | SUB | MUL | DIV | REM
BinaryOps : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE
ShiftOps  : SHL | SHR

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType :  LONG |  INT |  SHORT | SBYTE
UIntType : ULONG | UINT | USHORT | UBYTE
IntType : SIntType | UIntType
FPType  : FLOAT | DOUBLE

// OptAssign - Value producing statements have an optional assignment component
OptAssign : VAR_ID '=' {
    $$ = $1;
  }
  | /*empty*/ { 
    $$ = 0; 
  }

// ConstVal - The various declarations that go into the constant pool.  This
// includes all forward declarations of types, constants, and functions.
//
// This is broken into two sections: ExtendedConstVal and ConstVal
//
ExtendedConstVal: '[' Types ']' '[' ConstVector ']' { // Nonempty unsized array
    // Verify all elements are correct type!
    const ArrayType *AT = ArrayType::getArrayType($2);
    for (unsigned i = 0; i < $5->size(); i++) {
      if ($2 != (*$5)[i]->getType())
	ThrowException("Element #" + utostr(i) + " is not of type '" + 
		       $2->getName() + "' as required!\nIt is of type '" +
		       (*$5)[i]->getType()->getName() + "'.");
    }

    $$ = new ConstPoolArray(AT, *$5);
    delete $5;
  }
  | '[' Types ']' '[' ']' {                  // Empty unsized array constant
    vector<ConstPoolVal*> Empty;
    $$ = new ConstPoolArray(ArrayType::getArrayType($2), Empty);
  }
  | '[' EUINT64VAL 'x' Types ']' '[' ConstVector ']' {
    // Verify all elements are correct type!
    const ArrayType *AT = ArrayType::getArrayType($4, (int)$2);
    if ($2 != $7->size())
      ThrowException("Type mismatch: constant sized array initialized with " +
		     utostr($7->size()) +  " arguments, but has size of " + 
		     itostr((int)$2) + "!");

    for (unsigned i = 0; i < $7->size(); i++) {
      if ($4 != (*$7)[i]->getType())
	ThrowException("Element #" + utostr(i) + " is not of type '" + 
		       $4->getName() + "' as required!\nIt is of type '" +
		       (*$7)[i]->getType()->getName() + "'.");
    }

    $$ = new ConstPoolArray(AT, *$7);
    delete $7;
  }
  | '[' EUINT64VAL 'x' Types ']' '[' ']' {
    if ($2 != 0) 
      ThrowException("Type mismatch: constant sized array initialized with 0"
		     " arguments, but has size of " + itostr((int)$2) + "!");
    vector<ConstPoolVal*> Empty;
    $$ = new ConstPoolArray(ArrayType::getArrayType($4, 0), Empty);
  }
  | '{' TypeList '}' '{' ConstVector '}' {
    StructType::ElementTypes Types($2->begin(), $2->end());
    delete $2;

    const StructType *St = StructType::getStructType(Types);
    $$ = new ConstPoolStruct(St, *$5);
    delete $5;
  }
  | '{' '}' '{' '}' {
    const StructType *St = 
      StructType::getStructType(StructType::ElementTypes());
    vector<ConstPoolVal*> Empty;
    $$ = new ConstPoolStruct(St, Empty);
  }
/*
  | Types '*' ConstVal {
    assert(0);
    $$ = 0;
  }
*/

ConstVal : ExtendedConstVal
  | TYPE Types {                    // Type constants
    $$ = new ConstPoolType($2);
  }
  | SIntType EINT64VAL {     // integral constants
    if (!ConstPoolSInt::isValueValidForType($1, $2))
      ThrowException("Constant value doesn't fit in type!");
    $$ = new ConstPoolSInt($1, $2);
  } 
  | UIntType EUINT64VAL {           // integral constants
    if (!ConstPoolUInt::isValueValidForType($1, $2))
      ThrowException("Constant value doesn't fit in type!");
    $$ = new ConstPoolUInt($1, $2);
  } 
  | BOOL TRUE {                     // Boolean constants
    $$ = new ConstPoolBool(true);
  }
  | BOOL FALSE {                    // Boolean constants
    $$ = new ConstPoolBool(false);
  }
  | FPType FPVAL {                   // Float & Double constants
    $$ = new ConstPoolFP($1, $2);
  }
  | STRING STRINGCONSTANT {         // String constants
    cerr << "FIXME: TODO: String constants [sbyte] not implemented yet!\n";
    abort();
    //$$ = new ConstPoolString($2);
    free($2);
  } 

// ConstVector - A list of comma seperated constants.
ConstVector : ConstVector ',' ConstVal {
    ($$ = $1)->push_back(addConstValToConstantPool($3));
  }
  | ConstVal {
    $$ = new vector<ConstPoolVal*>();
    $$->push_back(addConstValToConstantPool($1));
  }


//ExternMethodDecl : EXTERNAL TypesV '(' TypeList ')' {
//  }
//ExternVarDecl : 

// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign ConstVal { 
    if ($2) {
      $3->setName($2);
      free($2);
    }

    addConstValToConstantPool($3);
  }
/*
  | ConstPool OptAssign GlobalDecl {     // Global declarations appear in CP
    if ($2) {
      $3->setName($2);
      free($2);
    }
    //CurModule.CurrentModule->
  }
*/
  | /* empty: end of list */ { 
  }


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module : MethodList {
  $$ = ParserResult = $1;
  CurModule.ModuleDone();
}

// MethodList - A list of methods, preceeded by a constant pool.
//
MethodList : MethodList Method {
    $$ = $1;
    if (!$2->getParent())
      $1->getMethodList().push_back($2);
    CurMeth.MethodDone();
  } 
  | MethodList MethodProto {
    $$ = $1;
    if (!$2->getParent())
      $1->getMethodList().push_back($2);
    CurMeth.MethodDone();
  }
  | ConstPool IMPLEMENTATION {
    $$ = CurModule.CurrentModule;
  }


//===----------------------------------------------------------------------===//
//                       Rules to match Method Headers
//===----------------------------------------------------------------------===//

OptVAR_ID : VAR_ID | /*empty*/ { $$ = 0; }

ArgVal : Types OptVAR_ID {
  $$ = new MethodArgument($1);
  if ($2) {      // Was the argument named?
    $$->setName($2); 
    free($2);    // The string was strdup'd, so free it now.
  }
}

ArgListH : ArgVal ',' ArgListH {
    $$ = $3;
    $3->push_front($1);
  }
  | ArgVal {
    $$ = new list<MethodArgument*>();
    $$->push_front($1);
  }
  | DOTDOTDOT {
    $$ = new list<MethodArgument*>();
    $$->push_back(new MethodArgument(Type::VoidTy));
  }

ArgList : ArgListH {
    $$ = $1;
  }
  | /* empty */ {
    $$ = 0;
  }

MethodHeaderH : TypesV STRINGCONSTANT '(' ArgList ')' {
  MethodType::ParamTypes ParamTypeList;
  if ($4)
    for (list<MethodArgument*>::iterator I = $4->begin(); I != $4->end(); ++I)
      ParamTypeList.push_back((*I)->getType());

  const MethodType *MT = MethodType::getMethodType($1, ParamTypeList);

  Method *M = 0;
  if (SymbolTable *ST = CurModule.CurrentModule->getSymbolTable()) {
    if (Value *V = ST->lookup(MT, $2)) {  // Method already in symtab?
      M = V->castMethodAsserting();

      // Yes it is.  If this is the case, either we need to be a forward decl,
      // or it needs to be.
      if (!CurMeth.isDeclare && !M->isExternal())
	ThrowException("Redefinition of method '" + string($2) + "'!");      
    }
  }

  if (M == 0) {  // Not already defined?
    M = new Method(MT, $2);
    InsertValue(M, CurModule.Values);
  }

  free($2);  // Free strdup'd memory!

  CurMeth.MethodStart(M);

  // Add all of the arguments we parsed to the method...
  if ($4 && !CurMeth.isDeclare) {        // Is null if empty...
    Method::ArgumentListType &ArgList = M->getArgumentList();

    for (list<MethodArgument*>::iterator I = $4->begin(); I != $4->end(); ++I) {
      InsertValue(*I);
      ArgList.push_back(*I);
    }
    delete $4;                     // We're now done with the argument list
  }
}

MethodHeader : MethodHeaderH ConstPool BEGINTOK {
  $$ = CurMeth.CurrentMethod;
}

Method : BasicBlockList END {
  $$ = $1;
}

MethodProto : DECLARE { CurMeth.isDeclare = true; } MethodHeaderH {
  $$ = CurMeth.CurrentMethod;
}

//===----------------------------------------------------------------------===//
//                        Rules to match Basic Blocks
//===----------------------------------------------------------------------===//

ConstValueRef : ESINT64VAL {    // A reference to a direct constant
    $$ = ValID::create($1);
  }
  | EUINT64VAL {
    $$ = ValID::create($1);
  }
  | FPVAL {                     // Perhaps it's an FP constant?
    $$ = ValID::create($1);
  }
  | TRUE {
    $$ = ValID::create((int64_t)1);
  } 
  | FALSE {
    $$ = ValID::create((int64_t)0);
  }
  | STRINGCONSTANT {        // Quoted strings work too... especially for methods
    $$ = ValID::create_conststr($1);
  }

// ValueRef - A reference to a definition... 
ValueRef : INTVAL {           // Is it an integer reference...?
    $$ = ValID::create($1);
  }
  | VAR_ID {                 // Is it a named reference...?
    $$ = ValID::create($1);
  }
  | ConstValueRef {
    $$ = $1;
  }

// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : ExtendedConstVal {
    $$ = addConstValToConstantPool($1);
  }
  | Types ValueRef {
    $$ = getVal($1, $2);
  }


// The user may refer to a user defined type by its typeplane... check for this
// now...
//
Types : ValueRef {
    Value *D = getVal(Type::TypeTy, $1, true);
    if (D == 0) ThrowException("Invalid user defined type: " + $1.getName());

    // User defined type not in const pool!
    ConstPoolType *CPT = (ConstPoolType*)D->castConstantAsserting();
    $$ = CPT->getValue();
  }
  | TypesV '(' ArgTypeList ')' {               // Method derived type?
    MethodType::ParamTypes Params($3->begin(), $3->end());
    delete $3;
    $$ = checkNewType(MethodType::getMethodType($1, Params));
  }
  | TypesV '(' ')' {               // Method derived type?
    MethodType::ParamTypes Params;     // Empty list
    $$ = checkNewType(MethodType::getMethodType($1, Params));
  }
  | '[' Types ']' {
    $$ = checkNewType(ArrayType::getArrayType($2));
  }
  | '[' EUINT64VAL 'x' Types ']' {
    $$ = checkNewType(ArrayType::getArrayType($4, (int)$2));
  }
  | '{' TypeList '}' {
    StructType::ElementTypes Elements($2->begin(), $2->end());
    delete $2;
    $$ = checkNewType(StructType::getStructType(Elements));
  }
  | '{' '}' {
    $$ = checkNewType(StructType::getStructType(StructType::ElementTypes()));
  }
  | Types '*' {
    $$ = checkNewType(PointerType::getPointerType($1));
  }

TypeList : Types {
    $$ = new list<const Type*>();
    $$->push_back($1);
  }
  | TypeList ',' Types {
    ($$=$1)->push_back($3);
  }

ArgTypeList : TypeList 
  | TypeList ',' DOTDOTDOT {
    ($$=$1)->push_back(Type::VoidTy);
  }


BasicBlockList : BasicBlockList BasicBlock {
    $1->getBasicBlocks().push_back($2);
    $$ = $1;
  }
  | MethodHeader BasicBlock { // Do not allow methods with 0 basic blocks   
    $$ = $1;                  // in them...
    $1->getBasicBlocks().push_back($2);
  }


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock : InstructionList BBTerminatorInst  {
    $1->getInstList().push_back($2);
    InsertValue($1);
    $$ = $1;
  }
  | LABELSTR InstructionList BBTerminatorInst  {
    $2->getInstList().push_back($3);
    $2->setName($1);
    free($1);         // Free the strdup'd memory...

    InsertValue($2);
    $$ = $2;
  }

InstructionList : InstructionList Inst {
    $1->getInstList().push_back($2);
    $$ = $1;
  }
  | /* empty */ {
    $$ = new BasicBlock();
  }

BBTerminatorInst : RET ResolvedVal {              // Return with a result...
    $$ = new ReturnInst($2);
  }
  | RET VOID {                                       // Return with no result...
    $$ = new ReturnInst();
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    $$ = new BranchInst(getVal(Type::LabelTy, $3)->castBasicBlockAsserting());
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    $$ = new BranchInst(getVal(Type::LabelTy, $6)->castBasicBlockAsserting(), 
			getVal(Type::LabelTy, $9)->castBasicBlockAsserting(),
			getVal(Type::BoolTy, $3));
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    SwitchInst *S = new SwitchInst(getVal($2, $3), 
                          getVal(Type::LabelTy, $6)->castBasicBlockAsserting());
    $$ = S;

    list<pair<ConstPoolVal*, BasicBlock*> >::iterator I = $8->begin(), 
                                                      end = $8->end();
    for (; I != end; ++I)
      S->dest_push_back(I->first, I->second);
  }

JumpTable : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    $$ = $1;
    ConstPoolVal *V = getVal($2, $3, true)->castConstantAsserting();
    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    $$->push_back(make_pair(V, getVal($5, $6)->castBasicBlockAsserting()));
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $$ = new list<pair<ConstPoolVal*, BasicBlock*> >();
    ConstPoolVal *V = getVal($1, $2, true)->castConstantAsserting();

    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    $$->push_back(make_pair(V, getVal($4, $5)->castBasicBlockAsserting()));
  }

Inst : OptAssign InstVal {
  if ($1)              // Is this definition named??
    $2->setName($1);   // if so, assign the name...

  InsertValue($2);
  $$ = $2;
}

PHIList : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    $$ = new list<pair<Value*, BasicBlock*> >();
    $$->push_back(make_pair(getVal($1, $3), 
			 getVal(Type::LabelTy, $5)->castBasicBlockAsserting()));
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    $$ = $1;
    $1->push_back(make_pair(getVal($1->front().first->getType(), $4),
			 getVal(Type::LabelTy, $6)->castBasicBlockAsserting()));
  }


ValueRefList : ResolvedVal {    // Used for call statements...
    $$ = new list<Value*>();
    $$->push_back($1);
  }
  | ValueRefList ',' ResolvedVal {
    $$ = $1;
    $1->push_back($3);
  }

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE : ValueRefList | /*empty*/ { $$ = 0; }

InstVal : BinaryOps Types ValueRef ',' ValueRef {
    $$ = BinaryOperator::create($1, getVal($2, $3), getVal($2, $5));
    if ($$ == 0)
      ThrowException("binary operator returned null!");
  }
  | UnaryOps ResolvedVal {
    $$ = UnaryOperator::create($1, $2);
    if ($$ == 0)
      ThrowException("unary operator returned null!");
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    if ($4->getType() != Type::UByteTy)
      ThrowException("Shift amount must be ubyte!");
    $$ = new ShiftInst($1, $2, $4);
  }
  | CAST ResolvedVal TO Types {
    $$ = new CastInst($2, $4);
  }
  | PHI PHIList {
    const Type *Ty = $2->front().first->getType();
    $$ = new PHINode(Ty);
    while ($2->begin() != $2->end()) {
      if ($2->front().first->getType() != Ty) 
	ThrowException("All elements of a PHI node must be of the same type!");
      ((PHINode*)$$)->addIncoming($2->front().first, $2->front().second);
      $2->pop_front();
    }
    delete $2;  // Free the list...
  } 
  | CALL Types ValueRef '(' ValueRefListE ')' {
    const MethodType *Ty;

    if (!(Ty = $2->isMethodType())) {
      // Pull out the types of all of the arguments...
      vector<const Type*> ParamTypes;
      for (list<Value*>::iterator I = $5->begin(), E = $5->end(); I != E; ++I)
	ParamTypes.push_back((*I)->getType());
      Ty = MethodType::get($2, ParamTypes);
    }

    Value *V = getVal(Ty, $3);   // Get the method we're calling...

    // Create the call node...
    if (!$5) {                                   // Has no arguments?
      $$ = new CallInst(V->castMethodAsserting(), vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through MethodType's arguments and ensure they are specified
      // correctly!
      //
      MethodType::ParamTypes::const_iterator I = Ty->getParamTypes().begin();
      MethodType::ParamTypes::const_iterator E = Ty->getParamTypes().end();
      list<Value*>::iterator ArgI = $5->begin(), ArgE = $5->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
	if ((*ArgI)->getType() != *I)
	  ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
			 (*I)->getName() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
	ThrowException("Invalid number of parameters detected!");

      $$ = new CallInst(V->castMethodAsserting(),
			vector<Value*>($5->begin(), $5->end()));
    }
    delete $5;
  }
  | MemoryInst {
    $$ = $1;
  }

// UByteList - List of ubyte values for load and store instructions
UByteList : ',' ConstVector { 
  $$ = $2; 
} | /* empty */ { 
  $$ = new vector<ConstPoolVal*>(); 
}

MemoryInst : MALLOC Types {
    $$ = new MallocInst(checkNewType(PointerType::getPointerType($2)));
  }
  | MALLOC Types ',' UINT ValueRef {
    if (!$2->isArrayType() || ((const ArrayType*)$2)->isSized())
      ThrowException("Trying to allocate " + $2->getName() + 
		     " as unsized array!");
    const Type *Ty = checkNewType(PointerType::getPointerType($2));
    $$ = new MallocInst(Ty, getVal($4, $5));
  }
  | ALLOCA Types {
    $$ = new AllocaInst(checkNewType(PointerType::getPointerType($2)));
  }
  | ALLOCA Types ',' UINT ValueRef {
    if (!$2->isArrayType() || ((const ArrayType*)$2)->isSized())
      ThrowException("Trying to allocate " + $2->getName() + 
		     " as unsized array!");
    const Type *Ty = checkNewType(PointerType::getPointerType($2));    
    Value *ArrSize = getVal($4, $5);
    $$ = new AllocaInst(Ty, ArrSize);
  }
  | FREE ResolvedVal {
    if (!$2->getType()->isPointerType())
      ThrowException("Trying to free nonpointer type " + 
                     $2->getType()->getName() + "!");
    $$ = new FreeInst($2);
  }

  | LOAD Types ValueRef UByteList {
    if (!$2->isPointerType())
      ThrowException("Can't load from nonpointer type: " + $2->getName());
    if (LoadInst::getIndexedType($2, *$4) == 0)
      ThrowException("Invalid indices for load instruction!");

    $$ = new LoadInst(getVal($2, $3), *$4);
    delete $4;   // Free the vector...
  }
  | STORE ResolvedVal ',' Types ValueRef UByteList {
    if (!$4->isPointerType())
      ThrowException("Can't store to a nonpointer type: " + $4->getName());
    const Type *ElTy = StoreInst::getIndexedType($4, *$6);
    if (ElTy == 0)
      ThrowException("Can't store into that field list!");
    if (ElTy != $2->getType())
      ThrowException("Can't store '" + $2->getType()->getName() +
                     "' into space of type '" + ElTy->getName() + "'!");
    $$ = new StoreInst($2, getVal($4, $5), *$6);
    delete $6;
  }
  | GETELEMENTPTR Types ValueRef UByteList {
    if (!$2->isPointerType())
      ThrowException("getelementptr insn requires pointer operand!");
    if (!GetElementPtrInst::getIndexedType($2, *$4, true))
      ThrowException("Can't get element ptr '" + $2->getName() + "'!");
    $$ = new GetElementPtrInst(getVal($2, $3), *$4);
    delete $4;
    checkNewType($$->getType());
  }

%%
int yyerror(const char *ErrorMsg) {
  ThrowException(string("Parse error: ") + ErrorMsg);
  return 0;
}
