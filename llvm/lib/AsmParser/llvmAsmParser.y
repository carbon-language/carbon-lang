//===-- llvmAsmParser.y - Parser for llvm assembly files ---------*- C++ -*--=//
//
//  This file implements the bison parser for LLVM assembly languages files.
//
//===------------------------------------------------------------------------=//

%{
#include "ParserInternals.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "Support/STLExtras.h"
#include "Support/DepthFirstIterator.h"
#include <list>
#include <utility>
#include <algorithm>

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit
int yylex();                       // declaration" of xxx warnings.
int yyparse();

static Module *ParserResult;
std::string CurFilename;

// DEBUG_UPREFS - Define this symbol if you want to enable debugging output
// relating to upreferences in the input stream.
//
//#define DEBUG_UPREFS 1
#ifdef DEBUG_UPREFS
#define UR_OUT(X) std::cerr << X
#else
#define UR_OUT(X)
#endif

#define YYERROR_VERBOSE 1

// HACK ALERT: This variable is used to implement the automatic conversion of
// load/store instructions with indexes into a load/store + getelementptr pair
// of instructions.  When this compatiblity "Feature" is removed, this should be
// too.
//
static BasicBlock *CurBB;


// This contains info used when building the body of a function.  It is
// destroyed when the function is completed.
//
typedef std::vector<Value *> ValueList;           // Numbered defs
static void ResolveDefinitions(std::vector<ValueList> &LateResolvers,
                               std::vector<ValueList> *FutureLateResolvers = 0);

static struct PerModuleInfo {
  Module *CurrentModule;
  std::vector<ValueList>    Values;     // Module level numbered definitions
  std::vector<ValueList>    LateResolveValues;
  std::vector<PATypeHolder> Types;
  std::map<ValID, PATypeHolder> LateResolveTypes;

  // GlobalRefs - This maintains a mapping between <Type, ValID>'s and forward
  // references to global values.  Global values may be referenced before they
  // are defined, and if so, the temporary object that they represent is held
  // here.  This is used for forward references of ConstantPointerRefs.
  //
  typedef std::map<std::pair<const PointerType *,
                             ValID>, GlobalVariable*> GlobalRefsType;
  GlobalRefsType GlobalRefs;

  void ModuleDone() {
    // If we could not resolve some functions at function compilation time
    // (calls to functions before they are defined), resolve them now...  Types
    // are resolved when the constant pool has been completely parsed.
    //
    ResolveDefinitions(LateResolveValues);

    // Check to make sure that all global value forward references have been
    // resolved!
    //
    if (!GlobalRefs.empty()) {
      std::string UndefinedReferences = "Unresolved global references exist:\n";
      
      for (GlobalRefsType::iterator I = GlobalRefs.begin(), E =GlobalRefs.end();
           I != E; ++I) {
        UndefinedReferences += "  " + I->first.first->getDescription() + " " +
                               I->first.second.getName() + "\n";
      }
      ThrowException(UndefinedReferences);
    }

    Values.clear();         // Clear out function local definitions
    Types.clear();
    CurrentModule = 0;
  }


  // DeclareNewGlobalValue - Called every time a new GV has been defined.  This
  // is used to remove things from the forward declaration map, resolving them
  // to the correct thing as needed.
  //
  void DeclareNewGlobalValue(GlobalValue *GV, ValID D) {
    // Check to see if there is a forward reference to this global variable...
    // if there is, eliminate it and patch the reference to use the new def'n.
    GlobalRefsType::iterator I =
      GlobalRefs.find(std::make_pair(GV->getType(), D));

    if (I != GlobalRefs.end()) {
      GlobalVariable *OldGV = I->second;   // Get the placeholder...
      I->first.second.destroy();  // Free string memory if neccesary
      
      // Loop over all of the uses of the GlobalValue.  The only thing they are
      // allowed to be is ConstantPointerRef's.
      assert(OldGV->use_size() == 1 && "Only one reference should exist!");
      User *U = OldGV->use_back();  // Must be a ConstantPointerRef...
      ConstantPointerRef *CPR = cast<ConstantPointerRef>(U);
        
      // Change the const pool reference to point to the real global variable
      // now.  This should drop a use from the OldGV.
      CPR->mutateReferences(OldGV, GV);
      assert(OldGV->use_empty() && "All uses should be gone now!");
      
      // Remove OldGV from the module...
      CurrentModule->getGlobalList().remove(OldGV);
      delete OldGV;                        // Delete the old placeholder
      
      // Remove the map entry for the global now that it has been created...
      GlobalRefs.erase(I);
    }
  }

} CurModule;

static struct PerFunctionInfo {
  Function *CurrentFunction;     // Pointer to current function being created

  std::vector<ValueList> Values;      // Keep track of numbered definitions
  std::vector<ValueList> LateResolveValues;
  std::vector<PATypeHolder> Types;
  std::map<ValID, PATypeHolder> LateResolveTypes;
  bool isDeclare;                // Is this function a forward declararation?

  inline PerFunctionInfo() {
    CurrentFunction = 0;
    isDeclare = false;
  }

  inline ~PerFunctionInfo() {}

  inline void FunctionStart(Function *M) {
    CurrentFunction = M;
  }

  void FunctionDone() {
    // If we could not resolve some blocks at parsing time (forward branches)
    // resolve the branches now...
    ResolveDefinitions(LateResolveValues, &CurModule.LateResolveValues);

    // Make sure to resolve any constant expr references that might exist within
    // the function we just declared itself.
    ValID FID;
    if (CurrentFunction->hasName()) {
      FID = ValID::create((char*)CurrentFunction->getName().c_str());
    } else {
      unsigned Slot = CurrentFunction->getType()->getUniqueID();
      assert(CurModule.Values.size() > Slot && "Function not inserted?");
      // Figure out which slot number if is...
      for (unsigned i = 0; ; ++i) {
        assert(i < CurModule.Values[Slot].size() && "Function not found!");
        if (CurModule.Values[Slot][i] == CurrentFunction) {
          FID = ValID::create((int)i);
          break;
        }
      }
    }
    CurModule.DeclareNewGlobalValue(CurrentFunction, FID);

    Values.clear();         // Clear out function local definitions
    Types.clear();
    CurrentFunction = 0;
    isDeclare = false;
  }
} CurMeth;  // Info for the current function...

static bool inFunctionScope() { return CurMeth.CurrentFunction != 0; }


//===----------------------------------------------------------------------===//
//               Code to handle definitions of all the types
//===----------------------------------------------------------------------===//

static int InsertValue(Value *D,
                       std::vector<ValueList> &ValueTab = CurMeth.Values) {
  if (D->hasName()) return -1;           // Is this a numbered definition?

  // Yes, insert the value into the value table...
  unsigned type = D->getType()->getUniqueID();
  if (ValueTab.size() <= type)
    ValueTab.resize(type+1, ValueList());
  //printf("Values[%d][%d] = %d\n", type, ValueTab[type].size(), D);
  ValueTab[type].push_back(D);
  return ValueTab[type].size()-1;
}

// TODO: FIXME when Type are not const
static void InsertType(const Type *Ty, std::vector<PATypeHolder> &Types) {
  Types.push_back(Ty);
}

static const Type *getTypeVal(const ValID &D, bool DoNotImprovise = false) {
  switch (D.Type) {
  case ValID::NumberVal: {                 // Is it a numbered definition?
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    if (Num < CurModule.Types.size()) 
      return CurModule.Types[Num];

    Num -= CurModule.Types.size();

    // Check that the number is within bounds...
    if (Num <= CurMeth.Types.size())
      return CurMeth.Types[Num];
    break;
  }
  case ValID::NameVal: {                // Is it a named definition?
    std::string Name(D.Name);
    SymbolTable *SymTab = 0;
    Value *N = 0;
    if (inFunctionScope()) {
      SymTab = &CurMeth.CurrentFunction->getSymbolTable();
      N = SymTab->lookup(Type::TypeTy, Name);
    }

    if (N == 0) {
      // Symbol table doesn't automatically chain yet... because the function
      // hasn't been added to the module...
      //
      SymTab = &CurModule.CurrentModule->getSymbolTable();
      N = SymTab->lookup(Type::TypeTy, Name);
      if (N == 0) break;
    }

    D.destroy();  // Free old strdup'd memory...
    return cast<const Type>(N);
  }
  default:
    ThrowException("Internal parser error: Invalid symbol type reference!");
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?

  std::map<ValID, PATypeHolder> &LateResolver = inFunctionScope() ? 
    CurMeth.LateResolveTypes : CurModule.LateResolveTypes;
  
  std::map<ValID, PATypeHolder>::iterator I = LateResolver.find(D);
  if (I != LateResolver.end()) {
    return I->second;
  }

  Type *Typ = OpaqueType::get();
  LateResolver.insert(std::make_pair(D, Typ));
  return Typ;
}

static Value *lookupInSymbolTable(const Type *Ty, const std::string &Name) {
  SymbolTable &SymTab = 
    inFunctionScope() ? CurMeth.CurrentFunction->getSymbolTable() :
                        CurModule.CurrentModule->getSymbolTable();
  return SymTab.lookup(Ty, Name);
}

// getValNonImprovising - Look up the value specified by the provided type and
// the provided ValID.  If the value exists and has already been defined, return
// it.  Otherwise return null.
//
static Value *getValNonImprovising(const Type *Ty, const ValID &D) {
  if (isa<FunctionType>(Ty))
    ThrowException("Functions are not values and "
                   "must be referenced as pointers");

  switch (D.Type) {
  case ValID::NumberVal: {                 // Is it a numbered definition?
    unsigned type = Ty->getUniqueID();
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    if (type < CurModule.Values.size()) {
      if (Num < CurModule.Values[type].size()) 
        return CurModule.Values[type][Num];

      Num -= CurModule.Values[type].size();
    }

    // Make sure that our type is within bounds
    if (CurMeth.Values.size() <= type) return 0;

    // Check that the number is within bounds...
    if (CurMeth.Values[type].size() <= Num) return 0;
  
    return CurMeth.Values[type][Num];
  }

  case ValID::NameVal: {                // Is it a named definition?
    Value *N = lookupInSymbolTable(Ty, std::string(D.Name));
    if (N == 0) return 0;

    D.destroy();  // Free old strdup'd memory...
    return N;
  }

  // Check to make sure that "Ty" is an integral type, and that our 
  // value will fit into the specified type...
  case ValID::ConstSIntVal:    // Is it a constant pool reference??
    if (!ConstantSInt::isValueValidForType(Ty, D.ConstPool64))
      ThrowException("Signed integral constant '" +
                     itostr(D.ConstPool64) + "' is invalid for type '" + 
                     Ty->getDescription() + "'!");
    return ConstantSInt::get(Ty, D.ConstPool64);

  case ValID::ConstUIntVal:     // Is it an unsigned const pool reference?
    if (!ConstantUInt::isValueValidForType(Ty, D.UConstPool64)) {
      if (!ConstantSInt::isValueValidForType(Ty, D.ConstPool64)) {
	ThrowException("Integral constant '" + utostr(D.UConstPool64) +
                       "' is invalid or out of range!");
      } else {     // This is really a signed reference.  Transmogrify.
	return ConstantSInt::get(Ty, D.ConstPool64);
      }
    } else {
      return ConstantUInt::get(Ty, D.UConstPool64);
    }

  case ValID::ConstFPVal:        // Is it a floating point const pool reference?
    if (!ConstantFP::isValueValidForType(Ty, D.ConstPoolFP))
      ThrowException("FP constant invalid for type!!");
    return ConstantFP::get(Ty, D.ConstPoolFP);
    
  case ValID::ConstNullVal:      // Is it a null value?
    if (!isa<PointerType>(Ty))
      ThrowException("Cannot create a a non pointer null!");
    return ConstantPointerNull::get(cast<PointerType>(Ty));
    
  case ValID::ConstantVal:       // Fully resolved constant?
    if (D.ConstantValue->getType() != Ty)
      ThrowException("Constant expression type different from required type!");
    return D.ConstantValue;

  default:
    assert(0 && "Unhandled case!");
    return 0;
  }   // End of switch

  assert(0 && "Unhandled case!");
  return 0;
}


// getVal - This function is identical to getValNonImprovising, except that if a
// value is not already defined, it "improvises" by creating a placeholder var
// that looks and acts just like the requested variable.  When the value is
// defined later, all uses of the placeholder variable are replaced with the
// real thing.
//
static Value *getVal(const Type *Ty, const ValID &D) {
  assert(Ty != Type::TypeTy && "Should use getTypeVal for types!");

  // See if the value has already been defined...
  Value *V = getValNonImprovising(Ty, D);
  if (V) return V;

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  Value *d = 0;
  switch (Ty->getPrimitiveID()) {
  case Type::LabelTyID:  d = new   BBPlaceHolder(Ty, D); break;
  default:               d = new ValuePlaceHolder(Ty, D); break;
  }

  assert(d != 0 && "How did we not make something?");
  if (inFunctionScope())
    InsertValue(d, CurMeth.LateResolveValues);
  else 
    InsertValue(d, CurModule.LateResolveValues);
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
static void ResolveDefinitions(std::vector<ValueList> &LateResolvers,
                               std::vector<ValueList> *FutureLateResolvers) {
  // Loop over LateResolveDefs fixing up stuff that couldn't be resolved
  for (unsigned ty = 0; ty < LateResolvers.size(); ty++) {
    while (!LateResolvers[ty].empty()) {
      Value *V = LateResolvers[ty].back();
      assert(!isa<Type>(V) && "Types should be in LateResolveTypes!");

      LateResolvers[ty].pop_back();
      ValID &DID = getValIDFromPlaceHolder(V);

      Value *TheRealValue = getValNonImprovising(Type::getUniqueIDType(ty),DID);
      if (TheRealValue) {
        V->replaceAllUsesWith(TheRealValue);
        delete V;
      } else if (FutureLateResolvers) {
        // Functions have their unresolved items forwarded to the module late
        // resolver table
        InsertValue(V, *FutureLateResolvers);
      } else {
	if (DID.Type == ValID::NameVal)
	  ThrowException("Reference to an invalid definition: '" +DID.getName()+
			 "' of type '" + V->getType()->getDescription() + "'",
			 getLineNumFromPlaceHolder(V));
	else
	  ThrowException("Reference to an invalid definition: #" +
			 itostr(DID.Num) + " of type '" + 
			 V->getType()->getDescription() + "'",
			 getLineNumFromPlaceHolder(V));
      }
    }
  }

  LateResolvers.clear();
}

// ResolveTypeTo - A brand new type was just declared.  This means that (if
// name is not null) things referencing Name can be resolved.  Otherwise, things
// refering to the number can be resolved.  Do this now.
//
static void ResolveTypeTo(char *Name, const Type *ToTy) {
  std::vector<PATypeHolder> &Types = inFunctionScope() ? 
     CurMeth.Types : CurModule.Types;

   ValID D;
   if (Name) D = ValID::create(Name);
   else      D = ValID::create((int)Types.size());

   std::map<ValID, PATypeHolder> &LateResolver = inFunctionScope() ? 
     CurMeth.LateResolveTypes : CurModule.LateResolveTypes;
  
   std::map<ValID, PATypeHolder>::iterator I = LateResolver.find(D);
   if (I != LateResolver.end()) {
     ((DerivedType*)I->second.get())->refineAbstractTypeTo(ToTy);
     LateResolver.erase(I);
   }
}

// ResolveTypes - At this point, all types should be resolved.  Any that aren't
// are errors.
//
static void ResolveTypes(std::map<ValID, PATypeHolder> &LateResolveTypes) {
  if (!LateResolveTypes.empty()) {
    const ValID &DID = LateResolveTypes.begin()->first;

    if (DID.Type == ValID::NameVal)
      ThrowException("Reference to an invalid type: '" +DID.getName() + "'");
    else
      ThrowException("Reference to an invalid type: #" + itostr(DID.Num));
  }
}


// setValueName - Set the specified value to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is freed by this function.
//
// This function returns true if the value has already been defined, but is
// allowed to be redefined in the specified context.  If the name is a new name
// for the typeplane, false is returned.
//
static bool setValueName(Value *V, char *NameStr) {
  if (NameStr == 0) return false;
  
  std::string Name(NameStr);      // Copy string
  free(NameStr);                  // Free old string

  if (V->getType() == Type::VoidTy) 
    ThrowException("Can't assign name '" + Name + 
		   "' to a null valued instruction!");

  SymbolTable &ST = inFunctionScope() ? 
    CurMeth.CurrentFunction->getSymbolTable() : 
    CurModule.CurrentModule->getSymbolTable();

  Value *Existing = ST.lookup(V->getType(), Name);
  if (Existing) {    // Inserting a name that is already defined???
    // There is only one case where this is allowed: when we are refining an
    // opaque type.  In this case, Existing will be an opaque type.
    if (const Type *Ty = dyn_cast<const Type>(Existing)) {
      if (const OpaqueType *OpTy = dyn_cast<OpaqueType>(Ty)) {
	// We ARE replacing an opaque type!
	((OpaqueType*)OpTy)->refineAbstractTypeTo(cast<Type>(V));
	return true;
      }
    }

    // Otherwise, we are a simple redefinition of a value, check to see if it
    // is defined the same as the old one...
    if (const Type *Ty = dyn_cast<Type>(Existing)) {
      if (Ty == cast<Type>(V)) return true;  // Yes, it's equal.
      // std::cerr << "Type: " << Ty->getDescription() << " != "
      //      << cast<const Type>(V)->getDescription() << "!\n";
    } else if (const Constant *C = dyn_cast<Constant>(Existing)) {
      if (C == V) return true;      // Constants are equal to themselves
    } else if (GlobalVariable *EGV = dyn_cast<GlobalVariable>(Existing)) {
      // We are allowed to redefine a global variable in two circumstances:
      // 1. If at least one of the globals is uninitialized or 
      // 2. If both initializers have the same value.
      //
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
        if (!EGV->hasInitializer() || !GV->hasInitializer() ||
             EGV->getInitializer() == GV->getInitializer()) {

          // Make sure the existing global version gets the initializer!  Make
          // sure that it also gets marked const if the new version is.
          if (GV->hasInitializer() && !EGV->hasInitializer())
            EGV->setInitializer(GV->getInitializer());
          if (GV->isConstant())
            EGV->setConstant(true);
          EGV->setLinkage(GV->getLinkage());
          
	  delete GV;     // Destroy the duplicate!
          return true;   // They are equivalent!
        }
      }
    }
    ThrowException("Redefinition of value named '" + Name + "' in the '" +
		   V->getType()->getDescription() + "' type plane!");
  }

  V->setName(Name, &ST);
  return false;
}


//===----------------------------------------------------------------------===//
// Code for handling upreferences in type names...
//

// TypeContains - Returns true if Ty contains E in it.
//
static bool TypeContains(const Type *Ty, const Type *E) {
  return find(df_begin(Ty), df_end(Ty), E) != df_end(Ty);
}


static std::vector<std::pair<unsigned, OpaqueType *> > UpRefs;

static PATypeHolder HandleUpRefs(const Type *ty) {
  PATypeHolder Ty(ty);
  UR_OUT("Type '" << ty->getDescription() << 
         "' newly formed.  Resolving upreferences.\n" <<
         UpRefs.size() << " upreferences active!\n");
  for (unsigned i = 0; i < UpRefs.size(); ) {
    UR_OUT("  UR#" << i << " - TypeContains(" << Ty->getDescription() << ", " 
	   << UpRefs[i].second->getDescription() << ") = " 
	   << (TypeContains(Ty, UpRefs[i].second) ? "true" : "false") << endl);
    if (TypeContains(Ty, UpRefs[i].second)) {
      unsigned Level = --UpRefs[i].first;   // Decrement level of upreference
      UR_OUT("  Uplevel Ref Level = " << Level << endl);
      if (Level == 0) {                     // Upreference should be resolved! 
	UR_OUT("  * Resolving upreference for "
               << UpRefs[i].second->getDescription() << endl;
	       std::string OldName = UpRefs[i].second->getDescription());
	UpRefs[i].second->refineAbstractTypeTo(Ty);
	UpRefs.erase(UpRefs.begin()+i);     // Remove from upreference list...
	UR_OUT("  * Type '" << OldName << "' refined upreference to: "
	       << (const void*)Ty << ", " << Ty->getDescription() << endl);
	continue;
      }
    }

    ++i;                                  // Otherwise, no resolve, move on...
  }
  // FIXME: TODO: this should return the updated type
  return Ty;
}


//===----------------------------------------------------------------------===//
//            RunVMAsmParser - Define an interface to this parser
//===----------------------------------------------------------------------===//
//
Module *RunVMAsmParser(const std::string &Filename, FILE *F) {
  llvmAsmin = F;
  CurFilename = Filename;
  llvmAsmlineno = 1;      // Reset the current line number...

  // Allocate a new module to read
  CurModule.CurrentModule = new Module(Filename);
  yyparse();       // Parse the file.
  Module *Result = ParserResult;
  llvmAsmin = stdin;    // F is about to go away, don't use it anymore...
  ParserResult = 0;

  return Result;
}

%}

%union {
  Module                           *ModuleVal;
  Function                         *FunctionVal;
  std::pair<PATypeHolder*, char*>  *ArgVal;
  BasicBlock                       *BasicBlockVal;
  TerminatorInst                   *TermInstVal;
  Instruction                      *InstVal;
  Constant                         *ConstVal;

  const Type                       *PrimType;
  PATypeHolder                     *TypeVal;
  Value                            *ValueVal;

  std::vector<std::pair<PATypeHolder*,char*> > *ArgList;
  std::vector<Value*>              *ValueList;
  std::list<PATypeHolder>          *TypeList;
  std::list<std::pair<Value*,
                      BasicBlock*> > *PHIList; // Represent the RHS of PHI node
  std::vector<std::pair<Constant*, BasicBlock*> > *JumpTable;
  std::vector<Constant*>           *ConstVector;

  GlobalValue::LinkageTypes         Linkage;
  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  double                            FPVal;
  bool                              BoolVal;

  char                             *StrVal;   // This memory is strdup'd!
  ValID                             ValIDVal; // strdup'd memory maybe!

  Instruction::BinaryOps            BinaryOpVal;
  Instruction::TermOps              TermOpVal;
  Instruction::MemoryOps            MemOpVal;
  Instruction::OtherOps             OtherOpVal;
  Module::Endianness                Endianness;
}

%type <ModuleVal>     Module FunctionList
%type <FunctionVal>   Function FunctionProto FunctionHeader BasicBlockList
%type <BasicBlockVal> BasicBlock InstructionList
%type <TermInstVal>   BBTerminatorInst
%type <InstVal>       Inst InstVal MemoryInst
%type <ConstVal>      ConstVal ConstExpr
%type <ConstVector>   ConstVector
%type <ArgList>       ArgList ArgListH
%type <ArgVal>        ArgVal
%type <PHIList>       PHIList
%type <ValueList>     ValueRefList ValueRefListE  // For call param lists
%type <ValueList>     IndexList                   // For GEP derived indices
%type <TypeList>      TypeListI ArgTypeListI
%type <JumpTable>     JumpTable
%type <BoolVal>       GlobalType                  // GLOBAL or CONSTANT?
%type <Linkage>       OptLinkage
%type <Endianness>    BigOrLittle

// ValueRef - Unresolved reference to a definition or BB
%type <ValIDVal>      ValueRef ConstValueRef SymbolicValueRef
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
%type  <TypeVal> Types TypesV UpRTypes UpRTypesV
%type  <PrimType> SIntType UIntType IntType FPType PrimType   // Classifications
%token <PrimType> VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token <PrimType> FLOAT DOUBLE TYPE LABEL

%token <StrVal>     VAR_ID LABELSTR STRINGCONSTANT
%type  <StrVal>  OptVAR_ID OptAssign FuncName


%token IMPLEMENTATION ZEROINITIALIZER TRUE FALSE BEGINTOK ENDTOK
%token  DECLARE GLOBAL CONSTANT
%token TO EXCEPT DOTDOTDOT NULL_TOK CONST INTERNAL LINKONCE APPENDING
%token OPAQUE NOT EXTERNAL TARGET ENDIAN POINTERSIZE LITTLE BIG

// Basic Block Terminating Operators 
%token <TermOpVal> RET BR SWITCH

// Binary Operators 
%type  <BinaryOpVal> BinaryOps  // all the binary operators
%type  <BinaryOpVal> ArithmeticOps LogicalOps SetCondOps // Binops Subcatagories
%token <BinaryOpVal> ADD SUB MUL DIV REM AND OR XOR
%token <BinaryOpVal> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comarators

// Memory Instructions
%token <MemOpVal> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR

// Other Operators
%type  <OtherOpVal> ShiftOps
%token <OtherOpVal> PHI CALL INVOKE CAST SHL SHR VA_ARG

%start Module
%%

// Handle constant integer size restriction and conversion...
//

INTVAL : SINTVAL;
INTVAL : UINTVAL {
  if ($1 > (uint32_t)INT32_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  $$ = (int32_t)$1;
};


EINT64VAL : ESINT64VAL;      // These have same type and can't cause problems...
EINT64VAL : EUINT64VAL {
  if ($1 > (uint64_t)INT64_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  $$ = (int64_t)$1;
};

// Operations that are notably excluded from this list include: 
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
//
ArithmeticOps: ADD | SUB | MUL | DIV | REM;
LogicalOps   : AND | OR | XOR;
SetCondOps   : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE;
BinaryOps : ArithmeticOps | LogicalOps | SetCondOps;

ShiftOps  : SHL | SHR;

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType :  LONG |  INT |  SHORT | SBYTE;
UIntType : ULONG | UINT | USHORT | UBYTE;
IntType  : SIntType | UIntType;
FPType   : FLOAT | DOUBLE;

// OptAssign - Value producing statements have an optional assignment component
OptAssign : VAR_ID '=' {
    $$ = $1;
  }
  | /*empty*/ { 
    $$ = 0; 
  };

OptLinkage : INTERNAL  { $$ = GlobalValue::InternalLinkage; } |
             LINKONCE  { $$ = GlobalValue::LinkOnceLinkage; } |
             APPENDING { $$ = GlobalValue::AppendingLinkage; } |
             /*empty*/ { $$ = GlobalValue::ExternalLinkage; };

//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (function returning void for example).  To have
// access to it, a user must explicitly use TypesV.
//

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    : Types    | VOID { $$ = new PATypeHolder($1); };
UpRTypesV : UpRTypes | VOID { $$ = new PATypeHolder($1); };

Types     : UpRTypes {
    if (UpRefs.size())
      ThrowException("Invalid upreference in type: " + (*$1)->getDescription());
    $$ = $1;
  };


// Derived types are added later...
//
PrimType : BOOL | SBYTE | UBYTE | SHORT  | USHORT | INT   | UINT ;
PrimType : LONG | ULONG | FLOAT | DOUBLE | TYPE   | LABEL;
UpRTypes : OPAQUE {
    $$ = new PATypeHolder(OpaqueType::get());
  }
  | PrimType {
    $$ = new PATypeHolder($1);
  };
UpRTypes : SymbolicValueRef {            // Named types are also simple types...
  $$ = new PATypeHolder(getTypeVal($1));
};

// Include derived types in the Types production.
//
UpRTypes : '\\' EUINT64VAL {                   // Type UpReference
    if ($2 > (uint64_t)INT64_MAX) ThrowException("Value out of range!");
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder
    UpRefs.push_back(std::make_pair((unsigned)$2, OT));  // Add to vector...
    $$ = new PATypeHolder(OT);
    UR_OUT("New Upreference!\n");
  }
  | UpRTypesV '(' ArgTypeListI ')' {           // Function derived type?
    std::vector<const Type*> Params;
    mapto($3->begin(), $3->end(), std::back_inserter(Params), 
	  std::mem_fun_ref(&PATypeHandle::get));
    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    $$ = new PATypeHolder(HandleUpRefs(FunctionType::get(*$1,Params,isVarArg)));
    delete $3;      // Delete the argument list
    delete $1;      // Delete the old type handle
  }
  | '[' EUINT64VAL 'x' UpRTypes ']' {          // Sized array type?
    $$ = new PATypeHolder(HandleUpRefs(ArrayType::get(*$4, (unsigned)$2)));
    delete $4;
  }
  | '{' TypeListI '}' {                        // Structure type?
    std::vector<const Type*> Elements;
    mapto($2->begin(), $2->end(), std::back_inserter(Elements), 
	std::mem_fun_ref(&PATypeHandle::get));

    $$ = new PATypeHolder(HandleUpRefs(StructType::get(Elements)));
    delete $2;
  }
  | '{' '}' {                                  // Empty structure type?
    $$ = new PATypeHolder(StructType::get(std::vector<const Type*>()));
  }
  | UpRTypes '*' {                             // Pointer type?
    $$ = new PATypeHolder(HandleUpRefs(PointerType::get(*$1)));
    delete $1;
  };

// TypeList - Used for struct declarations and as a basis for function type 
// declaration type lists
//
TypeListI : UpRTypes {
    $$ = new std::list<PATypeHolder>();
    $$->push_back(*$1); delete $1;
  }
  | TypeListI ',' UpRTypes {
    ($$=$1)->push_back(*$3); delete $3;
  };

// ArgTypeList - List of types for a function type declaration...
ArgTypeListI : TypeListI
  | TypeListI ',' DOTDOTDOT {
    ($$=$1)->push_back(Type::VoidTy);
  }
  | DOTDOTDOT {
    ($$ = new std::list<PATypeHolder>())->push_back(Type::VoidTy);
  }
  | /*empty*/ {
    $$ = new std::list<PATypeHolder>();
  };

// ConstVal - The various declarations that go into the constant pool.  This
// production is used ONLY to represent constants that show up AFTER a 'const',
// 'constant' or 'global' token at global scope.  Constants that can be inlined
// into other expressions (such as integers and constexprs) are handled by the
// ResolvedVal, ValueRef and ConstValueRef productions.
//
ConstVal: Types '[' ConstVector ']' { // Nonempty unsized arr
    const ArrayType *ATy = dyn_cast<const ArrayType>($1->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*$1)->getDescription() + "'!");
    const Type *ETy = ATy->getElementType();
    int NumElements = ATy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)$3->size())
      ThrowException("Type mismatch: constant sized array initialized with " +
		     utostr($3->size()) +  " arguments, but has size of " + 
		     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < $3->size(); i++) {
      if (ETy != (*$3)[i]->getType())
	ThrowException("Element #" + utostr(i) + " is not of type '" + 
		       ETy->getDescription() +"' as required!\nIt is of type '"+
		       (*$3)[i]->getType()->getDescription() + "'.");
    }

    $$ = ConstantArray::get(ATy, *$3);
    delete $1; delete $3;
  }
  | Types '[' ']' {
    const ArrayType *ATy = dyn_cast<const ArrayType>($1->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    if (NumElements != -1 && NumElements != 0) 
      ThrowException("Type mismatch: constant sized array initialized with 0"
		     " arguments, but has size of " + itostr(NumElements) +"!");
    $$ = ConstantArray::get(ATy, std::vector<Constant*>());
    delete $1;
  }
  | Types 'c' STRINGCONSTANT {
    const ArrayType *ATy = dyn_cast<const ArrayType>($1->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    const Type *ETy = ATy->getElementType();
    char *EndStr = UnEscapeLexed($3, true);
    if (NumElements != -1 && NumElements != (EndStr-$3))
      ThrowException("Can't build string constant of size " + 
		     itostr((int)(EndStr-$3)) +
		     " when array has size " + itostr(NumElements) + "!");
    std::vector<Constant*> Vals;
    if (ETy == Type::SByteTy) {
      for (char *C = $3; C != EndStr; ++C)
	Vals.push_back(ConstantSInt::get(ETy, *C));
    } else if (ETy == Type::UByteTy) {
      for (char *C = $3; C != EndStr; ++C)
	Vals.push_back(ConstantUInt::get(ETy, (unsigned char)*C));
    } else {
      free($3);
      ThrowException("Cannot build string arrays of non byte sized elements!");
    }
    free($3);
    $$ = ConstantArray::get(ATy, Vals);
    delete $1;
  }
  | Types '{' ConstVector '}' {
    const StructType *STy = dyn_cast<const StructType>($1->get());
    if (STy == 0)
      ThrowException("Cannot make struct constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    if ($3->size() != STy->getNumContainedTypes())
      ThrowException("Illegal number of initializers for structure type!");

    // Check to ensure that constants are compatible with the type initializer!
    for (unsigned i = 0, e = $3->size(); i != e; ++i)
      if ((*$3)[i]->getType() != STy->getElementTypes()[i])
        ThrowException("Expected type '" +
                       STy->getElementTypes()[i]->getDescription() +
                       "' for element #" + utostr(i) +
                       " of structure initializer!");

    $$ = ConstantStruct::get(STy, *$3);
    delete $1; delete $3;
  }
  | Types '{' '}' {
    const StructType *STy = dyn_cast<const StructType>($1->get());
    if (STy == 0)
      ThrowException("Cannot make struct constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    if (STy->getNumContainedTypes() != 0)
      ThrowException("Illegal number of initializers for structure type!");

    $$ = ConstantStruct::get(STy, std::vector<Constant*>());
    delete $1;
  }
  | Types NULL_TOK {
    const PointerType *PTy = dyn_cast<const PointerType>($1->get());
    if (PTy == 0)
      ThrowException("Cannot make null pointer constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    $$ = ConstantPointerNull::get(PTy);
    delete $1;
  }
  | Types SymbolicValueRef {
    const PointerType *Ty = dyn_cast<const PointerType>($1->get());
    if (Ty == 0)
      ThrowException("Global const reference must be a pointer type!");

    // ConstExprs can exist in the body of a function, thus creating
    // ConstantPointerRefs whenever they refer to a variable.  Because we are in
    // the context of a function, getValNonImprovising will search the functions
    // symbol table instead of the module symbol table for the global symbol,
    // which throws things all off.  To get around this, we just tell
    // getValNonImprovising that we are at global scope here.
    //
    Function *SavedCurFn = CurMeth.CurrentFunction;
    CurMeth.CurrentFunction = 0;

    Value *V = getValNonImprovising(Ty, $2);

    CurMeth.CurrentFunction = SavedCurFn;

    // If this is an initializer for a constant pointer, which is referencing a
    // (currently) undefined variable, create a stub now that shall be replaced
    // in the future with the right type of variable.
    //
    if (V == 0) {
      assert(isa<PointerType>(Ty) && "Globals may only be used as pointers!");
      const PointerType *PT = cast<PointerType>(Ty);

      // First check to see if the forward references value is already created!
      PerModuleInfo::GlobalRefsType::iterator I =
	CurModule.GlobalRefs.find(std::make_pair(PT, $2));
    
      if (I != CurModule.GlobalRefs.end()) {
	V = I->second;             // Placeholder already exists, use it...
      } else {
	// TODO: Include line number info by creating a subclass of
	// TODO: GlobalVariable here that includes the said information!
	
	// Create a placeholder for the global variable reference...
	GlobalVariable *GV = new GlobalVariable(PT->getElementType(),
                                                false,
                                                GlobalValue::ExternalLinkage);
	// Keep track of the fact that we have a forward ref to recycle it
	CurModule.GlobalRefs.insert(std::make_pair(std::make_pair(PT, $2), GV));

	// Must temporarily push this value into the module table...
	CurModule.CurrentModule->getGlobalList().push_back(GV);
	V = GV;
      }
    }

    GlobalValue *GV = cast<GlobalValue>(V);
    $$ = ConstantPointerRef::get(GV);
    delete $1;            // Free the type handle
  }
  | Types ConstExpr {
    if ($1->get() != $2->getType())
      ThrowException("Mismatched types for constant expression!");
    $$ = $2;
    delete $1;
  }
  | Types ZEROINITIALIZER {
    $$ = Constant::getNullValue($1->get());
    delete $1;
  };

ConstVal : SIntType EINT64VAL {      // integral constants
    if (!ConstantSInt::isValueValidForType($1, $2))
      ThrowException("Constant value doesn't fit in type!");
    $$ = ConstantSInt::get($1, $2);
  }
  | UIntType EUINT64VAL {            // integral constants
    if (!ConstantUInt::isValueValidForType($1, $2))
      ThrowException("Constant value doesn't fit in type!");
    $$ = ConstantUInt::get($1, $2);
  }
  | BOOL TRUE {                      // Boolean constants
    $$ = ConstantBool::True;
  }
  | BOOL FALSE {                     // Boolean constants
    $$ = ConstantBool::False;
  }
  | FPType FPVAL {                   // Float & Double constants
    $$ = ConstantFP::get($1, $2);
  };


ConstExpr: CAST '(' ConstVal TO Types ')' {
    $$ = ConstantExpr::getCast($3, $5->get());
    delete $5;
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
    if (!isa<PointerType>($3->getType()))
      ThrowException("GetElementPtr requires a pointer operand!");

    const Type *IdxTy =
      GetElementPtrInst::getIndexedType($3->getType(), *$4, true);
    if (!IdxTy)
      ThrowException("Index list invalid for constant getelementptr!");

    std::vector<Constant*> IdxVec;
    for (unsigned i = 0, e = $4->size(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>((*$4)[i]))
        IdxVec.push_back(C);
      else
        ThrowException("Indices to constant getelementptr must be constants!");

    delete $4;

    $$ = ConstantExpr::getGetElementPtr($3, IdxVec);
  }
  | BinaryOps '(' ConstVal ',' ConstVal ')' {
    if ($3->getType() != $5->getType())
      ThrowException("Binary operator types must match!");
    $$ = ConstantExpr::get($1, $3, $5);
  }
  | ShiftOps '(' ConstVal ',' ConstVal ')' {
    if ($5->getType() != Type::UByteTy)
      ThrowException("Shift count for shift constant must be unsigned byte!");
    if (!$3->getType()->isIntegral())
      ThrowException("Shift constant expression requires integral operand!");
    $$ = ConstantExpr::getShift($1, $3, $5);
  };


// ConstVector - A list of comma seperated constants.
ConstVector : ConstVector ',' ConstVal {
    ($$ = $1)->push_back($3);
  }
  | ConstVal {
    $$ = new std::vector<Constant*>();
    $$->push_back($1);
  };


// GlobalType - Match either GLOBAL or CONSTANT for global declarations...
GlobalType : GLOBAL { $$ = false; } | CONSTANT { $$ = true; };


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module : FunctionList {
  $$ = ParserResult = $1;
  CurModule.ModuleDone();
};

// FunctionList - A list of functions, preceeded by a constant pool.
//
FunctionList : FunctionList Function {
    $$ = $1;
    assert($2->getParent() == 0 && "Function already in module!");
    $1->getFunctionList().push_back($2);
    CurMeth.FunctionDone();
  } 
  | FunctionList FunctionProto {
    $$ = $1;
  }
  | FunctionList IMPLEMENTATION {
    $$ = $1;
  }
  | ConstPool {
    $$ = CurModule.CurrentModule;
    // Resolve circular types before we parse the body of the module
    ResolveTypes(CurModule.LateResolveTypes);
  };

// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign CONST ConstVal { 
    if (!setValueName($4, $2))
      InsertValue($4);
  }
  | ConstPool OptAssign TYPE TypesV {  // Types can be defined in the const pool
    // Eagerly resolve types.  This is not an optimization, this is a
    // requirement that is due to the fact that we could have this:
    //
    // %list = type { %list * }
    // %list = type { %list * }    ; repeated type decl
    //
    // If types are not resolved eagerly, then the two types will not be
    // determined to be the same type!
    //
    ResolveTypeTo($2, $4->get());

    // TODO: FIXME when Type are not const
    if (!setValueName(const_cast<Type*>($4->get()), $2)) {
      // If this is not a redefinition of a type...
      if (!$2) {
        InsertType($4->get(),
                   inFunctionScope() ? CurMeth.Types : CurModule.Types);
      }
    }

    delete $4;
  }
  | ConstPool FunctionProto {       // Function prototypes can be in const pool
  }
  | ConstPool OptAssign OptLinkage GlobalType ConstVal {
    const Type *Ty = $5->getType();
    // Global declarations appear in Constant Pool
    Constant *Initializer = $5;
    if (Initializer == 0)
      ThrowException("Global value initializer is not a constant!");
    
    GlobalVariable *GV = new GlobalVariable(Ty, $4, $3, Initializer);
    if (!setValueName(GV, $2)) {   // If not redefining...
      CurModule.CurrentModule->getGlobalList().push_back(GV);
      int Slot = InsertValue(GV, CurModule.Values);

      if (Slot != -1) {
	CurModule.DeclareNewGlobalValue(GV, ValID::create(Slot));
      } else {
	CurModule.DeclareNewGlobalValue(GV, ValID::create(
				                (char*)GV->getName().c_str()));
      }
    }
  }
  | ConstPool OptAssign EXTERNAL GlobalType Types {
    const Type *Ty = *$5;
    // Global declarations appear in Constant Pool
    GlobalVariable *GV = new GlobalVariable(Ty,$4,GlobalValue::ExternalLinkage);
    if (!setValueName(GV, $2)) {   // If not redefining...
      CurModule.CurrentModule->getGlobalList().push_back(GV);
      int Slot = InsertValue(GV, CurModule.Values);

      if (Slot != -1) {
	CurModule.DeclareNewGlobalValue(GV, ValID::create(Slot));
      } else {
	assert(GV->hasName() && "Not named and not numbered!?");
	CurModule.DeclareNewGlobalValue(GV, ValID::create(
				                (char*)GV->getName().c_str()));
      }
    }
    delete $5;
  }
  | ConstPool TARGET TargetDefinition { 
  }
  | /* empty: end of list */ { 
  };



BigOrLittle : BIG    { $$ = Module::BigEndian; };
BigOrLittle : LITTLE { $$ = Module::LittleEndian; };

TargetDefinition : ENDIAN '=' BigOrLittle {
    CurModule.CurrentModule->setEndianness($3);
  }
  | POINTERSIZE '=' EUINT64VAL {
    if ($3 == 32)
      CurModule.CurrentModule->setPointerSize(Module::Pointer32);
    else if ($3 == 64)
      CurModule.CurrentModule->setPointerSize(Module::Pointer64);
    else
      ThrowException("Invalid pointer size: '" + utostr($3) + "'!");
  };


//===----------------------------------------------------------------------===//
//                       Rules to match Function Headers
//===----------------------------------------------------------------------===//

OptVAR_ID : VAR_ID | /*empty*/ { $$ = 0; };

ArgVal : Types OptVAR_ID {
  if (*$1 == Type::VoidTy)
    ThrowException("void typed arguments are invalid!");
  $$ = new std::pair<PATypeHolder*, char*>($1, $2);
};

ArgListH : ArgListH ',' ArgVal {
    $$ = $1;
    $1->push_back(*$3);
    delete $3;
  }
  | ArgVal {
    $$ = new std::vector<std::pair<PATypeHolder*,char*> >();
    $$->push_back(*$1);
    delete $1;
  };

ArgList : ArgListH {
    $$ = $1;
  }
  | ArgListH ',' DOTDOTDOT {
    $$ = $1;
    $$->push_back(std::pair<PATypeHolder*,
                            char*>(new PATypeHolder(Type::VoidTy), 0));
  }
  | DOTDOTDOT {
    $$ = new std::vector<std::pair<PATypeHolder*,char*> >();
    $$->push_back(std::make_pair(new PATypeHolder(Type::VoidTy), (char*)0));
  }
  | /* empty */ {
    $$ = 0;
  };

FuncName : VAR_ID | STRINGCONSTANT;

FunctionHeaderH : TypesV FuncName '(' ArgList ')' {
  UnEscapeLexed($2);
  std::string FunctionName($2);
  
  std::vector<const Type*> ParamTypeList;
  if ($4) {   // If there are arguments...
    for (std::vector<std::pair<PATypeHolder*,char*> >::iterator I = $4->begin();
         I != $4->end(); ++I)
      ParamTypeList.push_back(I->first->get());
  }

  bool isVarArg = ParamTypeList.size() && ParamTypeList.back() == Type::VoidTy;
  if (isVarArg) ParamTypeList.pop_back();

  const FunctionType *FT = FunctionType::get(*$1, ParamTypeList, isVarArg);
  const PointerType *PFT = PointerType::get(FT);
  delete $1;

  Function *Fn = 0;
  // Is the function already in symtab?
  if ((Fn = CurModule.CurrentModule->getFunction(FunctionName, FT))) {
    // Yes it is.  If this is the case, either we need to be a forward decl,
    // or it needs to be.
    if (!CurMeth.isDeclare && !Fn->isExternal())
      ThrowException("Redefinition of function '" + FunctionName + "'!");
    
    // If we found a preexisting function prototype, remove it from the
    // module, so that we don't get spurious conflicts with global & local
    // variables.
    //
    CurModule.CurrentModule->getFunctionList().remove(Fn);

    // Make sure to strip off any argument names so we can't get conflicts...
    for (Function::aiterator AI = Fn->abegin(), AE = Fn->aend(); AI != AE; ++AI)
      AI->setName("");

  } else  {  // Not already defined?
    Fn = new Function(FT, GlobalValue::ExternalLinkage, FunctionName);
    InsertValue(Fn, CurModule.Values);
    CurModule.DeclareNewGlobalValue(Fn, ValID::create($2));
  }
  free($2);  // Free strdup'd memory!

  CurMeth.FunctionStart(Fn);

  // Add all of the arguments we parsed to the function...
  if ($4) {                     // Is null if empty...
    if (isVarArg) {  // Nuke the last entry
      assert($4->back().first->get() == Type::VoidTy && $4->back().second == 0&&
             "Not a varargs marker!");
      delete $4->back().first;
      $4->pop_back();  // Delete the last entry
    }
    Function::aiterator ArgIt = Fn->abegin();
    for (std::vector<std::pair<PATypeHolder*, char*> >::iterator I =$4->begin();
         I != $4->end(); ++I, ++ArgIt) {
      delete I->first;                          // Delete the typeholder...

      if (setValueName(ArgIt, I->second))       // Insert arg into symtab...
        assert(0 && "No arg redef allowed!");
      
      InsertValue(ArgIt);
    }

    delete $4;                     // We're now done with the argument list
  }
};

BEGIN : BEGINTOK | '{';                // Allow BEGIN or '{' to start a function

FunctionHeader : OptLinkage FunctionHeaderH BEGIN {
  $$ = CurMeth.CurrentFunction;

  // Make sure that we keep track of the linkage type even if there was a
  // previous "declare".
  $$->setLinkage($1);

  // Resolve circular types before we parse the body of the function.
  ResolveTypes(CurMeth.LateResolveTypes);
};

END : ENDTOK | '}';                    // Allow end of '}' to end a function

Function : BasicBlockList END {
  $$ = $1;
};

FunctionProto : DECLARE { CurMeth.isDeclare = true; } FunctionHeaderH {
  $$ = CurMeth.CurrentFunction;
  assert($$->getParent() == 0 && "Function already in module!");
  CurModule.CurrentModule->getFunctionList().push_back($$);
  CurMeth.FunctionDone();
};

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
    $$ = ValID::create(ConstantBool::True);
  } 
  | FALSE {
    $$ = ValID::create(ConstantBool::False);
  }
  | NULL_TOK {
    $$ = ValID::createNull();
  }
  | ConstExpr {
    $$ = ValID::create($1);
  };

// SymbolicValueRef - Reference to one of two ways of symbolically refering to
// another value.
//
SymbolicValueRef : INTVAL {  // Is it an integer reference...?
    $$ = ValID::create($1);
  }
  | VAR_ID {                 // Is it a named reference...?
    $$ = ValID::create($1);
  };

// ValueRef - A reference to a definition... either constant or symbolic
ValueRef : SymbolicValueRef | ConstValueRef;


// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : Types ValueRef {
    $$ = getVal(*$1, $2); delete $1;
  };

BasicBlockList : BasicBlockList BasicBlock {
    ($$ = $1)->getBasicBlockList().push_back($2);
  }
  | FunctionHeader BasicBlock { // Do not allow functions with 0 basic blocks   
    ($$ = $1)->getBasicBlockList().push_back($2);
  };


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock : InstructionList OptAssign BBTerminatorInst  {
    if (setValueName($3, $2)) { assert(0 && "No redefn allowed!"); }
    InsertValue($3);

    $1->getInstList().push_back($3);
    InsertValue($1);
    $$ = $1;
  }
  | LABELSTR InstructionList OptAssign BBTerminatorInst  {
    if (setValueName($4, $3)) { assert(0 && "No redefn allowed!"); }
    InsertValue($4);

    $2->getInstList().push_back($4);
    if (setValueName($2, $1)) { assert(0 && "No label redef allowed!"); }

    InsertValue($2);
    $$ = $2;
  };

InstructionList : InstructionList Inst {
    $1->getInstList().push_back($2);
    $$ = $1;
  }
  | /* empty */ {
    $$ = CurBB = new BasicBlock();
  };

BBTerminatorInst : RET ResolvedVal {              // Return with a result...
    $$ = new ReturnInst($2);
  }
  | RET VOID {                                       // Return with no result...
    $$ = new ReturnInst();
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    $$ = new BranchInst(cast<BasicBlock>(getVal(Type::LabelTy, $3)));
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    $$ = new BranchInst(cast<BasicBlock>(getVal(Type::LabelTy, $6)), 
			cast<BasicBlock>(getVal(Type::LabelTy, $9)),
			getVal(Type::BoolTy, $3));
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    SwitchInst *S = new SwitchInst(getVal($2, $3), 
                                   cast<BasicBlock>(getVal(Type::LabelTy, $6)));
    $$ = S;

    std::vector<std::pair<Constant*,BasicBlock*> >::iterator I = $8->begin(),
      E = $8->end();
    for (; I != E; ++I)
      S->dest_push_back(I->first, I->second);
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    SwitchInst *S = new SwitchInst(getVal($2, $3), 
                                   cast<BasicBlock>(getVal(Type::LabelTy, $6)));
    $$ = S;
  }
  | INVOKE TypesV ValueRef '(' ValueRefListE ')' TO ResolvedVal 
    EXCEPT ResolvedVal {
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>($2->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if ($5) {
        for (std::vector<Value*>::iterator I = $5->begin(), E = $5->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      Ty = FunctionType::get($2->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }
    delete $2;

    Value *V = getVal(PFTy, $3);   // Get the function we're calling...

    BasicBlock *Normal = dyn_cast<BasicBlock>($8);
    BasicBlock *Except = dyn_cast<BasicBlock>($10);

    if (Normal == 0 || Except == 0)
      ThrowException("Invoke instruction without label destinations!");

    // Create the call node...
    if (!$5) {                                   // Has no arguments?
      $$ = new InvokeInst(V, Normal, Except, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::ParamTypes::const_iterator I = Ty->getParamTypes().begin();
      FunctionType::ParamTypes::const_iterator E = Ty->getParamTypes().end();
      std::vector<Value*>::iterator ArgI = $5->begin(), ArgE = $5->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
	if ((*ArgI)->getType() != *I)
	  ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
			 (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
	ThrowException("Invalid number of parameters detected!");

      $$ = new InvokeInst(V, Normal, Except, *$5);
    }
    delete $5;
  };



JumpTable : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    $$ = $1;
    Constant *V = cast<Constant>(getValNonImprovising($2, $3));
    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    $$->push_back(std::make_pair(V, cast<BasicBlock>(getVal($5, $6))));
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $$ = new std::vector<std::pair<Constant*, BasicBlock*> >();
    Constant *V = cast<Constant>(getValNonImprovising($1, $2));

    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    $$->push_back(std::make_pair(V, cast<BasicBlock>(getVal($4, $5))));
  };

Inst : OptAssign InstVal {
  // Is this definition named?? if so, assign the name...
  if (setValueName($2, $1)) { assert(0 && "No redefin allowed!"); }
  InsertValue($2);
  $$ = $2;
};

PHIList : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    $$ = new std::list<std::pair<Value*, BasicBlock*> >();
    $$->push_back(std::make_pair(getVal(*$1, $3), 
                                 cast<BasicBlock>(getVal(Type::LabelTy, $5))));
    delete $1;
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    $$ = $1;
    $1->push_back(std::make_pair(getVal($1->front().first->getType(), $4),
                                 cast<BasicBlock>(getVal(Type::LabelTy, $6))));
  };


ValueRefList : ResolvedVal {    // Used for call statements, and memory insts...
    $$ = new std::vector<Value*>();
    $$->push_back($1);
  }
  | ValueRefList ',' ResolvedVal {
    $$ = $1;
    $1->push_back($3);
  };

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE : ValueRefList | /*empty*/ { $$ = 0; };

InstVal : ArithmeticOps Types ValueRef ',' ValueRef {
    if (!(*$2)->isInteger() && !(*$2)->isFloatingPoint())
      ThrowException("Arithmetic operator requires integer or FP operands!");
    $$ = BinaryOperator::create($1, getVal(*$2, $3), getVal(*$2, $5));
    if ($$ == 0)
      ThrowException("binary operator returned null!");
    delete $2;
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    if (!(*$2)->isIntegral())
      ThrowException("Logical operator requires integral operands!");
    $$ = BinaryOperator::create($1, getVal(*$2, $3), getVal(*$2, $5));
    if ($$ == 0)
      ThrowException("binary operator returned null!");
    delete $2;
  }
  | SetCondOps Types ValueRef ',' ValueRef {
    $$ = new SetCondInst($1, getVal(*$2, $3), getVal(*$2, $5));
    if ($$ == 0)
      ThrowException("binary operator returned null!");
    delete $2;
  }
  | NOT ResolvedVal {
    std::cerr << "WARNING: Use of eliminated 'not' instruction:"
              << " Replacing with 'xor'.\n";

    Value *Ones = ConstantIntegral::getAllOnesValue($2->getType());
    if (Ones == 0)
      ThrowException("Expected integral type for not instruction!");

    $$ = BinaryOperator::create(Instruction::Xor, $2, Ones);
    if ($$ == 0)
      ThrowException("Could not create a xor instruction!");
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    if ($4->getType() != Type::UByteTy)
      ThrowException("Shift amount must be ubyte!");
    $$ = new ShiftInst($1, $2, $4);
  }
  | CAST ResolvedVal TO Types {
    $$ = new CastInst($2, *$4);
    delete $4;
  }
  | VA_ARG ResolvedVal ',' Types {
    $$ = new VarArgInst($2, *$4);
    delete $4;
  }
  | PHI PHIList {
    const Type *Ty = $2->front().first->getType();
    $$ = new PHINode(Ty);
    while ($2->begin() != $2->end()) {
      if ($2->front().first->getType() != Ty) 
	ThrowException("All elements of a PHI node must be of the same type!");
      cast<PHINode>($$)->addIncoming($2->front().first, $2->front().second);
      $2->pop_front();
    }
    delete $2;  // Free the list...
  } 
  | CALL TypesV ValueRef '(' ValueRefListE ')' {
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>($2->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if ($5) {
        for (std::vector<Value*>::iterator I = $5->begin(), E = $5->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      Ty = FunctionType::get($2->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }
    delete $2;

    Value *V = getVal(PFTy, $3);   // Get the function we're calling...

    // Create the call node...
    if (!$5) {                                   // Has no arguments?
      // Make sure no arguments is a good thing!
      if (Ty->getNumParams() != 0)
        ThrowException("No arguments passed to a function that "
                       "expects arguments!");

      $$ = new CallInst(V, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::ParamTypes::const_iterator I = Ty->getParamTypes().begin();
      FunctionType::ParamTypes::const_iterator E = Ty->getParamTypes().end();
      std::vector<Value*>::iterator ArgI = $5->begin(), ArgE = $5->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
	if ((*ArgI)->getType() != *I)
	  ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
			 (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
	ThrowException("Invalid number of parameters detected!");

      $$ = new CallInst(V, *$5);
    }
    delete $5;
  }
  | MemoryInst {
    $$ = $1;
  };


// IndexList - List of indices for GEP based instructions...
IndexList : ',' ValueRefList { 
  $$ = $2; 
} | /* empty */ { 
  $$ = new std::vector<Value*>(); 
};

MemoryInst : MALLOC Types {
    $$ = new MallocInst(*$2);
    delete $2;
  }
  | MALLOC Types ',' UINT ValueRef {
    $$ = new MallocInst(*$2, getVal($4, $5));
    delete $2;
  }
  | ALLOCA Types {
    $$ = new AllocaInst(*$2);
    delete $2;
  }
  | ALLOCA Types ',' UINT ValueRef {
    $$ = new AllocaInst(*$2, getVal($4, $5));
    delete $2;
  }
  | FREE ResolvedVal {
    if (!isa<PointerType>($2->getType()))
      ThrowException("Trying to free nonpointer type " + 
                     $2->getType()->getDescription() + "!");
    $$ = new FreeInst($2);
  }

  | LOAD Types ValueRef IndexList {
    if (!isa<PointerType>($2->get()))
      ThrowException("Can't load from nonpointer type: " +
		     (*$2)->getDescription());
    if (GetElementPtrInst::getIndexedType(*$2, *$4) == 0)
      ThrowException("Invalid indices for load instruction!");

    Value *Src = getVal(*$2, $3);
    if (!$4->empty()) {
      std::cerr << "WARNING: Use of index load instruction:"
                << " replacing with getelementptr/load pair.\n";
      // Create a getelementptr hack instruction to do the right thing for
      // compatibility.
      //
      Instruction *I = new GetElementPtrInst(Src, *$4);
      CurBB->getInstList().push_back(I);
      Src = I;
    }

    $$ = new LoadInst(Src);
    delete $4;   // Free the vector...
    delete $2;
  }
  | STORE ResolvedVal ',' Types ValueRef IndexList {
    if (!isa<PointerType>($4->get()))
      ThrowException("Can't store to a nonpointer type: " +
                     (*$4)->getDescription());
    const Type *ElTy = GetElementPtrInst::getIndexedType(*$4, *$6);
    if (ElTy == 0)
      ThrowException("Can't store into that field list!");
    if (ElTy != $2->getType())
      ThrowException("Can't store '" + $2->getType()->getDescription() +
                     "' into space of type '" + ElTy->getDescription() + "'!");

    Value *Ptr = getVal(*$4, $5);
    if (!$6->empty()) {
      std::cerr << "WARNING: Use of index store instruction:"
                << " replacing with getelementptr/store pair.\n";
      // Create a getelementptr hack instruction to do the right thing for
      // compatibility.
      //
      Instruction *I = new GetElementPtrInst(Ptr, *$6);
      CurBB->getInstList().push_back(I);
      Ptr = I;
    }

    $$ = new StoreInst($2, Ptr);
    delete $4; delete $6;
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    for (unsigned i = 0, e = $4->size(); i != e; ++i) {
      if ((*$4)[i]->getType() == Type::UIntTy) {
        std::cerr << "WARNING: Use of uint type indexes to getelementptr "
                  << "instruction: replacing with casts to long type.\n";
        Instruction *I = new CastInst((*$4)[i], Type::LongTy);
        CurBB->getInstList().push_back(I);
        (*$4)[i] = I;
      }
    }

    if (!isa<PointerType>($2->get()))
      ThrowException("getelementptr insn requires pointer operand!");
    if (!GetElementPtrInst::getIndexedType(*$2, *$4, true))
      ThrowException("Can't get element ptr '" + (*$2)->getDescription()+ "'!");
    $$ = new GetElementPtrInst(getVal(*$2, $3), *$4);
    delete $2; delete $4;
  };

%%
int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + utostr((unsigned) llvmAsmlineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(llvmAsmtext, llvmAsmleng) + "'";
  ThrowException(errMsg);
  return 0;
}
