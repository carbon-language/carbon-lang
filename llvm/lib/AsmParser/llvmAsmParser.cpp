
/*  A Bison parser, made from /Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y
    by GNU Bison version 1.28  */

#define YYBISON 1  /* Identify Bison output.  */

#define yyparse llvmAsmparse
#define yylex llvmAsmlex
#define yyerror llvmAsmerror
#define yylval llvmAsmlval
#define yychar llvmAsmchar
#define yydebug llvmAsmdebug
#define yynerrs llvmAsmnerrs
#define	ESINT64VAL	257
#define	EUINT64VAL	258
#define	SINTVAL	259
#define	UINTVAL	260
#define	FPVAL	261
#define	VOID	262
#define	BOOL	263
#define	SBYTE	264
#define	UBYTE	265
#define	SHORT	266
#define	USHORT	267
#define	INT	268
#define	UINT	269
#define	LONG	270
#define	ULONG	271
#define	FLOAT	272
#define	DOUBLE	273
#define	TYPE	274
#define	LABEL	275
#define	VAR_ID	276
#define	LABELSTR	277
#define	STRINGCONSTANT	278
#define	IMPLEMENTATION	279
#define	ZEROINITIALIZER	280
#define	TRUETOK	281
#define	FALSETOK	282
#define	BEGINTOK	283
#define	ENDTOK	284
#define	DECLARE	285
#define	GLOBAL	286
#define	CONSTANT	287
#define	SECTION	288
#define	VOLATILE	289
#define	TO	290
#define	DOTDOTDOT	291
#define	NULL_TOK	292
#define	UNDEF	293
#define	CONST	294
#define	INTERNAL	295
#define	LINKONCE	296
#define	WEAK	297
#define	APPENDING	298
#define	OPAQUE	299
#define	NOT	300
#define	EXTERNAL	301
#define	TARGET	302
#define	TRIPLE	303
#define	ENDIAN	304
#define	POINTERSIZE	305
#define	LITTLE	306
#define	BIG	307
#define	ALIGN	308
#define	DEPLIBS	309
#define	CALL	310
#define	TAIL	311
#define	ASM_TOK	312
#define	MODULE	313
#define	CC_TOK	314
#define	CCC_TOK	315
#define	FASTCC_TOK	316
#define	COLDCC_TOK	317
#define	RET	318
#define	BR	319
#define	SWITCH	320
#define	INVOKE	321
#define	UNWIND	322
#define	UNREACHABLE	323
#define	ADD	324
#define	SUB	325
#define	MUL	326
#define	DIV	327
#define	REM	328
#define	AND	329
#define	OR	330
#define	XOR	331
#define	SETLE	332
#define	SETGE	333
#define	SETLT	334
#define	SETGT	335
#define	SETEQ	336
#define	SETNE	337
#define	MALLOC	338
#define	ALLOCA	339
#define	FREE	340
#define	LOAD	341
#define	STORE	342
#define	GETELEMENTPTR	343
#define	PHI_TOK	344
#define	CAST	345
#define	SELECT	346
#define	SHL	347
#define	SHR	348
#define	VAARG	349
#define	EXTRACTELEMENT	350
#define	INSERTELEMENT	351
#define	VAARG_old	352
#define	VANEXT_old	353

#line 14 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"

#include "ParserInternals.h"
#include "llvm/CallingConv.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Assembly/AutoUpgrade.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <utility>

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit
int yylex();                       // declaration" of xxx warnings.
int yyparse();

namespace llvm {
  std::string CurFilename;
}
using namespace llvm;

static Module *ParserResult;

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

static bool ObsoleteVarArgs;
static bool NewVarArgs;
static BasicBlock *CurBB;
static GlobalVariable *CurGV;


// This contains info used when building the body of a function.  It is
// destroyed when the function is completed.
//
typedef std::vector<Value *> ValueList;           // Numbered defs
static void 
ResolveDefinitions(std::map<const Type *,ValueList> &LateResolvers,
                   std::map<const Type *,ValueList> *FutureLateResolvers = 0);

static struct PerModuleInfo {
  Module *CurrentModule;
  std::map<const Type *, ValueList> Values; // Module level numbered definitions
  std::map<const Type *,ValueList> LateResolveValues;
  std::vector<PATypeHolder>    Types;
  std::map<ValID, PATypeHolder> LateResolveTypes;

  /// PlaceHolderInfo - When temporary placeholder objects are created, remember
  /// how they were referenced and one which line of the input they came from so
  /// that we can resolve them later and print error messages as appropriate.
  std::map<Value*, std::pair<ValID, int> > PlaceHolderInfo;

  // GlobalRefs - This maintains a mapping between <Type, ValID>'s and forward
  // references to global values.  Global values may be referenced before they
  // are defined, and if so, the temporary object that they represent is held
  // here.  This is used for forward references of GlobalValues.
  //
  typedef std::map<std::pair<const PointerType *,
                             ValID>, GlobalValue*> GlobalRefsType;
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

    // Look for intrinsic functions and CallInst that need to be upgraded
    for (Module::iterator FI = CurrentModule->begin(),FE = CurrentModule->end();
         FI != FE; ++FI)
      UpgradeCallsToIntrinsic(FI);

    Values.clear();         // Clear out function local definitions
    Types.clear();
    CurrentModule = 0;
  }

  // GetForwardRefForGlobal - Check to see if there is a forward reference
  // for this global.  If so, remove it from the GlobalRefs map and return it.
  // If not, just return null.
  GlobalValue *GetForwardRefForGlobal(const PointerType *PTy, ValID ID) {
    // Check to see if there is a forward reference to this global variable...
    // if there is, eliminate it and patch the reference to use the new def'n.
    GlobalRefsType::iterator I = GlobalRefs.find(std::make_pair(PTy, ID));
    GlobalValue *Ret = 0;
    if (I != GlobalRefs.end()) {
      Ret = I->second;
      GlobalRefs.erase(I);
    }
    return Ret;
  }
} CurModule;

static struct PerFunctionInfo {
  Function *CurrentFunction;     // Pointer to current function being created

  std::map<const Type*, ValueList> Values;   // Keep track of #'d definitions
  std::map<const Type*, ValueList> LateResolveValues;
  bool isDeclare;                // Is this function a forward declararation?

  /// BBForwardRefs - When we see forward references to basic blocks, keep
  /// track of them here.
  std::map<BasicBlock*, std::pair<ValID, int> > BBForwardRefs;
  std::vector<BasicBlock*> NumberedBlocks;
  unsigned NextBBNum;

  inline PerFunctionInfo() {
    CurrentFunction = 0;
    isDeclare = false;
  }

  inline void FunctionStart(Function *M) {
    CurrentFunction = M;
    NextBBNum = 0;
  }

  void FunctionDone() {
    NumberedBlocks.clear();

    // Any forward referenced blocks left?
    if (!BBForwardRefs.empty())
      ThrowException("Undefined reference to label " +
                     BBForwardRefs.begin()->first->getName());

    // Resolve all forward references now.
    ResolveDefinitions(LateResolveValues, &CurModule.LateResolveValues);

    Values.clear();         // Clear out function local definitions
    CurrentFunction = 0;
    isDeclare = false;
  }
} CurFun;  // Info for the current function...

static bool inFunctionScope() { return CurFun.CurrentFunction != 0; }


//===----------------------------------------------------------------------===//
//               Code to handle definitions of all the types
//===----------------------------------------------------------------------===//

static int InsertValue(Value *V,
                  std::map<const Type*,ValueList> &ValueTab = CurFun.Values) {
  if (V->hasName()) return -1;           // Is this a numbered definition?

  // Yes, insert the value into the value table...
  ValueList &List = ValueTab[V->getType()];
  List.push_back(V);
  return List.size()-1;
}

static const Type *getTypeVal(const ValID &D, bool DoNotImprovise = false) {
  switch (D.Type) {
  case ValID::NumberVal:               // Is it a numbered definition?
    // Module constants occupy the lowest numbered slots...
    if ((unsigned)D.Num < CurModule.Types.size())
      return CurModule.Types[(unsigned)D.Num];
    break;
  case ValID::NameVal:                 // Is it a named definition?
    if (const Type *N = CurModule.CurrentModule->getTypeByName(D.Name)) {
      D.destroy();  // Free old strdup'd memory...
      return N;
    }
    break;
  default:
    ThrowException("Internal parser error: Invalid symbol type reference!");
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?


  if (inFunctionScope()) {
    if (D.Type == ValID::NameVal)
      ThrowException("Reference to an undefined type: '" + D.getName() + "'");
    else
      ThrowException("Reference to an undefined type: #" + itostr(D.Num));
  }

  std::map<ValID, PATypeHolder>::iterator I =CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end())
    return I->second;

  Type *Typ = OpaqueType::get();
  CurModule.LateResolveTypes.insert(std::make_pair(D, Typ));
  return Typ;
 }

static Value *lookupInSymbolTable(const Type *Ty, const std::string &Name) {
  SymbolTable &SymTab =
    inFunctionScope() ? CurFun.CurrentFunction->getSymbolTable() :
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
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    std::map<const Type*,ValueList>::iterator VI = CurModule.Values.find(Ty);
    if (VI != CurModule.Values.end()) {
      if (Num < VI->second.size())
        return VI->second[Num];
      Num -= VI->second.size();
    }

    // Make sure that our type is within bounds
    VI = CurFun.Values.find(Ty);
    if (VI == CurFun.Values.end()) return 0;

    // Check that the number is within bounds...
    if (VI->second.size() <= Num) return 0;

    return VI->second[Num];
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

  case ValID::ConstUndefVal:      // Is it an undef value?
    return UndefValue::get(Ty);

  case ValID::ConstZeroVal:      // Is it a zero value?
    return Constant::getNullValue(Ty);
    
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
static Value *getVal(const Type *Ty, const ValID &ID) {
  if (Ty == Type::LabelTy)
    ThrowException("Cannot use a basic block here");

  // See if the value has already been defined.
  Value *V = getValNonImprovising(Ty, ID);
  if (V) return V;

  if (!Ty->isFirstClassType() && !isa<OpaqueType>(Ty))
    ThrowException("Invalid use of a composite type!");

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  V = new Argument(Ty);

  // Remember where this forward reference came from.  FIXME, shouldn't we try
  // to recycle these things??
  CurModule.PlaceHolderInfo.insert(std::make_pair(V, std::make_pair(ID,
                                                               llvmAsmlineno)));

  if (inFunctionScope())
    InsertValue(V, CurFun.LateResolveValues);
  else
    InsertValue(V, CurModule.LateResolveValues);
  return V;
}

/// getBBVal - This is used for two purposes:
///  * If isDefinition is true, a new basic block with the specified ID is being
///    defined.
///  * If isDefinition is true, this is a reference to a basic block, which may
///    or may not be a forward reference.
///
static BasicBlock *getBBVal(const ValID &ID, bool isDefinition = false) {
  assert(inFunctionScope() && "Can't get basic block at global scope!");

  std::string Name;
  BasicBlock *BB = 0;
  switch (ID.Type) {
  default: ThrowException("Illegal label reference " + ID.getName());
  case ValID::NumberVal:                // Is it a numbered definition?
    if (unsigned(ID.Num) >= CurFun.NumberedBlocks.size())
      CurFun.NumberedBlocks.resize(ID.Num+1);
    BB = CurFun.NumberedBlocks[ID.Num];
    break;
  case ValID::NameVal:                  // Is it a named definition?
    Name = ID.Name;
    if (Value *N = CurFun.CurrentFunction->
                   getSymbolTable().lookup(Type::LabelTy, Name))
      BB = cast<BasicBlock>(N);
    break;
  }

  // See if the block has already been defined.
  if (BB) {
    // If this is the definition of the block, make sure the existing value was
    // just a forward reference.  If it was a forward reference, there will be
    // an entry for it in the PlaceHolderInfo map.
    if (isDefinition && !CurFun.BBForwardRefs.erase(BB))
      // The existing value was a definition, not a forward reference.
      ThrowException("Redefinition of label " + ID.getName());

    ID.destroy();                       // Free strdup'd memory.
    return BB;
  }

  // Otherwise this block has not been seen before.
  BB = new BasicBlock("", CurFun.CurrentFunction);
  if (ID.Type == ValID::NameVal) {
    BB->setName(ID.Name);
  } else {
    CurFun.NumberedBlocks[ID.Num] = BB;
  }

  // If this is not a definition, keep track of it so we can use it as a forward
  // reference.
  if (!isDefinition) {
    // Remember where this forward reference came from.
    CurFun.BBForwardRefs[BB] = std::make_pair(ID, llvmAsmlineno);
  } else {
    // The forward declaration could have been inserted anywhere in the
    // function: insert it into the correct place now.
    CurFun.CurrentFunction->getBasicBlockList().remove(BB);
    CurFun.CurrentFunction->getBasicBlockList().push_back(BB);
  }
  ID.destroy();
  return BB;
}


//===----------------------------------------------------------------------===//
//              Code to handle forward references in instructions
//===----------------------------------------------------------------------===//
//
// This code handles the late binding needed with statements that reference
// values not defined yet... for example, a forward branch, or the PHI node for
// a loop body.
//
// This keeps a table (CurFun.LateResolveValues) of all such forward references
// and back patchs after we are done.
//

// ResolveDefinitions - If we could not resolve some defs at parsing
// time (forward branches, phi functions for loops, etc...) resolve the
// defs now...
//
static void 
ResolveDefinitions(std::map<const Type*,ValueList> &LateResolvers,
                   std::map<const Type*,ValueList> *FutureLateResolvers) {
  // Loop over LateResolveDefs fixing up stuff that couldn't be resolved
  for (std::map<const Type*,ValueList>::iterator LRI = LateResolvers.begin(),
         E = LateResolvers.end(); LRI != E; ++LRI) {
    ValueList &List = LRI->second;
    while (!List.empty()) {
      Value *V = List.back();
      List.pop_back();

      std::map<Value*, std::pair<ValID, int> >::iterator PHI =
        CurModule.PlaceHolderInfo.find(V);
      assert(PHI != CurModule.PlaceHolderInfo.end() && "Placeholder error!");

      ValID &DID = PHI->second.first;

      Value *TheRealValue = getValNonImprovising(LRI->first, DID);
      if (TheRealValue) {
        V->replaceAllUsesWith(TheRealValue);
        delete V;
        CurModule.PlaceHolderInfo.erase(PHI);
      } else if (FutureLateResolvers) {
        // Functions have their unresolved items forwarded to the module late
        // resolver table
        InsertValue(V, *FutureLateResolvers);
      } else {
        if (DID.Type == ValID::NameVal)
          ThrowException("Reference to an invalid definition: '" +DID.getName()+
                         "' of type '" + V->getType()->getDescription() + "'",
                         PHI->second.second);
        else
          ThrowException("Reference to an invalid definition: #" +
                         itostr(DID.Num) + " of type '" +
                         V->getType()->getDescription() + "'",
                         PHI->second.second);
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
  ValID D;
  if (Name) D = ValID::create(Name);
  else      D = ValID::create((int)CurModule.Types.size());

  std::map<ValID, PATypeHolder>::iterator I =
    CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end()) {
    ((DerivedType*)I->second.get())->refineAbstractTypeTo(ToTy);
    CurModule.LateResolveTypes.erase(I);
  }
}

// setValueName - Set the specified value to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is free'd by this function.
//
static void setValueName(Value *V, char *NameStr) {
  if (NameStr) {
    std::string Name(NameStr);      // Copy string
    free(NameStr);                  // Free old string

    if (V->getType() == Type::VoidTy)
      ThrowException("Can't assign name '" + Name+"' to value with void type!");

    assert(inFunctionScope() && "Must be in function scope!");
    SymbolTable &ST = CurFun.CurrentFunction->getSymbolTable();
    if (ST.lookup(V->getType(), Name))
      ThrowException("Redefinition of value named '" + Name + "' in the '" +
                     V->getType()->getDescription() + "' type plane!");

    // Set the name.
    V->setName(Name);
  }
}

/// ParseGlobalVariable - Handle parsing of a global.  If Initializer is null,
/// this is a declaration, otherwise it is a definition.
static GlobalVariable *
ParseGlobalVariable(char *NameStr,GlobalValue::LinkageTypes Linkage,
                    bool isConstantGlobal, const Type *Ty,
                    Constant *Initializer) {
  if (isa<FunctionType>(Ty))
    ThrowException("Cannot declare global vars of function type!");

  const PointerType *PTy = PointerType::get(Ty);

  std::string Name;
  if (NameStr) {
    Name = NameStr;      // Copy string
    free(NameStr);       // Free old string
  }

  // See if this global value was forward referenced.  If so, recycle the
  // object.
  ValID ID;
  if (!Name.empty()) {
    ID = ValID::create((char*)Name.c_str());
  } else {
    ID = ValID::create((int)CurModule.Values[PTy].size());
  }

  if (GlobalValue *FWGV = CurModule.GetForwardRefForGlobal(PTy, ID)) {
    // Move the global to the end of the list, from whereever it was
    // previously inserted.
    GlobalVariable *GV = cast<GlobalVariable>(FWGV);
    CurModule.CurrentModule->getGlobalList().remove(GV);
    CurModule.CurrentModule->getGlobalList().push_back(GV);
    GV->setInitializer(Initializer);
    GV->setLinkage(Linkage);
    GV->setConstant(isConstantGlobal);
    InsertValue(GV, CurModule.Values);
    return GV;
  }

  // If this global has a name, check to see if there is already a definition
  // of this global in the module.  If so, merge as appropriate.  Note that
  // this is really just a hack around problems in the CFE.  :(
  if (!Name.empty()) {
    // We are a simple redefinition of a value, check to see if it is defined
    // the same as the old one.
    if (GlobalVariable *EGV =
                CurModule.CurrentModule->getGlobalVariable(Name, Ty)) {
      // We are allowed to redefine a global variable in two circumstances:
      // 1. If at least one of the globals is uninitialized or
      // 2. If both initializers have the same value.
      //
      if (!EGV->hasInitializer() || !Initializer ||
          EGV->getInitializer() == Initializer) {

        // Make sure the existing global version gets the initializer!  Make
        // sure that it also gets marked const if the new version is.
        if (Initializer && !EGV->hasInitializer())
          EGV->setInitializer(Initializer);
        if (isConstantGlobal)
          EGV->setConstant(true);
        EGV->setLinkage(Linkage);
        return EGV;
      }

      ThrowException("Redefinition of global variable named '" + Name +
                     "' in the '" + Ty->getDescription() + "' type plane!");
    }
  }

  // Otherwise there is no existing GV to use, create one now.
  GlobalVariable *GV =
    new GlobalVariable(Ty, isConstantGlobal, Linkage, Initializer, Name,
                       CurModule.CurrentModule);
  InsertValue(GV, CurModule.Values);
  return GV;
}

// setTypeName - Set the specified type to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is freed by this function.
//
// This function returns true if the type has already been defined, but is
// allowed to be redefined in the specified context.  If the name is a new name
// for the type plane, it is inserted and false is returned.
static bool setTypeName(const Type *T, char *NameStr) {
  assert(!inFunctionScope() && "Can't give types function-local names!");
  if (NameStr == 0) return false;
 
  std::string Name(NameStr);      // Copy string
  free(NameStr);                  // Free old string

  // We don't allow assigning names to void type
  if (T == Type::VoidTy)
    ThrowException("Can't assign name '" + Name + "' to the void type!");

  // Set the type name, checking for conflicts as we do so.
  bool AlreadyExists = CurModule.CurrentModule->addTypeName(Name, T);

  if (AlreadyExists) {   // Inserting a name that is already defined???
    const Type *Existing = CurModule.CurrentModule->getTypeByName(Name);
    assert(Existing && "Conflict but no matching type?");

    // There is only one case where this is allowed: when we are refining an
    // opaque type.  In this case, Existing will be an opaque type.
    if (const OpaqueType *OpTy = dyn_cast<OpaqueType>(Existing)) {
      // We ARE replacing an opaque type!
      const_cast<OpaqueType*>(OpTy)->refineAbstractTypeTo(T);
      return true;
    }

    // Otherwise, this is an attempt to redefine a type. That's okay if
    // the redefinition is identical to the original. This will be so if
    // Existing and T point to the same Type object. In this one case we
    // allow the equivalent redefinition.
    if (Existing == T) return true;  // Yes, it's equal.

    // Any other kind of (non-equivalent) redefinition is an error.
    ThrowException("Redefinition of type named '" + Name + "' in the '" +
                   T->getDescription() + "' type plane!");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Code for handling upreferences in type names...
//

// TypeContains - Returns true if Ty directly contains E in it.
//
static bool TypeContains(const Type *Ty, const Type *E) {
  return std::find(Ty->subtype_begin(), Ty->subtype_end(),
                   E) != Ty->subtype_end();
}

namespace {
  struct UpRefRecord {
    // NestingLevel - The number of nesting levels that need to be popped before
    // this type is resolved.
    unsigned NestingLevel;

    // LastContainedTy - This is the type at the current binding level for the
    // type.  Every time we reduce the nesting level, this gets updated.
    const Type *LastContainedTy;

    // UpRefTy - This is the actual opaque type that the upreference is
    // represented with.
    OpaqueType *UpRefTy;

    UpRefRecord(unsigned NL, OpaqueType *URTy)
      : NestingLevel(NL), LastContainedTy(URTy), UpRefTy(URTy) {}
  };
}

// UpRefs - A list of the outstanding upreferences that need to be resolved.
static std::vector<UpRefRecord> UpRefs;

/// HandleUpRefs - Every time we finish a new layer of types, this function is
/// called.  It loops through the UpRefs vector, which is a list of the
/// currently active types.  For each type, if the up reference is contained in
/// the newly completed type, we decrement the level count.  When the level
/// count reaches zero, the upreferenced type is the type that is passed in:
/// thus we can complete the cycle.
///
static PATypeHolder HandleUpRefs(const Type *ty) {
  if (!ty->isAbstract()) return ty;
  PATypeHolder Ty(ty);
  UR_OUT("Type '" << Ty->getDescription() <<
         "' newly formed.  Resolving upreferences.\n" <<
         UpRefs.size() << " upreferences active!\n");

  // If we find any resolvable upreferences (i.e., those whose NestingLevel goes
  // to zero), we resolve them all together before we resolve them to Ty.  At
  // the end of the loop, if there is anything to resolve to Ty, it will be in
  // this variable.
  OpaqueType *TypeToResolve = 0;

  for (unsigned i = 0; i != UpRefs.size(); ++i) {
    UR_OUT("  UR#" << i << " - TypeContains(" << Ty->getDescription() << ", "
           << UpRefs[i].second->getDescription() << ") = "
           << (TypeContains(Ty, UpRefs[i].second) ? "true" : "false") << "\n");
    if (TypeContains(Ty, UpRefs[i].LastContainedTy)) {
      // Decrement level of upreference
      unsigned Level = --UpRefs[i].NestingLevel;
      UpRefs[i].LastContainedTy = Ty;
      UR_OUT("  Uplevel Ref Level = " << Level << "\n");
      if (Level == 0) {                     // Upreference should be resolved!
        if (!TypeToResolve) {
          TypeToResolve = UpRefs[i].UpRefTy;
        } else {
          UR_OUT("  * Resolving upreference for "
                 << UpRefs[i].second->getDescription() << "\n";
                 std::string OldName = UpRefs[i].UpRefTy->getDescription());
          UpRefs[i].UpRefTy->refineAbstractTypeTo(TypeToResolve);
          UR_OUT("  * Type '" << OldName << "' refined upreference to: "
                 << (const void*)Ty << ", " << Ty->getDescription() << "\n");
        }
        UpRefs.erase(UpRefs.begin()+i);     // Remove from upreference list...
        --i;                                // Do not skip the next element...
      }
    }
  }

  if (TypeToResolve) {
    UR_OUT("  * Resolving upreference for "
           << UpRefs[i].second->getDescription() << "\n";
           std::string OldName = TypeToResolve->getDescription());
    TypeToResolve->refineAbstractTypeTo(Ty);
  }

  return Ty;
}


// common code from the two 'RunVMAsmParser' functions
 static Module * RunParser(Module * M) {

  llvmAsmlineno = 1;      // Reset the current line number...
  ObsoleteVarArgs = false;
  NewVarArgs = false;

  CurModule.CurrentModule = M;
  yyparse();       // Parse the file, potentially throwing exception

  Module *Result = ParserResult;
  ParserResult = 0;

  //Not all functions use vaarg, so make a second check for ObsoleteVarArgs
  {
    Function* F;
    if ((F = Result->getNamedFunction("llvm.va_start"))
        && F->getFunctionType()->getNumParams() == 0)
      ObsoleteVarArgs = true;
    if((F = Result->getNamedFunction("llvm.va_copy"))
       && F->getFunctionType()->getNumParams() == 1)
      ObsoleteVarArgs = true;
  }

  if (ObsoleteVarArgs && NewVarArgs)
    ThrowException("This file is corrupt: it uses both new and old style varargs");

  if(ObsoleteVarArgs) {
    if(Function* F = Result->getNamedFunction("llvm.va_start")) {
      if (F->arg_size() != 0)
        ThrowException("Obsolete va_start takes 0 argument!");
      
      //foo = va_start()
      // ->
      //bar = alloca typeof(foo)
      //va_start(bar)
      //foo = load bar

      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getReturnType();
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_start", 
                                                 RetTy, ArgTyPtr, (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vastart.fix.1", CI);
        new CallInst(NF, bar, "", CI);
        Value* foo = new LoadInst(bar, "vastart.fix.2", CI);
        CI->replaceAllUsesWith(foo);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }
    
    if(Function* F = Result->getNamedFunction("llvm.va_end")) {
      if(F->arg_size() != 1)
        ThrowException("Obsolete va_end takes 1 argument!");

      //vaend foo
      // ->
      //bar = alloca 1 of typeof(foo)
      //vaend bar
      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getParamType(0);
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_end", 
                                                 RetTy, ArgTyPtr, (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vaend.fix.1", CI);
        new StoreInst(CI->getOperand(1), bar, CI);
        new CallInst(NF, bar, "", CI);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }

    if(Function* F = Result->getNamedFunction("llvm.va_copy")) {
      if(F->arg_size() != 1)
        ThrowException("Obsolete va_copy takes 1 argument!");
      //foo = vacopy(bar)
      // ->
      //a = alloca 1 of typeof(foo)
      //b = alloca 1 of typeof(foo)
      //store bar -> b
      //vacopy(a, b)
      //foo = load a
      
      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getReturnType();
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_copy", 
                                                 RetTy, ArgTyPtr, ArgTyPtr,
                                                 (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* a = new AllocaInst(ArgTy, 0, "vacopy.fix.1", CI);
        AllocaInst* b = new AllocaInst(ArgTy, 0, "vacopy.fix.2", CI);
        new StoreInst(CI->getOperand(1), b, CI);
        new CallInst(NF, a, b, "", CI);
        Value* foo = new LoadInst(a, "vacopy.fix.3", CI);
        CI->replaceAllUsesWith(foo);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }
  }

  return Result;

 }

//===----------------------------------------------------------------------===//
//            RunVMAsmParser - Define an interface to this parser
//===----------------------------------------------------------------------===//
//
Module *llvm::RunVMAsmParser(const std::string &Filename, FILE *F) {
  set_scan_file(F);

  CurFilename = Filename;
  return RunParser(new Module(CurFilename));
}

Module *llvm::RunVMAsmParser(const char * AsmString, Module * M) {
  set_scan_string(AsmString);

  CurFilename = "from_memory";
  if (M == NULL) {
    return RunParser(new Module (CurFilename));
  } else {
    return RunParser(M);
  }
}


#line 878 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
typedef union {
  llvm::Module                           *ModuleVal;
  llvm::Function                         *FunctionVal;
  std::pair<llvm::PATypeHolder*, char*>  *ArgVal;
  llvm::BasicBlock                       *BasicBlockVal;
  llvm::TerminatorInst                   *TermInstVal;
  llvm::Instruction                      *InstVal;
  llvm::Constant                         *ConstVal;

  const llvm::Type                       *PrimType;
  llvm::PATypeHolder                     *TypeVal;
  llvm::Value                            *ValueVal;

  std::vector<std::pair<llvm::PATypeHolder*,char*> > *ArgList;
  std::vector<llvm::Value*>              *ValueList;
  std::list<llvm::PATypeHolder>          *TypeList;
  // Represent the RHS of PHI node
  std::list<std::pair<llvm::Value*,
                      llvm::BasicBlock*> > *PHIList;
  std::vector<std::pair<llvm::Constant*, llvm::BasicBlock*> > *JumpTable;
  std::vector<llvm::Constant*>           *ConstVector;

  llvm::GlobalValue::LinkageTypes         Linkage;
  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  double                            FPVal;
  bool                              BoolVal;

  char                             *StrVal;   // This memory is strdup'd!
  llvm::ValID                             ValIDVal; // strdup'd memory maybe!

  llvm::Instruction::BinaryOps            BinaryOpVal;
  llvm::Instruction::TermOps              TermOpVal;
  llvm::Instruction::MemoryOps            MemOpVal;
  llvm::Instruction::OtherOps             OtherOpVal;
  llvm::Module::Endianness                Endianness;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		467
#define	YYFLAG		-32768
#define	YYNTBASE	114

#define YYTRANSLATE(x) ((unsigned)(x) <= 353 ? yytranslate[x] : 184)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,   103,
   104,   112,     2,   101,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,   108,
   100,   109,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
   105,   102,   107,     2,     2,     2,     2,     2,   113,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,   106,
     2,     2,   110,     2,   111,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     1,     3,     4,     5,     6,
     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
    27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
    37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
    47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
    57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
    77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
    87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
    97,    98,    99
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     4,     6,     8,    10,    12,    14,    16,    18,
    20,    22,    24,    26,    28,    30,    32,    34,    36,    38,
    40,    42,    44,    46,    48,    50,    52,    54,    56,    58,
    60,    62,    64,    67,    68,    70,    72,    74,    76,    77,
    78,    80,    82,    84,    87,    88,    91,    92,    96,    99,
   100,   102,   103,   107,   109,   112,   114,   116,   118,   120,
   122,   124,   126,   128,   130,   132,   134,   136,   138,   140,
   142,   144,   146,   148,   150,   152,   154,   157,   162,   168,
   174,   178,   181,   184,   186,   190,   192,   196,   198,   199,
   204,   208,   212,   217,   222,   226,   229,   232,   235,   238,
   241,   244,   247,   250,   253,   256,   263,   269,   278,   285,
   292,   299,   306,   313,   317,   319,   321,   323,   325,   328,
   331,   336,   339,   341,   346,   349,   354,   355,   363,   364,
   372,   376,   381,   382,   384,   386,   388,   392,   396,   400,
   404,   408,   410,   411,   413,   415,   417,   418,   421,   425,
   427,   429,   433,   435,   436,   445,   447,   449,   453,   455,
   457,   460,   461,   465,   467,   469,   471,   473,   475,   477,
   479,   481,   485,   487,   489,   491,   493,   495,   498,   501,
   504,   508,   511,   512,   514,   517,   520,   524,   534,   544,
   553,   567,   569,   571,   578,   584,   587,   594,   602,   604,
   608,   610,   611,   614,   616,   622,   628,   634,   637,   642,
   647,   654,   659,   664,   669,   674,   681,   684,   692,   694,
   697,   698,   700,   701,   705,   712,   716,   723,   726,   731,
   738
};

static const short yyrhs[] = {     5,
     0,     6,     0,     3,     0,     4,     0,    70,     0,    71,
     0,    72,     0,    73,     0,    74,     0,    75,     0,    76,
     0,    77,     0,    78,     0,    79,     0,    80,     0,    81,
     0,    82,     0,    83,     0,    93,     0,    94,     0,    16,
     0,    14,     0,    12,     0,    10,     0,    17,     0,    15,
     0,    13,     0,    11,     0,   120,     0,   121,     0,    18,
     0,    19,     0,   154,   100,     0,     0,    41,     0,    42,
     0,    43,     0,    44,     0,     0,     0,    61,     0,    62,
     0,    63,     0,    60,     4,     0,     0,    54,     4,     0,
     0,   101,    54,     4,     0,    34,    24,     0,     0,   129,
     0,     0,   101,   132,   131,     0,   129,     0,    54,     4,
     0,   135,     0,     8,     0,   137,     0,     8,     0,   137,
     0,     9,     0,    10,     0,    11,     0,    12,     0,    13,
     0,    14,     0,    15,     0,    16,     0,    17,     0,    18,
     0,    19,     0,    20,     0,    21,     0,    45,     0,   136,
     0,   167,     0,   102,     4,     0,   134,   103,   139,   104,
     0,   105,     4,   106,   137,   107,     0,   108,     4,   106,
   137,   109,     0,   110,   138,   111,     0,   110,   111,     0,
   137,   112,     0,   137,     0,   138,   101,   137,     0,   138,
     0,   138,   101,    37,     0,    37,     0,     0,   135,   105,
   142,   107,     0,   135,   105,   107,     0,   135,   113,    24,
     0,   135,   108,   142,   109,     0,   135,   110,   142,   111,
     0,   135,   110,   111,     0,   135,    38,     0,   135,    39,
     0,   135,   167,     0,   135,   141,     0,   135,    26,     0,
   120,   115,     0,   121,     4,     0,     9,    27,     0,     9,
    28,     0,   123,     7,     0,    91,   103,   140,    36,   135,
   104,     0,    89,   103,   140,   181,   104,     0,    92,   103,
   140,   101,   140,   101,   140,   104,     0,   116,   103,   140,
   101,   140,   104,     0,   117,   103,   140,   101,   140,   104,
     0,   118,   103,   140,   101,   140,   104,     0,   119,   103,
   140,   101,   140,   104,     0,    96,   103,   140,   101,   140,
   104,     0,   142,   101,   140,     0,   140,     0,    32,     0,
    33,     0,   145,     0,   145,   163,     0,   145,   164,     0,
   145,    59,    58,   149,     0,   145,    25,     0,   146,     0,
   146,   124,    20,   133,     0,   146,   164,     0,   146,    59,
    58,   149,     0,     0,   146,   124,   125,   143,   140,   147,
   131,     0,     0,   146,   124,    47,   143,   135,   148,   131,
     0,   146,    48,   151,     0,   146,    55,   100,   152,     0,
     0,    24,     0,    53,     0,    52,     0,    50,   100,   150,
     0,    51,   100,     4,     0,    49,   100,    24,     0,   105,
   153,   107,     0,   153,   101,    24,     0,    24,     0,     0,
    22,     0,    24,     0,   154,     0,     0,   135,   155,     0,
   157,   101,   156,     0,   156,     0,   157,     0,   157,   101,
    37,     0,    37,     0,     0,   126,   133,   154,   103,   158,
   104,   130,   127,     0,    29,     0,   110,     0,   125,   159,
   160,     0,    30,     0,   111,     0,   170,   162,     0,     0,
    31,   165,   159,     0,     3,     0,     4,     0,     7,     0,
    27,     0,    28,     0,    38,     0,    39,     0,    26,     0,
   108,   142,   109,     0,   141,     0,   114,     0,   154,     0,
   167,     0,   166,     0,   135,   168,     0,   170,   171,     0,
   161,   171,     0,   172,   124,   173,     0,   172,   175,     0,
     0,    23,     0,    64,   169,     0,    64,     8,     0,    65,
    21,   168,     0,    65,     9,   168,   101,    21,   168,   101,
    21,   168,     0,    66,   122,   168,   101,    21,   168,   105,
   174,   107,     0,    66,   122,   168,   101,    21,   168,   105,
   107,     0,    67,   126,   133,   168,   103,   178,   104,    36,
    21,   168,    68,    21,   168,     0,    68,     0,    69,     0,
   174,   122,   166,   101,    21,   168,     0,   122,   166,   101,
    21,   168,     0,   124,   180,     0,   135,   105,   168,   101,
   168,   107,     0,   176,   101,   105,   168,   101,   168,   107,
     0,   169,     0,   177,   101,   169,     0,   177,     0,     0,
    57,    56,     0,    56,     0,   116,   135,   168,   101,   168,
     0,   117,   135,   168,   101,   168,     0,   118,   135,   168,
   101,   168,     0,    46,   169,     0,   119,   169,   101,   169,
     0,    91,   169,    36,   135,     0,    92,   169,   101,   169,
   101,   169,     0,    95,   169,   101,   135,     0,    98,   169,
   101,   135,     0,    99,   169,   101,   135,     0,    96,   169,
   101,   169,     0,    97,   169,   101,   169,   101,   169,     0,
    90,   176,     0,   179,   126,   133,   168,   103,   178,   104,
     0,   183,     0,   101,   177,     0,     0,    35,     0,     0,
    84,   135,   128,     0,    84,   135,   101,    15,   168,   128,
     0,    85,   135,   128,     0,    85,   135,   101,    15,   168,
   128,     0,    86,   169,     0,   182,    87,   135,   168,     0,
   182,    88,   169,   101,   135,   168,     0,    89,   135,   168,
   181,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   997,   998,  1005,  1006,  1015,  1015,  1015,  1015,  1015,  1016,
  1016,  1016,  1017,  1017,  1017,  1017,  1017,  1017,  1019,  1019,
  1023,  1023,  1023,  1023,  1024,  1024,  1024,  1024,  1025,  1025,
  1026,  1026,  1029,  1032,  1036,  1036,  1037,  1038,  1039,  1042,
  1042,  1043,  1044,  1045,  1054,  1054,  1060,  1060,  1068,  1075,
  1075,  1081,  1081,  1083,  1087,  1100,  1100,  1101,  1101,  1103,
  1112,  1112,  1112,  1112,  1112,  1112,  1112,  1113,  1113,  1113,
  1113,  1113,  1113,  1114,  1117,  1120,  1126,  1133,  1145,  1149,
  1160,  1169,  1172,  1180,  1184,  1189,  1190,  1193,  1196,  1206,
  1231,  1244,  1273,  1298,  1318,  1330,  1339,  1343,  1402,  1408,
  1416,  1421,  1426,  1429,  1432,  1439,  1449,  1480,  1487,  1508,
  1518,  1523,  1530,  1540,  1543,  1550,  1550,  1560,  1567,  1571,
  1574,  1577,  1580,  1593,  1613,  1615,  1617,  1620,  1623,  1627,
  1630,  1632,  1634,  1638,  1650,  1651,  1653,  1656,  1664,  1669,
  1671,  1675,  1679,  1687,  1687,  1688,  1688,  1690,  1696,  1701,
  1707,  1710,  1715,  1719,  1723,  1809,  1809,  1811,  1819,  1819,
  1821,  1825,  1825,  1834,  1837,  1840,  1843,  1846,  1849,  1852,
  1855,  1858,  1882,  1889,  1892,  1897,  1897,  1903,  1907,  1910,
  1918,  1927,  1931,  1941,  1952,  1955,  1958,  1961,  1964,  1978,
  1982,  2035,  2038,  2044,  2052,  2062,  2069,  2074,  2081,  2085,
  2091,  2091,  2093,  2096,  2102,  2114,  2125,  2135,  2147,  2154,
  2161,  2168,  2173,  2192,  2214,  2222,  2234,  2248,  2305,  2311,
  2313,  2317,  2320,  2326,  2330,  2334,  2338,  2342,  2349,  2359,
  2372
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","ESINT64VAL",
"EUINT64VAL","SINTVAL","UINTVAL","FPVAL","VOID","BOOL","SBYTE","UBYTE","SHORT",
"USHORT","INT","UINT","LONG","ULONG","FLOAT","DOUBLE","TYPE","LABEL","VAR_ID",
"LABELSTR","STRINGCONSTANT","IMPLEMENTATION","ZEROINITIALIZER","TRUETOK","FALSETOK",
"BEGINTOK","ENDTOK","DECLARE","GLOBAL","CONSTANT","SECTION","VOLATILE","TO",
"DOTDOTDOT","NULL_TOK","UNDEF","CONST","INTERNAL","LINKONCE","WEAK","APPENDING",
"OPAQUE","NOT","EXTERNAL","TARGET","TRIPLE","ENDIAN","POINTERSIZE","LITTLE",
"BIG","ALIGN","DEPLIBS","CALL","TAIL","ASM_TOK","MODULE","CC_TOK","CCC_TOK",
"FASTCC_TOK","COLDCC_TOK","RET","BR","SWITCH","INVOKE","UNWIND","UNREACHABLE",
"ADD","SUB","MUL","DIV","REM","AND","OR","XOR","SETLE","SETGE","SETLT","SETGT",
"SETEQ","SETNE","MALLOC","ALLOCA","FREE","LOAD","STORE","GETELEMENTPTR","PHI_TOK",
"CAST","SELECT","SHL","SHR","VAARG","EXTRACTELEMENT","INSERTELEMENT","VAARG_old",
"VANEXT_old","'='","','","'\\\\'","'('","')'","'['","'x'","']'","'<'","'>'",
"'{'","'}'","'*'","'c'","INTVAL","EINT64VAL","ArithmeticOps","LogicalOps","SetCondOps",
"ShiftOps","SIntType","UIntType","IntType","FPType","OptAssign","OptLinkage",
"OptCallingConv","OptAlign","OptCAlign","SectionString","OptSection","GlobalVarAttributes",
"GlobalVarAttribute","TypesV","UpRTypesV","Types","PrimType","UpRTypes","TypeListI",
"ArgTypeListI","ConstVal","ConstExpr","ConstVector","GlobalType","Module","FunctionList",
"ConstPool","@1","@2","AsmBlock","BigOrLittle","TargetDefinition","LibrariesDefinition",
"LibList","Name","OptName","ArgVal","ArgListH","ArgList","FunctionHeaderH","BEGIN",
"FunctionHeader","END","Function","FunctionProto","@3","ConstValueRef","SymbolicValueRef",
"ValueRef","ResolvedVal","BasicBlockList","BasicBlock","InstructionList","BBTerminatorInst",
"JumpTable","Inst","PHIList","ValueRefList","ValueRefListE","OptTailCall","InstVal",
"IndexList","OptVolatile","MemoryInst", NULL
};
#endif

static const short yyr1[] = {     0,
   114,   114,   115,   115,   116,   116,   116,   116,   116,   117,
   117,   117,   118,   118,   118,   118,   118,   118,   119,   119,
   120,   120,   120,   120,   121,   121,   121,   121,   122,   122,
   123,   123,   124,   124,   125,   125,   125,   125,   125,   126,
   126,   126,   126,   126,   127,   127,   128,   128,   129,   130,
   130,   131,   131,   132,   132,   133,   133,   134,   134,   135,
   136,   136,   136,   136,   136,   136,   136,   136,   136,   136,
   136,   136,   136,   137,   137,   137,   137,   137,   137,   137,
   137,   137,   137,   138,   138,   139,   139,   139,   139,   140,
   140,   140,   140,   140,   140,   140,   140,   140,   140,   140,
   140,   140,   140,   140,   140,   141,   141,   141,   141,   141,
   141,   141,   141,   142,   142,   143,   143,   144,   145,   145,
   145,   145,   145,   146,   146,   146,   147,   146,   148,   146,
   146,   146,   146,   149,   150,   150,   151,   151,   151,   152,
   153,   153,   153,   154,   154,   155,   155,   156,   157,   157,
   158,   158,   158,   158,   159,   160,   160,   161,   162,   162,
   163,   165,   164,   166,   166,   166,   166,   166,   166,   166,
   166,   166,   166,   167,   167,   168,   168,   169,   170,   170,
   171,   172,   172,   172,   173,   173,   173,   173,   173,   173,
   173,   173,   173,   174,   174,   175,   176,   176,   177,   177,
   178,   178,   179,   179,   180,   180,   180,   180,   180,   180,
   180,   180,   180,   180,   180,   180,   180,   180,   180,   181,
   181,   182,   182,   183,   183,   183,   183,   183,   183,   183,
   183
};

static const short yyr2[] = {     0,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     2,     0,     1,     1,     1,     1,     0,     0,
     1,     1,     1,     2,     0,     2,     0,     3,     2,     0,
     1,     0,     3,     1,     2,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     2,     4,     5,     5,
     3,     2,     2,     1,     3,     1,     3,     1,     0,     4,
     3,     3,     4,     4,     3,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     6,     5,     8,     6,     6,
     6,     6,     6,     3,     1,     1,     1,     1,     2,     2,
     4,     2,     1,     4,     2,     4,     0,     7,     0,     7,
     3,     4,     0,     1,     1,     1,     3,     3,     3,     3,
     3,     1,     0,     1,     1,     1,     0,     2,     3,     1,
     1,     3,     1,     0,     8,     1,     1,     3,     1,     1,
     2,     0,     3,     1,     1,     1,     1,     1,     1,     1,
     1,     3,     1,     1,     1,     1,     1,     2,     2,     2,
     3,     2,     0,     1,     2,     2,     3,     9,     9,     8,
    13,     1,     1,     6,     5,     2,     6,     7,     1,     3,
     1,     0,     2,     1,     5,     5,     5,     2,     4,     4,
     6,     4,     4,     4,     4,     6,     2,     7,     1,     2,
     0,     1,     0,     3,     6,     3,     6,     2,     4,     6,
     4
};

static const short yydefact[] = {   133,
    39,   123,   122,   162,    35,    36,    37,    38,     0,    40,
   183,   119,   120,   183,   144,   145,     0,     0,     0,    39,
     0,   125,    40,     0,     0,    41,    42,    43,     0,     0,
   184,   180,    34,   159,   160,   161,   179,     0,     0,     0,
   131,     0,     0,     0,     0,     0,    33,   163,   134,   121,
    44,     1,     2,    57,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    74,     0,     0,
     0,     0,   174,     0,     0,    56,    75,    60,   175,    76,
   156,   157,   158,   223,   182,     0,     0,     0,   143,   132,
   126,   124,   116,   117,     0,     0,    77,     0,     0,    59,
    82,    84,     0,     0,    89,    83,   222,     0,   204,     0,
     0,     0,     0,    40,   192,   193,     5,     6,     7,     8,
     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
     0,     0,     0,     0,     0,     0,     0,    19,    20,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   181,    40,
   196,     0,   219,   139,   136,   135,   137,   138,   142,     0,
   129,    61,    62,    63,    64,    65,    66,    67,    68,    69,
    70,    71,     0,     0,     0,     0,   127,     0,     0,     0,
    81,   154,    88,    86,     0,     0,   208,   203,   186,   185,
     0,     0,    24,    28,    23,    27,    22,    26,    21,    25,
    29,    30,     0,     0,    47,    47,   228,     0,     0,   217,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   140,    52,   103,   104,     3,
     4,   101,   102,   105,   100,    96,    97,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,    99,
    98,    52,    58,    58,    85,   153,   147,   150,   151,     0,
     0,    78,   164,   165,   166,   171,   167,   168,   169,   170,
     0,   173,   177,   176,   178,     0,   187,     0,     0,     0,
   224,     0,   226,   221,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   141,     0,   130,     0,     0,     0,     0,    91,   115,     0,
     0,    95,     0,    92,     0,     0,     0,     0,   128,    79,
    80,   146,   148,     0,    50,    87,     0,     0,     0,     0,
     0,     0,     0,     0,   231,     0,     0,   210,     0,   212,
   215,     0,   213,   214,     0,     0,     0,   209,     0,   229,
     0,     0,     0,    54,    52,   221,     0,     0,     0,     0,
    90,    93,    94,     0,     0,     0,     0,   152,   149,    51,
    45,   172,     0,     0,   202,    47,    48,    47,   199,   220,
     0,     0,     0,     0,   205,   206,   207,   202,     0,    49,
    55,    53,     0,     0,     0,     0,   114,     0,     0,     0,
     0,     0,   155,     0,     0,   201,     0,     0,   225,   227,
     0,     0,     0,   211,   216,     0,   230,   107,     0,     0,
     0,     0,     0,     0,     0,    46,     0,     0,     0,   200,
   197,     0,   218,   106,     0,   113,   109,   110,   111,   112,
     0,   190,     0,     0,     0,   198,     0,   188,     0,   189,
     0,     0,   108,     0,     0,     0,     0,     0,     0,   195,
     0,     0,   194,   191,     0,     0,     0
};

static const short yydefgoto[] = {    73,
   232,   246,   247,   248,   249,   173,   174,   203,   175,    20,
    10,    29,   403,   281,   354,   371,   303,   355,    74,    75,
   186,    77,    78,   103,   185,   309,   272,   310,    95,   465,
     1,     2,   252,   227,    50,   157,    41,    90,   160,    79,
   323,   258,   259,   260,    30,    83,    11,    36,    12,    13,
    23,   273,    80,   275,   379,    14,    32,    33,   149,   444,
    85,   210,   406,   407,   150,   151,   335,   152,   153
};

static const short yypact[] = {-32768,
    37,   361,-32768,-32768,-32768,-32768,-32768,-32768,   -44,    49,
    38,-32768,-32768,     3,-32768,-32768,    70,    -2,    42,    50,
     4,-32768,    49,    89,   125,-32768,-32768,-32768,  1054,   -21,
-32768,-32768,    64,-32768,-32768,-32768,-32768,    26,    45,    47,
-32768,    69,    89,  1054,   117,   117,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,    73,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   174,   179,
   180,   122,-32768,    64,    82,-32768,-32768,   -27,-32768,-32768,
-32768,-32768,-32768,  1215,-32768,   163,   -25,   186,   168,-32768,
-32768,-32768,-32768,-32768,  1092,  1130,-32768,    85,    90,-32768,
-32768,   -27,   -38,    94,   864,-32768,-32768,  1092,-32768,   142,
  1168,    23,   144,    49,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
  1092,  1092,  1092,  1092,  1092,  1092,  1092,-32768,-32768,  1092,
  1092,  1092,  1092,  1092,  1092,  1092,  1092,  1092,-32768,    49,
-32768,    75,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   -36,
-32768,   124,   162,   195,   165,   196,   167,   197,   169,   198,
   200,   201,   177,   199,   209,   531,-32768,  1092,  1092,  1092,
-32768,   902,-32768,   103,   101,   680,-32768,-32768,    73,-32768,
   680,   680,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   680,  1054,   116,   118,-32768,   680,   113,   127,
   189,   130,   133,   134,   135,   136,   137,   680,   680,   680,
   139,  1054,  1092,  1092,   202,-32768,   140,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,   143,   145,   146,
   147,   940,  1130,   637,   219,   148,   149,   150,   151,-32768,
-32768,   140,   -35,   -64,   -27,-32768,    64,-32768,   154,   153,
   978,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
  1130,-32768,-32768,-32768,-32768,   158,-32768,   161,   680,    -5,
-32768,    15,-32768,   173,   680,   164,  1092,  1092,  1092,  1092,
  1092,  1092,  1092,   178,   182,   183,  1092,   680,   680,   184,
-32768,   -18,-32768,  1130,  1130,  1130,  1130,-32768,-32768,    -6,
   -26,-32768,   -37,-32768,  1130,  1130,  1130,  1130,-32768,-32768,
-32768,-32768,-32768,  1016,   213,-32768,   -19,   245,   247,   187,
   680,   266,   680,  1092,-32768,   190,   680,-32768,   191,-32768,
-32768,   192,-32768,-32768,   680,   680,   680,-32768,   193,-32768,
  1092,   256,   290,-32768,   140,   173,   261,   206,   207,  1130,
-32768,-32768,-32768,   210,   211,   212,   214,-32768,-32768,-32768,
   244,-32768,   680,   680,  1092,   215,-32768,   215,-32768,   218,
   680,   220,  1092,  1092,-32768,-32768,-32768,  1092,   680,-32768,
-32768,-32768,   216,  1092,  1130,  1130,-32768,  1130,  1130,  1130,
  1130,   295,-32768,   222,   205,   218,   221,   248,-32768,-32768,
  1092,   217,   680,-32768,-32768,   223,-32768,-32768,   224,   225,
   231,   232,   233,   235,   238,-32768,   293,     8,   307,-32768,
-32768,   237,-32768,-32768,  1130,-32768,-32768,-32768,-32768,-32768,
   680,-32768,   763,    41,   324,-32768,   243,-32768,   249,-32768,
   763,   680,-32768,   327,   252,   281,   680,   333,   334,-32768,
   680,   680,-32768,-32768,   359,   360,-32768
};

static const short yypgoto[] = {-32768,
-32768,   277,   278,   279,   280,  -111,  -110,  -415,-32768,   335,
   353,  -103,-32768,  -199,    51,-32768,  -247,-32768,   -40,-32768,
   -29,-32768,   -55,   269,-32768,   -95,   203,  -184,   329,-32768,
-32768,-32768,-32768,-32768,   337,-32768,-32768,-32768,-32768,    10,
-32768,    54,-32768,-32768,   354,-32768,-32768,-32768,-32768,   380,
-32768,  -344,    53,   138,  -102,-32768,   373,-32768,-32768,-32768,
-32768,-32768,    55,     0,-32768,-32768,    35,-32768,-32768
};


#define	YYLAST		1314


static const short yytable[] = {    76,
   177,   201,   202,    92,   319,   187,   283,    81,   190,   331,
   204,    21,   443,    24,    76,   352,   102,   193,   194,   195,
   196,   197,   198,   199,   200,    31,   155,   156,   451,   333,
   207,   191,    34,   211,   212,   353,  -118,   213,   214,   215,
   216,   217,    21,   192,   321,   221,   222,   106,   332,   102,
   193,   194,   195,   196,   197,   198,   199,   200,   311,   313,
    31,     3,   180,   360,   225,   161,   176,     4,   332,    44,
   226,   320,   181,   363,   360,   -58,   106,     5,     6,     7,
     8,   360,   362,   104,   106,    15,   327,    16,    82,   372,
     5,     6,     7,     8,   360,     9,    45,    42,   449,    43,
   361,   205,   206,    47,   208,   209,   455,   392,    25,    26,
    27,    28,    49,    35,   442,   218,   219,   220,    38,    39,
    40,   300,   253,   254,   255,    86,    52,    53,    51,   100,
    55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
    65,    66,    67,    15,    87,    16,    88,   450,    93,    94,
   228,   229,   257,   193,   194,   195,   196,   197,   198,   199,
   200,   223,   224,   279,   -24,   -24,    68,   -23,   -23,   -22,
   -22,   -21,   -21,    89,    76,   -59,   409,    97,   410,   230,
   231,   298,    98,    99,   105,   339,   154,   341,   342,   158,
   178,   159,    76,   299,   348,   179,   182,   188,   -28,   -27,
   -26,   -25,   233,   261,   262,   255,   -31,   -32,   356,   357,
   358,   359,   176,   176,   176,   234,   280,   285,   282,   364,
   365,   366,   367,    69,   287,   301,    70,   286,   251,    71,
   288,    72,   101,   289,   290,   291,   292,   293,   274,   297,
   302,   176,   314,   274,   274,   304,   352,   305,   306,   307,
   315,   316,   317,   318,   324,   274,   325,   338,   328,   340,
   274,   329,   343,   344,   397,   373,   322,   374,   337,   377,
   274,   274,   274,   334,   176,   176,   176,   176,   345,   390,
   414,   415,   346,   347,   351,   176,   176,   176,   176,   375,
   381,   383,   384,   391,   257,   388,   394,   402,   426,   420,
   421,   332,   422,   423,   424,   425,   395,   396,   430,   428,
   398,   399,   400,   441,   401,   408,   201,   202,   411,   418,
   413,   389,   427,   431,   429,   435,   433,   434,   276,   277,
   176,   274,   201,   202,   436,   437,   438,   274,   439,   447,
   278,   440,   445,   446,   452,   284,   453,   457,   459,   454,
   274,   274,   458,   461,   462,   294,   295,   296,   466,   467,
   145,   146,   147,   148,   419,   176,   176,    84,   176,   176,
   176,   176,    46,   184,    96,   370,    48,   369,   250,    91,
   -34,    22,    15,   274,    16,   274,    37,   416,   380,   274,
   393,     4,   -34,   -34,     0,     0,     0,   274,   274,   274,
     0,   -34,   -34,   -34,   -34,   176,     0,   -34,    17,     0,
     0,     0,     0,     0,     0,    18,   330,     0,     0,    19,
     0,     0,   336,     0,     0,   274,   274,     0,     0,     0,
     0,     0,     0,   274,     0,   349,   350,     0,     0,     0,
     0,   274,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   274,     0,     0,   376,     0,
   378,     0,     0,     0,   382,     0,     0,     0,     0,     0,
     0,     0,   385,   386,   387,     0,     0,     0,     0,     0,
     0,     0,     0,   274,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   274,     0,     0,     0,     0,   274,
   404,   405,     0,   274,   274,     0,     0,     0,   412,     0,
     0,     0,     0,     0,     0,     0,   417,     0,     0,     0,
     0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   432,     0,    15,     0,    16,     0,   235,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   236,   237,
     0,     0,     0,     0,     0,     0,     0,     0,   448,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   456,
     0,     0,     0,     0,   460,     0,     0,     0,   463,   464,
   117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
   127,   128,   129,   130,     0,     0,     0,     0,     0,   238,
     0,   239,   240,   138,   139,     0,   241,     0,     0,     0,
     0,     0,     0,     0,     0,   242,     0,     0,   243,     0,
   244,    52,    53,   245,   100,   162,   163,   164,   165,   166,
   167,   168,   169,   170,   171,   172,    66,    67,    15,     0,
    16,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    68,   263,   264,    52,    53,   265,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    15,     0,    16,     0,   266,   267,   268,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   269,   270,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    69,     0,
     0,    70,     0,     0,    71,     0,    72,   312,     0,   117,
   118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
   128,   129,   130,     0,     0,   263,   264,     0,   238,   265,
   239,   240,   138,   139,     0,   241,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   271,   266,   267,
   268,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   269,   270,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   117,   118,   119,   120,   121,   122,   123,   124,
   125,   126,   127,   128,   129,   130,     0,     0,     0,     0,
     0,   238,     0,   239,   240,   138,   139,     0,   241,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    52,    53,
   271,   100,    55,    56,    57,    58,    59,    60,    61,    62,
    63,    64,    65,    66,    67,    15,     0,    16,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   183,     0,     0,     0,     0,     0,    52,    53,    68,   100,
    55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
    65,    66,    67,    15,     0,    16,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   256,     0,
     0,     0,     0,     0,    52,    53,    68,   100,   162,   163,
   164,   165,   166,   167,   168,   169,   170,   171,   172,    66,
    67,    15,     0,    16,     0,    69,     0,     0,    70,     0,
     0,    71,     0,    72,     0,     0,     0,     0,     0,     0,
     0,     0,    52,    53,    68,   100,    55,    56,    57,    58,
    59,    60,    61,    62,    63,    64,    65,    66,    67,    15,
     0,    16,     0,    69,     0,     0,    70,     0,     0,    71,
     0,    72,     0,     0,   326,     0,     0,     0,     0,     0,
    52,    53,    68,   100,    55,    56,    57,    58,    59,    60,
    61,    62,    63,    64,    65,    66,    67,    15,     0,    16,
     0,    69,     0,     0,    70,     0,   308,    71,     0,    72,
     0,     0,   368,     0,     0,     0,     0,     0,    52,    53,
    68,    54,    55,    56,    57,    58,    59,    60,    61,    62,
    63,    64,    65,    66,    67,    15,     0,    16,     0,    69,
     0,     0,    70,     0,     0,    71,     0,    72,     0,     0,
     0,     0,     0,     0,     0,     0,    52,    53,    68,   100,
    55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
    65,    66,    67,    15,     0,    16,     0,    69,     0,     0,
    70,     0,     0,    71,     0,    72,     0,     0,     0,     0,
     0,     0,     0,     0,    52,    53,    68,   100,   162,   163,
   164,   165,   166,   167,   168,   169,   170,   171,   172,    66,
    67,    15,     0,    16,     0,    69,     0,     0,    70,     0,
     0,    71,     0,    72,     0,     0,     0,     0,     0,     0,
     0,     0,    52,    53,    68,   189,    55,    56,    57,    58,
    59,    60,    61,    62,    63,    64,    65,    66,    67,    15,
     0,    16,     0,    69,     0,     0,    70,     0,     0,    71,
     0,    72,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,    68,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    69,     0,     0,    70,     0,     0,    71,     0,    72,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   107,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   108,     0,     0,     0,     0,     0,     0,     0,     0,    69,
   109,   110,    70,     0,     0,    71,     0,    72,   111,   112,
   113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
   123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
   133,     0,     0,   134,   135,   136,   137,   138,   139,   140,
   141,   142,   143,   144
};

static const short yycheck[] = {    29,
    96,   113,   113,    44,   252,   108,   206,    29,   111,    15,
   114,     2,   428,    58,    44,    34,    72,    10,    11,    12,
    13,    14,    15,    16,    17,    23,    52,    53,   444,    15,
   133,     9,    30,   136,   137,    54,     0,   140,   141,   142,
   143,   144,    33,    21,   109,   148,   150,   112,    54,   105,
    10,    11,    12,    13,    14,    15,    16,    17,   243,   244,
    23,    25,   101,   101,   101,    95,    96,    31,    54,    20,
   107,   107,   111,   111,   101,   103,   112,    41,    42,    43,
    44,   101,   109,    74,   112,    22,   271,    24,   110,   109,
    41,    42,    43,    44,   101,    59,    47,   100,   443,    58,
   107,   131,   132,   100,   134,   135,   451,   355,    60,    61,
    62,    63,    24,   111,   107,   145,   146,   147,    49,    50,
    51,   224,   178,   179,   180,   100,     5,     6,     4,     8,
     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
    19,    20,    21,    22,   100,    24,   100,   107,    32,    33,
    27,    28,   182,    10,    11,    12,    13,    14,    15,    16,
    17,    87,    88,   204,     3,     4,    45,     3,     4,     3,
     4,     3,     4,   105,   204,   103,   376,     4,   378,     3,
     4,   222,     4,     4,   103,   288,    24,   290,   291,     4,
   106,    24,   222,   223,   297,   106,   103,    56,     4,     4,
     4,     4,     4,   101,   104,   261,     7,     7,   304,   305,
   306,   307,   242,   243,   244,     7,   101,   105,   101,   315,
   316,   317,   318,   102,    36,    24,   105,   101,   176,   108,
   101,   110,   111,   101,   101,   101,   101,   101,   186,   101,
   101,   271,    24,   191,   192,   103,    34,   103,   103,   103,
   103,   103,   103,   103,   101,   203,   104,   287,   101,   289,
   208,   101,   292,   293,   360,    21,   257,    21,   105,     4,
   218,   219,   220,   101,   304,   305,   306,   307,   101,    24,
   383,   384,   101,   101,   101,   315,   316,   317,   318,   103,
   101,   101,   101,     4,   324,   103,    36,    54,     4,   395,
   396,    54,   398,   399,   400,   401,   101,   101,   411,   105,
   101,   101,   101,    21,   101,   101,   428,   428,   101,   104,
   101,   351,   101,   107,   104,   101,   104,   104,   191,   192,
   360,   279,   444,   444,   104,   104,   104,   285,   104,   435,
   203,   104,    36,   107,    21,   208,   104,    21,    68,   101,
   298,   299,   101,    21,    21,   218,   219,   220,     0,     0,
    84,    84,    84,    84,   394,   395,   396,    33,   398,   399,
   400,   401,    20,   105,    46,   325,    23,   324,   176,    43,
    20,     2,    22,   331,    24,   333,    14,   388,   334,   337,
   356,    31,    32,    33,    -1,    -1,    -1,   345,   346,   347,
    -1,    41,    42,    43,    44,   435,    -1,    47,    48,    -1,
    -1,    -1,    -1,    -1,    -1,    55,   279,    -1,    -1,    59,
    -1,    -1,   285,    -1,    -1,   373,   374,    -1,    -1,    -1,
    -1,    -1,    -1,   381,    -1,   298,   299,    -1,    -1,    -1,
    -1,   389,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   413,    -1,    -1,   331,    -1,
   333,    -1,    -1,    -1,   337,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,   345,   346,   347,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,   441,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   452,    -1,    -1,    -1,    -1,   457,
   373,   374,    -1,   461,   462,    -1,    -1,    -1,   381,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   389,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,     5,     6,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   413,    -1,    22,    -1,    24,    -1,    26,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    39,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   441,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   452,
    -1,    -1,    -1,    -1,   457,    -1,    -1,    -1,   461,   462,
    70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
    80,    81,    82,    83,    -1,    -1,    -1,    -1,    -1,    89,
    -1,    91,    92,    93,    94,    -1,    96,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   105,    -1,    -1,   108,    -1,
   110,     5,     6,   113,     8,     9,    10,    11,    12,    13,
    14,    15,    16,    17,    18,    19,    20,    21,    22,    -1,
    24,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    45,     3,     4,     5,     6,     7,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    22,    -1,    24,    -1,    26,    27,    28,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    39,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
    -1,   105,    -1,    -1,   108,    -1,   110,   111,    -1,    70,
    71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
    81,    82,    83,    -1,    -1,     3,     4,    -1,    89,     7,
    91,    92,    93,    94,    -1,    96,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   108,    26,    27,
    28,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    38,    39,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    70,    71,    72,    73,    74,    75,    76,    77,
    78,    79,    80,    81,    82,    83,    -1,    -1,    -1,    -1,
    -1,    89,    -1,    91,    92,    93,    94,    -1,    96,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     5,     6,
   108,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    17,    18,    19,    20,    21,    22,    -1,    24,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    37,    -1,    -1,    -1,    -1,    -1,     5,     6,    45,     8,
     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
    19,    20,    21,    22,    -1,    24,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,
    -1,    -1,    -1,    -1,     5,     6,    45,     8,     9,    10,
    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
    21,    22,    -1,    24,    -1,   102,    -1,    -1,   105,    -1,
    -1,   108,    -1,   110,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,     5,     6,    45,     8,     9,    10,    11,    12,
    13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
    -1,    24,    -1,   102,    -1,    -1,   105,    -1,    -1,   108,
    -1,   110,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
     5,     6,    45,     8,     9,    10,    11,    12,    13,    14,
    15,    16,    17,    18,    19,    20,    21,    22,    -1,    24,
    -1,   102,    -1,    -1,   105,    -1,   107,   108,    -1,   110,
    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,     5,     6,
    45,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    17,    18,    19,    20,    21,    22,    -1,    24,    -1,   102,
    -1,    -1,   105,    -1,    -1,   108,    -1,   110,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,     5,     6,    45,     8,
     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
    19,    20,    21,    22,    -1,    24,    -1,   102,    -1,    -1,
   105,    -1,    -1,   108,    -1,   110,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,     5,     6,    45,     8,     9,    10,
    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
    21,    22,    -1,    24,    -1,   102,    -1,    -1,   105,    -1,
    -1,   108,    -1,   110,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,     5,     6,    45,     8,     9,    10,    11,    12,
    13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
    -1,    24,    -1,   102,    -1,    -1,   105,    -1,    -1,   108,
    -1,   110,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    45,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   102,    -1,    -1,   105,    -1,    -1,   108,    -1,   110,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    35,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    46,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
    56,    57,   105,    -1,    -1,   108,    -1,   110,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
    76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
    86,    -1,    -1,    89,    90,    91,    92,    93,    94,    95,
    96,    97,    98,    99
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/usr/share/bison.simple"
/* This file comes from bison-1.28.  */

/* Skeleton output parser for bison,
   Copyright (C) 1984, 1989, 1990 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

#ifndef YYSTACK_USE_ALLOCA
#ifdef alloca
#define YYSTACK_USE_ALLOCA
#else /* alloca not defined */
#ifdef __GNUC__
#define YYSTACK_USE_ALLOCA
#define alloca __builtin_alloca
#else /* not GNU C.  */
#if (!defined (__STDC__) && defined (sparc)) || defined (__sparc__) || defined (__sparc) || defined (__sgi) || (defined (__sun) && defined (__i386))
#define YYSTACK_USE_ALLOCA
#include <alloca.h>
#else /* not sparc */
/* We think this test detects Watcom and Microsoft C.  */
/* This used to test MSDOS, but that is a bad idea
   since that symbol is in the user namespace.  */
#if (defined (_MSDOS) || defined (_MSDOS_)) && !defined (__TURBOC__)
#if 0 /* No need for malloc.h, which pollutes the namespace;
	 instead, just don't use alloca.  */
#include <malloc.h>
#endif
#else /* not MSDOS, or __TURBOC__ */
#if defined(_AIX)
/* I don't know what this was needed for, but it pollutes the namespace.
   So I turned it off.   rms, 2 May 1997.  */
/* #include <malloc.h>  */
 #pragma alloca
#define YYSTACK_USE_ALLOCA
#else /* not MSDOS, or __TURBOC__, or _AIX */
#if 0
#ifdef __hpux /* haible@ilog.fr says this works for HPUX 9.05 and up,
		 and on HPUX 10.  Eventually we can turn this on.  */
#define YYSTACK_USE_ALLOCA
#define alloca __builtin_alloca
#endif /* __hpux */
#endif
#endif /* not _AIX */
#endif /* not MSDOS, or __TURBOC__ */
#endif /* not sparc */
#endif /* not GNU C */
#endif /* alloca not defined */
#endif /* YYSTACK_USE_ALLOCA not defined */

#ifdef YYSTACK_USE_ALLOCA
#define YYSTACK_ALLOC alloca
#else
#define YYSTACK_ALLOC malloc
#endif

/* Note: there must be only one dollar sign in this file.
   It is replaced by the list of actions, each action
   as one case of the switch.  */

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		-2
#define YYEOF		0
#define YYACCEPT	goto yyacceptlab
#define YYABORT 	goto yyabortlab
#define YYERROR		goto yyerrlab1
/* Like YYERROR except do call yyerror.
   This remains here temporarily to ease the
   transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */
#define YYFAIL		goto yyerrlab
#define YYRECOVERING()  (!!yyerrstatus)
#define YYBACKUP(token, value) \
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    { yychar = (token), yylval = (value);			\
      yychar1 = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { yyerror ("syntax error: cannot back up"); YYERROR; }	\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

#ifndef YYPURE
#define YYLEX		yylex()
#endif

#ifdef YYPURE
#ifdef YYLSP_NEEDED
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, &yylloc, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval, &yylloc)
#endif
#else /* not YYLSP_NEEDED */
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval)
#endif
#endif /* not YYLSP_NEEDED */
#endif

/* If nonreentrant, generate the variables here */

#ifndef YYPURE

int	yychar;			/*  the lookahead symbol		*/
YYSTYPE	yylval;			/*  the semantic value of the		*/
				/*  lookahead symbol			*/

#ifdef YYLSP_NEEDED
YYLTYPE yylloc;			/*  location data for the lookahead	*/
				/*  symbol				*/
#endif

int yynerrs;			/*  number of parse errors so far       */
#endif  /* not YYPURE */

#if YYDEBUG != 0
int yydebug;			/*  nonzero means print parse trace	*/
/* Since this is uninitialized, it does not stop multiple parsers
   from coexisting.  */
#endif

/*  YYINITDEPTH indicates the initial size of the parser's stacks	*/

#ifndef	YYINITDEPTH
#define YYINITDEPTH 200
#endif

/*  YYMAXDEPTH is the maximum size the stacks can grow to
    (effective only if the built-in stack extension method is used).  */

#if YYMAXDEPTH == 0
#undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

/* Define __yy_memcpy.  Note that the size argument
   should be passed with type unsigned int, because that is what the non-GCC
   definitions require.  With GCC, __builtin_memcpy takes an arg
   of type size_t, but it can handle unsigned int.  */

#if __GNUC__ > 1		/* GNU C and GNU C++ define this.  */
#define __yy_memcpy(TO,FROM,COUNT)	__builtin_memcpy(TO,FROM,COUNT)
#else				/* not GNU C or C++ */
#ifndef __cplusplus

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (to, from, count)
     char *to;
     char *from;
     unsigned int count;
{
  register char *f = from;
  register char *t = to;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#else /* __cplusplus */

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (char *to, char *from, unsigned int count)
{
  register char *t = to;
  register char *f = from;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#endif
#endif

#line 217 "/usr/share/bison.simple"

/* The user can define YYPARSE_PARAM as the name of an argument to be passed
   into yyparse.  The argument should have type void *.
   It should actually point to an object.
   Grammar actions can access the variable by casting it
   to the proper pointer type.  */

#ifdef YYPARSE_PARAM
#ifdef __cplusplus
#define YYPARSE_PARAM_ARG void *YYPARSE_PARAM
#define YYPARSE_PARAM_DECL
#else /* not __cplusplus */
#define YYPARSE_PARAM_ARG YYPARSE_PARAM
#define YYPARSE_PARAM_DECL void *YYPARSE_PARAM;
#endif /* not __cplusplus */
#else /* not YYPARSE_PARAM */
#define YYPARSE_PARAM_ARG
#define YYPARSE_PARAM_DECL
#endif /* not YYPARSE_PARAM */

/* Prevent warning if -Wstrict-prototypes.  */
#ifdef __GNUC__
#ifdef YYPARSE_PARAM
int yyparse (void *);
#else
int yyparse (void);
#endif
#endif

int
yyparse(YYPARSE_PARAM_ARG)
     YYPARSE_PARAM_DECL
{
  register int yystate;
  register int yyn;
  register short *yyssp;
  register YYSTYPE *yyvsp;
  int yyerrstatus;	/*  number of tokens to shift before error messages enabled */
  int yychar1 = 0;		/*  lookahead token as an internal (translated) token number */

  short	yyssa[YYINITDEPTH];	/*  the state stack			*/
  YYSTYPE yyvsa[YYINITDEPTH];	/*  the semantic value stack		*/

  short *yyss = yyssa;		/*  refer to the stacks thru separate pointers */
  YYSTYPE *yyvs = yyvsa;	/*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YYLSP_NEEDED
  YYLTYPE yylsa[YYINITDEPTH];	/*  the location stack			*/
  YYLTYPE *yyls = yylsa;
  YYLTYPE *yylsp;

#define YYPOPSTACK   (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK   (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;
  int yyfree_stacks = 0;

#ifdef YYPURE
  int yychar;
  YYSTYPE yylval;
  int yynerrs;
#ifdef YYLSP_NEEDED
  YYLTYPE yylloc;
#endif
#endif

  YYSTYPE yyval;		/*  the variable used to return		*/
				/*  semantic values from the action	*/
				/*  routines				*/

  int yylen;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Starting parse\n");
#endif

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YYLSP_NEEDED
  yylsp = yyls;
#endif

/* Push a new state, which is found in  yystate  .  */
/* In all cases, when you get here, the value and location stacks
   have just been pushed. so pushing a state here evens the stacks.  */
yynewstate:

  *++yyssp = yystate;

  if (yyssp >= yyss + yystacksize - 1)
    {
      /* Give user a chance to reallocate the stack */
      /* Use copies of these so that the &'s don't force the real ones into memory. */
      YYSTYPE *yyvs1 = yyvs;
      short *yyss1 = yyss;
#ifdef YYLSP_NEEDED
      YYLTYPE *yyls1 = yyls;
#endif

      /* Get the current used size of the three stacks, in elements.  */
      int size = yyssp - yyss + 1;

#ifdef yyoverflow
      /* Each stack pointer address is followed by the size of
	 the data in use in that stack, in bytes.  */
#ifdef YYLSP_NEEDED
      /* This used to be a conditional around just the two extra args,
	 but that might be undefined if yyoverflow is a macro.  */
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yyls1, size * sizeof (*yylsp),
		 &yystacksize);
#else
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yystacksize);
#endif

      yyss = yyss1; yyvs = yyvs1;
#ifdef YYLSP_NEEDED
      yyls = yyls1;
#endif
#else /* no yyoverflow */
      /* Extend the stack our own way.  */
      if (yystacksize >= YYMAXDEPTH)
	{
	  yyerror("parser stack overflow");
	  if (yyfree_stacks)
	    {
	      free (yyss);
	      free (yyvs);
#ifdef YYLSP_NEEDED
	      free (yyls);
#endif
	    }
	  return 2;
	}
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
	yystacksize = YYMAXDEPTH;
#ifndef YYSTACK_USE_ALLOCA
      yyfree_stacks = 1;
#endif
      yyss = (short *) YYSTACK_ALLOC (yystacksize * sizeof (*yyssp));
      __yy_memcpy ((char *)yyss, (char *)yyss1,
		   size * (unsigned int) sizeof (*yyssp));
      yyvs = (YYSTYPE *) YYSTACK_ALLOC (yystacksize * sizeof (*yyvsp));
      __yy_memcpy ((char *)yyvs, (char *)yyvs1,
		   size * (unsigned int) sizeof (*yyvsp));
#ifdef YYLSP_NEEDED
      yyls = (YYLTYPE *) YYSTACK_ALLOC (yystacksize * sizeof (*yylsp));
      __yy_memcpy ((char *)yyls, (char *)yyls1,
		   size * (unsigned int) sizeof (*yylsp));
#endif
#endif /* no yyoverflow */

      yyssp = yyss + size - 1;
      yyvsp = yyvs + size - 1;
#ifdef YYLSP_NEEDED
      yylsp = yyls + size - 1;
#endif

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

      if (yyssp >= yyss + yystacksize - 1)
	YYABORT;
    }

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Entering state %d\n", yystate);
#endif

  goto yybackup;
 yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* yychar is either YYEMPTY or YYEOF
     or a valid token in external form.  */

  if (yychar == YYEMPTY)
    {
#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Reading a token: ");
#endif
      yychar = YYLEX;
    }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (yychar <= 0)		/* This means end of input. */
    {
      yychar1 = 0;
      yychar = YYEOF;		/* Don't call YYLEX any more */

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Now at end of input.\n");
#endif
    }
  else
    {
      yychar1 = YYTRANSLATE(yychar);

#if YYDEBUG != 0
      if (yydebug)
	{
	  fprintf (stderr, "Next token is %d (%s", yychar, yytname[yychar1]);
	  /* Give the individual parser a way to print the precise meaning
	     of a token, for further debugging info.  */
#ifdef YYPRINT
	  YYPRINT (stderr, yychar, yylval);
#endif
	  fprintf (stderr, ")\n");
	}
#endif
    }

  yyn += yychar1;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != yychar1)
    goto yydefault;

  yyn = yytable[yyn];

  /* yyn is what to do for this token type in this state.
     Negative => reduce, -yyn is rule number.
     Positive => shift, yyn is new state.
       New state is final state => don't bother to shift,
       just return success.
     0, or most negative number => error.  */

  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrlab;

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting token %d (%s), ", yychar, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  /* count tokens shifted since error; after three, turn off error status.  */
  if (yyerrstatus) yyerrstatus--;

  yystate = yyn;
  goto yynewstate;

/* Do the default action for the current state.  */
yydefault:

  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;

/* Do a reduction.  yyn is the number of a rule to reduce with.  */
yyreduce:
  yylen = yyr2[yyn];
  if (yylen > 0)
    yyval = yyvsp[1-yylen]; /* implement default value of the action */

#if YYDEBUG != 0
  if (yydebug)
    {
      int i;

      fprintf (stderr, "Reducing via rule %d (line %d), ",
	       yyn, yyrline[yyn]);

      /* Print the symbols being reduced, and their result.  */
      for (i = yyprhs[yyn]; yyrhs[i] > 0; i++)
	fprintf (stderr, "%s ", yytname[yyrhs[i]]);
      fprintf (stderr, " -> %s\n", yytname[yyr1[yyn]]);
    }
#endif


  switch (yyn) {

case 2:
#line 998 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  if (yyvsp[0].UIntVal > (uint32_t)INT32_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  yyval.SIntVal = (int32_t)yyvsp[0].UIntVal;
;
    break;}
case 4:
#line 1006 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  if (yyvsp[0].UInt64Val > (uint64_t)INT64_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  yyval.SInt64Val = (int64_t)yyvsp[0].UInt64Val;
;
    break;}
case 33:
#line 1029 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.StrVal = yyvsp[-1].StrVal;
  ;
    break;}
case 34:
#line 1032 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.StrVal = 0;
  ;
    break;}
case 35:
#line 1036 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.Linkage = GlobalValue::InternalLinkage; ;
    break;}
case 36:
#line 1037 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.Linkage = GlobalValue::LinkOnceLinkage; ;
    break;}
case 37:
#line 1038 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.Linkage = GlobalValue::WeakLinkage; ;
    break;}
case 38:
#line 1039 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.Linkage = GlobalValue::AppendingLinkage; ;
    break;}
case 39:
#line 1040 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.Linkage = GlobalValue::ExternalLinkage; ;
    break;}
case 40:
#line 1042 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.UIntVal = CallingConv::C; ;
    break;}
case 41:
#line 1043 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.UIntVal = CallingConv::C; ;
    break;}
case 42:
#line 1044 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.UIntVal = CallingConv::Fast; ;
    break;}
case 43:
#line 1045 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.UIntVal = CallingConv::Cold; ;
    break;}
case 44:
#line 1046 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
                   if ((unsigned)yyvsp[0].UInt64Val != yyvsp[0].UInt64Val)
                     ThrowException("Calling conv too large!");
                   yyval.UIntVal = yyvsp[0].UInt64Val;
                 ;
    break;}
case 45:
#line 1054 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.UIntVal = 0; ;
    break;}
case 46:
#line 1055 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  yyval.UIntVal = yyvsp[0].UInt64Val;
  if (yyval.UIntVal != 0 && !isPowerOf2_32(yyval.UIntVal))
    ThrowException("Alignment must be a power of two!");
;
    break;}
case 47:
#line 1060 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.UIntVal = 0; ;
    break;}
case 48:
#line 1061 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  yyval.UIntVal = yyvsp[0].UInt64Val;
  if (yyval.UIntVal != 0 && !isPowerOf2_32(yyval.UIntVal))
    ThrowException("Alignment must be a power of two!");
;
    break;}
case 49:
#line 1068 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  for (unsigned i = 0, e = strlen(yyvsp[0].StrVal); i != e; ++i)
    if (yyvsp[0].StrVal[i] == '"' || yyvsp[0].StrVal[i] == '\\')
      ThrowException("Invalid character in section name!");
  yyval.StrVal = yyvsp[0].StrVal;
;
    break;}
case 50:
#line 1075 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.StrVal = 0; ;
    break;}
case 51:
#line 1076 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.StrVal = yyvsp[0].StrVal; ;
    break;}
case 52:
#line 1081 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{;
    break;}
case 53:
#line 1082 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{;
    break;}
case 54:
#line 1083 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    CurGV->setSection(yyvsp[0].StrVal);
    free(yyvsp[0].StrVal);
  ;
    break;}
case 55:
#line 1087 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[0].UInt64Val != 0 && !isPowerOf2_32(yyvsp[0].UInt64Val))
      ThrowException("Alignment must be a power of two!");
    CurGV->setAlignment(yyvsp[0].UInt64Val);
  ;
    break;}
case 57:
#line 1100 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.TypeVal = new PATypeHolder(yyvsp[0].PrimType); ;
    break;}
case 59:
#line 1101 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.TypeVal = new PATypeHolder(yyvsp[0].PrimType); ;
    break;}
case 60:
#line 1103 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!UpRefs.empty())
      ThrowException("Invalid upreference in type: " + (*yyvsp[0].TypeVal)->getDescription());
    yyval.TypeVal = yyvsp[0].TypeVal;
  ;
    break;}
case 74:
#line 1114 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.TypeVal = new PATypeHolder(OpaqueType::get());
  ;
    break;}
case 75:
#line 1117 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.TypeVal = new PATypeHolder(yyvsp[0].PrimType);
  ;
    break;}
case 76:
#line 1120 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{            // Named types are also simple types...
  yyval.TypeVal = new PATypeHolder(getTypeVal(yyvsp[0].ValIDVal));
;
    break;}
case 77:
#line 1126 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                   // Type UpReference
    if (yyvsp[0].UInt64Val > (uint64_t)~0U) ThrowException("Value out of range!");
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder
    UpRefs.push_back(UpRefRecord((unsigned)yyvsp[0].UInt64Val, OT));  // Add to vector...
    yyval.TypeVal = new PATypeHolder(OT);
    UR_OUT("New Upreference!\n");
  ;
    break;}
case 78:
#line 1133 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{           // Function derived type?
    std::vector<const Type*> Params;
    for (std::list<llvm::PATypeHolder>::iterator I = yyvsp[-1].TypeList->begin(),
           E = yyvsp[-1].TypeList->end(); I != E; ++I)
      Params.push_back(*I);
    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    yyval.TypeVal = new PATypeHolder(HandleUpRefs(FunctionType::get(*yyvsp[-3].TypeVal,Params,isVarArg)));
    delete yyvsp[-1].TypeList;      // Delete the argument list
    delete yyvsp[-3].TypeVal;      // Delete the return type handle
  ;
    break;}
case 79:
#line 1145 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{          // Sized array type?
    yyval.TypeVal = new PATypeHolder(HandleUpRefs(ArrayType::get(*yyvsp[-1].TypeVal, (unsigned)yyvsp[-3].UInt64Val)));
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 80:
#line 1149 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{          // Packed array type?
     const llvm::Type* ElemTy = yyvsp[-1].TypeVal->get();
     if ((unsigned)yyvsp[-3].UInt64Val != yyvsp[-3].UInt64Val)
        ThrowException("Unsigned result not equal to signed result");
     if (!ElemTy->isPrimitiveType())
        ThrowException("Elemental type of a PackedType must be primitive");
     if (!isPowerOf2_32(yyvsp[-3].UInt64Val))
       ThrowException("Vector length should be a power of 2!");
     yyval.TypeVal = new PATypeHolder(HandleUpRefs(PackedType::get(*yyvsp[-1].TypeVal, (unsigned)yyvsp[-3].UInt64Val)));
     delete yyvsp[-1].TypeVal;
  ;
    break;}
case 81:
#line 1160 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                        // Structure type?
    std::vector<const Type*> Elements;
    for (std::list<llvm::PATypeHolder>::iterator I = yyvsp[-1].TypeList->begin(),
           E = yyvsp[-1].TypeList->end(); I != E; ++I)
      Elements.push_back(*I);

    yyval.TypeVal = new PATypeHolder(HandleUpRefs(StructType::get(Elements)));
    delete yyvsp[-1].TypeList;
  ;
    break;}
case 82:
#line 1169 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                                  // Empty structure type?
    yyval.TypeVal = new PATypeHolder(StructType::get(std::vector<const Type*>()));
  ;
    break;}
case 83:
#line 1172 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                             // Pointer type?
    yyval.TypeVal = new PATypeHolder(HandleUpRefs(PointerType::get(*yyvsp[-1].TypeVal)));
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 84:
#line 1180 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.TypeList = new std::list<PATypeHolder>();
    yyval.TypeList->push_back(*yyvsp[0].TypeVal); delete yyvsp[0].TypeVal;
  ;
    break;}
case 85:
#line 1184 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    (yyval.TypeList=yyvsp[-2].TypeList)->push_back(*yyvsp[0].TypeVal); delete yyvsp[0].TypeVal;
  ;
    break;}
case 87:
#line 1190 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    (yyval.TypeList=yyvsp[-2].TypeList)->push_back(Type::VoidTy);
  ;
    break;}
case 88:
#line 1193 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    (yyval.TypeList = new std::list<PATypeHolder>())->push_back(Type::VoidTy);
  ;
    break;}
case 89:
#line 1196 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.TypeList = new std::list<PATypeHolder>();
  ;
    break;}
case 90:
#line 1206 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ // Nonempty unsized arr
    const ArrayType *ATy = dyn_cast<ArrayType>(yyvsp[-3].TypeVal->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*yyvsp[-3].TypeVal)->getDescription() + "'!");
    const Type *ETy = ATy->getElementType();
    int NumElements = ATy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)yyvsp[-1].ConstVector->size())
      ThrowException("Type mismatch: constant sized array initialized with " +
                     utostr(yyvsp[-1].ConstVector->size()) +  " arguments, but has size of " + 
                     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < yyvsp[-1].ConstVector->size(); i++) {
      if (ETy != (*yyvsp[-1].ConstVector)[i]->getType())
        ThrowException("Element #" + utostr(i) + " is not of type '" + 
                       ETy->getDescription() +"' as required!\nIt is of type '"+
                       (*yyvsp[-1].ConstVector)[i]->getType()->getDescription() + "'.");
    }

    yyval.ConstVal = ConstantArray::get(ATy, *yyvsp[-1].ConstVector);
    delete yyvsp[-3].TypeVal; delete yyvsp[-1].ConstVector;
  ;
    break;}
case 91:
#line 1231 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const ArrayType *ATy = dyn_cast<ArrayType>(yyvsp[-2].TypeVal->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*yyvsp[-2].TypeVal)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    if (NumElements != -1 && NumElements != 0) 
      ThrowException("Type mismatch: constant sized array initialized with 0"
                     " arguments, but has size of " + itostr(NumElements) +"!");
    yyval.ConstVal = ConstantArray::get(ATy, std::vector<Constant*>());
    delete yyvsp[-2].TypeVal;
  ;
    break;}
case 92:
#line 1244 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const ArrayType *ATy = dyn_cast<ArrayType>(yyvsp[-2].TypeVal->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*yyvsp[-2].TypeVal)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    const Type *ETy = ATy->getElementType();
    char *EndStr = UnEscapeLexed(yyvsp[0].StrVal, true);
    if (NumElements != -1 && NumElements != (EndStr-yyvsp[0].StrVal))
      ThrowException("Can't build string constant of size " + 
                     itostr((int)(EndStr-yyvsp[0].StrVal)) +
                     " when array has size " + itostr(NumElements) + "!");
    std::vector<Constant*> Vals;
    if (ETy == Type::SByteTy) {
      for (signed char *C = (signed char *)yyvsp[0].StrVal; C != (signed char *)EndStr; ++C)
        Vals.push_back(ConstantSInt::get(ETy, *C));
    } else if (ETy == Type::UByteTy) {
      for (unsigned char *C = (unsigned char *)yyvsp[0].StrVal; 
           C != (unsigned char*)EndStr; ++C)
        Vals.push_back(ConstantUInt::get(ETy, *C));
    } else {
      free(yyvsp[0].StrVal);
      ThrowException("Cannot build string arrays of non byte sized elements!");
    }
    free(yyvsp[0].StrVal);
    yyval.ConstVal = ConstantArray::get(ATy, Vals);
    delete yyvsp[-2].TypeVal;
  ;
    break;}
case 93:
#line 1273 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ // Nonempty unsized arr
    const PackedType *PTy = dyn_cast<PackedType>(yyvsp[-3].TypeVal->get());
    if (PTy == 0)
      ThrowException("Cannot make packed constant with type: '" + 
                     (*yyvsp[-3].TypeVal)->getDescription() + "'!");
    const Type *ETy = PTy->getElementType();
    int NumElements = PTy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)yyvsp[-1].ConstVector->size())
      ThrowException("Type mismatch: constant sized packed initialized with " +
                     utostr(yyvsp[-1].ConstVector->size()) +  " arguments, but has size of " + 
                     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < yyvsp[-1].ConstVector->size(); i++) {
      if (ETy != (*yyvsp[-1].ConstVector)[i]->getType())
        ThrowException("Element #" + utostr(i) + " is not of type '" + 
           ETy->getDescription() +"' as required!\nIt is of type '"+
           (*yyvsp[-1].ConstVector)[i]->getType()->getDescription() + "'.");
    }

    yyval.ConstVal = ConstantPacked::get(PTy, *yyvsp[-1].ConstVector);
    delete yyvsp[-3].TypeVal; delete yyvsp[-1].ConstVector;
  ;
    break;}
case 94:
#line 1298 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const StructType *STy = dyn_cast<StructType>(yyvsp[-3].TypeVal->get());
    if (STy == 0)
      ThrowException("Cannot make struct constant with type: '" + 
                     (*yyvsp[-3].TypeVal)->getDescription() + "'!");

    if (yyvsp[-1].ConstVector->size() != STy->getNumContainedTypes())
      ThrowException("Illegal number of initializers for structure type!");

    // Check to ensure that constants are compatible with the type initializer!
    for (unsigned i = 0, e = yyvsp[-1].ConstVector->size(); i != e; ++i)
      if ((*yyvsp[-1].ConstVector)[i]->getType() != STy->getElementType(i))
        ThrowException("Expected type '" +
                       STy->getElementType(i)->getDescription() +
                       "' for element #" + utostr(i) +
                       " of structure initializer!");

    yyval.ConstVal = ConstantStruct::get(STy, *yyvsp[-1].ConstVector);
    delete yyvsp[-3].TypeVal; delete yyvsp[-1].ConstVector;
  ;
    break;}
case 95:
#line 1318 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const StructType *STy = dyn_cast<StructType>(yyvsp[-2].TypeVal->get());
    if (STy == 0)
      ThrowException("Cannot make struct constant with type: '" + 
                     (*yyvsp[-2].TypeVal)->getDescription() + "'!");

    if (STy->getNumContainedTypes() != 0)
      ThrowException("Illegal number of initializers for structure type!");

    yyval.ConstVal = ConstantStruct::get(STy, std::vector<Constant*>());
    delete yyvsp[-2].TypeVal;
  ;
    break;}
case 96:
#line 1330 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const PointerType *PTy = dyn_cast<PointerType>(yyvsp[-1].TypeVal->get());
    if (PTy == 0)
      ThrowException("Cannot make null pointer constant with type: '" + 
                     (*yyvsp[-1].TypeVal)->getDescription() + "'!");

    yyval.ConstVal = ConstantPointerNull::get(PTy);
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 97:
#line 1339 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ConstVal = UndefValue::get(yyvsp[-1].TypeVal->get());
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 98:
#line 1343 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const PointerType *Ty = dyn_cast<PointerType>(yyvsp[-1].TypeVal->get());
    if (Ty == 0)
      ThrowException("Global const reference must be a pointer type!");

    // ConstExprs can exist in the body of a function, thus creating
    // GlobalValues whenever they refer to a variable.  Because we are in
    // the context of a function, getValNonImprovising will search the functions
    // symbol table instead of the module symbol table for the global symbol,
    // which throws things all off.  To get around this, we just tell
    // getValNonImprovising that we are at global scope here.
    //
    Function *SavedCurFn = CurFun.CurrentFunction;
    CurFun.CurrentFunction = 0;

    Value *V = getValNonImprovising(Ty, yyvsp[0].ValIDVal);

    CurFun.CurrentFunction = SavedCurFn;

    // If this is an initializer for a constant pointer, which is referencing a
    // (currently) undefined variable, create a stub now that shall be replaced
    // in the future with the right type of variable.
    //
    if (V == 0) {
      assert(isa<PointerType>(Ty) && "Globals may only be used as pointers!");
      const PointerType *PT = cast<PointerType>(Ty);

      // First check to see if the forward references value is already created!
      PerModuleInfo::GlobalRefsType::iterator I =
        CurModule.GlobalRefs.find(std::make_pair(PT, yyvsp[0].ValIDVal));
    
      if (I != CurModule.GlobalRefs.end()) {
        V = I->second;             // Placeholder already exists, use it...
        yyvsp[0].ValIDVal.destroy();
      } else {
        std::string Name;
        if (yyvsp[0].ValIDVal.Type == ValID::NameVal) Name = yyvsp[0].ValIDVal.Name;

        // Create the forward referenced global.
        GlobalValue *GV;
        if (const FunctionType *FTy = 
                 dyn_cast<FunctionType>(PT->getElementType())) {
          GV = new Function(FTy, GlobalValue::ExternalLinkage, Name,
                            CurModule.CurrentModule);
        } else {
          GV = new GlobalVariable(PT->getElementType(), false,
                                  GlobalValue::ExternalLinkage, 0,
                                  Name, CurModule.CurrentModule);
        }

        // Keep track of the fact that we have a forward ref to recycle it
        CurModule.GlobalRefs.insert(std::make_pair(std::make_pair(PT, yyvsp[0].ValIDVal), GV));
        V = GV;
      }
    }

    yyval.ConstVal = cast<GlobalValue>(V);
    delete yyvsp[-1].TypeVal;            // Free the type handle
  ;
    break;}
case 99:
#line 1402 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[-1].TypeVal->get() != yyvsp[0].ConstVal->getType())
      ThrowException("Mismatched types for constant expression!");
    yyval.ConstVal = yyvsp[0].ConstVal;
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 100:
#line 1408 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const Type *Ty = yyvsp[-1].TypeVal->get();
    if (isa<FunctionType>(Ty) || Ty == Type::LabelTy || isa<OpaqueType>(Ty))
      ThrowException("Cannot create a null initialized value of this type!");
    yyval.ConstVal = Constant::getNullValue(Ty);
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 101:
#line 1416 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{      // integral constants
    if (!ConstantSInt::isValueValidForType(yyvsp[-1].PrimType, yyvsp[0].SInt64Val))
      ThrowException("Constant value doesn't fit in type!");
    yyval.ConstVal = ConstantSInt::get(yyvsp[-1].PrimType, yyvsp[0].SInt64Val);
  ;
    break;}
case 102:
#line 1421 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{            // integral constants
    if (!ConstantUInt::isValueValidForType(yyvsp[-1].PrimType, yyvsp[0].UInt64Val))
      ThrowException("Constant value doesn't fit in type!");
    yyval.ConstVal = ConstantUInt::get(yyvsp[-1].PrimType, yyvsp[0].UInt64Val);
  ;
    break;}
case 103:
#line 1426 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                      // Boolean constants
    yyval.ConstVal = ConstantBool::True;
  ;
    break;}
case 104:
#line 1429 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                     // Boolean constants
    yyval.ConstVal = ConstantBool::False;
  ;
    break;}
case 105:
#line 1432 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                   // Float & Double constants
    if (!ConstantFP::isValueValidForType(yyvsp[-1].PrimType, yyvsp[0].FPVal))
      ThrowException("Floating point constant invalid for type!!");
    yyval.ConstVal = ConstantFP::get(yyvsp[-1].PrimType, yyvsp[0].FPVal);
  ;
    break;}
case 106:
#line 1439 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!yyvsp[-3].ConstVal->getType()->isFirstClassType())
      ThrowException("cast constant expression from a non-primitive type: '" +
                     yyvsp[-3].ConstVal->getType()->getDescription() + "'!");
    if (!yyvsp[-1].TypeVal->get()->isFirstClassType())
      ThrowException("cast constant expression to a non-primitive type: '" +
                     yyvsp[-1].TypeVal->get()->getDescription() + "'!");
    yyval.ConstVal = ConstantExpr::getCast(yyvsp[-3].ConstVal, yyvsp[-1].TypeVal->get());
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 107:
#line 1449 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!isa<PointerType>(yyvsp[-2].ConstVal->getType()))
      ThrowException("GetElementPtr requires a pointer operand!");

    // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte struct
    // indices to uint struct indices for compatibility.
    generic_gep_type_iterator<std::vector<Value*>::iterator>
      GTI = gep_type_begin(yyvsp[-2].ConstVal->getType(), yyvsp[-1].ValueList->begin(), yyvsp[-1].ValueList->end()),
      GTE = gep_type_end(yyvsp[-2].ConstVal->getType(), yyvsp[-1].ValueList->begin(), yyvsp[-1].ValueList->end());
    for (unsigned i = 0, e = yyvsp[-1].ValueList->size(); i != e && GTI != GTE; ++i, ++GTI)
      if (isa<StructType>(*GTI))        // Only change struct indices
        if (ConstantUInt *CUI = dyn_cast<ConstantUInt>((*yyvsp[-1].ValueList)[i]))
          if (CUI->getType() == Type::UByteTy)
            (*yyvsp[-1].ValueList)[i] = ConstantExpr::getCast(CUI, Type::UIntTy);

    const Type *IdxTy =
      GetElementPtrInst::getIndexedType(yyvsp[-2].ConstVal->getType(), *yyvsp[-1].ValueList, true);
    if (!IdxTy)
      ThrowException("Index list invalid for constant getelementptr!");

    std::vector<Constant*> IdxVec;
    for (unsigned i = 0, e = yyvsp[-1].ValueList->size(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>((*yyvsp[-1].ValueList)[i]))
        IdxVec.push_back(C);
      else
        ThrowException("Indices to constant getelementptr must be constants!");

    delete yyvsp[-1].ValueList;

    yyval.ConstVal = ConstantExpr::getGetElementPtr(yyvsp[-2].ConstVal, IdxVec);
  ;
    break;}
case 108:
#line 1480 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[-5].ConstVal->getType() != Type::BoolTy)
      ThrowException("Select condition must be of boolean type!");
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("Select operand types must match!");
    yyval.ConstVal = ConstantExpr::getSelect(yyvsp[-5].ConstVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;
    break;}
case 109:
#line 1487 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("Binary operator types must match!");
    // HACK: llvm 1.3 and earlier used to emit invalid pointer constant exprs.
    // To retain backward compatibility with these early compilers, we emit a
    // cast to the appropriate integer type automatically if we are in the
    // broken case.  See PR424 for more information.
    if (!isa<PointerType>(yyvsp[-3].ConstVal->getType())) {
      yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
    } else {
      const Type *IntPtrTy = 0;
      switch (CurModule.CurrentModule->getPointerSize()) {
      case Module::Pointer32: IntPtrTy = Type::IntTy; break;
      case Module::Pointer64: IntPtrTy = Type::LongTy; break;
      default: ThrowException("invalid pointer binary constant expr!");
      }
      yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, ConstantExpr::getCast(yyvsp[-3].ConstVal, IntPtrTy),
                             ConstantExpr::getCast(yyvsp[-1].ConstVal, IntPtrTy));
      yyval.ConstVal = ConstantExpr::getCast(yyval.ConstVal, yyvsp[-3].ConstVal->getType());
    }
  ;
    break;}
case 110:
#line 1508 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("Logical operator types must match!");
    if (!yyvsp[-3].ConstVal->getType()->isIntegral()) {
      if (!isa<PackedType>(yyvsp[-3].ConstVal->getType()) || 
          !cast<PackedType>(yyvsp[-3].ConstVal->getType())->getElementType()->isIntegral())
        ThrowException("Logical operator requires integral operands!");
    }
    yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;
    break;}
case 111:
#line 1518 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("setcc operand types must match!");
    yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;
    break;}
case 112:
#line 1523 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[-1].ConstVal->getType() != Type::UByteTy)
      ThrowException("Shift count for shift constant must be unsigned byte!");
    if (!yyvsp[-3].ConstVal->getType()->isInteger())
      ThrowException("Shift constant expression requires integer operand!");
    yyval.ConstVal = ConstantExpr::get(yyvsp[-5].OtherOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;
    break;}
case 113:
#line 1530 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
        if (!isa<PackedType>(yyvsp[-3].ConstVal->getType()))
      ThrowException("First operand of extractelement must be "
                     "packed type!");
    if (yyvsp[-1].ConstVal->getType() != Type::UIntTy)
      ThrowException("Second operand of extractelement must be uint!");
    yyval.ConstVal = ConstantExpr::getExtractElement(yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;
    break;}
case 114:
#line 1540 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    (yyval.ConstVector = yyvsp[-2].ConstVector)->push_back(yyvsp[0].ConstVal);
  ;
    break;}
case 115:
#line 1543 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ConstVector = new std::vector<Constant*>();
    yyval.ConstVector->push_back(yyvsp[0].ConstVal);
  ;
    break;}
case 116:
#line 1550 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.BoolVal = false; ;
    break;}
case 117:
#line 1550 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.BoolVal = true; ;
    break;}
case 118:
#line 1560 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  yyval.ModuleVal = ParserResult = yyvsp[0].ModuleVal;
  CurModule.ModuleDone();
;
    break;}
case 119:
#line 1567 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ModuleVal = yyvsp[-1].ModuleVal;
    CurFun.FunctionDone();
  ;
    break;}
case 120:
#line 1571 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ModuleVal = yyvsp[-1].ModuleVal;
  ;
    break;}
case 121:
#line 1574 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ModuleVal = yyvsp[-3].ModuleVal;
  ;
    break;}
case 122:
#line 1577 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ModuleVal = yyvsp[-1].ModuleVal;
  ;
    break;}
case 123:
#line 1580 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ModuleVal = CurModule.CurrentModule;
    // Emit an error if there are any unresolved types left.
    if (!CurModule.LateResolveTypes.empty()) {
      const ValID &DID = CurModule.LateResolveTypes.begin()->first;
      if (DID.Type == ValID::NameVal)
        ThrowException("Reference to an undefined type: '"+DID.getName() + "'");
      else
        ThrowException("Reference to an undefined type: #" + itostr(DID.Num));
    }
  ;
    break;}
case 124:
#line 1593 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    // Eagerly resolve types.  This is not an optimization, this is a
    // requirement that is due to the fact that we could have this:
    //
    // %list = type { %list * }
    // %list = type { %list * }    ; repeated type decl
    //
    // If types are not resolved eagerly, then the two types will not be
    // determined to be the same type!
    //
    ResolveTypeTo(yyvsp[-2].StrVal, *yyvsp[0].TypeVal);

    if (!setTypeName(*yyvsp[0].TypeVal, yyvsp[-2].StrVal) && !yyvsp[-2].StrVal) {
      // If this is a named type that is not a redefinition, add it to the slot
      // table.
      CurModule.Types.push_back(*yyvsp[0].TypeVal);
    }

    delete yyvsp[0].TypeVal;
  ;
    break;}
case 125:
#line 1613 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{       // Function prototypes can be in const pool
  ;
    break;}
case 126:
#line 1615 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{  // Asm blocks can be in the const pool
  ;
    break;}
case 127:
#line 1617 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[0].ConstVal == 0) ThrowException("Global value initializer is not a constant!");
    CurGV = ParseGlobalVariable(yyvsp[-3].StrVal, yyvsp[-2].Linkage, yyvsp[-1].BoolVal, yyvsp[0].ConstVal->getType(), yyvsp[0].ConstVal);
                                                       ;
    break;}
case 128:
#line 1620 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    CurGV = 0;
  ;
    break;}
case 129:
#line 1623 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    CurGV = ParseGlobalVariable(yyvsp[-3].StrVal, GlobalValue::ExternalLinkage,
                                             yyvsp[-1].BoolVal, *yyvsp[0].TypeVal, 0);
    delete yyvsp[0].TypeVal;
                                                   ;
    break;}
case 130:
#line 1627 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    CurGV = 0;
  ;
    break;}
case 131:
#line 1630 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ 
  ;
    break;}
case 132:
#line 1632 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  ;
    break;}
case 133:
#line 1634 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ 
  ;
    break;}
case 134:
#line 1638 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  const std::string &AsmSoFar = CurModule.CurrentModule->getModuleInlineAsm();
  char *EndStr = UnEscapeLexed(yyvsp[0].StrVal, true);
  std::string NewAsm(yyvsp[0].StrVal, EndStr);
  free(yyvsp[0].StrVal);

  if (AsmSoFar.empty())
    CurModule.CurrentModule->setModuleInlineAsm(NewAsm);
  else
    CurModule.CurrentModule->setModuleInlineAsm(AsmSoFar+"\n"+NewAsm);
;
    break;}
case 135:
#line 1650 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.Endianness = Module::BigEndian; ;
    break;}
case 136:
#line 1651 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.Endianness = Module::LittleEndian; ;
    break;}
case 137:
#line 1653 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    CurModule.CurrentModule->setEndianness(yyvsp[0].Endianness);
  ;
    break;}
case 138:
#line 1656 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[0].UInt64Val == 32)
      CurModule.CurrentModule->setPointerSize(Module::Pointer32);
    else if (yyvsp[0].UInt64Val == 64)
      CurModule.CurrentModule->setPointerSize(Module::Pointer64);
    else
      ThrowException("Invalid pointer size: '" + utostr(yyvsp[0].UInt64Val) + "'!");
  ;
    break;}
case 139:
#line 1664 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    CurModule.CurrentModule->setTargetTriple(yyvsp[0].StrVal);
    free(yyvsp[0].StrVal);
  ;
    break;}
case 141:
#line 1671 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
          CurModule.CurrentModule->addLibrary(yyvsp[0].StrVal);
          free(yyvsp[0].StrVal);
        ;
    break;}
case 142:
#line 1675 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
          CurModule.CurrentModule->addLibrary(yyvsp[0].StrVal);
          free(yyvsp[0].StrVal);
        ;
    break;}
case 143:
#line 1679 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
        ;
    break;}
case 147:
#line 1688 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.StrVal = 0; ;
    break;}
case 148:
#line 1690 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  if (*yyvsp[-1].TypeVal == Type::VoidTy)
    ThrowException("void typed arguments are invalid!");
  yyval.ArgVal = new std::pair<PATypeHolder*, char*>(yyvsp[-1].TypeVal, yyvsp[0].StrVal);
;
    break;}
case 149:
#line 1696 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ArgList = yyvsp[-2].ArgList;
    yyvsp[-2].ArgList->push_back(*yyvsp[0].ArgVal);
    delete yyvsp[0].ArgVal;
  ;
    break;}
case 150:
#line 1701 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ArgList = new std::vector<std::pair<PATypeHolder*,char*> >();
    yyval.ArgList->push_back(*yyvsp[0].ArgVal);
    delete yyvsp[0].ArgVal;
  ;
    break;}
case 151:
#line 1707 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ArgList = yyvsp[0].ArgList;
  ;
    break;}
case 152:
#line 1710 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ArgList = yyvsp[-2].ArgList;
    yyval.ArgList->push_back(std::pair<PATypeHolder*,
                            char*>(new PATypeHolder(Type::VoidTy), 0));
  ;
    break;}
case 153:
#line 1715 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ArgList = new std::vector<std::pair<PATypeHolder*,char*> >();
    yyval.ArgList->push_back(std::make_pair(new PATypeHolder(Type::VoidTy), (char*)0));
  ;
    break;}
case 154:
#line 1719 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ArgList = 0;
  ;
    break;}
case 155:
#line 1724 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  UnEscapeLexed(yyvsp[-5].StrVal);
  std::string FunctionName(yyvsp[-5].StrVal);
  free(yyvsp[-5].StrVal);  // Free strdup'd memory!
  
  if (!(*yyvsp[-6].TypeVal)->isFirstClassType() && *yyvsp[-6].TypeVal != Type::VoidTy)
    ThrowException("LLVM functions cannot return aggregate types!");

  std::vector<const Type*> ParamTypeList;
  if (yyvsp[-3].ArgList) {   // If there are arguments...
    for (std::vector<std::pair<PATypeHolder*,char*> >::iterator I = yyvsp[-3].ArgList->begin();
         I != yyvsp[-3].ArgList->end(); ++I)
      ParamTypeList.push_back(I->first->get());
  }

  bool isVarArg = ParamTypeList.size() && ParamTypeList.back() == Type::VoidTy;
  if (isVarArg) ParamTypeList.pop_back();

  const FunctionType *FT = FunctionType::get(*yyvsp[-6].TypeVal, ParamTypeList, isVarArg);
  const PointerType *PFT = PointerType::get(FT);
  delete yyvsp[-6].TypeVal;

  ValID ID;
  if (!FunctionName.empty()) {
    ID = ValID::create((char*)FunctionName.c_str());
  } else {
    ID = ValID::create((int)CurModule.Values[PFT].size());
  }

  Function *Fn = 0;
  // See if this function was forward referenced.  If so, recycle the object.
  if (GlobalValue *FWRef = CurModule.GetForwardRefForGlobal(PFT, ID)) {
    // Move the function to the end of the list, from whereever it was 
    // previously inserted.
    Fn = cast<Function>(FWRef);
    CurModule.CurrentModule->getFunctionList().remove(Fn);
    CurModule.CurrentModule->getFunctionList().push_back(Fn);
  } else if (!FunctionName.empty() &&     // Merge with an earlier prototype?
             (Fn = CurModule.CurrentModule->getFunction(FunctionName, FT))) {
    // If this is the case, either we need to be a forward decl, or it needs 
    // to be.
    if (!CurFun.isDeclare && !Fn->isExternal())
      ThrowException("Redefinition of function '" + FunctionName + "'!");
    
    // Make sure to strip off any argument names so we can't get conflicts.
    if (Fn->isExternal())
      for (Function::arg_iterator AI = Fn->arg_begin(), AE = Fn->arg_end();
           AI != AE; ++AI)
        AI->setName("");

  } else  {  // Not already defined?
    Fn = new Function(FT, GlobalValue::ExternalLinkage, FunctionName,
                      CurModule.CurrentModule);
    InsertValue(Fn, CurModule.Values);
  }

  CurFun.FunctionStart(Fn);
  Fn->setCallingConv(yyvsp[-7].UIntVal);
  Fn->setAlignment(yyvsp[0].UIntVal);
  if (yyvsp[-1].StrVal) {
    Fn->setSection(yyvsp[-1].StrVal);
    free(yyvsp[-1].StrVal);
  }

  // Add all of the arguments we parsed to the function...
  if (yyvsp[-3].ArgList) {                     // Is null if empty...
    if (isVarArg) {  // Nuke the last entry
      assert(yyvsp[-3].ArgList->back().first->get() == Type::VoidTy && yyvsp[-3].ArgList->back().second == 0&&
             "Not a varargs marker!");
      delete yyvsp[-3].ArgList->back().first;
      yyvsp[-3].ArgList->pop_back();  // Delete the last entry
    }
    Function::arg_iterator ArgIt = Fn->arg_begin();
    for (std::vector<std::pair<PATypeHolder*,char*> >::iterator I = yyvsp[-3].ArgList->begin();
         I != yyvsp[-3].ArgList->end(); ++I, ++ArgIt) {
      delete I->first;                          // Delete the typeholder...

      setValueName(ArgIt, I->second);           // Insert arg into symtab...
      InsertValue(ArgIt);
    }

    delete yyvsp[-3].ArgList;                     // We're now done with the argument list
  }
;
    break;}
case 158:
#line 1811 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  yyval.FunctionVal = CurFun.CurrentFunction;

  // Make sure that we keep track of the linkage type even if there was a
  // previous "declare".
  yyval.FunctionVal->setLinkage(yyvsp[-2].Linkage);
;
    break;}
case 161:
#line 1821 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  yyval.FunctionVal = yyvsp[-1].FunctionVal;
;
    break;}
case 162:
#line 1825 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ CurFun.isDeclare = true; ;
    break;}
case 163:
#line 1825 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  yyval.FunctionVal = CurFun.CurrentFunction;
  CurFun.FunctionDone();
;
    break;}
case 164:
#line 1834 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{    // A reference to a direct constant
    yyval.ValIDVal = ValID::create(yyvsp[0].SInt64Val);
  ;
    break;}
case 165:
#line 1837 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValIDVal = ValID::create(yyvsp[0].UInt64Val);
  ;
    break;}
case 166:
#line 1840 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                     // Perhaps it's an FP constant?
    yyval.ValIDVal = ValID::create(yyvsp[0].FPVal);
  ;
    break;}
case 167:
#line 1843 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValIDVal = ValID::create(ConstantBool::True);
  ;
    break;}
case 168:
#line 1846 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValIDVal = ValID::create(ConstantBool::False);
  ;
    break;}
case 169:
#line 1849 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValIDVal = ValID::createNull();
  ;
    break;}
case 170:
#line 1852 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValIDVal = ValID::createUndef();
  ;
    break;}
case 171:
#line 1855 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{     // A vector zero constant.
    yyval.ValIDVal = ValID::createZeroInit();
  ;
    break;}
case 172:
#line 1858 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ // Nonempty unsized packed vector
    const Type *ETy = (*yyvsp[-1].ConstVector)[0]->getType();
    int NumElements = yyvsp[-1].ConstVector->size(); 
    
    PackedType* pt = PackedType::get(ETy, NumElements);
    PATypeHolder* PTy = new PATypeHolder(
                                         HandleUpRefs(
                                            PackedType::get(
                                                ETy, 
                                                NumElements)
                                            )
                                         );
    
    // Verify all elements are correct type!
    for (unsigned i = 0; i < yyvsp[-1].ConstVector->size(); i++) {
      if (ETy != (*yyvsp[-1].ConstVector)[i]->getType())
        ThrowException("Element #" + utostr(i) + " is not of type '" + 
                     ETy->getDescription() +"' as required!\nIt is of type '" +
                     (*yyvsp[-1].ConstVector)[i]->getType()->getDescription() + "'.");
    }

    yyval.ValIDVal = ValID::create(ConstantPacked::get(pt, *yyvsp[-1].ConstVector));
    delete PTy; delete yyvsp[-1].ConstVector;
  ;
    break;}
case 173:
#line 1882 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValIDVal = ValID::create(yyvsp[0].ConstVal);
  ;
    break;}
case 174:
#line 1889 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{  // Is it an integer reference...?
    yyval.ValIDVal = ValID::create(yyvsp[0].SIntVal);
  ;
    break;}
case 175:
#line 1892 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                   // Is it a named reference...?
    yyval.ValIDVal = ValID::create(yyvsp[0].StrVal);
  ;
    break;}
case 178:
#line 1903 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValueVal = getVal(*yyvsp[-1].TypeVal, yyvsp[0].ValIDVal); delete yyvsp[-1].TypeVal;
  ;
    break;}
case 179:
#line 1907 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.FunctionVal = yyvsp[-1].FunctionVal;
  ;
    break;}
case 180:
#line 1910 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ // Do not allow functions with 0 basic blocks   
    yyval.FunctionVal = yyvsp[-1].FunctionVal;
  ;
    break;}
case 181:
#line 1918 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    setValueName(yyvsp[0].TermInstVal, yyvsp[-1].StrVal);
    InsertValue(yyvsp[0].TermInstVal);

    yyvsp[-2].BasicBlockVal->getInstList().push_back(yyvsp[0].TermInstVal);
    InsertValue(yyvsp[-2].BasicBlockVal);
    yyval.BasicBlockVal = yyvsp[-2].BasicBlockVal;
  ;
    break;}
case 182:
#line 1927 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyvsp[-1].BasicBlockVal->getInstList().push_back(yyvsp[0].InstVal);
    yyval.BasicBlockVal = yyvsp[-1].BasicBlockVal;
  ;
    break;}
case 183:
#line 1931 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.BasicBlockVal = CurBB = getBBVal(ValID::create((int)CurFun.NextBBNum++), true);

    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, yyval.BasicBlockVal);
  ;
    break;}
case 184:
#line 1941 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.BasicBlockVal = CurBB = getBBVal(ValID::create(yyvsp[0].StrVal), true);

    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, yyval.BasicBlockVal);
  ;
    break;}
case 185:
#line 1952 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{              // Return with a result...
    yyval.TermInstVal = new ReturnInst(yyvsp[0].ValueVal);
  ;
    break;}
case 186:
#line 1955 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                                       // Return with no result...
    yyval.TermInstVal = new ReturnInst();
  ;
    break;}
case 187:
#line 1958 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{                         // Unconditional Branch...
    yyval.TermInstVal = new BranchInst(getBBVal(yyvsp[0].ValIDVal));
  ;
    break;}
case 188:
#line 1961 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{  
    yyval.TermInstVal = new BranchInst(getBBVal(yyvsp[-3].ValIDVal), getBBVal(yyvsp[0].ValIDVal), getVal(Type::BoolTy, yyvsp[-6].ValIDVal));
  ;
    break;}
case 189:
#line 1964 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    SwitchInst *S = new SwitchInst(getVal(yyvsp[-7].PrimType, yyvsp[-6].ValIDVal), getBBVal(yyvsp[-3].ValIDVal), yyvsp[-1].JumpTable->size());
    yyval.TermInstVal = S;

    std::vector<std::pair<Constant*,BasicBlock*> >::iterator I = yyvsp[-1].JumpTable->begin(),
      E = yyvsp[-1].JumpTable->end();
    for (; I != E; ++I) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(I->first))
          S->addCase(CI, I->second);
      else
        ThrowException("Switch case is constant, but not a simple integer!");
    }
    delete yyvsp[-1].JumpTable;
  ;
    break;}
case 190:
#line 1978 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    SwitchInst *S = new SwitchInst(getVal(yyvsp[-6].PrimType, yyvsp[-5].ValIDVal), getBBVal(yyvsp[-2].ValIDVal), 0);
    yyval.TermInstVal = S;
  ;
    break;}
case 191:
#line 1983 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>(yyvsp[-10].TypeVal->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if (yyvsp[-7].ValueList) {
        for (std::vector<Value*>::iterator I = yyvsp[-7].ValueList->begin(), E = yyvsp[-7].ValueList->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      Ty = FunctionType::get(yyvsp[-10].TypeVal->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }

    Value *V = getVal(PFTy, yyvsp[-9].ValIDVal);   // Get the function we're calling...

    BasicBlock *Normal = getBBVal(yyvsp[-3].ValIDVal);
    BasicBlock *Except = getBBVal(yyvsp[0].ValIDVal);

    // Create the call node...
    if (!yyvsp[-7].ValueList) {                                   // Has no arguments?
      yyval.TermInstVal = new InvokeInst(V, Normal, Except, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      std::vector<Value*>::iterator ArgI = yyvsp[-7].ValueList->begin(), ArgE = yyvsp[-7].ValueList->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
        if ((*ArgI)->getType() != *I)
          ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
        ThrowException("Invalid number of parameters detected!");

      yyval.TermInstVal = new InvokeInst(V, Normal, Except, *yyvsp[-7].ValueList);
    }
    cast<InvokeInst>(yyval.TermInstVal)->setCallingConv(yyvsp[-11].UIntVal);
  
    delete yyvsp[-10].TypeVal;
    delete yyvsp[-7].ValueList;
  ;
    break;}
case 192:
#line 2035 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.TermInstVal = new UnwindInst();
  ;
    break;}
case 193:
#line 2038 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.TermInstVal = new UnreachableInst();
  ;
    break;}
case 194:
#line 2044 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.JumpTable = yyvsp[-5].JumpTable;
    Constant *V = cast<Constant>(getValNonImprovising(yyvsp[-4].PrimType, yyvsp[-3].ValIDVal));
    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    yyval.JumpTable->push_back(std::make_pair(V, getBBVal(yyvsp[0].ValIDVal)));
  ;
    break;}
case 195:
#line 2052 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.JumpTable = new std::vector<std::pair<Constant*, BasicBlock*> >();
    Constant *V = cast<Constant>(getValNonImprovising(yyvsp[-4].PrimType, yyvsp[-3].ValIDVal));

    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    yyval.JumpTable->push_back(std::make_pair(V, getBBVal(yyvsp[0].ValIDVal)));
  ;
    break;}
case 196:
#line 2062 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
  // Is this definition named?? if so, assign the name...
  setValueName(yyvsp[0].InstVal, yyvsp[-1].StrVal);
  InsertValue(yyvsp[0].InstVal);
  yyval.InstVal = yyvsp[0].InstVal;
;
    break;}
case 197:
#line 2069 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{    // Used for PHI nodes
    yyval.PHIList = new std::list<std::pair<Value*, BasicBlock*> >();
    yyval.PHIList->push_back(std::make_pair(getVal(*yyvsp[-5].TypeVal, yyvsp[-3].ValIDVal), getBBVal(yyvsp[-1].ValIDVal)));
    delete yyvsp[-5].TypeVal;
  ;
    break;}
case 198:
#line 2074 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.PHIList = yyvsp[-6].PHIList;
    yyvsp[-6].PHIList->push_back(std::make_pair(getVal(yyvsp[-6].PHIList->front().first->getType(), yyvsp[-3].ValIDVal),
                                 getBBVal(yyvsp[-1].ValIDVal)));
  ;
    break;}
case 199:
#line 2081 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{    // Used for call statements, and memory insts...
    yyval.ValueList = new std::vector<Value*>();
    yyval.ValueList->push_back(yyvsp[0].ValueVal);
  ;
    break;}
case 200:
#line 2085 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.ValueList = yyvsp[-2].ValueList;
    yyvsp[-2].ValueList->push_back(yyvsp[0].ValueVal);
  ;
    break;}
case 202:
#line 2091 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ yyval.ValueList = 0; ;
    break;}
case 203:
#line 2093 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.BoolVal = true;
  ;
    break;}
case 204:
#line 2096 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.BoolVal = false;
  ;
    break;}
case 205:
#line 2102 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!(*yyvsp[-3].TypeVal)->isInteger() && !(*yyvsp[-3].TypeVal)->isFloatingPoint() && 
        !isa<PackedType>((*yyvsp[-3].TypeVal).get()))
      ThrowException(
        "Arithmetic operator requires integer, FP, or packed operands!");
    if (isa<PackedType>((*yyvsp[-3].TypeVal).get()) && yyvsp[-4].BinaryOpVal == Instruction::Rem)
      ThrowException("Rem not supported on packed types!");
    yyval.InstVal = BinaryOperator::create(yyvsp[-4].BinaryOpVal, getVal(*yyvsp[-3].TypeVal, yyvsp[-2].ValIDVal), getVal(*yyvsp[-3].TypeVal, yyvsp[0].ValIDVal));
    if (yyval.InstVal == 0)
      ThrowException("binary operator returned null!");
    delete yyvsp[-3].TypeVal;
  ;
    break;}
case 206:
#line 2114 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!(*yyvsp[-3].TypeVal)->isIntegral()) {
      if (!isa<PackedType>(yyvsp[-3].TypeVal->get()) ||
          !cast<PackedType>(yyvsp[-3].TypeVal->get())->getElementType()->isIntegral())
        ThrowException("Logical operator requires integral operands!");
    }
    yyval.InstVal = BinaryOperator::create(yyvsp[-4].BinaryOpVal, getVal(*yyvsp[-3].TypeVal, yyvsp[-2].ValIDVal), getVal(*yyvsp[-3].TypeVal, yyvsp[0].ValIDVal));
    if (yyval.InstVal == 0)
      ThrowException("binary operator returned null!");
    delete yyvsp[-3].TypeVal;
  ;
    break;}
case 207:
#line 2125 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if(isa<PackedType>((*yyvsp[-3].TypeVal).get())) {
      ThrowException(
        "PackedTypes currently not supported in setcc instructions!");
    }
    yyval.InstVal = new SetCondInst(yyvsp[-4].BinaryOpVal, getVal(*yyvsp[-3].TypeVal, yyvsp[-2].ValIDVal), getVal(*yyvsp[-3].TypeVal, yyvsp[0].ValIDVal));
    if (yyval.InstVal == 0)
      ThrowException("binary operator returned null!");
    delete yyvsp[-3].TypeVal;
  ;
    break;}
case 208:
#line 2135 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    std::cerr << "WARNING: Use of eliminated 'not' instruction:"
              << " Replacing with 'xor'.\n";

    Value *Ones = ConstantIntegral::getAllOnesValue(yyvsp[0].ValueVal->getType());
    if (Ones == 0)
      ThrowException("Expected integral type for not instruction!");

    yyval.InstVal = BinaryOperator::create(Instruction::Xor, yyvsp[0].ValueVal, Ones);
    if (yyval.InstVal == 0)
      ThrowException("Could not create a xor instruction!");
  ;
    break;}
case 209:
#line 2147 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[0].ValueVal->getType() != Type::UByteTy)
      ThrowException("Shift amount must be ubyte!");
    if (!yyvsp[-2].ValueVal->getType()->isInteger())
      ThrowException("Shift constant expression requires integer operand!");
    yyval.InstVal = new ShiftInst(yyvsp[-3].OtherOpVal, yyvsp[-2].ValueVal, yyvsp[0].ValueVal);
  ;
    break;}
case 210:
#line 2154 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!yyvsp[0].TypeVal->get()->isFirstClassType())
      ThrowException("cast instruction to a non-primitive type: '" +
                     yyvsp[0].TypeVal->get()->getDescription() + "'!");
    yyval.InstVal = new CastInst(yyvsp[-2].ValueVal, *yyvsp[0].TypeVal);
    delete yyvsp[0].TypeVal;
  ;
    break;}
case 211:
#line 2161 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (yyvsp[-4].ValueVal->getType() != Type::BoolTy)
      ThrowException("select condition must be boolean!");
    if (yyvsp[-2].ValueVal->getType() != yyvsp[0].ValueVal->getType())
      ThrowException("select value types should match!");
    yyval.InstVal = new SelectInst(yyvsp[-4].ValueVal, yyvsp[-2].ValueVal, yyvsp[0].ValueVal);
  ;
    break;}
case 212:
#line 2168 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    NewVarArgs = true;
    yyval.InstVal = new VAArgInst(yyvsp[-2].ValueVal, *yyvsp[0].TypeVal);
    delete yyvsp[0].TypeVal;
  ;
    break;}
case 213:
#line 2173 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    ObsoleteVarArgs = true;
    const Type* ArgTy = yyvsp[-2].ValueVal->getType();
    Function* NF = CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0);

    //b = vaarg a, t -> 
    //foo = alloca 1 of t
    //bar = vacopy a 
    //store bar -> foo
    //b = vaarg foo, t
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vaarg.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, yyvsp[-2].ValueVal);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    yyval.InstVal = new VAArgInst(foo, *yyvsp[0].TypeVal);
    delete yyvsp[0].TypeVal;
  ;
    break;}
case 214:
#line 2192 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    ObsoleteVarArgs = true;
    const Type* ArgTy = yyvsp[-2].ValueVal->getType();
    Function* NF = CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0);

    //b = vanext a, t ->
    //foo = alloca 1 of t
    //bar = vacopy a
    //store bar -> foo
    //tmp = vaarg foo, t
    //b = load foo
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vanext.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, yyvsp[-2].ValueVal);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    Instruction* tmp = new VAArgInst(foo, *yyvsp[0].TypeVal);
    CurBB->getInstList().push_back(tmp);
    yyval.InstVal = new LoadInst(foo);
    delete yyvsp[0].TypeVal;
  ;
    break;}
case 215:
#line 2214 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!isa<PackedType>(yyvsp[-2].ValueVal->getType()))
      ThrowException("First operand of extractelement must be "
                     "packed type!");
    if (yyvsp[0].ValueVal->getType() != Type::UIntTy)
      ThrowException("Second operand of extractelement must be uint!");
    yyval.InstVal = new ExtractElementInst(yyvsp[-2].ValueVal, yyvsp[0].ValueVal);
  ;
    break;}
case 216:
#line 2222 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!isa<PackedType>(yyvsp[-4].ValueVal->getType()))
      ThrowException("First operand of insertelement must be "
                     "packed type!");
    if (yyvsp[-2].ValueVal->getType() != 
        cast<PackedType>(yyvsp[-4].ValueVal->getType())->getElementType())
      ThrowException("Second operand of insertelement must be "
                     "packed element type!");
    if (yyvsp[0].ValueVal->getType() != Type::UIntTy)
      ThrowException("Third operand of insertelement must be uint!");
    yyval.InstVal = new InsertElementInst(yyvsp[-4].ValueVal, yyvsp[-2].ValueVal, yyvsp[0].ValueVal);
  ;
    break;}
case 217:
#line 2234 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const Type *Ty = yyvsp[0].PHIList->front().first->getType();
    if (!Ty->isFirstClassType())
      ThrowException("PHI node operands must be of first class type!");
    yyval.InstVal = new PHINode(Ty);
    ((PHINode*)yyval.InstVal)->reserveOperandSpace(yyvsp[0].PHIList->size());
    while (yyvsp[0].PHIList->begin() != yyvsp[0].PHIList->end()) {
      if (yyvsp[0].PHIList->front().first->getType() != Ty) 
        ThrowException("All elements of a PHI node must be of the same type!");
      cast<PHINode>(yyval.InstVal)->addIncoming(yyvsp[0].PHIList->front().first, yyvsp[0].PHIList->front().second);
      yyvsp[0].PHIList->pop_front();
    }
    delete yyvsp[0].PHIList;  // Free the list...
  ;
    break;}
case 218:
#line 2248 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>(yyvsp[-4].TypeVal->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if (yyvsp[-1].ValueList) {
        for (std::vector<Value*>::iterator I = yyvsp[-1].ValueList->begin(), E = yyvsp[-1].ValueList->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      if (!(*yyvsp[-4].TypeVal)->isFirstClassType() && *yyvsp[-4].TypeVal != Type::VoidTy)
        ThrowException("LLVM functions cannot return aggregate types!");

      Ty = FunctionType::get(yyvsp[-4].TypeVal->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }

    Value *V = getVal(PFTy, yyvsp[-3].ValIDVal);   // Get the function we're calling...

    // Create the call node...
    if (!yyvsp[-1].ValueList) {                                   // Has no arguments?
      // Make sure no arguments is a good thing!
      if (Ty->getNumParams() != 0)
        ThrowException("No arguments passed to a function that "
                       "expects arguments!");

      yyval.InstVal = new CallInst(V, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      std::vector<Value*>::iterator ArgI = yyvsp[-1].ValueList->begin(), ArgE = yyvsp[-1].ValueList->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
        if ((*ArgI)->getType() != *I)
          ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
        ThrowException("Invalid number of parameters detected!");

      yyval.InstVal = new CallInst(V, *yyvsp[-1].ValueList);
    }
    cast<CallInst>(yyval.InstVal)->setTailCall(yyvsp[-6].BoolVal);
    cast<CallInst>(yyval.InstVal)->setCallingConv(yyvsp[-5].UIntVal);
    delete yyvsp[-4].TypeVal;
    delete yyvsp[-1].ValueList;
  ;
    break;}
case 219:
#line 2305 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.InstVal = yyvsp[0].InstVal;
  ;
    break;}
case 220:
#line 2311 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ 
    yyval.ValueList = yyvsp[0].ValueList; 
  ;
    break;}
case 221:
#line 2313 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{ 
    yyval.ValueList = new std::vector<Value*>(); 
  ;
    break;}
case 222:
#line 2317 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.BoolVal = true;
  ;
    break;}
case 223:
#line 2320 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.BoolVal = false;
  ;
    break;}
case 224:
#line 2326 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.InstVal = new MallocInst(*yyvsp[-1].TypeVal, 0, yyvsp[0].UIntVal);
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 225:
#line 2330 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.InstVal = new MallocInst(*yyvsp[-4].TypeVal, getVal(yyvsp[-2].PrimType, yyvsp[-1].ValIDVal), yyvsp[0].UIntVal);
    delete yyvsp[-4].TypeVal;
  ;
    break;}
case 226:
#line 2334 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.InstVal = new AllocaInst(*yyvsp[-1].TypeVal, 0, yyvsp[0].UIntVal);
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 227:
#line 2338 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    yyval.InstVal = new AllocaInst(*yyvsp[-4].TypeVal, getVal(yyvsp[-2].PrimType, yyvsp[-1].ValIDVal), yyvsp[0].UIntVal);
    delete yyvsp[-4].TypeVal;
  ;
    break;}
case 228:
#line 2342 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!isa<PointerType>(yyvsp[0].ValueVal->getType()))
      ThrowException("Trying to free nonpointer type " + 
                     yyvsp[0].ValueVal->getType()->getDescription() + "!");
    yyval.InstVal = new FreeInst(yyvsp[0].ValueVal);
  ;
    break;}
case 229:
#line 2349 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!isa<PointerType>(yyvsp[-1].TypeVal->get()))
      ThrowException("Can't load from nonpointer type: " +
                     (*yyvsp[-1].TypeVal)->getDescription());
    if (!cast<PointerType>(yyvsp[-1].TypeVal->get())->getElementType()->isFirstClassType())
      ThrowException("Can't load from pointer of non-first-class type: " +
                     (*yyvsp[-1].TypeVal)->getDescription());
    yyval.InstVal = new LoadInst(getVal(*yyvsp[-1].TypeVal, yyvsp[0].ValIDVal), "", yyvsp[-3].BoolVal);
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 230:
#line 2359 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    const PointerType *PT = dyn_cast<PointerType>(yyvsp[-1].TypeVal->get());
    if (!PT)
      ThrowException("Can't store to a nonpointer type: " +
                     (*yyvsp[-1].TypeVal)->getDescription());
    const Type *ElTy = PT->getElementType();
    if (ElTy != yyvsp[-3].ValueVal->getType())
      ThrowException("Can't store '" + yyvsp[-3].ValueVal->getType()->getDescription() +
                     "' into space of type '" + ElTy->getDescription() + "'!");

    yyval.InstVal = new StoreInst(yyvsp[-3].ValueVal, getVal(*yyvsp[-1].TypeVal, yyvsp[0].ValIDVal), yyvsp[-5].BoolVal);
    delete yyvsp[-1].TypeVal;
  ;
    break;}
case 231:
#line 2372 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"
{
    if (!isa<PointerType>(yyvsp[-2].TypeVal->get()))
      ThrowException("getelementptr insn requires pointer operand!");

    // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte struct
    // indices to uint struct indices for compatibility.
    generic_gep_type_iterator<std::vector<Value*>::iterator>
      GTI = gep_type_begin(yyvsp[-2].TypeVal->get(), yyvsp[0].ValueList->begin(), yyvsp[0].ValueList->end()),
      GTE = gep_type_end(yyvsp[-2].TypeVal->get(), yyvsp[0].ValueList->begin(), yyvsp[0].ValueList->end());
    for (unsigned i = 0, e = yyvsp[0].ValueList->size(); i != e && GTI != GTE; ++i, ++GTI)
      if (isa<StructType>(*GTI))        // Only change struct indices
        if (ConstantUInt *CUI = dyn_cast<ConstantUInt>((*yyvsp[0].ValueList)[i]))
          if (CUI->getType() == Type::UByteTy)
            (*yyvsp[0].ValueList)[i] = ConstantExpr::getCast(CUI, Type::UIntTy);

    if (!GetElementPtrInst::getIndexedType(*yyvsp[-2].TypeVal, *yyvsp[0].ValueList, true))
      ThrowException("Invalid getelementptr indices for type '" +
                     (*yyvsp[-2].TypeVal)->getDescription()+ "'!");
    yyval.InstVal = new GetElementPtrInst(getVal(*yyvsp[-2].TypeVal, yyvsp[-1].ValIDVal), *yyvsp[0].ValueList);
    delete yyvsp[-2].TypeVal; delete yyvsp[0].ValueList;
  ;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 543 "/usr/share/bison.simple"

  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YYLSP_NEEDED
  yylsp -= yylen;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

  *++yyvsp = yyval;

#ifdef YYLSP_NEEDED
  yylsp++;
  if (yylen == 0)
    {
      yylsp->first_line = yylloc.first_line;
      yylsp->first_column = yylloc.first_column;
      yylsp->last_line = (yylsp-1)->last_line;
      yylsp->last_column = (yylsp-1)->last_column;
      yylsp->text = 0;
    }
  else
    {
      yylsp->last_line = (yylsp+yylen-1)->last_line;
      yylsp->last_column = (yylsp+yylen-1)->last_column;
    }
#endif

  /* Now "shift" the result of the reduction.
     Determine what state that goes to,
     based on the state we popped back to
     and the rule number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTBASE] + *yyssp;
  if (yystate >= 0 && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTBASE];

  goto yynewstate;

yyerrlab:   /* here on detecting error */

  if (! yyerrstatus)
    /* If not already recovering from an error, report this error.  */
    {
      ++yynerrs;

#ifdef YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (yyn > YYFLAG && yyn < YYLAST)
	{
	  int size = 0;
	  char *msg;
	  int x, count;

	  count = 0;
	  /* Start X at -yyn if nec to avoid negative indexes in yycheck.  */
	  for (x = (yyn < 0 ? -yyn : 0);
	       x < (sizeof(yytname) / sizeof(char *)); x++)
	    if (yycheck[x + yyn] == x)
	      size += strlen(yytname[x]) + 15, count++;
	  msg = (char *) malloc(size + 15);
	  if (msg != 0)
	    {
	      strcpy(msg, "parse error");

	      if (count < 5)
		{
		  count = 0;
		  for (x = (yyn < 0 ? -yyn : 0);
		       x < (sizeof(yytname) / sizeof(char *)); x++)
		    if (yycheck[x + yyn] == x)
		      {
			strcat(msg, count == 0 ? ", expecting `" : " or `");
			strcat(msg, yytname[x]);
			strcat(msg, "'");
			count++;
		      }
		}
	      yyerror(msg);
	      free(msg);
	    }
	  else
	    yyerror ("parse error; also virtual memory exceeded");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror("parse error");
    }

  goto yyerrlab1;
yyerrlab1:   /* here on error raised explicitly by an action */

  if (yyerrstatus == 3)
    {
      /* if just tried and failed to reuse lookahead token after an error, discard it.  */

      /* return failure if at end of input */
      if (yychar == YYEOF)
	YYABORT;

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Discarding token %d (%s).\n", yychar, yytname[yychar1]);
#endif

      yychar = YYEMPTY;
    }

  /* Else will try to reuse lookahead token
     after shifting the error token.  */

  yyerrstatus = 3;		/* Each real token shifted decrements this */

  goto yyerrhandle;

yyerrdefault:  /* current state does not do anything special for the error token. */

#if 0
  /* This is wrong; only states that explicitly want error tokens
     should shift them.  */
  yyn = yydefact[yystate];  /* If its default is to accept any token, ok.  Otherwise pop it.*/
  if (yyn) goto yydefault;
#endif

yyerrpop:   /* pop the current state because it cannot handle the error token */

  if (yyssp == yyss) YYABORT;
  yyvsp--;
  yystate = *--yyssp;
#ifdef YYLSP_NEEDED
  yylsp--;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "Error: state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

yyerrhandle:

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yyerrdefault;

  yyn += YYTERROR;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != YYTERROR)
    goto yyerrdefault;

  yyn = yytable[yyn];
  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrpop;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrpop;

  if (yyn == YYFINAL)
    YYACCEPT;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  yystate = yyn;
  goto yynewstate;

 yyacceptlab:
  /* YYACCEPT comes here.  */
  if (yyfree_stacks)
    {
      free (yyss);
      free (yyvs);
#ifdef YYLSP_NEEDED
      free (yyls);
#endif
    }
  return 0;

 yyabortlab:
  /* YYABORT comes here.  */
  if (yyfree_stacks)
    {
      free (yyss);
      free (yyvs);
#ifdef YYLSP_NEEDED
      free (yyls);
#endif
    }
  return 1;
}
#line 2395 "/Users/sabre/cvs/llvm/lib/AsmParser/llvmAsmParser.y"

int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + utostr((unsigned) llvmAsmlineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(llvmAsmtext, llvmAsmleng) + "'";
  ThrowException(errMsg);
  return 0;
}
