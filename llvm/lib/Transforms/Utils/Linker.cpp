//===- Linker.cpp - Module Linker Implementation --------------------------===//
//
// This file implements the LLVM module linker.
//
// Specifically, this:
//  * Merges global variables between the two modules
//    * Uninit + Uninit = Init, Init + Uninit = Init, Init + Init = Error if !=
//  * Merges functions between two modules
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Linker.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/Constants.h"

// Error - Simple wrapper function to conditionally assign to E and return true.
// This just makes error return conditions a little bit simpler...
//
static inline bool Error(std::string *E, const std::string &Message) {
  if (E) *E = Message;
  return true;
}

// ResolveTypes - Attempt to link the two specified types together.  Return true
// if there is an error and they cannot yet be linked.
//
static bool ResolveTypes(const Type *DestTy, const Type *SrcTy,
                         SymbolTable *DestST, const std::string &Name) {
  if (DestTy == SrcTy) return false;       // If already equal, noop

  // Does the type already exist in the module?
  if (DestTy && !isa<OpaqueType>(DestTy)) {  // Yup, the type already exists...
    if (const OpaqueType *OT = dyn_cast<OpaqueType>(SrcTy)) {
      const_cast<OpaqueType*>(OT)->refineAbstractTypeTo(DestTy);
    } else {
      return true;  // Cannot link types... neither is opaque and not-equal
    }
  } else {                       // Type not in dest module.  Add it now.
    if (DestTy)                  // Type _is_ in module, just opaque...
      const_cast<OpaqueType*>(cast<OpaqueType>(DestTy))
                           ->refineAbstractTypeTo(SrcTy);
    else if (!Name.empty())
      DestST->insert(Name, const_cast<Type*>(SrcTy));
  }
  return false;
}

static const FunctionType *getFT(const PATypeHolder &TH) {
  return cast<FunctionType>(TH.get());
}
static const StructType *getST(const PATypeHolder &TH) {
  return cast<StructType>(TH.get());
}

// RecursiveResolveTypes - This is just like ResolveTypes, except that it
// recurses down into derived types, merging the used types if the parent types
// are compatible.
//
static bool RecursiveResolveTypesI(const PATypeHolder &DestTy,
                                   const PATypeHolder &SrcTy,
                                   SymbolTable *DestST, const std::string &Name,
                std::vector<std::pair<PATypeHolder, PATypeHolder> > &Pointers) {
  const Type *SrcTyT = SrcTy.get();
  const Type *DestTyT = DestTy.get();
  if (DestTyT == SrcTyT) return false;       // If already equal, noop
  
  // If we found our opaque type, resolve it now!
  if (isa<OpaqueType>(DestTyT) || isa<OpaqueType>(SrcTyT))
    return ResolveTypes(DestTyT, SrcTyT, DestST, Name);
  
  // Two types cannot be resolved together if they are of different primitive
  // type.  For example, we cannot resolve an int to a float.
  if (DestTyT->getPrimitiveID() != SrcTyT->getPrimitiveID()) return true;

  // Otherwise, resolve the used type used by this derived type...
  switch (DestTyT->getPrimitiveID()) {
  case Type::FunctionTyID: {
    if (cast<FunctionType>(DestTyT)->isVarArg() !=
        cast<FunctionType>(SrcTyT)->isVarArg())
      return true;
    for (unsigned i = 0, e = getFT(DestTy)->getNumContainedTypes(); i != e; ++i)
      if (RecursiveResolveTypesI(getFT(DestTy)->getContainedType(i),
                                 getFT(SrcTy)->getContainedType(i), DestST, "",
                                 Pointers))
        return true;
    return false;
  }
  case Type::StructTyID: {
    if (getST(DestTy)->getNumContainedTypes() != 
        getST(SrcTy)->getNumContainedTypes()) return 1;
    for (unsigned i = 0, e = getST(DestTy)->getNumContainedTypes(); i != e; ++i)
      if (RecursiveResolveTypesI(getST(DestTy)->getContainedType(i),
                                 getST(SrcTy)->getContainedType(i), DestST, "",
                                 Pointers))
        return true;
    return false;
  }
  case Type::ArrayTyID: {
    const ArrayType *DAT = cast<ArrayType>(DestTy.get());
    const ArrayType *SAT = cast<ArrayType>(SrcTy.get());
    if (DAT->getNumElements() != SAT->getNumElements()) return true;
    return RecursiveResolveTypesI(DAT->getElementType(), SAT->getElementType(),
                                  DestST, "", Pointers);
  }
  case Type::PointerTyID: {
    // If this is a pointer type, check to see if we have already seen it.  If
    // so, we are in a recursive branch.  Cut off the search now.  We cannot use
    // an associative container for this search, because the type pointers (keys
    // in the container) change whenever types get resolved...
    //
    for (unsigned i = 0, e = Pointers.size(); i != e; ++i)
      if (Pointers[i].first == DestTy)
        return Pointers[i].second != SrcTy;

    // Otherwise, add the current pointers to the vector to stop recursion on
    // this pair.
    Pointers.push_back(std::make_pair(DestTyT, SrcTyT));
    bool Result =
      RecursiveResolveTypesI(cast<PointerType>(DestTy.get())->getElementType(),
                             cast<PointerType>(SrcTy.get())->getElementType(),
                             DestST, "", Pointers);
    Pointers.pop_back();
    return Result;
  }
  default: assert(0 && "Unexpected type!"); return true;
  }  
}

static bool RecursiveResolveTypes(const PATypeHolder &DestTy,
                                  const PATypeHolder &SrcTy,
                                  SymbolTable *DestST, const std::string &Name){
  std::vector<std::pair<PATypeHolder, PATypeHolder> > PointerTypes;
  return RecursiveResolveTypesI(DestTy, SrcTy, DestST, Name, PointerTypes);
}


// LinkTypes - Go through the symbol table of the Src module and see if any
// types are named in the src module that are not named in the Dst module.
// Make sure there are no type name conflicts.
//
static bool LinkTypes(Module *Dest, const Module *Src, std::string *Err) {
  SymbolTable       *DestST = &Dest->getSymbolTable();
  const SymbolTable *SrcST  = &Src->getSymbolTable();

  // Look for a type plane for Type's...
  SymbolTable::const_iterator PI = SrcST->find(Type::TypeTy);
  if (PI == SrcST->end()) return false;  // No named types, do nothing.

  // Some types cannot be resolved immediately becuse they depend on other types
  // being resolved to each other first.  This contains a list of types we are
  // waiting to recheck.
  std::vector<std::string> DelayedTypesToResolve;

  const SymbolTable::VarMap &VM = PI->second;
  for (SymbolTable::type_const_iterator I = VM.begin(), E = VM.end();
       I != E; ++I) {
    const std::string &Name = I->first;
    Type *RHS = cast<Type>(I->second);

    // Check to see if this type name is already in the dest module...
    Type *Entry = cast_or_null<Type>(DestST->lookup(Type::TypeTy, Name));

    if (ResolveTypes(Entry, RHS, DestST, Name)) {
      // They look different, save the types 'till later to resolve.
      DelayedTypesToResolve.push_back(Name);
    }
  }

  // Iteratively resolve types while we can...
  while (!DelayedTypesToResolve.empty()) {
    // Loop over all of the types, attempting to resolve them if possible...
    unsigned OldSize = DelayedTypesToResolve.size();

    // Try direct resolution by name...
    for (unsigned i = 0; i != DelayedTypesToResolve.size(); ++i) {
      const std::string &Name = DelayedTypesToResolve[i];
      Type *T1 = cast<Type>(VM.find(Name)->second);
      Type *T2 = cast<Type>(DestST->lookup(Type::TypeTy, Name));
      if (!ResolveTypes(T2, T1, DestST, Name)) {
        // We are making progress!
        DelayedTypesToResolve.erase(DelayedTypesToResolve.begin()+i);
        --i;
      }
    }

    // Did we not eliminate any types?
    if (DelayedTypesToResolve.size() == OldSize) {
      // Attempt to resolve subelements of types.  This allows us to merge these
      // two types: { int* } and { opaque* }
      for (unsigned i = 0, e = DelayedTypesToResolve.size(); i != e; ++i) {
        const std::string &Name = DelayedTypesToResolve[i];
        PATypeHolder T1(cast<Type>(VM.find(Name)->second));
        PATypeHolder T2(cast<Type>(DestST->lookup(Type::TypeTy, Name)));

        if (!RecursiveResolveTypes(T2, T1, DestST, Name)) {
          // We are making progress!
          DelayedTypesToResolve.erase(DelayedTypesToResolve.begin()+i);
          
          // Go back to the main loop, perhaps we can resolve directly by name
          // now...
          break;
        }
      }

      // If we STILL cannot resolve the types, then there is something wrong.
      // Report the error.
      if (DelayedTypesToResolve.size() == OldSize) {
        // Build up an error message of all of the mismatched types.
        std::string ErrorMessage;
        for (unsigned i = 0, e = DelayedTypesToResolve.size(); i != e; ++i) {
          const std::string &Name = DelayedTypesToResolve[i];
          const Type *T1 = cast<Type>(VM.find(Name)->second);
          const Type *T2 = cast<Type>(DestST->lookup(Type::TypeTy, Name));
          ErrorMessage += "  Type named '" + Name + 
                          "' conflicts.\n    Src='" + T1->getDescription() +
                          "'.\n   Dest='" + T2->getDescription() + "'\n";
        }
        return Error(Err, "Type conflict between types in modules:\n" +
                     ErrorMessage);
      }
    }
  }


  return false;
}

static void PrintMap(const std::map<const Value*, Value*> &M) {
  for (std::map<const Value*, Value*>::const_iterator I = M.begin(), E =M.end();
       I != E; ++I) {
    std::cerr << " Fr: " << (void*)I->first << " ";
    I->first->dump();
    std::cerr << " To: " << (void*)I->second << " ";
    I->second->dump();
    std::cerr << "\n";
  }
}


// RemapOperand - Use LocalMap and GlobalMap to convert references from one
// module to another.  This is somewhat sophisticated in that it can
// automatically handle constant references correctly as well...
//
static Value *RemapOperand(const Value *In,
                           std::map<const Value*, Value*> &LocalMap,
                           std::map<const Value*, Value*> *GlobalMap) {
  std::map<const Value*,Value*>::const_iterator I = LocalMap.find(In);
  if (I != LocalMap.end()) return I->second;

  if (GlobalMap) {
    I = GlobalMap->find(In);
    if (I != GlobalMap->end()) return I->second;
  }

  // Check to see if it's a constant that we are interesting in transforming...
  if (const Constant *CPV = dyn_cast<Constant>(In)) {
    if (!isa<DerivedType>(CPV->getType()) && !isa<ConstantExpr>(CPV))
      return const_cast<Constant*>(CPV);   // Simple constants stay identical...

    Constant *Result = 0;

    if (const ConstantArray *CPA = dyn_cast<ConstantArray>(CPV)) {
      const std::vector<Use> &Ops = CPA->getValues();
      std::vector<Constant*> Operands(Ops.size());
      for (unsigned i = 0, e = Ops.size(); i != e; ++i)
        Operands[i] = 
          cast<Constant>(RemapOperand(Ops[i], LocalMap, GlobalMap));
      Result = ConstantArray::get(cast<ArrayType>(CPA->getType()), Operands);
    } else if (const ConstantStruct *CPS = dyn_cast<ConstantStruct>(CPV)) {
      const std::vector<Use> &Ops = CPS->getValues();
      std::vector<Constant*> Operands(Ops.size());
      for (unsigned i = 0; i < Ops.size(); ++i)
        Operands[i] = 
          cast<Constant>(RemapOperand(Ops[i], LocalMap, GlobalMap));
      Result = ConstantStruct::get(cast<StructType>(CPS->getType()), Operands);
    } else if (isa<ConstantPointerNull>(CPV)) {
      Result = const_cast<Constant*>(CPV);
    } else if (const ConstantPointerRef *CPR =
                      dyn_cast<ConstantPointerRef>(CPV)) {
      Value *V = RemapOperand(CPR->getValue(), LocalMap, GlobalMap);
      Result = ConstantPointerRef::get(cast<GlobalValue>(V));
    } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        Value *Ptr = RemapOperand(CE->getOperand(0), LocalMap, GlobalMap);
        std::vector<Constant*> Indices;
        Indices.reserve(CE->getNumOperands()-1);
        for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
          Indices.push_back(cast<Constant>(RemapOperand(CE->getOperand(i),
                                                        LocalMap, GlobalMap)));

        Result = ConstantExpr::getGetElementPtr(cast<Constant>(Ptr), Indices);
      } else if (CE->getNumOperands() == 1) {
        // Cast instruction
        assert(CE->getOpcode() == Instruction::Cast);
        Value *V = RemapOperand(CE->getOperand(0), LocalMap, GlobalMap);
        Result = ConstantExpr::getCast(cast<Constant>(V), CE->getType());
      } else if (CE->getNumOperands() == 2) {
        // Binary operator...
        Value *V1 = RemapOperand(CE->getOperand(0), LocalMap, GlobalMap);
        Value *V2 = RemapOperand(CE->getOperand(1), LocalMap, GlobalMap);

        Result = ConstantExpr::get(CE->getOpcode(), cast<Constant>(V1),
                                   cast<Constant>(V2));        
      } else {
        assert(0 && "Unknown constant expr type!");
      }

    } else {
      assert(0 && "Unknown type of derived type constant value!");
    }

    // Cache the mapping in our local map structure...
    if (GlobalMap)
      GlobalMap->insert(std::make_pair(In, Result));
    else
      LocalMap.insert(std::make_pair(In, Result));
    return Result;
  }

  std::cerr << "XXX LocalMap: \n";
  PrintMap(LocalMap);

  if (GlobalMap) {
    std::cerr << "XXX GlobalMap: \n";
    PrintMap(*GlobalMap);
  }

  std::cerr << "Couldn't remap value: " << (void*)In << " " << *In << "\n";
  assert(0 && "Couldn't remap value!");
  return 0;
}

/// FindGlobalNamed - Look in the specified symbol table for a global with the
/// specified name and type.  If an exactly matching global does not exist, see
/// if there is a global which is "type compatible" with the specified
/// name/type.  This allows us to resolve things like '%x = global int*' with
/// '%x = global opaque*'.
///
static GlobalValue *FindGlobalNamed(const std::string &Name, const Type *Ty,
                                    SymbolTable *ST) {
  // See if an exact match exists in the symbol table...
  if (Value *V = ST->lookup(Ty, Name)) return cast<GlobalValue>(V);
  
  // It doesn't exist exactly, scan through all of the type planes in the symbol
  // table, checking each of them for a type-compatible version.
  //
  for (SymbolTable::iterator I = ST->begin(), E = ST->end(); I != E; ++I)
    if (I->first->getType() != Type::TypeTy) {
      SymbolTable::VarMap &VM = I->second;
      // Does this type plane contain an entry with the specified name?
      SymbolTable::type_iterator TI = VM.find(Name);
      if (TI != VM.end()) {
        // Determine whether we can fold the two types together, resolving them.
        // If so, we can use this value.
        if (!RecursiveResolveTypes(Ty, I->first, ST, ""))
          return cast<GlobalValue>(TI->second);
      }
    }
  return 0;  // Otherwise, nothing could be found.
}


// LinkGlobals - Loop through the global variables in the src module and merge
// them into the dest module.
//
static bool LinkGlobals(Module *Dest, const Module *Src,
                        std::map<const Value*, Value*> &ValueMap,
                    std::multimap<std::string, GlobalVariable *> &AppendingVars,
                        std::string *Err) {
  // We will need a module level symbol table if the src module has a module
  // level symbol table...
  SymbolTable *ST = (SymbolTable*)&Dest->getSymbolTable();
  
  // Loop over all of the globals in the src module, mapping them over as we go
  //
  for (Module::const_giterator I = Src->gbegin(), E = Src->gend(); I != E; ++I){
    const GlobalVariable *SGV = I;
    GlobalVariable *DGV = 0;
    if (SGV->hasName()) {
      // A same named thing is a global variable, because the only two things
      // that may be in a module level symbol table are Global Vars and
      // Functions, and they both have distinct, nonoverlapping, possible types.
      // 
      DGV = cast_or_null<GlobalVariable>(FindGlobalNamed(SGV->getName(), 
                                                         SGV->getType(), ST));
    }

    assert(SGV->hasInitializer() || SGV->hasExternalLinkage() &&
           "Global must either be external or have an initializer!");

    bool SGExtern = SGV->isExternal();
    bool DGExtern = DGV ? DGV->isExternal() : false;

    if (!DGV || DGV->hasInternalLinkage() || SGV->hasInternalLinkage()) {
      // No linking to be performed, simply create an identical version of the
      // symbol over in the dest module... the initializer will be filled in
      // later by LinkGlobalInits...
      //
      GlobalVariable *NewDGV =
        new GlobalVariable(SGV->getType()->getElementType(),
                           SGV->isConstant(), SGV->getLinkage(), /*init*/0,
                           SGV->getName(), Dest);

      // If the LLVM runtime renamed the global, but it is an externally visible
      // symbol, DGV must be an existing global with internal linkage.  Rename
      // it.
      if (NewDGV->getName() != SGV->getName() && !NewDGV->hasInternalLinkage()){
        assert(DGV && DGV->getName() == SGV->getName() &&
               DGV->hasInternalLinkage());
        DGV->setName("");
        NewDGV->setName(SGV->getName());  // Force the name back
        DGV->setName(SGV->getName());     // This will cause a renaming
        assert(NewDGV->getName() == SGV->getName() &&
               DGV->getName() != SGV->getName());
      }

      // Make sure to remember this mapping...
      ValueMap.insert(std::make_pair(SGV, NewDGV));
      if (SGV->hasAppendingLinkage())
        // Keep track that this is an appending variable...
        AppendingVars.insert(std::make_pair(SGV->getName(), NewDGV));

    } else if (SGV->isExternal()) {
      // If SGV is external or if both SGV & DGV are external..  Just link the
      // external globals, we aren't adding anything.
      ValueMap.insert(std::make_pair(SGV, DGV));

    } else if (DGV->isExternal()) {   // If DGV is external but SGV is not...
      ValueMap.insert(std::make_pair(SGV, DGV));
      DGV->setLinkage(SGV->getLinkage());    // Inherit linkage!
    } else if (SGV->getLinkage() != DGV->getLinkage()) {
      return Error(Err, "Global variables named '" + SGV->getName() +
                   "' have different linkage specifiers!");
    } else if (SGV->hasExternalLinkage()) {
      // Allow linking two exactly identical external global variables...
      if (SGV->isConstant() != DGV->isConstant() ||
          SGV->getInitializer() != DGV->getInitializer())
        return Error(Err, "Global Variable Collision on '" + 
                     SGV->getType()->getDescription() + " %" + SGV->getName() +
                     "' - Global variables differ in const'ness");
      ValueMap.insert(std::make_pair(SGV, DGV));
    } else if (SGV->hasLinkOnceLinkage()) {
      // If the global variable has a name, and that name is already in use in
      // the Dest module, make sure that the name is a compatible global
      // variable...
      //
      // Check to see if the two GV's have the same Const'ness...
      if (SGV->isConstant() != DGV->isConstant())
        return Error(Err, "Global Variable Collision on '" + 
                     SGV->getType()->getDescription() + " %" + SGV->getName() +
                     "' - Global variables differ in const'ness");

      // Okay, everything is cool, remember the mapping...
      ValueMap.insert(std::make_pair(SGV, DGV));
    } else if (SGV->hasAppendingLinkage()) {
      // No linking is performed yet.  Just insert a new copy of the global, and
      // keep track of the fact that it is an appending variable in the
      // AppendingVars map.  The name is cleared out so that no linkage is
      // performed.
      GlobalVariable *NewDGV =
        new GlobalVariable(SGV->getType()->getElementType(),
                           SGV->isConstant(), SGV->getLinkage(), /*init*/0,
                           "", Dest);

      // Make sure to remember this mapping...
      ValueMap.insert(std::make_pair(SGV, NewDGV));

      // Keep track that this is an appending variable...
      AppendingVars.insert(std::make_pair(SGV->getName(), NewDGV));
    } else {
      assert(0 && "Unknown linkage!");
    }
  }
  return false;
}


// LinkGlobalInits - Update the initializers in the Dest module now that all
// globals that may be referenced are in Dest.
//
static bool LinkGlobalInits(Module *Dest, const Module *Src,
                            std::map<const Value*, Value*> &ValueMap,
                            std::string *Err) {

  // Loop over all of the globals in the src module, mapping them over as we go
  //
  for (Module::const_giterator I = Src->gbegin(), E = Src->gend(); I != E; ++I){
    const GlobalVariable *SGV = I;

    if (SGV->hasInitializer()) {      // Only process initialized GV's
      // Figure out what the initializer looks like in the dest module...
      Constant *SInit =
        cast<Constant>(RemapOperand(SGV->getInitializer(), ValueMap, 0));

      GlobalVariable *DGV = cast<GlobalVariable>(ValueMap[SGV]);    
      if (DGV->hasInitializer()) {
        assert(SGV->getLinkage() == DGV->getLinkage());
        if (SGV->hasExternalLinkage()) {
          if (DGV->getInitializer() != SInit)
            return Error(Err, "Global Variable Collision on '" + 
                         SGV->getType()->getDescription() +"':%"+SGV->getName()+
                         " - Global variables have different initializers");
        } else if (DGV->hasLinkOnceLinkage()) {
          // Nothing is required, mapped values will take the new global
          // automatically.
        } else if (DGV->hasAppendingLinkage()) {
          assert(0 && "Appending linkage unimplemented!");
        } else {
          assert(0 && "Unknown linkage!");
        }
      } else {
        // Copy the initializer over now...
        DGV->setInitializer(SInit);
      }
    }
  }
  return false;
}

// LinkFunctionProtos - Link the functions together between the two modules,
// without doing function bodies... this just adds external function prototypes
// to the Dest function...
//
static bool LinkFunctionProtos(Module *Dest, const Module *Src,
                               std::map<const Value*, Value*> &ValueMap,
                               std::string *Err) {
  SymbolTable *ST = (SymbolTable*)&Dest->getSymbolTable();
  
  // Loop over all of the functions in the src module, mapping them over as we
  // go
  //
  for (Module::const_iterator I = Src->begin(), E = Src->end(); I != E; ++I) {
    const Function *SF = I;   // SrcFunction
    Function *DF = 0;
    if (SF->hasName())
      // The same named thing is a Function, because the only two things
      // that may be in a module level symbol table are Global Vars and
      // Functions, and they both have distinct, nonoverlapping, possible types.
      // 
      DF = cast_or_null<Function>(FindGlobalNamed(SF->getName(), SF->getType(),
                                                  ST));

    if (!DF || SF->hasInternalLinkage() || DF->hasInternalLinkage()) {
      // Function does not already exist, simply insert an function signature
      // identical to SF into the dest module...
      Function *NewDF = new Function(SF->getFunctionType(), SF->getLinkage(),
                                     SF->getName(), Dest);

      // If the LLVM runtime renamed the function, but it is an externally
      // visible symbol, DF must be an existing function with internal linkage.
      // Rename it.
      if (NewDF->getName() != SF->getName() && !NewDF->hasInternalLinkage()) {
        assert(DF && DF->getName() == SF->getName() &&DF->hasInternalLinkage());
        DF->setName("");
        NewDF->setName(SF->getName());  // Force the name back
        DF->setName(SF->getName());     // This will cause a renaming
        assert(NewDF->getName() == SF->getName() &&
               DF->getName() != SF->getName());
      }

      // ... and remember this mapping...
      ValueMap.insert(std::make_pair(SF, NewDF));
    } else if (SF->isExternal()) {
      // If SF is external or if both SF & DF are external..  Just link the
      // external functions, we aren't adding anything.
      ValueMap.insert(std::make_pair(SF, DF));
    } else if (DF->isExternal()) {   // If DF is external but SF is not...
      // Link the external functions, update linkage qualifiers
      ValueMap.insert(std::make_pair(SF, DF));
      DF->setLinkage(SF->getLinkage());

    } else if (SF->getLinkage() != DF->getLinkage()) {
      return Error(Err, "Functions named '" + SF->getName() +
                   "' have different linkage specifiers!");
    } else if (SF->hasExternalLinkage()) {
      // The function is defined in both modules!!
      return Error(Err, "Function '" + 
                   SF->getFunctionType()->getDescription() + "':\"" + 
                   SF->getName() + "\" - Function is already defined!");
    } else if (SF->hasLinkOnceLinkage()) {
      // Completely ignore the source function.
      ValueMap.insert(std::make_pair(SF, DF));
    } else {
      assert(0 && "Unknown linkage configuration found!");
    }
  }
  return false;
}

// LinkFunctionBody - Copy the source function over into the dest function and
// fix up references to values.  At this point we know that Dest is an external
// function, and that Src is not.
//
static bool LinkFunctionBody(Function *Dest, const Function *Src,
                             std::map<const Value*, Value*> &GlobalMap,
                             std::string *Err) {
  assert(Src && Dest && Dest->isExternal() && !Src->isExternal());
  std::map<const Value*, Value*> LocalMap;   // Map for function local values

  // Go through and convert function arguments over...
  Function::aiterator DI = Dest->abegin();
  for (Function::const_aiterator I = Src->abegin(), E = Src->aend();
       I != E; ++I, ++DI) {
    DI->setName(I->getName());  // Copy the name information over...

    // Add a mapping to our local map
    LocalMap.insert(std::make_pair(I, DI));
  }

  // Loop over all of the basic blocks, copying the instructions over...
  //
  for (Function::const_iterator I = Src->begin(), E = Src->end(); I != E; ++I) {
    // Create new basic block and add to mapping and the Dest function...
    BasicBlock *DBB = new BasicBlock(I->getName(), Dest);
    LocalMap.insert(std::make_pair(I, DBB));

    // Loop over all of the instructions in the src basic block, copying them
    // over.  Note that this is broken in a strict sense because the cloned
    // instructions will still be referencing values in the Src module, not
    // the remapped values.  In our case, however, we will not get caught and 
    // so we can delay patching the values up until later...
    //
    for (BasicBlock::const_iterator II = I->begin(), IE = I->end(); 
         II != IE; ++II) {
      Instruction *DI = II->clone();
      DI->setName(II->getName());
      DBB->getInstList().push_back(DI);
      LocalMap.insert(std::make_pair(II, DI));
    }
  }

  // At this point, all of the instructions and values of the function are now
  // copied over.  The only problem is that they are still referencing values in
  // the Source function as operands.  Loop through all of the operands of the
  // functions and patch them up to point to the local versions...
  //
  for (Function::iterator BB = Dest->begin(), BE = Dest->end(); BB != BE; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
           OI != OE; ++OI)
        *OI = RemapOperand(*OI, LocalMap, &GlobalMap);

  return false;
}


// LinkFunctionBodies - Link in the function bodies that are defined in the
// source module into the DestModule.  This consists basically of copying the
// function over and fixing up references to values.
//
static bool LinkFunctionBodies(Module *Dest, const Module *Src,
                               std::map<const Value*, Value*> &ValueMap,
                               std::string *Err) {

  // Loop over all of the functions in the src module, mapping them over as we
  // go
  //
  for (Module::const_iterator SF = Src->begin(), E = Src->end(); SF != E; ++SF){
    if (!SF->isExternal()) {                  // No body if function is external
      Function *DF = cast<Function>(ValueMap[SF]); // Destination function

      // DF not external SF external?
      if (!DF->isExternal()) {
        if (DF->hasLinkOnceLinkage()) continue; // No relinkage for link-once!
        if (Err)
          *Err = "Function '" + (SF->hasName() ? SF->getName() :std::string(""))
               + "' body multiply defined!";
        return true;
      }

      if (LinkFunctionBody(DF, SF, ValueMap, Err)) return true;
    }
  }
  return false;
}

// LinkAppendingVars - If there were any appending global variables, link them
// together now.  Return true on error.
//
static bool LinkAppendingVars(Module *M,
                  std::multimap<std::string, GlobalVariable *> &AppendingVars,
                              std::string *ErrorMsg) {
  if (AppendingVars.empty()) return false; // Nothing to do.
  
  // Loop over the multimap of appending vars, processing any variables with the
  // same name, forming a new appending global variable with both of the
  // initializers merged together, then rewrite references to the old variables
  // and delete them.
  //
  std::vector<Constant*> Inits;
  while (AppendingVars.size() > 1) {
    // Get the first two elements in the map...
    std::multimap<std::string,
      GlobalVariable*>::iterator Second = AppendingVars.begin(), First=Second++;

    // If the first two elements are for different names, there is no pair...
    // Otherwise there is a pair, so link them together...
    if (First->first == Second->first) {
      GlobalVariable *G1 = First->second, *G2 = Second->second;
      const ArrayType *T1 = cast<ArrayType>(G1->getType()->getElementType());
      const ArrayType *T2 = cast<ArrayType>(G2->getType()->getElementType());
      
      // Check to see that they two arrays agree on type...
      if (T1->getElementType() != T2->getElementType())
        return Error(ErrorMsg,
         "Appending variables with different element types need to be linked!");
      if (G1->isConstant() != G2->isConstant())
        return Error(ErrorMsg,
                     "Appending variables linked with different const'ness!");

      unsigned NewSize = T1->getNumElements() + T2->getNumElements();
      ArrayType *NewType = ArrayType::get(T1->getElementType(), NewSize);

      // Create the new global variable...
      GlobalVariable *NG =
        new GlobalVariable(NewType, G1->isConstant(), G1->getLinkage(),
                           /*init*/0, First->first, M);

      // Merge the initializer...
      Inits.reserve(NewSize);
      ConstantArray *I = cast<ConstantArray>(G1->getInitializer());
      for (unsigned i = 0, e = T1->getNumElements(); i != e; ++i)
        Inits.push_back(cast<Constant>(I->getValues()[i]));
      I = cast<ConstantArray>(G2->getInitializer());
      for (unsigned i = 0, e = T2->getNumElements(); i != e; ++i)
        Inits.push_back(cast<Constant>(I->getValues()[i]));
      NG->setInitializer(ConstantArray::get(NewType, Inits));
      Inits.clear();

      // Replace any uses of the two global variables with uses of the new
      // global...

      // FIXME: This should rewrite simple/straight-forward uses such as
      // getelementptr instructions to not use the Cast!
      ConstantPointerRef *NGCP = ConstantPointerRef::get(NG);
      G1->replaceAllUsesWith(ConstantExpr::getCast(NGCP, G1->getType()));
      G2->replaceAllUsesWith(ConstantExpr::getCast(NGCP, G2->getType()));

      // Remove the two globals from the module now...
      M->getGlobalList().erase(G1);
      M->getGlobalList().erase(G2);

      // Put the new global into the AppendingVars map so that we can handle
      // linking of more than two vars...
      Second->second = NG;
    }
    AppendingVars.erase(First);
  }

  return false;
}


// LinkModules - This function links two modules together, with the resulting
// left module modified to be the composite of the two input modules.  If an
// error occurs, true is returned and ErrorMsg (if not null) is set to indicate
// the problem.  Upon failure, the Dest module could be in a modified state, and
// shouldn't be relied on to be consistent.
//
bool LinkModules(Module *Dest, const Module *Src, std::string *ErrorMsg) {
  if (Dest->getEndianness() == Module::AnyEndianness)
    Dest->setEndianness(Src->getEndianness());
  if (Dest->getPointerSize() == Module::AnyPointerSize)
    Dest->setPointerSize(Src->getPointerSize());

  if (Src->getEndianness() != Module::AnyEndianness &&
      Dest->getEndianness() != Src->getEndianness())
    std::cerr << "WARNING: Linking two modules of different endianness!\n";
  if (Src->getPointerSize() != Module::AnyPointerSize &&
      Dest->getPointerSize() != Src->getPointerSize())
    std::cerr << "WARNING: Linking two modules of different pointer size!\n";

  // LinkTypes - Go through the symbol table of the Src module and see if any
  // types are named in the src module that are not named in the Dst module.
  // Make sure there are no type name conflicts.
  //
  if (LinkTypes(Dest, Src, ErrorMsg)) return true;

  // ValueMap - Mapping of values from what they used to be in Src, to what they
  // are now in Dest.
  //
  std::map<const Value*, Value*> ValueMap;

  // AppendingVars - Keep track of global variables in the destination module
  // with appending linkage.  After the module is linked together, they are
  // appended and the module is rewritten.
  //
  std::multimap<std::string, GlobalVariable *> AppendingVars;

  // Add all of the appending globals already in the Dest module to
  // AppendingVars.
  for (Module::giterator I = Dest->gbegin(), E = Dest->gend(); I != E; ++I)
    if (I->hasAppendingLinkage())
      AppendingVars.insert(std::make_pair(I->getName(), I));

  // Insert all of the globals in src into the Dest module... without linking
  // initializers (which could refer to functions not yet mapped over).
  //
  if (LinkGlobals(Dest, Src, ValueMap, AppendingVars, ErrorMsg)) return true;

  // Link the functions together between the two modules, without doing function
  // bodies... this just adds external function prototypes to the Dest
  // function...  We do this so that when we begin processing function bodies,
  // all of the global values that may be referenced are available in our
  // ValueMap.
  //
  if (LinkFunctionProtos(Dest, Src, ValueMap, ErrorMsg)) return true;

  // Update the initializers in the Dest module now that all globals that may
  // be referenced are in Dest.
  //
  if (LinkGlobalInits(Dest, Src, ValueMap, ErrorMsg)) return true;

  // Link in the function bodies that are defined in the source module into the
  // DestModule.  This consists basically of copying the function over and
  // fixing up references to values.
  //
  if (LinkFunctionBodies(Dest, Src, ValueMap, ErrorMsg)) return true;

  // If there were any appending global variables, link them together now.
  //
  if (LinkAppendingVars(Dest, AppendingVars, ErrorMsg)) return true;

  return false;
}

