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
static inline bool Error(std::string *E, std::string Message) {
  if (E) *E = Message;
  return true;
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

  const SymbolTable::VarMap &VM = PI->second;
  for (SymbolTable::type_const_iterator I = VM.begin(), E = VM.end();
       I != E; ++I) {
    const std::string &Name = I->first;
    const Type *RHS = cast<Type>(I->second);

    // Check to see if this type name is already in the dest module...
    const Type *Entry = cast_or_null<Type>(DestST->lookup(Type::TypeTy, Name));
    if (Entry && !isa<OpaqueType>(Entry)) {  // Yup, the value already exists...
      if (Entry != RHS) {
        if (OpaqueType *OT = dyn_cast<OpaqueType>(const_cast<Type*>(RHS))) {
          OT->refineAbstractTypeTo(Entry);
        } else {
          // If it's the same, noop.  Otherwise, error.
          return Error(Err, "Type named '" + Name + 
                       "' of different shape in modules.\n  Src='" + 
                       Entry->getDescription() + "'.\n  Dst='" + 
                       RHS->getDescription() + "'");
        }
      }
    } else {                       // Type not in dest module.  Add it now.
      if (Entry) {
        OpaqueType *OT = cast<OpaqueType>(const_cast<Type*>(Entry));
        OT->refineAbstractTypeTo(RHS);
      }

      // TODO: FIXME WHEN TYPES AREN'T CONST
      DestST->insert(Name, const_cast<Type*>(RHS));
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


// LinkGlobals - Loop through the global variables in the src module and merge
// them into the dest module...
//
static bool LinkGlobals(Module *Dest, const Module *Src,
                        std::map<const Value*, Value*> &ValueMap,
                        std::string *Err) {
  // We will need a module level symbol table if the src module has a module
  // level symbol table...
  SymbolTable *ST = (SymbolTable*)&Dest->getSymbolTable();
  
  // Loop over all of the globals in the src module, mapping them over as we go
  //
  for (Module::const_giterator I = Src->gbegin(), E = Src->gend(); I != E; ++I){
    const GlobalVariable *SGV = I;
    Value *V;

    // If the global variable has a name, and that name is already in use in the
    // Dest module, make sure that the name is a compatible global variable...
    //
    if (SGV->hasExternalLinkage() && SGV->hasName() &&
	(V = ST->lookup(SGV->getType(), SGV->getName())) &&
	cast<GlobalVariable>(V)->hasExternalLinkage()) {
      // The same named thing is a global variable, because the only two things
      // that may be in a module level symbol table are Global Vars and
      // Functions, and they both have distinct, nonoverlapping, possible types.
      // 
      GlobalVariable *DGV = cast<GlobalVariable>(V);

      // Check to see if the two GV's have the same Const'ness...
      if (SGV->isConstant() != DGV->isConstant())
        return Error(Err, "Global Variable Collision on '" + 
                     SGV->getType()->getDescription() + "':%" + SGV->getName() +
                     " - Global variables differ in const'ness");

      // Okay, everything is cool, remember the mapping...
      ValueMap.insert(std::make_pair(SGV, DGV));
    } else {
      // No linking to be performed, simply create an identical version of the
      // symbol over in the dest module... the initializer will be filled in
      // later by LinkGlobalInits...
      //
      GlobalVariable *DGV = 
        new GlobalVariable(SGV->getType()->getElementType(), SGV->isConstant(),
                           SGV->hasInternalLinkage(), 0, SGV->getName());

      // Add the new global to the dest module
      Dest->getGlobalList().push_back(DGV);

      // Make sure to remember this mapping...
      ValueMap.insert(std::make_pair(SGV, DGV));
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
      Constant *DInit =
        cast<Constant>(RemapOperand(SGV->getInitializer(), ValueMap, 0));

      GlobalVariable *DGV = cast<GlobalVariable>(ValueMap[SGV]);    
      if (DGV->hasInitializer() && SGV->hasExternalLinkage() &&
	  DGV->hasExternalLinkage()) {
        if (DGV->getInitializer() != DInit)
          return Error(Err, "Global Variable Collision on '" + 
                       SGV->getType()->getDescription() + "':%" +SGV->getName()+
                       " - Global variables have different initializers");
      } else {
        // Copy the initializer over now...
        DGV->setInitializer(DInit);
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
    Value *V;

    // If the function has a name, and that name is already in use in the Dest
    // module, make sure that the name is a compatible function...
    //
    if (SF->hasExternalLinkage() && SF->hasName() &&
	(V = ST->lookup(SF->getType(), SF->getName())) &&
	cast<Function>(V)->hasExternalLinkage()) {
      // The same named thing is a Function, because the only two things
      // that may be in a module level symbol table are Global Vars and
      // Functions, and they both have distinct, nonoverlapping, possible types.
      // 
      Function *DF = cast<Function>(V);   // DestFunction

      // Check to make sure the function is not defined in both modules...
      if (!SF->isExternal() && !DF->isExternal())
        return Error(Err, "Function '" + 
                     SF->getFunctionType()->getDescription() + "':\"" + 
                     SF->getName() + "\" - Function is already defined!");

      // Otherwise, just remember this mapping...
      ValueMap.insert(std::make_pair(SF, DF));
    } else {
      // Function does not already exist, simply insert an external function
      // signature identical to SF into the dest module...
      Function *DF = new Function(SF->getFunctionType(),
                                  SF->hasInternalLinkage(),
                                  SF->getName());

      // Add the function signature to the dest module...
      Dest->getFunctionList().push_back(DF);

      // ... and remember this mapping...
      ValueMap.insert(std::make_pair(SF, DF));
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



// LinkModules - This function links two modules together, with the resulting
// left module modified to be the composite of the two input modules.  If an
// error occurs, true is returned and ErrorMsg (if not null) is set to indicate
// the problem.  Upon failure, the Dest module could be in a modified state, and
// shouldn't be relied on to be consistent.
//
bool LinkModules(Module *Dest, const Module *Src, std::string *ErrorMsg) {

  // LinkTypes - Go through the symbol table of the Src module and see if any
  // types are named in the src module that are not named in the Dst module.
  // Make sure there are no type name conflicts.
  //
  if (LinkTypes(Dest, Src, ErrorMsg)) return true;

  // ValueMap - Mapping of values from what they used to be in Src, to what they
  // are now in Dest.
  //
  std::map<const Value*, Value*> ValueMap;

  // Insert all of the globals in src into the Dest module... without
  // initializers
  if (LinkGlobals(Dest, Src, ValueMap, ErrorMsg)) return true;

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

  return false;
}

