//===- Linker.cpp - Module Linker Implementation --------------------------===//
//
// This file implements the LLVM module linker.
//
// Specifically, this:
//  * Merges global variables between the two modules
//    * Uninit + Uninit = Init, Init + Uninit = Init, Init + Init = Error if !=
//  * Merges methods between two modules
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Linker.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/GlobalVariable.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/ConstantVals.h"
#include "llvm/Argument.h"
#include <iostream>
using std::cerr;
using std::string;
using std::map;

// Error - Simple wrapper function to conditionally assign to E and return true.
// This just makes error return conditions a little bit simpler...
//
static inline bool Error(string *E, string Message) {
  if (E) *E = Message;
  return true;
}

// LinkTypes - Go through the symbol table of the Src module and see if any
// types are named in the src module that are not named in the Dst module.
// Make sure there are no type name conflicts.
//
static bool LinkTypes(Module *Dest, const Module *Src, string *Err = 0) {
  // No symbol table?  Can't have named types.
  if (!Src->hasSymbolTable()) return false;

  SymbolTable       *DestST = Dest->getSymbolTableSure();
  const SymbolTable *SrcST  = Src->getSymbolTable();

  // Look for a type plane for Type's...
  SymbolTable::const_iterator PI = SrcST->find(Type::TypeTy);
  if (PI == SrcST->end()) return false;  // No named types, do nothing.

  const SymbolTable::VarMap &VM = PI->second;
  for (SymbolTable::type_const_iterator I = VM.begin(), E = VM.end();
       I != E; ++I) {
    const string &Name = I->first;
    const Type *RHS = cast<Type>(I->second);

    // Check to see if this type name is already in the dest module...
    const Type *Entry = cast_or_null<Type>(DestST->lookup(Type::TypeTy, Name));
    if (Entry) {     // Yup, the value already exists...
      if (Entry != RHS)            // If it's the same, noop.  Otherwise, error.
        return Error(Err, "Type named '" + Name + 
                     "' of different shape in modules.\n  Src='" + 
                     Entry->getDescription() + "'.\n  Dst='" + 
                     RHS->getDescription() + "'");
    } else {                       // Type not in dest module.  Add it now.
      // TODO: FIXME WHEN TYPES AREN'T CONST
      DestST->insert(Name, const_cast<Type*>(RHS));
    }
  }
  return false;
}

static void PrintMap(const map<const Value*, Value*> &M) {
  for (map<const Value*, Value*>::const_iterator I = M.begin(), E = M.end();
       I != E; ++I) {
    cerr << " Fr: " << (void*)I->first << " ";
    I->first->dump();
    cerr << " To: " << (void*)I->second << " ";
    I->second->dump();
    cerr << "\n";
  }
}


// RemapOperand - Use LocalMap and GlobalMap to convert references from one
// module to another.  This is somewhat sophisticated in that it can
// automatically handle constant references correctly as well...
//
static Value *RemapOperand(const Value *In, map<const Value*, Value*> &LocalMap,
                           const map<const Value*, Value*> *GlobalMap = 0) {
  map<const Value*,Value*>::const_iterator I = LocalMap.find(In);
  if (I != LocalMap.end()) return I->second;

  if (GlobalMap) {
    I = GlobalMap->find(In);
    if (I != GlobalMap->end()) return I->second;
  }

  // Check to see if it's a constant that we are interesting in transforming...
  if (Constant *CPV = dyn_cast<Constant>(In)) {
    if (!isa<DerivedType>(CPV->getType()))
      return CPV;              // Simple constants stay identical...

    Constant *Result = 0;

    if (ConstantArray *CPA = dyn_cast<ConstantArray>(CPV)) {
      const std::vector<Use> &Ops = CPA->getValues();
      std::vector<Constant*> Operands(Ops.size());
      for (unsigned i = 0; i < Ops.size(); ++i)
        Operands[i] = 
          cast<Constant>(RemapOperand(Ops[i], LocalMap, GlobalMap));
      Result = ConstantArray::get(cast<ArrayType>(CPA->getType()), Operands);
    } else if (ConstantStruct *CPS = dyn_cast<ConstantStruct>(CPV)) {
      const std::vector<Use> &Ops = CPS->getValues();
      std::vector<Constant*> Operands(Ops.size());
      for (unsigned i = 0; i < Ops.size(); ++i)
        Operands[i] = 
          cast<Constant>(RemapOperand(Ops[i], LocalMap, GlobalMap));
      Result = ConstantStruct::get(cast<StructType>(CPS->getType()), Operands);
    } else if (isa<ConstantPointerNull>(CPV)) {
      Result = CPV;
    } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(CPV)) {
      Value *V = RemapOperand(CPR->getValue(), LocalMap, GlobalMap);
      Result = ConstantPointerRef::get(cast<GlobalValue>(V));
    } else {
      assert(0 && "Unknown type of derived type constant value!");
    }

    // Cache the mapping in our local map structure...
    LocalMap.insert(std::make_pair(In, CPV));
    return Result;
  }

  cerr << "XXX LocalMap: \n";
  PrintMap(LocalMap);

  if (GlobalMap) {
    cerr << "XXX GlobalMap: \n";
    PrintMap(*GlobalMap);
  }

  cerr << "Couldn't remap value: " << (void*)In << " ";
  In->dump();
  cerr << "\n";
  assert(0 && "Couldn't remap value!");
  return 0;
}


// LinkGlobals - Loop through the global variables in the src module and merge
// them into the dest module...
//
static bool LinkGlobals(Module *Dest, const Module *Src,
                        map<const Value*, Value*> &ValueMap, string *Err = 0) {
  // We will need a module level symbol table if the src module has a module
  // level symbol table...
  SymbolTable *ST = Src->getSymbolTable() ? Dest->getSymbolTableSure() : 0;
  
  // Loop over all of the globals in the src module, mapping them over as we go
  //
  for (Module::const_giterator I = Src->gbegin(), E = Src->gend(); I != E; ++I){
    const GlobalVariable *SGV = *I;
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
                            map<const Value*, Value*> &ValueMap,
                            string *Err = 0) {

  // Loop over all of the globals in the src module, mapping them over as we go
  //
  for (Module::const_giterator I = Src->gbegin(), E = Src->gend(); I != E; ++I){
    const GlobalVariable *SGV = *I;

    if (SGV->hasInitializer()) {      // Only process initialized GV's
      // Figure out what the initializer looks like in the dest module...
      Constant *DInit =
        cast<Constant>(RemapOperand(SGV->getInitializer(), ValueMap));

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
// without doing method bodies... this just adds external method prototypes to
// the Dest function...
//
static bool LinkFunctionProtos(Module *Dest, const Module *Src,
                               map<const Value*, Value*> &ValueMap,
                               string *Err = 0) {
  // We will need a module level symbol table if the src module has a module
  // level symbol table...
  SymbolTable *ST = Src->getSymbolTable() ? Dest->getSymbolTableSure() : 0;
  
  // Loop over all of the methods in the src module, mapping them over as we go
  //
  for (Module::const_iterator I = Src->begin(), E = Src->end(); I != E; ++I) {
    const Function *SM = *I;   // SrcFunction
    Value *V;

    // If the method has a name, and that name is already in use in the
    // Dest module, make sure that the name is a compatible method...
    //
    if (SM->hasExternalLinkage() && SM->hasName() &&
	(V = ST->lookup(SM->getType(), SM->getName())) &&
	cast<Function>(V)->hasExternalLinkage()) {
      // The same named thing is a Function, because the only two things
      // that may be in a module level symbol table are Global Vars and
      // Functions, and they both have distinct, nonoverlapping, possible types.
      // 
      Function *DM = cast<Function>(V);   // DestFunction

      // Check to make sure the method is not defined in both modules...
      if (!SM->isExternal() && !DM->isExternal())
        return Error(Err, "Function '" + 
                     SM->getFunctionType()->getDescription() + "':\"" + 
                     SM->getName() + "\" - Function is already defined!");

      // Otherwise, just remember this mapping...
      ValueMap.insert(std::make_pair(SM, DM));
    } else {
      // Function does not already exist, simply insert an external method
      // signature identical to SM into the dest module...
      Function *DM = new Function(SM->getFunctionType(),
                                  SM->hasInternalLinkage(),
                                  SM->getName());

      // Add the method signature to the dest module...
      Dest->getFunctionList().push_back(DM);

      // ... and remember this mapping...
      ValueMap.insert(std::make_pair(SM, DM));
    }
  }
  return false;
}

// LinkFunctionBody - Copy the source method over into the dest method
// and fix up references to values.  At this point we know that Dest
// is an external method, and that Src is not.
//
static bool LinkFunctionBody(Function *Dest, const Function *Src,
                             const map<const Value*, Value*> &GlobalMap,
                             string *Err = 0) {
  assert(Src && Dest && Dest->isExternal() && !Src->isExternal());
  map<const Value*, Value*> LocalMap;   // Map for method local values

  // Go through and convert method arguments over...
  for (Function::ArgumentListType::const_iterator 
         I = Src->getArgumentList().begin(),
         E = Src->getArgumentList().end(); I != E; ++I) {
    const Argument *SMA = *I;

    // Create the new method argument and add to the dest method...
    Argument *DMA = new Argument(SMA->getType(), SMA->getName());
    Dest->getArgumentList().push_back(DMA);

    // Add a mapping to our local map
    LocalMap.insert(std::make_pair(SMA, DMA));
  }

  // Loop over all of the basic blocks, copying the instructions over...
  //
  for (Function::const_iterator I = Src->begin(), E = Src->end(); I != E; ++I) {
    const BasicBlock *SBB = *I;

    // Create new basic block and add to mapping and the Dest method...
    BasicBlock *DBB = new BasicBlock(SBB->getName(), Dest);
    LocalMap.insert(std::make_pair(SBB, DBB));

    // Loop over all of the instructions in the src basic block, copying them
    // over.  Note that this is broken in a strict sense because the cloned
    // instructions will still be referencing values in the Src module, not
    // the remapped values.  In our case, however, we will not get caught and 
    // so we can delay patching the values up until later...
    //
    for (BasicBlock::const_iterator II = SBB->begin(), IE = SBB->end(); 
         II != IE; ++II) {
      const Instruction *SI = *II;
      Instruction *DI = SI->clone();
      DI->setName(SI->getName());
      DBB->getInstList().push_back(DI);
      LocalMap.insert(std::make_pair(SI, DI));
    }
  }

  // At this point, all of the instructions and values of the method are now
  // copied over.  The only problem is that they are still referencing values
  // in the Source method as operands.  Loop through all of the operands of the
  // methods and patch them up to point to the local versions...
  //
  for (Function::iterator BI = Dest->begin(), BE = Dest->end();
       BI != BE; ++BI) {
    BasicBlock *BB = *BI;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      Instruction *Inst = *I;
      
      for (Instruction::op_iterator OI = Inst->op_begin(), OE = Inst->op_end();
           OI != OE; ++OI)
        *OI = RemapOperand(*OI, LocalMap, &GlobalMap);
    }
  }

  return false;
}


// LinkFunctionBodies - Link in the method bodies that are defined in the source
// module into the DestModule.  This consists basically of copying the method
// over and fixing up references to values.
//
static bool LinkFunctionBodies(Module *Dest, const Module *Src,
                               map<const Value*, Value*> &ValueMap,
                               string *Err = 0) {

  // Loop over all of the methods in the src module, mapping them over as we go
  //
  for (Module::const_iterator I = Src->begin(), E = Src->end(); I != E; ++I) {
    const Function *SM = *I;                   // Source Function
    if (!SM->isExternal()) {                   // No body if method is external
      Function *DM = cast<Function>(ValueMap[SM]); // Destination method

      // DM not external SM external?
      if (!DM->isExternal()) {
        if (Err)
          *Err = "Function '" + (SM->hasName() ? SM->getName() : string("")) +
                 "' body multiply defined!";
        return true;
      }

      if (LinkFunctionBody(DM, SM, ValueMap, Err)) return true;
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
bool LinkModules(Module *Dest, const Module *Src, string *ErrorMsg = 0) {

  // LinkTypes - Go through the symbol table of the Src module and see if any
  // types are named in the src module that are not named in the Dst module.
  // Make sure there are no type name conflicts.
  //
  if (LinkTypes(Dest, Src, ErrorMsg)) return true;

  // ValueMap - Mapping of values from what they used to be in Src, to what they
  // are now in Dest.
  //
  map<const Value*, Value*> ValueMap;

  // Insert all of the globals in src into the Dest module... without
  // initializers
  if (LinkGlobals(Dest, Src, ValueMap, ErrorMsg)) return true;

  // Update the initializers in the Dest module now that all globals that may
  // be referenced are in Dest.
  //
  if (LinkGlobalInits(Dest, Src, ValueMap, ErrorMsg)) return true;

  // Link the methods together between the two modules, without doing method
  // bodies... this just adds external method prototypes to the Dest method...
  // We do this so that when we begin processing method bodies, all of the
  // global values that may be referenced are available in our ValueMap.
  //
  if (LinkFunctionProtos(Dest, Src, ValueMap, ErrorMsg)) return true;

  // Link in the method bodies that are defined in the source module into the
  // DestModule.  This consists basically of copying the method over and fixing
  // up references to values.
  //
  if (LinkFunctionBodies(Dest, Src, ValueMap, ErrorMsg)) return true;

  return false;
}

