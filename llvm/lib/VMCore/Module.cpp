//===-- Module.cpp - Implement the Module class ------------------*- C++ -*--=//
//
// This file implements the Module class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InstrTypes.h"
#include "llvm/ValueHolderImpl.h"
#include "llvm/Type.h"
#include "llvm/ConstantVals.h"
#include "Support/STLExtras.h"
#include <map>

// Instantiate Templates - This ugliness is the price we have to pay
// for having a DefHolderImpl.h file seperate from DefHolder.h!  :(
//
template class ValueHolder<GlobalVariable, Module, Module>;
template class ValueHolder<Function, Module, Module>;

// Define the GlobalValueRefMap as a struct that wraps a map so that we don't
// have Module.h depend on <map>
//
struct GlobalValueRefMap : public std::map<GlobalValue*, ConstantPointerRef*>{
};


Module::Module()
  : Value(Type::VoidTy, Value::ModuleVal, ""), SymTabValue(this),
    GlobalList(this, this), FunctionList(this, this), GVRefMap(0) {
}

Module::~Module() {
  dropAllReferences();
  GlobalList.delete_all();
  GlobalList.setParent(0);
  FunctionList.delete_all();
  FunctionList.setParent(0);
}


// dropAllReferences() - This function causes all the subinstructions to "let
// go" of all references that they are maintaining.  This allows one to
// 'delete' a whole class at a time, even though there may be circular
// references... first all references are dropped, and all use counts go to
// zero.  Then everything is delete'd for real.  Note that no operations are
// valid on an object that has "dropped all references", except operator 
// delete.
//
void Module::dropAllReferences() {
  for_each(FunctionList.begin(), FunctionList.end(),
	   std::mem_fun(&Function::dropAllReferences));

  for_each(GlobalList.begin(), GlobalList.end(),
	   std::mem_fun(&GlobalVariable::dropAllReferences));

  // If there are any GlobalVariable references still out there, nuke them now.
  // Since all references are hereby dropped, nothing could possibly reference
  // them still.
  if (GVRefMap) {
    for (GlobalValueRefMap::iterator I = GVRefMap->begin(), E = GVRefMap->end();
	 I != E; ++I) {
      // Delete the ConstantPointerRef node...
      I->second->destroyConstant();
    }

    // Since the table is empty, we can now delete it...
    delete GVRefMap;
  }
}

// Accessor for the underlying GlobalValRefMap...
ConstantPointerRef *Module::getConstantPointerRef(GlobalValue *V){
  // Create ref map lazily on demand...
  if (GVRefMap == 0) GVRefMap = new GlobalValueRefMap();

  GlobalValueRefMap::iterator I = GVRefMap->find(V);
  if (I != GVRefMap->end()) return I->second;

  ConstantPointerRef *Ref = new ConstantPointerRef(V);
  GVRefMap->insert(std::make_pair(V, Ref));

  return Ref;
}

void Module::mutateConstantPointerRef(GlobalValue *OldGV, GlobalValue *NewGV) {
  GlobalValueRefMap::iterator I = GVRefMap->find(OldGV);
  assert(I != GVRefMap->end() && 
	 "mutateConstantPointerRef; OldGV not in table!");
  ConstantPointerRef *Ref = I->second;

  // Remove the old entry...
  GVRefMap->erase(I);

  // Insert the new entry...
  GVRefMap->insert(std::make_pair(NewGV, Ref));
}
