//===-- Module.cpp - Implement the Module class ------------------*- C++ -*--=//
//
// This file implements the Module class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InstrTypes.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "Support/STLExtras.h"
#include "ValueHolderImpl.h"
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


Module::Module() : GlobalList(this, this), FunctionList(this, this) {
  GVRefMap = 0;
  SymTab = 0;
}

Module::~Module() {
  dropAllReferences();
  GlobalList.delete_all();
  GlobalList.setParent(0);
  FunctionList.delete_all();
  FunctionList.setParent(0);
  delete SymTab;
}

SymbolTable *Module::getSymbolTableSure() {
  if (!SymTab) SymTab = new SymbolTable(0);
  return SymTab;
}

// hasSymbolTable() - Returns true if there is a symbol table allocated to
// this object AND if there is at least one name in it!
//
bool Module::hasSymbolTable() const {
  if (!SymTab) return false;

  for (SymbolTable::const_iterator I = SymTab->begin(), E = SymTab->end();
       I != E; ++I)
    if (I->second.begin() != I->second.end())
      return true;                                // Found nonempty type plane!
  
  return false;
}


// getOrInsertFunction - Look up the specified function in the module symbol
// table.  If it does not exist, add a prototype for the function and return
// it.  This is nice because it allows most passes to get away with not handling
// the symbol table directly for this common task.
//
Function *Module::getOrInsertFunction(const std::string &Name,
                                      const FunctionType *Ty) {
  SymbolTable *SymTab = getSymbolTableSure();

  // See if we have a definitions for the specified function already...
  if (Value *V = SymTab->lookup(PointerType::get(Ty), Name)) {
    return cast<Function>(V);      // Yup, got it
  } else {                         // Nope, add one
    Function *New = new Function(Ty, false, Name);
    FunctionList.push_back(New);
    return New;                    // Return the new prototype...
  }
}

// getFunction - Look up the specified function in the module symbol table.
// If it does not exist, return null.
//
Function *Module::getFunction(const std::string &Name, const FunctionType *Ty) {
  SymbolTable *SymTab = getSymbolTable();
  if (SymTab == 0) return 0;  // No symtab, no symbols...

  return cast_or_null<Function>(SymTab->lookup(PointerType::get(Ty), Name));
}

// addTypeName - Insert an entry in the symbol table mapping Str to Type.  If
// there is already an entry for this name, true is returned and the symbol
// table is not modified.
//
bool Module::addTypeName(const std::string &Name, const Type *Ty) {
  SymbolTable *ST = getSymbolTableSure();

  if (ST->lookup(Type::TypeTy, Name)) return true;  // Already in symtab...
  
  // Not in symbol table?  Set the name with the Symtab as an argument so the
  // type knows what to update...
  ((Value*)Ty)->setName(Name, ST);

  return false;
}

// getTypeName - If there is at least one entry in the symbol table for the
// specified type, return it.
//
std::string Module::getTypeName(const Type *Ty) {
  const SymbolTable *ST = getSymbolTable();
  if (ST == 0) return "";  // No symbol table, must not have an entry...
  if (ST->find(Type::TypeTy) == ST->end())
    return ""; // No names for types...

  SymbolTable::type_const_iterator TI = ST->type_begin(Type::TypeTy);
  SymbolTable::type_const_iterator TE = ST->type_end(Type::TypeTy);

  while (TI != TE && TI->second != (const Value*)Ty)
    ++TI;

  if (TI != TE)  // Must have found an entry!
    return TI->first;
  return "";     // Must not have found anything...
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
