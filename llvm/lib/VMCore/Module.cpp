//===-- Module.cpp - Implement the Module class ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Module class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/InstrTypes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/LeakDetector.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/TypeSymbolTable.h"
#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <map>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Methods to implement the globals and functions lists.
//

Function *ilist_traits<Function>::createSentinel() {
  FunctionType *FTy =
    FunctionType::get(Type::VoidTy, std::vector<const Type*>(), false, 
                      std::vector<FunctionType::ParameterAttributes>() );
  Function *Ret = new Function(FTy, GlobalValue::ExternalLinkage);
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}
GlobalVariable *ilist_traits<GlobalVariable>::createSentinel() {
  GlobalVariable *Ret = new GlobalVariable(Type::Int32Ty, false,
                                           GlobalValue::ExternalLinkage);
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}

iplist<Function> &ilist_traits<Function>::getList(Module *M) {
  return M->getFunctionList();
}
iplist<GlobalVariable> &ilist_traits<GlobalVariable>::getList(Module *M) {
  return M->getGlobalList();
}

// Explicit instantiations of SymbolTableListTraits since some of the methods
// are not in the public header file.
template class SymbolTableListTraits<GlobalVariable, Module, Module>;
template class SymbolTableListTraits<Function, Module, Module>;

//===----------------------------------------------------------------------===//
// Primitive Module methods.
//

Module::Module(const std::string &MID)
  : ModuleID(MID), DataLayout("") {
  FunctionList.setItemParent(this);
  FunctionList.setParent(this);
  GlobalList.setItemParent(this);
  GlobalList.setParent(this);
  ValSymTab = new SymbolTable();
  TypeSymTab = new TypeSymbolTable();
}

Module::~Module() {
  dropAllReferences();
  GlobalList.clear();
  GlobalList.setParent(0);
  FunctionList.clear();
  FunctionList.setParent(0);
  LibraryList.clear();
  delete ValSymTab;
  delete TypeSymTab;
}

// Module::dump() - Allow printing from debugger
void Module::dump() const {
  print(*cerr.stream());
}

/// Target endian information...
Module::Endianness Module::getEndianness() const {
  std::string temp = DataLayout;
  Module::Endianness ret = AnyEndianness;
  
  while (!temp.empty()) {
    std::string token = getToken(temp, "-");
    
    if (token[0] == 'e') {
      ret = LittleEndian;
    } else if (token[0] == 'E') {
      ret = BigEndian;
    }
  }
  
  return ret;
}

/// Target Pointer Size information...
Module::PointerSize Module::getPointerSize() const {
  std::string temp = DataLayout;
  Module::PointerSize ret = AnyPointerSize;
  
  while (!temp.empty()) {
    std::string token = getToken(temp, "-");
    char signal = getToken(token, ":")[0];
    
    if (signal == 'p') {
      int size = atoi(getToken(token, ":").c_str());
      if (size == 32)
        ret = Pointer32;
      else if (size == 64)
        ret = Pointer64;
    }
  }
  
  return ret;
}

//===----------------------------------------------------------------------===//
// Methods for easy access to the functions in the module.
//

Constant *Module::getOrInsertFunction(const std::string &Name,
                                      const FunctionType *Ty) {
  SymbolTable &SymTab = getValueSymbolTable();

  // See if we have a definitions for the specified function already.
  Function *F =
    dyn_cast_or_null<Function>(SymTab.lookup(PointerType::get(Ty), Name));
  if (F == 0) {
    // Nope, add it.
    Function *New = new Function(Ty, GlobalVariable::ExternalLinkage, Name);
    FunctionList.push_back(New);
    return New;                    // Return the new prototype.
  }

  // Okay, the function exists.  Does it have externally visible linkage?
  if (F->hasInternalLinkage()) {
    // Rename the function.
    F->setName(SymTab.getUniqueName(F->getType(), F->getName()));
    // Retry, now there won't be a conflict.
    return getOrInsertFunction(Name, Ty);
  }

  // If the function exists but has the wrong type, return a bitcast to the
  // right type.
  if (F->getFunctionType() != Ty)
    return ConstantExpr::getBitCast(F, PointerType::get(Ty));
  
  // Otherwise, we just found the existing function or a prototype.
  return F;  
}

// getOrInsertFunction - Look up the specified function in the module symbol
// table.  If it does not exist, add a prototype for the function and return it.
// This version of the method takes a null terminated list of function
// arguments, which makes it easier for clients to use.
//
Constant *Module::getOrInsertFunction(const std::string &Name,
                                      const Type *RetTy, ...) {
  va_list Args;
  va_start(Args, RetTy);

  // Build the list of argument types...
  std::vector<const Type*> ArgTys;
  while (const Type *ArgTy = va_arg(Args, const Type*))
    ArgTys.push_back(ArgTy);

  va_end(Args);

  // Build the function type and chain to the other getOrInsertFunction...
  return getOrInsertFunction(Name, FunctionType::get(RetTy, ArgTys, false));
}


// getFunction - Look up the specified function in the module symbol table.
// If it does not exist, return null.
//
Function *Module::getFunction(const std::string &Name, const FunctionType *Ty) {
  SymbolTable &SymTab = getValueSymbolTable();
  return cast_or_null<Function>(SymTab.lookup(PointerType::get(Ty), Name));
}


/// getMainFunction - This function looks up main efficiently.  This is such a
/// common case, that it is a method in Module.  If main cannot be found, a
/// null pointer is returned.
///
Function *Module::getMainFunction() {
  std::vector<const Type*> Params;

  // int main(void)...
  if (Function *F = getFunction("main", FunctionType::get(Type::Int32Ty,
                                                          Params, false)))
    return F;

  // void main(void)...
  if (Function *F = getFunction("main", FunctionType::get(Type::VoidTy,
                                                          Params, false)))
    return F;

  Params.push_back(Type::Int32Ty);

  // int main(int argc)...
  if (Function *F = getFunction("main", FunctionType::get(Type::Int32Ty,
                                                          Params, false)))
    return F;

  // void main(int argc)...
  if (Function *F = getFunction("main", FunctionType::get(Type::VoidTy,
                                                          Params, false)))
    return F;

  for (unsigned i = 0; i != 2; ++i) {  // Check argv and envp
    Params.push_back(PointerType::get(PointerType::get(Type::Int8Ty)));

    // int main(int argc, char **argv)...
    if (Function *F = getFunction("main", FunctionType::get(Type::Int32Ty,
                                                            Params, false)))
      return F;

    // void main(int argc, char **argv)...
    if (Function *F = getFunction("main", FunctionType::get(Type::VoidTy,
                                                            Params, false)))
      return F;
  }

  // Ok, try to find main the hard way...
  return getNamedFunction("main");
}

/// getNamedFunction - Return the first function in the module with the
/// specified name, of arbitrary type.  This method returns null if a function
/// with the specified name is not found.
///
Function *Module::getNamedFunction(const std::string &Name) const {
  // Loop over all of the functions, looking for the function desired
  const Function *Found = 0;
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    if (I->getName() == Name)
      if (I->isExternal())
        Found = I;
      else
        return const_cast<Function*>(&(*I));
  return const_cast<Function*>(Found); // Non-external function not found...
}

//===----------------------------------------------------------------------===//
// Methods for easy access to the global variables in the module.
//

/// getGlobalVariable - Look up the specified global variable in the module
/// symbol table.  If it does not exist, return null.  The type argument
/// should be the underlying type of the global, i.e., it should not have
/// the top-level PointerType, which represents the address of the global.
/// If AllowInternal is set to true, this function will return types that
/// have InternalLinkage. By default, these types are not returned.
///
GlobalVariable *Module::getGlobalVariable(const std::string &Name,
                                          const Type *Ty, bool AllowInternal) {
  if (Value *V = getValueSymbolTable().lookup(PointerType::get(Ty), Name)) {
    GlobalVariable *Result = cast<GlobalVariable>(V);
    if (AllowInternal || !Result->hasInternalLinkage())
      return Result;
  }
  return 0;
}

/// getNamedGlobal - Return the first global variable in the module with the
/// specified name, of arbitrary type.  This method returns null if a global
/// with the specified name is not found.
///
GlobalVariable *Module::getNamedGlobal(const std::string &Name) const {
  // FIXME: This would be much faster with a symbol table that doesn't
  // discriminate based on type!
  for (const_global_iterator I = global_begin(), E = global_end();
       I != E; ++I)
    if (I->getName() == Name) 
      return const_cast<GlobalVariable*>(&(*I));
  return 0;
}



//===----------------------------------------------------------------------===//
// Methods for easy access to the types in the module.
//


// addTypeName - Insert an entry in the symbol table mapping Str to Type.  If
// there is already an entry for this name, true is returned and the symbol
// table is not modified.
//
bool Module::addTypeName(const std::string &Name, const Type *Ty) {
  TypeSymbolTable &ST = getTypeSymbolTable();

  if (ST.lookup(Name)) return true;  // Already in symtab...

  // Not in symbol table?  Set the name with the Symtab as an argument so the
  // type knows what to update...
  ST.insert(Name, Ty);

  return false;
}

/// getTypeByName - Return the type with the specified name in this module, or
/// null if there is none by that name.
const Type *Module::getTypeByName(const std::string &Name) const {
  const TypeSymbolTable &ST = getTypeSymbolTable();
  return cast_or_null<Type>(ST.lookup(Name));
}

// getTypeName - If there is at least one entry in the symbol table for the
// specified type, return it.
//
std::string Module::getTypeName(const Type *Ty) const {
  const TypeSymbolTable &ST = getTypeSymbolTable();

  TypeSymbolTable::const_iterator TI = ST.begin();
  TypeSymbolTable::const_iterator TE = ST.end();
  if ( TI == TE ) return ""; // No names for types

  while (TI != TE && TI->second != Ty)
    ++TI;

  if (TI != TE)  // Must have found an entry!
    return TI->first;
  return "";     // Must not have found anything...
}

//===----------------------------------------------------------------------===//
// Other module related stuff.
//


// dropAllReferences() - This function causes all the subelementss to "let go"
// of all references that they are maintaining.  This allows one to 'delete' a
// whole module at a time, even though there may be circular references... first
// all references are dropped, and all use counts go to zero.  Then everything
// is deleted for real.  Note that no operations are valid on an object that
// has "dropped all references", except operator delete.
//
void Module::dropAllReferences() {
  for(Module::iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();

  for(Module::global_iterator I = global_begin(), E = global_end(); I != E; ++I)
    I->dropAllReferences();
}

