//===-- Function.cpp - Implement the Global object classes -------*- C++ -*--=//
//
// This file implements the Function & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/GlobalVariable.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOther.h"
#include "llvm/Argument.h"
#include "SymbolTableListTraitsImpl.h"

iplist<BasicBlock> &ilist_traits<BasicBlock>::getList(Function *F) {
  return F->getBasicBlockList();
}

Argument *ilist_traits<Argument>::createNode() {
  return new Argument(Type::IntTy);
}

iplist<Argument> &ilist_traits<Argument>::getList(Function *F) {
  return F->getArgumentList();
}

// Explicit instantiations of SymbolTableListTraits since some of the methods
// are not in the public header file...
template SymbolTableListTraits<Argument, Function, Function>;
template SymbolTableListTraits<BasicBlock, Function, Function>;

//===----------------------------------------------------------------------===//
// Argument Implementation
//===----------------------------------------------------------------------===//

// Specialize setName to take care of symbol table majik
void Argument::setName(const std::string &name, SymbolTable *ST) {
  Function *P;
  assert((ST == 0 || (!getParent() || ST == getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && hasName()) P->getSymbolTable()->insert(this);
}

//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//


Function::Function(const FunctionType *Ty, bool isInternal,
                   const std::string &name)
  : GlobalValue(PointerType::get(Ty), Value::FunctionVal, isInternal, name) {
  BasicBlocks.setItemParent(this);
  BasicBlocks.setParent(this);
  ArgumentList.setItemParent(this);
  ArgumentList.setParent(this);
  ParentSymTab = SymTab = 0;
}

Function::~Function() {
  dropAllReferences();    // After this it is safe to delete instructions.

  BasicBlocks.clear();    // Delete all basic blocks...

  // Delete all of the method arguments and unlink from symbol table...
  ArgumentList.clear();
  ArgumentList.setParent(0);
  delete SymTab;
}

// Specialize setName to take care of symbol table majik
void Function::setName(const std::string &name, SymbolTable *ST) {
  Module *P;
  assert((ST == 0 || (!getParent() || ST == getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && getName() != "") P->getSymbolTableSure()->insert(this);
}

void Function::setParent(Module *parent) {
  Parent = parent;

  // Relink symbol tables together...
  ParentSymTab = Parent ? Parent->getSymbolTableSure() : 0;
  if (SymTab) SymTab->setParentSymTab(ParentSymTab);
}

const FunctionType *Function::getFunctionType() const {
  return cast<FunctionType>(getType()->getElementType());
}

const Type *Function::getReturnType() const { 
  return getFunctionType()->getReturnType();
}

SymbolTable *Function::getSymbolTableSure() {
  if (!SymTab) SymTab = new SymbolTable(ParentSymTab);
  return SymTab;
}

// hasSymbolTable() - Returns true if there is a symbol table allocated to
// this object AND if there is at least one name in it!
//
bool Function::hasSymbolTable() const {
  if (!SymTab) return false;

  for (SymbolTable::const_iterator I = SymTab->begin(); 
       I != SymTab->end(); ++I) {
    if (I->second.begin() != I->second.end())
      return true;                                // Found nonempty type plane!
  }
  
  return false;
}


// dropAllReferences() - This function causes all the subinstructions to "let
// go" of all references that they are maintaining.  This allows one to
// 'delete' a whole class at a time, even though there may be circular
// references... first all references are dropped, and all use counts go to
// zero.  Then everything is delete'd for real.  Note that no operations are
// valid on an object that has "dropped all references", except operator 
// delete.
//
void Function::dropAllReferences() {
  for (iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();
}

//===----------------------------------------------------------------------===//
// GlobalVariable Implementation
//===----------------------------------------------------------------------===//

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, bool isIntern,
			       Constant *Initializer,
			       const std::string &Name)
  : GlobalValue(PointerType::get(Ty), Value::GlobalVariableVal, isIntern, Name),
    isConstantGlobal(constant) {
  if (Initializer) Operands.push_back(Use((Value*)Initializer, this));
}

// Specialize setName to take care of symbol table majik
void GlobalVariable::setName(const std::string &name, SymbolTable *ST) {
  Module *P;
  assert((ST == 0 || (!getParent() || ST == getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && getName() != "") P->getSymbolTableSure()->insert(this);
}
