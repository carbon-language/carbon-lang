//===-- Function.cpp - Implement the Global object classes -------*- C++ -*--=//
//
// This file implements the Function & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ValueHolderImpl.h"
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOther.h"

//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//


// Instantiate Templates - This ugliness is the price we have to pay
// for having a ValueHolderImpl.h file seperate from ValueHolder.h!  :(
//
template class ValueHolder<FunctionArgument, Function, Function>;
template class ValueHolder<BasicBlock    , Function, Function>;

Function::Function(const MethodType *Ty, bool isInternal,
                   const std::string &name)
  : GlobalValue(PointerType::get(Ty), Value::FunctionVal, isInternal, name),
    SymTabValue(this), BasicBlocks(this), ArgumentList(this, this) {
  assert(::isa<MethodType>(Ty) && "Function signature must be of method type!");
}

Function::~Function() {
  dropAllReferences();    // After this it is safe to delete instructions.

  // TODO: Should remove from the end, not the beginning of vector!
  iterator BI = begin();
  while ((BI = begin()) != end())
    delete BasicBlocks.remove(BI);

  // Delete all of the method arguments and unlink from symbol table...
  ArgumentList.delete_all();
  ArgumentList.setParent(0);
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
  setParentSymTab(Parent ? Parent->getSymbolTableSure() : 0);
}

const MethodType *Function::getMethodType() const {
  return cast<MethodType>(cast<PointerType>(getType())->getElementType());
}

const Type *Function::getReturnType() const { 
  return getMethodType()->getReturnType();
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
  for_each(begin(), end(), std::mem_fun(&BasicBlock::dropAllReferences));
}

//===----------------------------------------------------------------------===//
// GlobalVariable Implementation
//===----------------------------------------------------------------------===//

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, bool isIntern,
			       Constant *Initializer = 0,
			       const std::string &Name = "")
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
