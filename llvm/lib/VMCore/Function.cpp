//===-- Method.cpp - Implement the Method class ------------------*- C++ -*--=//
//
// This file implements the Method & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ValueHolderImpl.h"
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/GlobalVariable.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOther.h"

//===----------------------------------------------------------------------===//
// Method Implementation
//===----------------------------------------------------------------------===//


// Instantiate Templates - This ugliness is the price we have to pay
// for having a ValueHolderImpl.h file seperate from ValueHolder.h!  :(
//
template class ValueHolder<MethodArgument, Method, Method>;
template class ValueHolder<BasicBlock    , Method, Method>;

Method::Method(const MethodType *Ty, const string &name) 
  : Value(Ty, Value::MethodVal, name), SymTabValue(this), BasicBlocks(this), 
    ArgumentList(this, this) {
  assert(::isa<MethodType>(Ty) && "Method signature must be of method type!");
  Parent = 0;
}

Method::~Method() {
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
void Method::setName(const string &name, SymbolTable *ST) {
  Module *P;
  assert((ST == 0 || (!getParent() || ST == getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && getName() != "") P->getSymbolTableSure()->insert(this);
}

void Method::setParent(Module *parent) {
  Parent = parent;

  // Relink symbol tables together...
  setParentSymTab(Parent ? Parent->getSymbolTableSure() : 0);
}

const Type *Method::getReturnType() const { 
  return ((const MethodType *)getType())->getReturnType(); 
}

// dropAllReferences() - This function causes all the subinstructions to "let
// go" of all references that they are maintaining.  This allows one to
// 'delete' a whole class at a time, even though there may be circular
// references... first all references are dropped, and all use counts go to
// zero.  Then everything is delete'd for real.  Note that no operations are
// valid on an object that has "dropped all references", except operator 
// delete.
//
void Method::dropAllReferences() {
  for_each(begin(), end(), std::mem_fun(&BasicBlock::dropAllReferences));
}

//===----------------------------------------------------------------------===//
// GlobalVariable Implementation
//===----------------------------------------------------------------------===//

GlobalVariable::GlobalVariable(const Type *Ty, bool isConstant,
			       ConstPoolVal *Initializer = 0, 
			       const string &Name = "")
  : User(Ty, Value::GlobalVal, Name), Parent(0), Constant(isConstant) {
  assert(Ty->isPointerType() && "Global Variables must be pointers!");
  if (Initializer) Operands.push_back(Use((Value*)Initializer, this));

  assert(!isConstant || hasInitializer() &&
	 "Globals Constants must have an initializer!"); 
}

// Specialize setName to take care of symbol table majik
void GlobalVariable::setName(const string &name, SymbolTable *ST) {
  Module *P;
  assert((ST == 0 || (!getParent() || ST == getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && getName() != "") P->getSymbolTableSure()->insert(this);
}
