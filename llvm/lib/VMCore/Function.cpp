//===-- Method.cpp - Implement the Method class ------------------*- C++ -*--=//
//
// This file implements the Method class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ValueHolderImpl.h"
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOther.h"

// Instantiate Templates - This ugliness is the price we have to pay
// for having a ValueHolderImpl.h file seperate from ValueHolder.h!  :(
//
template class ValueHolder<MethodArgument, Method>;
template class ValueHolder<BasicBlock    , Method>;

Method::Method(const MethodType *Ty, const string &name) 
  : SymTabValue(Ty, Value::MethodVal, name), BasicBlocks(this), 
    ArgumentList(this, this) {
  assert(Ty->isMethodType() && "Method signature must be of method type!");
  Parent = 0;
}

Method::~Method() {
  dropAllReferences();    // After this it is safe to delete instructions.

  // TODO: Should remove from the end, not the beginning of vector!
  BasicBlocksType::iterator BI = BasicBlocks.begin();
  while ((BI = BasicBlocks.begin()) != BasicBlocks.end())
    delete BasicBlocks.remove(BI);

  // Delete all of the method arguments and unlink from symbol table...
  ArgumentList.delete_all();
  ArgumentList.setParent(0);
}

// Specialize setName to take care of symbol table majik
void Method::setName(const string &name) {
  Module *P;
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

const MethodType *Method::getMethodType() const { 
  return (const MethodType *)getType();
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
  for_each(BasicBlocks.begin(), BasicBlocks.end(), 
	   std::mem_fun(&BasicBlock::dropAllReferences));
}
