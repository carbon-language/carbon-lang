//===-- Function.cpp - Implement the Global object classes -------*- C++ -*--=//
//
// This file implements the Function & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/Intrinsics.h"
#include "Support/LeakDetector.h"
#include "SymbolTableListTraitsImpl.h"

BasicBlock *ilist_traits<BasicBlock>::createNode() {
  BasicBlock *Ret = new BasicBlock();
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}

iplist<BasicBlock> &ilist_traits<BasicBlock>::getList(Function *F) {
  return F->getBasicBlockList();
}

Argument *ilist_traits<Argument>::createNode() {
  Argument *Ret = new Argument(Type::IntTy);
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
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

Argument::Argument(const Type *Ty, const std::string &Name, Function *Par) 
  : Value(Ty, Value::ArgumentVal, Name) {
  Parent = 0;

  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (Par)
    Par->getArgumentList().push_back(this);
}


// Specialize setName to take care of symbol table majik
void Argument::setName(const std::string &name, SymbolTable *ST) {
  Function *P;
  assert((ST == 0 || (!getParent() || ST == &getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable().remove(this);
  Value::setName(name);
  if (P && hasName()) P->getSymbolTable().insert(this);
}

void Argument::setParent(Function *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}


//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//

Function::Function(const FunctionType *Ty, LinkageTypes Linkage,
                   const std::string &name, Module *ParentModule)
  : GlobalValue(PointerType::get(Ty), Value::FunctionVal, Linkage, name) {
  BasicBlocks.setItemParent(this);
  BasicBlocks.setParent(this);
  ArgumentList.setItemParent(this);
  ArgumentList.setParent(this);
  SymTab = new SymbolTable();

  // Create the arguments vector, all arguments start out unnamed.
  for (unsigned i = 0, e = Ty->getNumParams(); i != e; ++i) {
    assert(Ty->getParamType(i) != Type::VoidTy &&
           "Cannot have void typed arguments!");
    ArgumentList.push_back(new Argument(Ty->getParamType(i)));
  }

  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getFunctionList().push_back(this);
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
  assert((ST == 0 || (!getParent() || ST == &getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable().remove(this);
  Value::setName(name);
  if (P && getName() != "") P->getSymbolTable().insert(this);
}

void Function::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

const FunctionType *Function::getFunctionType() const {
  return cast<FunctionType>(getType()->getElementType());
}

const Type *Function::getReturnType() const { 
  return getFunctionType()->getReturnType();
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

/// getIntrinsicID - This method returns the ID number of the specified
/// function, or LLVMIntrinsic::not_intrinsic if the function is not an
/// instrinsic, or if the pointer is null.  This value is always defined to be
/// zero to allow easy checking for whether a function is intrinsic or not.  The
/// particular intrinsic functions which correspond to this value are defined in
/// llvm/Intrinsics.h.
///
unsigned Function::getIntrinsicID() const {
  if (getName().size() <= 5 || getName()[4] != '.' || getName()[0] != 'l' ||
      getName()[1] != 'l' || getName()[2] != 'v' || getName()[3] != 'm')
    return 0;  // All intrinsics start with 'llvm.'
  
  switch (getName()[5]) {
  case 'v':
    if (getName().size() >= 9) {
      switch (getName()[8]) {
      case 's':
        if (getName() == "llvm.va_start") return LLVMIntrinsic::va_start;
        break;
      case 'e':
        if (getName() == "llvm.va_end") return LLVMIntrinsic::va_end;
        break;
      case 'c':
        if (getName() == "llvm.va_copy") return LLVMIntrinsic::va_copy;
        break;
      }
    }
    break;
  }
  // The "llvm." namespace is reserved!
  assert(0 && "Unknown LLVM intrinsic function!");
  return 0;
}


//===----------------------------------------------------------------------===//
// GlobalVariable Implementation
//===----------------------------------------------------------------------===//

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, LinkageTypes Link,
			       Constant *Initializer,
			       const std::string &Name, Module *ParentModule)
  : GlobalValue(PointerType::get(Ty), Value::GlobalVariableVal, Link, Name),
    isConstantGlobal(constant) {
  if (Initializer) Operands.push_back(Use((Value*)Initializer, this));

  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getGlobalList().push_back(this);
}

void GlobalVariable::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

// Specialize setName to take care of symbol table majik
void GlobalVariable::setName(const std::string &name, SymbolTable *ST) {
  Module *P;
  assert((ST == 0 || (!getParent() || ST == &getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable().remove(this);
  Value::setName(name);
  if (P && getName() != "") P->getSymbolTable().insert(this);
}
