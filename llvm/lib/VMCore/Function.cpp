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
  BasicBlocks.clear();    // Delete all basic blocks...
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
  
  // a table of all Alpha intrinsic functions
  struct {
   std::string name;  // The name of the intrinsic 
   unsigned id;       // Its ID number
  } alpha_intrinsics[] = {
     { "llvm.alpha.ctlz",      LLVMIntrinsic::alpha_ctlz },
     { "llvm.alpha.cttz",      LLVMIntrinsic::alpha_cttz },
     { "llvm.alpha.ctpop",     LLVMIntrinsic::alpha_ctpop },
     { "llvm.alpha.umulh",     LLVMIntrinsic::alpha_umulh },
     { "llvm.alpha.vecop",     LLVMIntrinsic::alpha_vecop },
     { "llvm.alpha.pup",       LLVMIntrinsic::alpha_pup },
     { "llvm.alpha.bytezap",   LLVMIntrinsic::alpha_bytezap },
     { "llvm.alpha.bytemanip", LLVMIntrinsic::alpha_bytemanip },
     { "llvm.alpha.dfp_bop",   LLVMIntrinsic::alpha_dfpbop }, 
     { "llvm.alpha.dfp_uop",   LLVMIntrinsic::alpha_dfpuop },
     { "llvm.alpha.unordered", LLVMIntrinsic::alpha_unordered },
     { "llvm.alpha.uqtodfp",   LLVMIntrinsic::alpha_uqtodfp },
     { "llvm.alpha.uqtosfp",   LLVMIntrinsic::alpha_uqtosfp },
     { "llvm.alpha.dfptosq",   LLVMIntrinsic::alpha_dfptosq },
     { "llvm.alpha.sfptosq",   LLVMIntrinsic::alpha_sfptosq },
  };
  const unsigned num_alpha_intrinsics = 
                 sizeof(alpha_intrinsics) / sizeof(*alpha_intrinsics);

  switch (getName()[5]) {
  case 'a':
    if (getName().size() > 11 &&
        std::string(getName().begin()+4, getName().begin()+11) == ".alpha.")
      for (unsigned i = 0; i < num_alpha_intrinsics; ++i)
        if (getName() == alpha_intrinsics[i].name)
          return alpha_intrinsics[i].id;
    break;
  case 'l':
    if (getName() == "llvm.longjmp")  return LLVMIntrinsic::longjmp;
    break;
  case 's':
    if (getName() == "llvm.setjmp")     return LLVMIntrinsic::setjmp;
    if (getName() == "llvm.sigsetjmp")  return LLVMIntrinsic::sigsetjmp;
    if (getName() == "llvm.siglongjmp") return LLVMIntrinsic::siglongjmp;
    break;
  case 'v':
    if (getName() == "llvm.va_copy")  return LLVMIntrinsic::va_copy;
    if (getName() == "llvm.va_end")   return LLVMIntrinsic::va_end;
    if (getName() == "llvm.va_start") return LLVMIntrinsic::va_start;
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
