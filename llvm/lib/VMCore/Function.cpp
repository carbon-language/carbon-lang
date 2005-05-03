//===-- Function.cpp - Implement the Global object classes ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Function & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/LeakDetector.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

BasicBlock *ilist_traits<BasicBlock>::createSentinel() {
  BasicBlock *Ret = new BasicBlock();
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}

iplist<BasicBlock> &ilist_traits<BasicBlock>::getList(Function *F) {
  return F->getBasicBlockList();
}

Argument *ilist_traits<Argument>::createSentinel() {
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
template class SymbolTableListTraits<Argument, Function, Function>;
template class SymbolTableListTraits<BasicBlock, Function, Function>;

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
  : GlobalValue(PointerType::get(Ty), Value::FunctionVal, 0, 0, Linkage, name) {
  BasicBlocks.setItemParent(this);
  BasicBlocks.setParent(this);
  ArgumentList.setItemParent(this);
  ArgumentList.setParent(this);
  SymTab = new SymbolTable();

  assert((getReturnType()->isFirstClassType() ||getReturnType() == Type::VoidTy)
         && "LLVM functions cannot return aggregate values!");

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

bool Function::isVarArg() const {
  return getFunctionType()->isVarArg();
}

const Type *Function::getReturnType() const {
  return getFunctionType()->getReturnType();
}

void Function::removeFromParent() {
  getParent()->getFunctionList().remove(this);
}

void Function::eraseFromParent() {
  getParent()->getFunctionList().erase(this);
}


/// renameLocalSymbols - This method goes through the Function's symbol table
/// and renames any symbols that conflict with symbols at global scope.  This is
/// required before printing out to a textual form, to ensure that there is no
/// ambiguity when parsing.
void Function::renameLocalSymbols() {
  SymbolTable &LST = getSymbolTable();                 // Local Symtab
  SymbolTable &GST = getParent()->getSymbolTable();    // Global Symtab

  for (SymbolTable::plane_iterator LPI = LST.plane_begin(), E = LST.plane_end();
       LPI != E; ++LPI)
    // All global symbols are of pointer type, ignore any non-pointer planes.
    if (const PointerType *CurTy = dyn_cast<PointerType>(LPI->first)) {
      // Only check if the global plane has any symbols of this type.
      SymbolTable::plane_iterator GPI = GST.find(LPI->first);
      if (GPI != GST.plane_end()) {
        SymbolTable::ValueMap &LVM       = LPI->second;
        const SymbolTable::ValueMap &GVM = GPI->second;

        // Loop over all local symbols, renaming those that are in the global
        // symbol table already.
        for (SymbolTable::value_iterator VI = LVM.begin(), E = LVM.end();
             VI != E;) {
          Value *V                = VI->second;
          const std::string &Name = VI->first;
          ++VI;
          if (GVM.count(Name)) {
            static unsigned UniqueNum = 0;
            // Find a name that does not conflict!
            while (GVM.count(Name + "_" + utostr(++UniqueNum)) ||
                   LVM.count(Name + "_" + utostr(UniqueNum)))
              /* scan for UniqueNum that works */;
            V->setName(Name + "_" + utostr(UniqueNum));
          }
        }
      }
    }
}


// dropAllReferences() - This function causes all the subinstructions to "let
// go" of all references that they are maintaining.  This allows one to
// 'delete' a whole class at a time, even though there may be circular
// references... first all references are dropped, and all use counts go to
// zero.  Then everything is deleted for real.  Note that no operations are
// valid on an object that has "dropped all references", except operator
// delete.
//
void Function::dropAllReferences() {
  for (iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();
  BasicBlocks.clear();    // Delete all basic blocks...
}

/// getIntrinsicID - This method returns the ID number of the specified
/// function, or Intrinsic::not_intrinsic if the function is not an
/// intrinsic, or if the pointer is null.  This value is always defined to be
/// zero to allow easy checking for whether a function is intrinsic or not.  The
/// particular intrinsic functions which correspond to this value are defined in
/// llvm/Intrinsics.h.
///
unsigned Function::getIntrinsicID() const {
  if (getName().size() < 5 || getName()[4] != '.' || getName()[0] != 'l' ||
      getName()[1] != 'l' || getName()[2] != 'v' || getName()[3] != 'm')
    return 0;  // All intrinsics start with 'llvm.'

  assert(getName().size() != 5 && "'llvm.' is an invalid intrinsic name!");

  switch (getName()[5]) {
  case 'c':
    if (getName() == "llvm.ctpop") return Intrinsic::ctpop;
    if (getName() == "llvm.cttz") return Intrinsic::cttz;
    if (getName() == "llvm.ctlz") return Intrinsic::ctlz;
    break;
  case 'd':
    if (getName() == "llvm.dbg.stoppoint")   return Intrinsic::dbg_stoppoint;
    if (getName() == "llvm.dbg.region.start")return Intrinsic::dbg_region_start;
    if (getName() == "llvm.dbg.region.end")  return Intrinsic::dbg_region_end;
    if (getName() == "llvm.dbg.func.start")  return Intrinsic::dbg_func_start;
    if (getName() == "llvm.dbg.declare")     return Intrinsic::dbg_declare;
    break;
  case 'f':
    if (getName() == "llvm.frameaddress")  return Intrinsic::frameaddress;
    break;
  case 'g':
    if (getName() == "llvm.gcwrite") return Intrinsic::gcwrite;
    if (getName() == "llvm.gcread")  return Intrinsic::gcread;
    if (getName() == "llvm.gcroot")  return Intrinsic::gcroot;
    break;
  case 'i':
    if (getName() == "llvm.isunordered") return Intrinsic::isunordered;
    break;
  case 'l':
    if (getName() == "llvm.longjmp")  return Intrinsic::longjmp;
    break;
  case 'm':
    if (getName() == "llvm.memcpy")  return Intrinsic::memcpy;
    if (getName() == "llvm.memmove")  return Intrinsic::memmove;
    if (getName() == "llvm.memset")  return Intrinsic::memset;
    break;
  case 'p':
    if (getName() == "llvm.prefetch")  return Intrinsic::prefetch;
    if (getName() == "llvm.pcmarker")  return Intrinsic::pcmarker;
    break;
  case 'r':
    if (getName() == "llvm.returnaddress")  return Intrinsic::returnaddress;
    if (getName() == "llvm.readport")       return Intrinsic::readport;
    if (getName() == "llvm.readio")         return Intrinsic::readio;
    break;
  case 's':
    if (getName() == "llvm.setjmp")     return Intrinsic::setjmp;
    if (getName() == "llvm.sigsetjmp")  return Intrinsic::sigsetjmp;
    if (getName() == "llvm.siglongjmp") return Intrinsic::siglongjmp;
    if (getName() == "llvm.sqrt")       return Intrinsic::sqrt;
    break;
  case 'v':
    if (getName() == "llvm.va_copy")  return Intrinsic::vacopy;
    if (getName() == "llvm.va_end")   return Intrinsic::vaend;
    if (getName() == "llvm.va_start") return Intrinsic::vastart;
  case 'w':
    if (getName() == "llvm.writeport") return Intrinsic::writeport;
    if (getName() == "llvm.writeio")   return Intrinsic::writeio;
    break;
  }
  // The "llvm." namespace is reserved!
  assert(0 && "Unknown LLVM intrinsic function!");
  return 0;
}

Value *IntrinsicInst::StripPointerCasts(Value *Ptr) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
    if (CE->getOpcode() == Instruction::Cast) {
      if (isa<PointerType>(CE->getOperand(0)->getType()))
        return StripPointerCasts(CE->getOperand(0));
    } else if (CE->getOpcode() == Instruction::GetElementPtr) {
      for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
        if (!CE->getOperand(i)->isNullValue())
          return Ptr;
      return StripPointerCasts(CE->getOperand(0));
    }
    return Ptr;
  }

  if (CastInst *CI = dyn_cast<CastInst>(Ptr)) {
    if (isa<PointerType>(CI->getOperand(0)->getType()))
      return StripPointerCasts(CI->getOperand(0));
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEP->getOperand(i)) ||
          !cast<Constant>(GEP->getOperand(i))->isNullValue())
        return Ptr;
    return StripPointerCasts(GEP->getOperand(0));
  }
  return Ptr;
}

// vim: sw=2 ai
