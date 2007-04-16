//===-- Function.cpp - Implement the Global object classes ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Function class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ParameterAttributes.h"
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
  Argument *Ret = new Argument(Type::Int32Ty);
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
  : Value(Ty, Value::ArgumentVal) {
  Parent = 0;

  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (Par)
    Par->getArgumentList().push_back(this);
  setName(Name);
}

void Argument::setParent(Function *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

//===----------------------------------------------------------------------===//
// ParamAttrsList Implementation
//===----------------------------------------------------------------------===//

uint16_t
ParamAttrsList::getParamAttrs(uint16_t Index) const {
  unsigned limit = attrs.size();
  for (unsigned i = 0; i < limit; ++i)
    if (attrs[i].index == Index)
      return attrs[i].attrs;
  return ParamAttr::None;
}


std::string 
ParamAttrsList::getParamAttrsText(uint16_t Attrs) {
  std::string Result;
  if (Attrs & ParamAttr::ZExt)
    Result += "zext ";
  if (Attrs & ParamAttr::SExt)
    Result += "sext ";
  if (Attrs & ParamAttr::NoReturn)
    Result += "noreturn ";
  if (Attrs & ParamAttr::NoUnwind)
    Result += "nounwind ";
  if (Attrs & ParamAttr::InReg)
    Result += "inreg ";
  if (Attrs & ParamAttr::StructRet)
    Result += "sret ";  
  return Result;
}

void
ParamAttrsList::addAttributes(uint16_t Index, uint16_t Attrs) {
  // First, try to replace an existing one
  for (unsigned i = 0; i < attrs.size(); ++i)
    if (attrs[i].index == Index) {
      attrs[i].attrs |= Attrs;
      return;
    }

  // If not found, add a new one
  ParamAttrsWithIndex Val;
  Val.attrs = Attrs;
  Val.index = Index;
  attrs.push_back(Val);
}

void
ParamAttrsList::removeAttributes(uint16_t Index, uint16_t Attrs) {
  // Find the index from which to remove the attributes
  for (unsigned i = 0; i < attrs.size(); ++i)
    if (attrs[i].index == Index) {
      attrs[i].attrs &= ~Attrs;
      if (attrs[i].attrs == ParamAttr::None)
        attrs.erase(&attrs[i]);
      return;
    }

  // The index wasn't found above
  assert(0 && "Index not found for removeAttributes");
}

//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//

Function::Function(const FunctionType *Ty, LinkageTypes Linkage,
                   const std::string &name, Module *ParentModule)
  : GlobalValue(PointerType::get(Ty), Value::FunctionVal, 0, 0, Linkage, name) {
  ParamAttrs = 0;
  CallingConvention = 0;
  BasicBlocks.setItemParent(this);
  BasicBlocks.setParent(this);
  ArgumentList.setItemParent(this);
  ArgumentList.setParent(this);
  SymTab = new ValueSymbolTable();

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
unsigned Function::getIntrinsicID(bool noAssert) const {
  const ValueName *ValName = this->getValueName();
  unsigned Len = ValName->getKeyLength();
  const char *Name = ValName->getKeyData();
  
  if (Len <= 5 || Name[4] != '.' || Name[0] != 'l' || Name[1] != 'l'
      || Name[2] != 'v' || Name[3] != 'm')
    return 0;  // All intrinsics start with 'llvm.'

  assert((Len != 5 || noAssert) && "'llvm.' is an invalid intrinsic name!");

#define GET_FUNCTION_RECOGNIZER
#include "llvm/Intrinsics.gen"
#undef GET_FUNCTION_RECOGNIZER
  assert(noAssert && "Invalid LLVM intrinsic name");
  return 0;
}

std::string Intrinsic::getName(ID id, const Type **Tys, unsigned numTys) { 
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  const char * const Table[] = {
    "not_intrinsic",
#define GET_INTRINSIC_NAME_TABLE
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_NAME_TABLE
  };
  if (numTys == 0)
    return Table[id];
  std::string Result(Table[id]);
  for (unsigned i = 0; i < numTys; ++i) 
    if (Tys[i])
      Result += "." + Tys[i]->getDescription();
  return Result;
}

const FunctionType *Intrinsic::getType(ID id, const Type **Tys, 
                                       uint32_t numTys) {
  const Type *ResultTy = NULL;
  std::vector<const Type*> ArgTys;
  bool IsVarArg = false;
  
#define GET_INTRINSIC_GENERATOR
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_GENERATOR

  return FunctionType::get(ResultTy, ArgTys, IsVarArg); 
}

Function *Intrinsic::getDeclaration(Module *M, ID id, const Type **Tys, 
                                    unsigned numTys) {
// There can never be multiple globals with the same name of different types,
// because intrinsics must be a specific type.
  return cast<Function>(M->getOrInsertFunction(getName(id, Tys, numTys), 
                                               getType(id, Tys, numTys)));
}

Value *IntrinsicInst::StripPointerCasts(Value *Ptr) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
    if (CE->getOpcode() == Instruction::BitCast) {
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

  if (BitCastInst *CI = dyn_cast<BitCastInst>(Ptr)) {
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
