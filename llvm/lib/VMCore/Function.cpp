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
#include "llvm/IntrinsicInst.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/StringPool.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
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
template class SymbolTableListTraits<Argument, Function>;
template class SymbolTableListTraits<BasicBlock, Function>;

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
  for (unsigned i = 0; i < limit && attrs[i].index <= Index; ++i)
    if (attrs[i].index == Index)
      return attrs[i].attrs;
  return ParamAttr::None;
}

std::string 
ParamAttrsList::getParamAttrsText(uint16_t Attrs) {
  std::string Result;
  if (Attrs & ParamAttr::ZExt)
    Result += "zeroext ";
  if (Attrs & ParamAttr::SExt)
    Result += "signext ";
  if (Attrs & ParamAttr::NoReturn)
    Result += "noreturn ";
  if (Attrs & ParamAttr::NoUnwind)
    Result += "nounwind ";
  if (Attrs & ParamAttr::InReg)
    Result += "inreg ";
  if (Attrs & ParamAttr::NoAlias)
    Result += "noalias ";
  if (Attrs & ParamAttr::StructRet)
    Result += "sret ";  
  if (Attrs & ParamAttr::ByVal)
    Result += "byval ";
  if (Attrs & ParamAttr::Nest)
    Result += "nest ";
  if (Attrs & ParamAttr::ReadNone)
    Result += "readnone ";
  if (Attrs & ParamAttr::ReadOnly)
    Result += "readonly ";
  return Result;
}

/// onlyInformative - Returns whether only informative attributes are set.
static inline bool onlyInformative(uint16_t attrs) {
  return !(attrs & ~ParamAttr::Informative);
}

bool
ParamAttrsList::areCompatible(const ParamAttrsList *A, const ParamAttrsList *B){
  if (A == B)
    return true;
  unsigned ASize = A ? A->size() : 0;
  unsigned BSize = B ? B->size() : 0;
  unsigned AIndex = 0;
  unsigned BIndex = 0;

  while (AIndex < ASize && BIndex < BSize) {
    uint16_t AIdx = A->getParamIndex(AIndex);
    uint16_t BIdx = B->getParamIndex(BIndex);
    uint16_t AAttrs = A->getParamAttrsAtIndex(AIndex);
    uint16_t BAttrs = B->getParamAttrsAtIndex(AIndex);

    if (AIdx < BIdx) {
      if (!onlyInformative(AAttrs))
        return false;
      ++AIndex;
    } else if (BIdx < AIdx) {
      if (!onlyInformative(BAttrs))
        return false;
      ++BIndex;
    } else {
      if (!onlyInformative(AAttrs ^ BAttrs))
        return false;
      ++AIndex;
      ++BIndex;
    }
  }
  for (; AIndex < ASize; ++AIndex)
    if (!onlyInformative(A->getParamAttrsAtIndex(AIndex)))
      return false;
  for (; BIndex < BSize; ++BIndex)
    if (!onlyInformative(B->getParamAttrsAtIndex(AIndex)))
      return false;
  return true;
}

void 
ParamAttrsList::Profile(FoldingSetNodeID &ID) const {
  for (unsigned i = 0; i < attrs.size(); ++i) {
    uint32_t val = uint32_t(attrs[i].attrs) << 16 | attrs[i].index;
    ID.AddInteger(val);
  }
}

static ManagedStatic<FoldingSet<ParamAttrsList> > ParamAttrsLists;

const ParamAttrsList *
ParamAttrsList::get(const ParamAttrsVector &attrVec) {
  // If there are no attributes then return a null ParamAttrsList pointer.
  if (attrVec.empty())
    return 0;

#ifndef NDEBUG
  for (unsigned i = 0, e = attrVec.size(); i < e; ++i) {
    assert(attrVec[i].attrs != ParamAttr::None
           && "Pointless parameter attribute!");
    assert((!i || attrVec[i-1].index < attrVec[i].index)
           && "Misordered ParamAttrsList!");
  }
#endif

  // Otherwise, build a key to look up the existing attributes.
  ParamAttrsList key(attrVec);
  FoldingSetNodeID ID;
  key.Profile(ID);
  void *InsertPos;
  ParamAttrsList* PAL = ParamAttrsLists->FindNodeOrInsertPos(ID, InsertPos);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PAL) {
    PAL = new ParamAttrsList(attrVec);
    ParamAttrsLists->InsertNode(PAL, InsertPos);
  }

  // Return the ParamAttrsList that we found or created.
  return PAL;
}

const ParamAttrsList *
ParamAttrsList::getModified(const ParamAttrsList *PAL,
                            const ParamAttrsVector &modVec) {
  if (modVec.empty())
    return PAL;

#ifndef NDEBUG
  for (unsigned i = 0, e = modVec.size(); i < e; ++i)
    assert((!i || modVec[i-1].index < modVec[i].index)
           && "Misordered ParamAttrsList!");
#endif

  if (!PAL) {
    // Strip any instances of ParamAttr::None from modVec before calling 'get'.
    ParamAttrsVector newVec;
    for (unsigned i = 0, e = modVec.size(); i < e; ++i)
      if (modVec[i].attrs != ParamAttr::None)
        newVec.push_back(modVec[i]);
    return get(newVec);
  }

  const ParamAttrsVector &oldVec = PAL->attrs;

  ParamAttrsVector newVec;
  unsigned oldI = 0;
  unsigned modI = 0;
  unsigned oldE = oldVec.size();
  unsigned modE = modVec.size();

  while (oldI < oldE && modI < modE) {
    uint16_t oldIndex = oldVec[oldI].index;
    uint16_t modIndex = modVec[modI].index;

    if (oldIndex < modIndex) {
      newVec.push_back(oldVec[oldI]);
      ++oldI;
    } else if (modIndex < oldIndex) {
      if (modVec[modI].attrs != ParamAttr::None)
        newVec.push_back(modVec[modI]);
      ++modI;
    } else {
      // Same index - overwrite or delete existing attributes.
      if (modVec[modI].attrs != ParamAttr::None)
        newVec.push_back(modVec[modI]);
      ++oldI;
      ++modI;
    }
  }

  for (; oldI < oldE; ++oldI)
    newVec.push_back(oldVec[oldI]);
  for (; modI < modE; ++modI)
    if (modVec[modI].attrs != ParamAttr::None)
      newVec.push_back(modVec[modI]);

  return get(newVec);
}

ParamAttrsList::~ParamAttrsList() {
  ParamAttrsLists->RemoveNode(this);
}

//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//

Function::Function(const FunctionType *Ty, LinkageTypes Linkage,
                   const std::string &name, Module *ParentModule)
  : GlobalValue(PointerType::get(Ty), Value::FunctionVal, 0, 0, Linkage, name),
    ParamAttrs(0) {
  SymTab = new ValueSymbolTable();

  assert((getReturnType()->isFirstClassType() ||getReturnType() == Type::VoidTy)
         && "LLVM functions cannot return aggregate values!");

  // If the function has arguments, mark them as lazily built.
  if (Ty->getNumParams())
    SubclassData = 1;   // Set the "has lazy arguments" bit.
  
  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getFunctionList().push_back(this);
}

Function::~Function() {
  dropAllReferences();    // After this it is safe to delete instructions.

  // Delete all of the method arguments and unlink from symbol table...
  ArgumentList.clear();
  delete SymTab;

  // Drop our reference to the parameter attributes, if any.
  if (ParamAttrs)
    ParamAttrs->dropRef();
  
  // Remove the function from the on-the-side collector table.
  clearCollector();
}

void Function::BuildLazyArguments() const {
  // Create the arguments vector, all arguments start out unnamed.
  const FunctionType *FT = getFunctionType();
  for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
    assert(FT->getParamType(i) != Type::VoidTy &&
           "Cannot have void typed arguments!");
    ArgumentList.push_back(new Argument(FT->getParamType(i)));
  }
  
  // Clear the lazy arguments bit.
  const_cast<Function*>(this)->SubclassData &= ~1;
}

size_t Function::arg_size() const {
  return getFunctionType()->getNumParams();
}
bool Function::arg_empty() const {
  return getFunctionType()->getNumParams() == 0;
}

void Function::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void Function::setParamAttrs(const ParamAttrsList *attrs) {
  // Avoid deleting the ParamAttrsList if they are setting the
  // attributes to the same list.
  if (ParamAttrs == attrs)
    return;

  // Drop reference on the old ParamAttrsList
  if (ParamAttrs)
    ParamAttrs->dropRef();

  // Add reference to the new ParamAttrsList
  if (attrs)
    attrs->addRef();

  // Set the new ParamAttrsList.
  ParamAttrs = attrs; 
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

// Maintain the collector name for each function in an on-the-side table. This
// saves allocating an additional word in Function for programs which do not use
// GC (i.e., most programs) at the cost of increased overhead for clients which
// do use GC.
static DenseMap<const Function*,PooledStringPtr> *CollectorNames;
static StringPool *CollectorNamePool;

bool Function::hasCollector() const {
  return CollectorNames && CollectorNames->count(this);
}

const char *Function::getCollector() const {
  assert(hasCollector() && "Function has no collector");
  return *(*CollectorNames)[this];
}

void Function::setCollector(const char *Str) {
  if (!CollectorNamePool)
    CollectorNamePool = new StringPool();
  if (!CollectorNames)
    CollectorNames = new DenseMap<const Function*,PooledStringPtr>();
  (*CollectorNames)[this] = CollectorNamePool->intern(Str);
}

void Function::clearCollector() {
  if (CollectorNames) {
    CollectorNames->erase(this);
    if (CollectorNames->empty()) {
      delete CollectorNames;
      CollectorNames = 0;
      if (CollectorNamePool->empty()) {
        delete CollectorNamePool;
        CollectorNamePool = 0;
      }
    }
  }
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
  if (!ValName)
    return 0;
  unsigned Len = ValName->getKeyLength();
  const char *Name = ValName->getKeyData();
  
  if (Len < 5 || Name[4] != '.' || Name[0] != 'l' || Name[1] != 'l'
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
      Result += "." + MVT::getValueTypeString(MVT::getValueType(Tys[i]));
  return Result;
}

const FunctionType *Intrinsic::getType(ID id, const Type **Tys, 
                                       unsigned numTys) {
  const Type *ResultTy = NULL;
  std::vector<const Type*> ArgTys;
  bool IsVarArg = false;
  
#define GET_INTRINSIC_GENERATOR
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_GENERATOR

  return FunctionType::get(ResultTy, ArgTys, IsVarArg); 
}

const ParamAttrsList *Intrinsic::getParamAttrs(ID id) {
  static const ParamAttrsList *IntrinsicAttributes[Intrinsic::num_intrinsics];

  if (IntrinsicAttributes[id])
    return IntrinsicAttributes[id];

  ParamAttrsVector Attrs;
  uint16_t Attr = ParamAttr::None;

#define GET_INTRINSIC_ATTRIBUTES
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_ATTRIBUTES

  // Intrinsics cannot throw exceptions.
  Attr |= ParamAttr::NoUnwind;

  Attrs.push_back(ParamAttrsWithIndex::get(0, Attr));
  IntrinsicAttributes[id] = ParamAttrsList::get(Attrs);
  return IntrinsicAttributes[id];
}

Function *Intrinsic::getDeclaration(Module *M, ID id, const Type **Tys, 
                                    unsigned numTys) {
  // There can never be multiple globals with the same name of different types,
  // because intrinsics must be a specific type.
  Function *F =
    cast<Function>(M->getOrInsertFunction(getName(id, Tys, numTys),
                                          getType(id, Tys, numTys)));
  F->setParamAttrs(getParamAttrs(id));
  return F;
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
    if (GEP->hasAllZeroIndices())
      return StripPointerCasts(GEP->getOperand(0));
  }
  return Ptr;
}

// vim: sw=2 ai
