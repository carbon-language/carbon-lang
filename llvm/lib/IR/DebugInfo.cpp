//===--- DebugInfo.cpp - Debug Information Helper Classes -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the helper classes used to build and interpret debug
// information in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::dwarf;

//===----------------------------------------------------------------------===//
// DIDescriptor
//===----------------------------------------------------------------------===//

unsigned DIDescriptor::getFlag(StringRef Flag) {
  return StringSwitch<unsigned>(Flag)
#define HANDLE_DI_FLAG(ID, NAME) .Case("DIFlag" #NAME, Flag##NAME)
#include "llvm/IR/DebugInfoFlags.def"
      .Default(0);
}

const char *DIDescriptor::getFlagString(unsigned Flag) {
  switch (Flag) {
  default:
    return "";
#define HANDLE_DI_FLAG(ID, NAME)                                               \
  case Flag##NAME:                                                             \
    return "DIFlag" #NAME;
#include "llvm/IR/DebugInfoFlags.def"
  }
}

unsigned DIDescriptor::splitFlags(unsigned Flags,
                                  SmallVectorImpl<unsigned> &SplitFlags) {
  // Accessibility flags need to be specially handled, since they're packed
  // together.
  if (unsigned A = Flags & FlagAccessibility) {
    if (A == FlagPrivate)
      SplitFlags.push_back(FlagPrivate);
    else if (A == FlagProtected)
      SplitFlags.push_back(FlagProtected);
    else
      SplitFlags.push_back(FlagPublic);
    Flags &= ~A;
  }

#define HANDLE_DI_FLAG(ID, NAME)                                               \
  if (unsigned Bit = Flags & ID) {                                             \
    SplitFlags.push_back(Bit);                                                 \
    Flags &= ~Bit;                                                             \
  }
#include "llvm/IR/DebugInfoFlags.def"

  return Flags;
}

bool DIDescriptor::Verify() const {
  return DbgNode &&
         (DIDerivedType(DbgNode).Verify() ||
          DICompositeType(DbgNode).Verify() || DIBasicType(DbgNode).Verify() ||
          DIVariable(DbgNode).Verify() || DISubprogram(DbgNode).Verify() ||
          DIGlobalVariable(DbgNode).Verify() || DIFile(DbgNode).Verify() ||
          DICompileUnit(DbgNode).Verify() || DINameSpace(DbgNode).Verify() ||
          DILexicalBlock(DbgNode).Verify() ||
          DILexicalBlockFile(DbgNode).Verify() ||
          DISubrange(DbgNode).Verify() || DIEnumerator(DbgNode).Verify() ||
          DIObjCProperty(DbgNode).Verify() ||
          DITemplateTypeParameter(DbgNode).Verify() ||
          DITemplateValueParameter(DbgNode).Verify() ||
          DIImportedEntity(DbgNode).Verify());
}

static Metadata *getField(const MDNode *DbgNode, unsigned Elt) {
  if (!DbgNode || Elt >= DbgNode->getNumOperands())
    return nullptr;
  return DbgNode->getOperand(Elt);
}

static MDNode *getNodeField(const MDNode *DbgNode, unsigned Elt) {
  return dyn_cast_or_null<MDNode>(getField(DbgNode, Elt));
}

static StringRef getStringField(const MDNode *DbgNode, unsigned Elt) {
  if (MDString *MDS = dyn_cast_or_null<MDString>(getField(DbgNode, Elt)))
    return MDS->getString();
  return StringRef();
}

StringRef DIDescriptor::getStringField(unsigned Elt) const {
  return ::getStringField(DbgNode, Elt);
}

uint64_t DIDescriptor::getUInt64Field(unsigned Elt) const {
  if (auto *C = getConstantField(Elt))
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return CI->getZExtValue();

  return 0;
}

int64_t DIDescriptor::getInt64Field(unsigned Elt) const {
  if (auto *C = getConstantField(Elt))
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return CI->getZExtValue();

  return 0;
}

DIDescriptor DIDescriptor::getDescriptorField(unsigned Elt) const {
  MDNode *Field = getNodeField(DbgNode, Elt);
  return DIDescriptor(Field);
}

GlobalVariable *DIDescriptor::getGlobalVariableField(unsigned Elt) const {
  return dyn_cast_or_null<GlobalVariable>(getConstantField(Elt));
}

Constant *DIDescriptor::getConstantField(unsigned Elt) const {
  if (!DbgNode)
    return nullptr;

  if (Elt < DbgNode->getNumOperands())
    if (auto *C =
            dyn_cast_or_null<ConstantAsMetadata>(DbgNode->getOperand(Elt)))
      return C->getValue();
  return nullptr;
}

Function *DIDescriptor::getFunctionField(unsigned Elt) const {
  return dyn_cast_or_null<Function>(getConstantField(Elt));
}

/// \brief Return the size reported by the variable's type.
unsigned DIVariable::getSizeInBits(const DITypeIdentifierMap &Map) {
  DIType Ty = getType().resolve(Map);
  // Follow derived types until we reach a type that
  // reports back a size.
  while (Ty.isDerivedType() && !Ty.getSizeInBits()) {
    DIDerivedType DT(&*Ty);
    Ty = DT.getTypeDerivedFrom().resolve(Map);
  }
  assert(Ty.getSizeInBits() && "type with size 0");
  return Ty.getSizeInBits();
}

bool DIExpression::isBitPiece() const {
  unsigned N = getNumElements();
  return N >=3 && getElement(N-3) == dwarf::DW_OP_bit_piece;
}

uint64_t DIExpression::getBitPieceOffset() const {
  assert(isBitPiece() && "not a piece");
  return getElement(getNumElements()-2);
}

uint64_t DIExpression::getBitPieceSize() const {
  assert(isBitPiece() && "not a piece");
  return getElement(getNumElements()-1);
}

DIExpression::iterator DIExpression::Operand::getNext() const {
  iterator it(I);
  return ++it;
}

//===----------------------------------------------------------------------===//
// Simple Descriptor Constructors and other Methods
//===----------------------------------------------------------------------===//

void DIDescriptor::replaceAllUsesWith(LLVMContext &, DIDescriptor D) {
  assert(DbgNode && "Trying to replace an unverified type!");
  assert(DbgNode->isTemporary() && "Expected temporary node");
  TempMDNode Temp(get());

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  if (Temp.get() == D.get()) {
    DbgNode = MDNode::replaceWithUniqued(std::move(Temp));
    return;
  }

  Temp->replaceAllUsesWith(D.get());
  DbgNode = D.get();
}

void DIDescriptor::replaceAllUsesWith(MDNode *D) {
  assert(DbgNode && "Trying to replace an unverified type!");
  assert(DbgNode != D && "This replacement should always happen");
  assert(DbgNode->isTemporary() && "Expected temporary node");
  TempMDNode Node(get());
  Node->replaceAllUsesWith(D);
}

bool DICompileUnit::Verify() const {
  if (!isCompileUnit())
    return false;

  // Don't bother verifying the compilation directory or producer string
  // as those could be empty.
  return !getFilename().empty();
}

bool DIObjCProperty::Verify() const { return isObjCProperty(); }

/// \brief Check if a value can be a reference to a type.
static bool isTypeRef(const Metadata *MD) {
  if (!MD)
    return true;
  if (auto *S = dyn_cast<MDString>(MD))
    return !S->getString().empty();
  return isa<MDType>(MD);
}

/// \brief Check if a value can be a ScopeRef.
static bool isScopeRef(const Metadata *MD) {
  if (!MD)
    return true;
  if (auto *S = dyn_cast<MDString>(MD))
    return !S->getString().empty();
  return isa<MDScope>(MD);
}

#ifndef NDEBUG
/// \brief Check if a value can be a DescriptorRef.
static bool isDescriptorRef(const Metadata *MD) {
  if (!MD)
    return true;
  if (auto *S = dyn_cast<MDString>(MD))
    return !S->getString().empty();
  return isa<MDNode>(MD);
}
#endif

bool DIType::Verify() const {
  auto *N = getRaw();
  if (!N)
    return false;
  if (!isScopeRef(N->getScope()))
    return false;

  // DIType is abstract, it should be a BasicType, a DerivedType or
  // a CompositeType.
  if (isBasicType())
    return DIBasicType(DbgNode).Verify();

  // FIXME: Sink this into the various subclass verifies.
  if (getFilename().empty()) {
    // Check whether the filename is allowed to be empty.
    uint16_t Tag = getTag();
    if (Tag != dwarf::DW_TAG_const_type && Tag != dwarf::DW_TAG_volatile_type &&
        Tag != dwarf::DW_TAG_pointer_type &&
        Tag != dwarf::DW_TAG_ptr_to_member_type &&
        Tag != dwarf::DW_TAG_reference_type &&
        Tag != dwarf::DW_TAG_rvalue_reference_type &&
        Tag != dwarf::DW_TAG_restrict_type && Tag != dwarf::DW_TAG_array_type &&
        Tag != dwarf::DW_TAG_enumeration_type &&
        Tag != dwarf::DW_TAG_subroutine_type &&
        Tag != dwarf::DW_TAG_inheritance && Tag != dwarf::DW_TAG_friend)
      return false;
  }

  if (isCompositeType())
    return DICompositeType(DbgNode).Verify();
  if (isDerivedType())
    return DIDerivedType(DbgNode).Verify();
  return false;
}

bool DIBasicType::Verify() const { return getRaw(); }

bool DIDerivedType::Verify() const {
  auto *N = getRaw();
  if (!N)
    return false;
  if (getTag() == dwarf::DW_TAG_ptr_to_member_type) {
    auto *D = dyn_cast<MDDerivedType>(N);
    if (!D)
      return false;
    if (!isTypeRef(D->getExtraData()))
      return false;
  }
  return isTypeRef(N->getBaseType());
}

bool DICompositeType::Verify() const {
  auto *N = getRaw();
  return N && isTypeRef(N->getBaseType()) && isTypeRef(N->getVTableHolder()) &&
         !(isLValueReference() && isRValueReference());
}

bool DISubprogram::Verify() const {
  auto *N = getRaw();
  if (!N)
    return false;

  if (!isScopeRef(N->getScope()))
    return false;

  if (auto *Op = N->getType())
    if (!isa<MDNode>(Op))
      return false;

  if (!isTypeRef(getContainingType()))
    return false;

  if (isLValueReference() && isRValueReference())
    return false;

  // If a DISubprogram has an llvm::Function*, then scope chains from all
  // instructions within the function should lead to this DISubprogram.
  if (auto *F = getFunction()) {
    for (auto &BB : *F) {
      for (auto &I : BB) {
        DebugLoc DL = I.getDebugLoc();
        if (DL.isUnknown())
          continue;

        MDNode *Scope = nullptr;
        MDNode *IA = nullptr;
        // walk the inlined-at scopes
        while ((IA = DL.getInlinedAt()))
          DL = DebugLoc::getFromDILocation(IA);
        DL.getScopeAndInlinedAt(Scope, IA);
        if (!Scope)
          return false;
        assert(!IA);
        while (!DIDescriptor(Scope).isSubprogram()) {
          DILexicalBlockFile D(Scope);
          Scope = D.isLexicalBlockFile()
                      ? D.getScope()
                      : DebugLoc::getFromDILexicalBlock(Scope).getScope();
          if (!Scope)
            return false;
        }
        if (!DISubprogram(Scope).describes(F))
          return false;
      }
    }
  }

  return true;
}

bool DIGlobalVariable::Verify() const {
  auto *N = getRaw();

  if (!N)
    return false;

  if (N->getDisplayName().empty())
    return false;

  if (auto *Op = N->getScope())
    if (!isa<MDNode>(Op))
      return false;

  if (auto *Op = N->getStaticDataMemberDeclaration())
    if (!isa<MDNode>(Op))
      return false;

  return isTypeRef(N->getType());
}

bool DIVariable::Verify() const {
  auto *N = getRaw();

  if (!N)
    return false;

  if (auto *Op = N->getScope())
    if (!isa<MDNode>(Op))
      return false;

  return isTypeRef(N->getType());
}

bool DILocation::Verify() const { return getRaw(); }
bool DINameSpace::Verify() const { return getRaw(); }
bool DIFile::Verify() const { return getRaw(); }
bool DIEnumerator::Verify() const { return getRaw(); }
bool DISubrange::Verify() const { return getRaw(); }
bool DILexicalBlock::Verify() const { return getRaw(); }
bool DILexicalBlockFile::Verify() const { return getRaw(); }
bool DITemplateTypeParameter::Verify() const { return getRaw(); }
bool DITemplateValueParameter::Verify() const { return getRaw(); }
bool DIImportedEntity::Verify() const { return getRaw(); }

void DICompositeType::setArraysHelper(MDNode *Elements, MDNode *TParams) {
  TypedTrackingMDRef<MDCompositeTypeBase> N(getRaw());
  if (Elements)
    N->replaceElements(cast<MDTuple>(Elements));
  if (TParams)
    N->replaceTemplateParams(cast<MDTuple>(TParams));
  DbgNode = N;
}

DIScopeRef DIScope::getRef() const {
  if (!isCompositeType())
    return DIScopeRef(*this);
  DICompositeType DTy(DbgNode);
  if (!DTy.getIdentifier())
    return DIScopeRef(*this);
  return DIScopeRef(DTy.getIdentifier());
}

void DICompositeType::setContainingType(DICompositeType ContainingType) {
  TypedTrackingMDRef<MDCompositeTypeBase> N(getRaw());
  N->replaceVTableHolder(ContainingType.getRef());
  DbgNode = N;
}

bool DIVariable::isInlinedFnArgument(const Function *CurFn) {
  assert(CurFn && "Invalid function");
  if (!getContext().isSubprogram())
    return false;
  // This variable is not inlined function argument if its scope
  // does not describe current function.
  return !DISubprogram(getContext()).describes(CurFn);
}

Function *DISubprogram::getFunction() const {
  if (auto *N = getRaw())
    if (auto *C = dyn_cast_or_null<ConstantAsMetadata>(N->getFunction()))
      return dyn_cast<Function>(C->getValue());
  return nullptr;
}

bool DISubprogram::describes(const Function *F) {
  assert(F && "Invalid function");
  if (F == getFunction())
    return true;
  StringRef Name = getLinkageName();
  if (Name.empty())
    Name = getName();
  if (F->getName() == Name)
    return true;
  return false;
}

GlobalVariable *DIGlobalVariable::getGlobal() const {
  return dyn_cast_or_null<GlobalVariable>(getConstant());
}

DIScopeRef DIScope::getContext() const {

  if (isType())
    return DIType(DbgNode).getContext();

  if (isSubprogram())
    return DIScopeRef(DISubprogram(DbgNode).getContext());

  if (isLexicalBlock())
    return DIScopeRef(DILexicalBlock(DbgNode).getContext());

  if (isLexicalBlockFile())
    return DIScopeRef(DILexicalBlockFile(DbgNode).getContext());

  if (isNameSpace())
    return DIScopeRef(DINameSpace(DbgNode).getContext());

  assert((isFile() || isCompileUnit()) && "Unhandled type of scope.");
  return DIScopeRef(nullptr);
}

StringRef DIScope::getName() const {
  if (isType())
    return DIType(DbgNode).getName();
  if (isSubprogram())
    return DISubprogram(DbgNode).getName();
  if (isNameSpace())
    return DINameSpace(DbgNode).getName();
  assert((isLexicalBlock() || isLexicalBlockFile() || isFile() ||
          isCompileUnit()) &&
         "Unhandled type of scope.");
  return StringRef();
}

StringRef DIScope::getFilename() const {
  if (auto *N = getRaw())
    return ::getStringField(dyn_cast_or_null<MDNode>(N->getFile()), 0);
  return "";
}

StringRef DIScope::getDirectory() const {
  if (auto *N = getRaw())
    return ::getStringField(dyn_cast_or_null<MDNode>(N->getFile()), 1);
  return "";
}

void DICompileUnit::replaceSubprograms(DIArray Subprograms) {
  assert(Verify() && "Expected compile unit");
  getRaw()->replaceSubprograms(cast_or_null<MDTuple>(Subprograms.get()));
}

void DICompileUnit::replaceGlobalVariables(DIArray GlobalVariables) {
  assert(Verify() && "Expected compile unit");
  getRaw()->replaceGlobalVariables(
      cast_or_null<MDTuple>(GlobalVariables.get()));
}

DILocation DILocation::copyWithNewScope(LLVMContext &Ctx,
                                        DILexicalBlockFile NewScope) {
  assert(Verify());
  assert(NewScope && "Expected valid scope");

  const auto *Old = cast<MDLocation>(DbgNode);
  return DILocation(MDLocation::get(Ctx, Old->getLine(), Old->getColumn(),
                                    NewScope, Old->getInlinedAt()));
}

unsigned DILocation::computeNewDiscriminator(LLVMContext &Ctx) {
  std::pair<const char *, unsigned> Key(getFilename().data(), getLineNumber());
  return ++Ctx.pImpl->DiscriminatorTable[Key];
}

DIVariable llvm::createInlinedVariable(MDNode *DV, MDNode *InlinedScope,
                                       LLVMContext &VMContext) {
  assert(DIVariable(DV).Verify() && "Expected a DIVariable");
  return cast<MDLocalVariable>(DV)
      ->withInline(cast_or_null<MDLocation>(InlinedScope));
}

DIVariable llvm::cleanseInlinedVariable(MDNode *DV, LLVMContext &VMContext) {
  assert(DIVariable(DV).Verify() && "Expected a DIVariable");
  return cast<MDLocalVariable>(DV)->withoutInline();
}

DISubprogram llvm::getDISubprogram(const MDNode *Scope) {
  DIDescriptor D(Scope);
  if (D.isSubprogram())
    return DISubprogram(Scope);

  if (D.isLexicalBlockFile())
    return getDISubprogram(DILexicalBlockFile(Scope).getContext());

  if (D.isLexicalBlock())
    return getDISubprogram(DILexicalBlock(Scope).getContext());

  return DISubprogram();
}

DISubprogram llvm::getDISubprogram(const Function *F) {
  // We look for the first instr that has a debug annotation leading back to F.
  for (auto &BB : *F) {
    auto Inst = std::find_if(BB.begin(), BB.end(), [](const Instruction &Inst) {
      return !Inst.getDebugLoc().isUnknown();
    });
    if (Inst == BB.end())
      continue;
    DebugLoc DLoc = Inst->getDebugLoc();
    const MDNode *Scope = DLoc.getScopeNode();
    DISubprogram Subprogram = getDISubprogram(Scope);
    return Subprogram.describes(F) ? Subprogram : DISubprogram();
  }

  return DISubprogram();
}

DICompositeType llvm::getDICompositeType(DIType T) {
  if (T.isCompositeType())
    return DICompositeType(T);

  if (T.isDerivedType()) {
    // This function is currently used by dragonegg and dragonegg does
    // not generate identifier for types, so using an empty map to resolve
    // DerivedFrom should be fine.
    DITypeIdentifierMap EmptyMap;
    return getDICompositeType(
        DIDerivedType(T).getTypeDerivedFrom().resolve(EmptyMap));
  }

  return DICompositeType();
}

DITypeIdentifierMap
llvm::generateDITypeIdentifierMap(const NamedMDNode *CU_Nodes) {
  DITypeIdentifierMap Map;
  for (unsigned CUi = 0, CUe = CU_Nodes->getNumOperands(); CUi != CUe; ++CUi) {
    DICompileUnit CU(CU_Nodes->getOperand(CUi));
    DIArray Retain = CU.getRetainedTypes();
    for (unsigned Ti = 0, Te = Retain.getNumElements(); Ti != Te; ++Ti) {
      if (!Retain.getElement(Ti).isCompositeType())
        continue;
      DICompositeType Ty(Retain.getElement(Ti));
      if (MDString *TypeId = Ty.getIdentifier()) {
        // Definition has priority over declaration.
        // Try to insert (TypeId, Ty) to Map.
        std::pair<DITypeIdentifierMap::iterator, bool> P =
            Map.insert(std::make_pair(TypeId, Ty));
        // If TypeId already exists in Map and this is a definition, replace
        // whatever we had (declaration or definition) with the definition.
        if (!P.second && !Ty.isForwardDecl())
          P.first->second = Ty;
      }
    }
  }
  return Map;
}

//===----------------------------------------------------------------------===//
// DebugInfoFinder implementations.
//===----------------------------------------------------------------------===//

void DebugInfoFinder::reset() {
  CUs.clear();
  SPs.clear();
  GVs.clear();
  TYs.clear();
  Scopes.clear();
  NodesSeen.clear();
  TypeIdentifierMap.clear();
  TypeMapInitialized = false;
}

void DebugInfoFinder::InitializeTypeMap(const Module &M) {
  if (!TypeMapInitialized)
    if (NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu")) {
      TypeIdentifierMap = generateDITypeIdentifierMap(CU_Nodes);
      TypeMapInitialized = true;
    }
}

void DebugInfoFinder::processModule(const Module &M) {
  InitializeTypeMap(M);
  if (NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu")) {
    for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
      DICompileUnit CU(CU_Nodes->getOperand(i));
      addCompileUnit(CU);
      DIArray GVs = CU.getGlobalVariables();
      for (unsigned i = 0, e = GVs.getNumElements(); i != e; ++i) {
        DIGlobalVariable DIG(GVs.getElement(i));
        if (addGlobalVariable(DIG)) {
          processScope(DIG.getContext());
          processType(DIG.getType().resolve(TypeIdentifierMap));
        }
      }
      DIArray SPs = CU.getSubprograms();
      for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i)
        processSubprogram(DISubprogram(SPs.getElement(i)));
      DIArray EnumTypes = CU.getEnumTypes();
      for (unsigned i = 0, e = EnumTypes.getNumElements(); i != e; ++i)
        processType(DIType(EnumTypes.getElement(i)));
      DIArray RetainedTypes = CU.getRetainedTypes();
      for (unsigned i = 0, e = RetainedTypes.getNumElements(); i != e; ++i)
        processType(DIType(RetainedTypes.getElement(i)));
      DIArray Imports = CU.getImportedEntities();
      for (unsigned i = 0, e = Imports.getNumElements(); i != e; ++i) {
        DIImportedEntity Import = DIImportedEntity(Imports.getElement(i));
        if (!Import)
          continue;
        DIDescriptor Entity = Import.getEntity().resolve(TypeIdentifierMap);
        if (Entity.isType())
          processType(DIType(Entity));
        else if (Entity.isSubprogram())
          processSubprogram(DISubprogram(Entity));
        else if (Entity.isNameSpace())
          processScope(DINameSpace(Entity).getContext());
      }
    }
  }
}

void DebugInfoFinder::processLocation(const Module &M, DILocation Loc) {
  if (!Loc)
    return;
  InitializeTypeMap(M);
  processScope(Loc.getScope());
  processLocation(M, Loc.getOrigLocation());
}

void DebugInfoFinder::processType(DIType DT) {
  if (!addType(DT))
    return;
  processScope(DT.getContext().resolve(TypeIdentifierMap));
  if (DT.isCompositeType()) {
    DICompositeType DCT(DT);
    processType(DCT.getTypeDerivedFrom().resolve(TypeIdentifierMap));
    if (DT.isSubroutineType()) {
      DITypeArray DTA = DISubroutineType(DT).getTypeArray();
      for (unsigned i = 0, e = DTA.getNumElements(); i != e; ++i)
        processType(DTA.getElement(i).resolve(TypeIdentifierMap));
      return;
    }
    DIArray DA = DCT.getElements();
    for (unsigned i = 0, e = DA.getNumElements(); i != e; ++i) {
      DIDescriptor D = DA.getElement(i);
      if (D.isType())
        processType(DIType(D));
      else if (D.isSubprogram())
        processSubprogram(DISubprogram(D));
    }
  } else if (DT.isDerivedType()) {
    DIDerivedType DDT(DT);
    processType(DDT.getTypeDerivedFrom().resolve(TypeIdentifierMap));
  }
}

void DebugInfoFinder::processScope(DIScope Scope) {
  if (Scope.isType()) {
    DIType Ty(Scope);
    processType(Ty);
    return;
  }
  if (Scope.isCompileUnit()) {
    addCompileUnit(DICompileUnit(Scope));
    return;
  }
  if (Scope.isSubprogram()) {
    processSubprogram(DISubprogram(Scope));
    return;
  }
  if (!addScope(Scope))
    return;
  if (Scope.isLexicalBlock()) {
    DILexicalBlock LB(Scope);
    processScope(LB.getContext());
  } else if (Scope.isLexicalBlockFile()) {
    DILexicalBlockFile LBF = DILexicalBlockFile(Scope);
    processScope(LBF.getScope());
  } else if (Scope.isNameSpace()) {
    DINameSpace NS(Scope);
    processScope(NS.getContext());
  }
}

void DebugInfoFinder::processSubprogram(DISubprogram SP) {
  if (!addSubprogram(SP))
    return;
  processScope(SP.getContext().resolve(TypeIdentifierMap));
  processType(SP.getType());
  DIArray TParams = SP.getTemplateParams();
  for (unsigned I = 0, E = TParams.getNumElements(); I != E; ++I) {
    DIDescriptor Element = TParams.getElement(I);
    if (Element.isTemplateTypeParameter()) {
      DITemplateTypeParameter TType(Element);
      processType(TType.getType().resolve(TypeIdentifierMap));
    } else if (Element.isTemplateValueParameter()) {
      DITemplateValueParameter TVal(Element);
      processType(TVal.getType().resolve(TypeIdentifierMap));
    }
  }
}

void DebugInfoFinder::processDeclare(const Module &M,
                                     const DbgDeclareInst *DDI) {
  MDNode *N = dyn_cast<MDNode>(DDI->getVariable());
  if (!N)
    return;
  InitializeTypeMap(M);

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV).second)
    return;
  processScope(DIVariable(N).getContext());
  processType(DIVariable(N).getType().resolve(TypeIdentifierMap));
}

void DebugInfoFinder::processValue(const Module &M, const DbgValueInst *DVI) {
  MDNode *N = dyn_cast<MDNode>(DVI->getVariable());
  if (!N)
    return;
  InitializeTypeMap(M);

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV).second)
    return;
  processScope(DIVariable(N).getContext());
  processType(DIVariable(N).getType().resolve(TypeIdentifierMap));
}

bool DebugInfoFinder::addType(DIType DT) {
  if (!DT)
    return false;

  if (!NodesSeen.insert(DT).second)
    return false;

  TYs.push_back(DT);
  return true;
}

bool DebugInfoFinder::addCompileUnit(DICompileUnit CU) {
  if (!CU)
    return false;
  if (!NodesSeen.insert(CU).second)
    return false;

  CUs.push_back(CU);
  return true;
}

bool DebugInfoFinder::addGlobalVariable(DIGlobalVariable DIG) {
  if (!DIG)
    return false;

  if (!NodesSeen.insert(DIG).second)
    return false;

  GVs.push_back(DIG);
  return true;
}

bool DebugInfoFinder::addSubprogram(DISubprogram SP) {
  if (!SP)
    return false;

  if (!NodesSeen.insert(SP).second)
    return false;

  SPs.push_back(SP);
  return true;
}

bool DebugInfoFinder::addScope(DIScope Scope) {
  if (!Scope)
    return false;
  // FIXME: Ocaml binding generates a scope with no content, we treat it
  // as null for now.
  if (Scope->getNumOperands() == 0)
    return false;
  if (!NodesSeen.insert(Scope).second)
    return false;
  Scopes.push_back(Scope);
  return true;
}

//===----------------------------------------------------------------------===//
// DIDescriptor: dump routines for all descriptors.
//===----------------------------------------------------------------------===//

void DIDescriptor::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

void DIDescriptor::print(raw_ostream &OS) const {
  if (!get())
    return;
  get()->print(OS);
}

static void printDebugLoc(DebugLoc DL, raw_ostream &CommentOS,
                          const LLVMContext &Ctx) {
  if (!DL.isUnknown()) { // Print source line info.
    DIScope Scope(DL.getScope(Ctx));
    assert(Scope.isScope() && "Scope of a DebugLoc should be a DIScope.");
    // Omit the directory, because it's likely to be long and uninteresting.
    CommentOS << Scope.getFilename();
    CommentOS << ':' << DL.getLine();
    if (DL.getCol() != 0)
      CommentOS << ':' << DL.getCol();
    DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(DL.getInlinedAt(Ctx));
    if (!InlinedAtDL.isUnknown()) {
      CommentOS << " @[ ";
      printDebugLoc(InlinedAtDL, CommentOS, Ctx);
      CommentOS << " ]";
    }
  }
}

void DIVariable::printExtendedName(raw_ostream &OS) const {
  const LLVMContext &Ctx = DbgNode->getContext();
  StringRef Res = getName();
  if (!Res.empty())
    OS << Res << "," << getLineNumber();
  if (MDNode *InlinedAt = getInlinedAt()) {
    DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(InlinedAt);
    if (!InlinedAtDL.isUnknown()) {
      OS << " @[";
      printDebugLoc(InlinedAtDL, OS, Ctx);
      OS << "]";
    }
  }
}

template <> DIRef<DIDescriptor>::DIRef(const Metadata *V) : Val(V) {
  assert(isDescriptorRef(V) &&
         "DIDescriptorRef should be a MDString or MDNode");
}
template <> DIRef<DIScope>::DIRef(const Metadata *V) : Val(V) {
  assert(isScopeRef(V) && "DIScopeRef should be a MDString or MDNode");
}
template <> DIRef<DIType>::DIRef(const Metadata *V) : Val(V) {
  assert(isTypeRef(V) && "DITypeRef should be a MDString or MDNode");
}

template <>
DIDescriptorRef DIDescriptor::getFieldAs<DIDescriptorRef>(unsigned Elt) const {
  return DIDescriptorRef(cast_or_null<Metadata>(getField(DbgNode, Elt)));
}
template <>
DIScopeRef DIDescriptor::getFieldAs<DIScopeRef>(unsigned Elt) const {
  return DIScopeRef(cast_or_null<Metadata>(getField(DbgNode, Elt)));
}
template <> DITypeRef DIDescriptor::getFieldAs<DITypeRef>(unsigned Elt) const {
  return DITypeRef(cast_or_null<Metadata>(getField(DbgNode, Elt)));
}

bool llvm::StripDebugInfo(Module &M) {
  bool Changed = false;

  // Remove all of the calls to the debugger intrinsics, and remove them from
  // the module.
  if (Function *Declare = M.getFunction("llvm.dbg.declare")) {
    while (!Declare->use_empty()) {
      CallInst *CI = cast<CallInst>(Declare->user_back());
      CI->eraseFromParent();
    }
    Declare->eraseFromParent();
    Changed = true;
  }

  if (Function *DbgVal = M.getFunction("llvm.dbg.value")) {
    while (!DbgVal->use_empty()) {
      CallInst *CI = cast<CallInst>(DbgVal->user_back());
      CI->eraseFromParent();
    }
    DbgVal->eraseFromParent();
    Changed = true;
  }

  for (Module::named_metadata_iterator NMI = M.named_metadata_begin(),
         NME = M.named_metadata_end(); NMI != NME;) {
    NamedMDNode *NMD = NMI;
    ++NMI;
    if (NMD->getName().startswith("llvm.dbg.")) {
      NMD->eraseFromParent();
      Changed = true;
    }
  }

  for (Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
    for (Function::iterator FI = MI->begin(), FE = MI->end(); FI != FE;
         ++FI)
      for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE;
           ++BI) {
        if (!BI->getDebugLoc().isUnknown()) {
          Changed = true;
          BI->setDebugLoc(DebugLoc());
        }
      }

  return Changed;
}

unsigned llvm::getDebugMetadataVersionFromModule(const Module &M) {
  if (auto *Val = mdconst::dyn_extract_or_null<ConstantInt>(
          M.getModuleFlag("Debug Info Version")))
    return Val->getZExtValue();
  return 0;
}

llvm::DenseMap<const llvm::Function *, llvm::DISubprogram>
llvm::makeSubprogramMap(const Module &M) {
  DenseMap<const Function *, DISubprogram> R;

  NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes)
    return R;

  for (MDNode *N : CU_Nodes->operands()) {
    DICompileUnit CUNode(N);
    DIArray SPs = CUNode.getSubprograms();
    for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i) {
      DISubprogram SP(SPs.getElement(i));
      if (Function *F = SP.getFunction())
        R.insert(std::make_pair(F, SP));
    }
  }
  return R;
}
