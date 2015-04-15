//===- DebugInfoMetadata.cpp - Implement debug info metadata --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the debug info Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "LLVMContextImpl.h"
#include "MetadataImpl.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"

using namespace llvm;

MDLocation::MDLocation(LLVMContext &C, StorageType Storage, unsigned Line,
                       unsigned Column, ArrayRef<Metadata *> MDs)
    : MDNode(C, MDLocationKind, Storage, MDs) {
  assert((MDs.size() == 1 || MDs.size() == 2) &&
         "Expected a scope and optional inlined-at");

  // Set line and column.
  assert(Column < (1u << 16) && "Expected 16-bit column");

  SubclassData32 = Line;
  SubclassData16 = Column;
}

static void adjustColumn(unsigned &Column) {
  // Set to unknown on overflow.  We only have 16 bits to play with here.
  if (Column >= (1u << 16))
    Column = 0;
}

MDLocation *MDLocation::getImpl(LLVMContext &Context, unsigned Line,
                                unsigned Column, Metadata *Scope,
                                Metadata *InlinedAt, StorageType Storage,
                                bool ShouldCreate) {
  // Fixup column.
  adjustColumn(Column);

  assert(Scope && "Expected scope");
  if (Storage == Uniqued) {
    if (auto *N =
            getUniqued(Context.pImpl->MDLocations,
                       MDLocationInfo::KeyTy(Line, Column, Scope, InlinedAt)))
      return N;
    if (!ShouldCreate)
      return nullptr;
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  SmallVector<Metadata *, 2> Ops;
  Ops.push_back(Scope);
  if (InlinedAt)
    Ops.push_back(InlinedAt);
  return storeImpl(new (Ops.size())
                       MDLocation(Context, Storage, Line, Column, Ops),
                   Storage, Context.pImpl->MDLocations);
}

unsigned MDLocation::computeNewDiscriminator() const {
  // FIXME: This seems completely wrong.
  //
  //  1. If two modules are generated in the same context, then the second
  //     Module will get different discriminators than it would have if it were
  //     generated in its own context.
  //  2. If this function is called after round-tripping to bitcode instead of
  //     before, it will give a different (and potentially incorrect!) return.
  //
  // The discriminator should instead be calculated from local information
  // where it's actually needed.  This logic should be moved to
  // AddDiscriminators::runOnFunction(), where it doesn't pollute the
  // LLVMContext.
  std::pair<const char *, unsigned> Key(getFilename().data(), getLine());
  return ++getContext().pImpl->DiscriminatorTable[Key];
}

unsigned DebugNode::getFlag(StringRef Flag) {
  return StringSwitch<unsigned>(Flag)
#define HANDLE_DI_FLAG(ID, NAME) .Case("DIFlag" #NAME, Flag##NAME)
#include "llvm/IR/DebugInfoFlags.def"
      .Default(0);
}

const char *DebugNode::getFlagString(unsigned Flag) {
  switch (Flag) {
  default:
    return "";
#define HANDLE_DI_FLAG(ID, NAME)                                               \
  case Flag##NAME:                                                             \
    return "DIFlag" #NAME;
#include "llvm/IR/DebugInfoFlags.def"
  }
}

unsigned DebugNode::splitFlags(unsigned Flags,
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

MDScopeRef MDScope::getScope() const {
  if (auto *T = dyn_cast<MDType>(this))
    return T->getScope();

  if (auto *SP = dyn_cast<MDSubprogram>(this))
    return SP->getScope();

  if (auto *LB = dyn_cast<MDLexicalBlockBase>(this))
    return MDScopeRef(LB->getScope());

  if (auto *NS = dyn_cast<MDNamespace>(this))
    return MDScopeRef(NS->getScope());

  assert((isa<MDFile>(this) || isa<MDCompileUnit>(this)) &&
         "Unhandled type of scope.");
  return nullptr;
}

StringRef MDScope::getName() const {
  if (auto *T = dyn_cast<MDType>(this))
    return T->getName();
  if (auto *SP = dyn_cast<MDSubprogram>(this))
    return SP->getName();
  if (auto *NS = dyn_cast<MDNamespace>(this))
    return NS->getName();
  assert((isa<MDLexicalBlockBase>(this) || isa<MDFile>(this) ||
          isa<MDCompileUnit>(this)) &&
         "Unhandled type of scope.");
  return "";
}

static StringRef getString(const MDString *S) {
  if (S)
    return S->getString();
  return StringRef();
}

#ifndef NDEBUG
static bool isCanonical(const MDString *S) {
  return !S || !S->getString().empty();
}
#endif

GenericDebugNode *GenericDebugNode::getImpl(LLVMContext &Context, unsigned Tag,
                                            MDString *Header,
                                            ArrayRef<Metadata *> DwarfOps,
                                            StorageType Storage,
                                            bool ShouldCreate) {
  unsigned Hash = 0;
  if (Storage == Uniqued) {
    GenericDebugNodeInfo::KeyTy Key(Tag, getString(Header), DwarfOps);
    if (auto *N = getUniqued(Context.pImpl->GenericDebugNodes, Key))
      return N;
    if (!ShouldCreate)
      return nullptr;
    Hash = Key.getHash();
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  // Use a nullptr for empty headers.
  assert(isCanonical(Header) && "Expected canonical MDString");
  Metadata *PreOps[] = {Header};
  return storeImpl(new (DwarfOps.size() + 1) GenericDebugNode(
                       Context, Storage, Hash, Tag, PreOps, DwarfOps),
                   Storage, Context.pImpl->GenericDebugNodes);
}

void GenericDebugNode::recalculateHash() {
  setHash(GenericDebugNodeInfo::KeyTy::calculateHash(this));
}

#define UNWRAP_ARGS_IMPL(...) __VA_ARGS__
#define UNWRAP_ARGS(ARGS) UNWRAP_ARGS_IMPL ARGS
#define DEFINE_GETIMPL_LOOKUP(CLASS, ARGS)                                     \
  do {                                                                         \
    if (Storage == Uniqued) {                                                  \
      if (auto *N = getUniqued(Context.pImpl->CLASS##s,                        \
                               CLASS##Info::KeyTy(UNWRAP_ARGS(ARGS))))         \
        return N;                                                              \
      if (!ShouldCreate)                                                       \
        return nullptr;                                                        \
    } else {                                                                   \
      assert(ShouldCreate &&                                                   \
             "Expected non-uniqued nodes to always be created");               \
    }                                                                          \
  } while (false)
#define DEFINE_GETIMPL_STORE(CLASS, ARGS, OPS)                                 \
  return storeImpl(new (ArrayRef<Metadata *>(OPS).size())                      \
                       CLASS(Context, Storage, UNWRAP_ARGS(ARGS), OPS),        \
                   Storage, Context.pImpl->CLASS##s)
#define DEFINE_GETIMPL_STORE_NO_OPS(CLASS, ARGS)                               \
  return storeImpl(new (0u) CLASS(Context, Storage, UNWRAP_ARGS(ARGS)),        \
                   Storage, Context.pImpl->CLASS##s)
#define DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(CLASS, OPS)                   \
  return storeImpl(new (ArrayRef<Metadata *>(OPS).size())                      \
                       CLASS(Context, Storage, OPS),                           \
                   Storage, Context.pImpl->CLASS##s)

MDSubrange *MDSubrange::getImpl(LLVMContext &Context, int64_t Count, int64_t Lo,
                                StorageType Storage, bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(MDSubrange, (Count, Lo));
  DEFINE_GETIMPL_STORE_NO_OPS(MDSubrange, (Count, Lo));
}

MDEnumerator *MDEnumerator::getImpl(LLVMContext &Context, int64_t Value,
                                    MDString *Name, StorageType Storage,
                                    bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDEnumerator, (Value, getString(Name)));
  Metadata *Ops[] = {Name};
  DEFINE_GETIMPL_STORE(MDEnumerator, (Value), Ops);
}

MDBasicType *MDBasicType::getImpl(LLVMContext &Context, unsigned Tag,
                                  MDString *Name, uint64_t SizeInBits,
                                  uint64_t AlignInBits, unsigned Encoding,
                                  StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(
      MDBasicType, (Tag, getString(Name), SizeInBits, AlignInBits, Encoding));
  Metadata *Ops[] = {nullptr, nullptr, Name};
  DEFINE_GETIMPL_STORE(MDBasicType, (Tag, SizeInBits, AlignInBits, Encoding),
                       Ops);
}

MDDerivedType *MDDerivedType::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
    unsigned Line, Metadata *Scope, Metadata *BaseType, uint64_t SizeInBits,
    uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags,
    Metadata *ExtraData, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDDerivedType, (Tag, getString(Name), File, Line, Scope,
                                        BaseType, SizeInBits, AlignInBits,
                                        OffsetInBits, Flags, ExtraData));
  Metadata *Ops[] = {File, Scope, Name, BaseType, ExtraData};
  DEFINE_GETIMPL_STORE(
      MDDerivedType, (Tag, Line, SizeInBits, AlignInBits, OffsetInBits, Flags),
      Ops);
}

MDCompositeType *MDCompositeType::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
    unsigned Line, Metadata *Scope, Metadata *BaseType, uint64_t SizeInBits,
    uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags,
    Metadata *Elements, unsigned RuntimeLang, Metadata *VTableHolder,
    Metadata *TemplateParams, MDString *Identifier, StorageType Storage,
    bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDCompositeType,
                        (Tag, getString(Name), File, Line, Scope, BaseType,
                         SizeInBits, AlignInBits, OffsetInBits, Flags, Elements,
                         RuntimeLang, VTableHolder, TemplateParams,
                         getString(Identifier)));
  Metadata *Ops[] = {File,     Scope,        Name,           BaseType,
                     Elements, VTableHolder, TemplateParams, Identifier};
  DEFINE_GETIMPL_STORE(MDCompositeType, (Tag, Line, RuntimeLang, SizeInBits,
                                         AlignInBits, OffsetInBits, Flags),
                       Ops);
}

MDSubroutineType *MDSubroutineType::getImpl(LLVMContext &Context,
                                            unsigned Flags, Metadata *TypeArray,
                                            StorageType Storage,
                                            bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(MDSubroutineType, (Flags, TypeArray));
  Metadata *Ops[] = {nullptr,   nullptr, nullptr, nullptr,
                     TypeArray, nullptr, nullptr, nullptr};
  DEFINE_GETIMPL_STORE(MDSubroutineType, (Flags), Ops);
}

MDFile *MDFile::getImpl(LLVMContext &Context, MDString *Filename,
                        MDString *Directory, StorageType Storage,
                        bool ShouldCreate) {
  assert(isCanonical(Filename) && "Expected canonical MDString");
  assert(isCanonical(Directory) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDFile, (getString(Filename), getString(Directory)));
  Metadata *Ops[] = {Filename, Directory};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(MDFile, Ops);
}

MDCompileUnit *MDCompileUnit::getImpl(
    LLVMContext &Context, unsigned SourceLanguage, Metadata *File,
    MDString *Producer, bool IsOptimized, MDString *Flags,
    unsigned RuntimeVersion, MDString *SplitDebugFilename,
    unsigned EmissionKind, Metadata *EnumTypes, Metadata *RetainedTypes,
    Metadata *Subprograms, Metadata *GlobalVariables,
    Metadata *ImportedEntities, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Producer) && "Expected canonical MDString");
  assert(isCanonical(Flags) && "Expected canonical MDString");
  assert(isCanonical(SplitDebugFilename) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(
      MDCompileUnit,
      (SourceLanguage, File, getString(Producer), IsOptimized, getString(Flags),
       RuntimeVersion, getString(SplitDebugFilename), EmissionKind, EnumTypes,
       RetainedTypes, Subprograms, GlobalVariables, ImportedEntities));
  Metadata *Ops[] = {File, Producer, Flags, SplitDebugFilename, EnumTypes,
                     RetainedTypes, Subprograms, GlobalVariables,
                     ImportedEntities};
  DEFINE_GETIMPL_STORE(
      MDCompileUnit,
      (SourceLanguage, IsOptimized, RuntimeVersion, EmissionKind), Ops);
}

MDSubprogram *MDLocalScope::getSubprogram() const {
  if (auto *Block = dyn_cast<MDLexicalBlockBase>(this))
    return Block->getScope()->getSubprogram();
  return const_cast<MDSubprogram *>(cast<MDSubprogram>(this));
}

MDSubprogram *MDSubprogram::getImpl(
    LLVMContext &Context, Metadata *Scope, MDString *Name,
    MDString *LinkageName, Metadata *File, unsigned Line, Metadata *Type,
    bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
    Metadata *ContainingType, unsigned Virtuality, unsigned VirtualIndex,
    unsigned Flags, bool IsOptimized, Metadata *Function,
    Metadata *TemplateParams, Metadata *Declaration, Metadata *Variables,
    StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(LinkageName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDSubprogram,
                        (Scope, getString(Name), getString(LinkageName), File,
                         Line, Type, IsLocalToUnit, IsDefinition, ScopeLine,
                         ContainingType, Virtuality, VirtualIndex, Flags,
                         IsOptimized, Function, TemplateParams, Declaration,
                         Variables));
  Metadata *Ops[] = {File,           Scope,       Name,           Name,
                     LinkageName,    Type,        ContainingType, Function,
                     TemplateParams, Declaration, Variables};
  DEFINE_GETIMPL_STORE(MDSubprogram,
                       (Line, ScopeLine, Virtuality, VirtualIndex, Flags,
                        IsLocalToUnit, IsDefinition, IsOptimized),
                       Ops);
}

Function *MDSubprogram::getFunction() const {
  // FIXME: Should this be looking through bitcasts?
  return dyn_cast_or_null<Function>(getFunctionConstant());
}

bool MDSubprogram::describes(const Function *F) const {
  assert(F && "Invalid function");
  if (F == getFunction())
    return true;
  StringRef Name = getLinkageName();
  if (Name.empty())
    Name = getName();
  return F->getName() == Name;
}

void MDSubprogram::replaceFunction(Function *F) {
  replaceFunction(F ? ConstantAsMetadata::get(F)
                    : static_cast<ConstantAsMetadata *>(nullptr));
}

MDLexicalBlock *MDLexicalBlock::getImpl(LLVMContext &Context, Metadata *Scope,
                                        Metadata *File, unsigned Line,
                                        unsigned Column, StorageType Storage,
                                        bool ShouldCreate) {
  assert(Scope && "Expected scope");
  DEFINE_GETIMPL_LOOKUP(MDLexicalBlock, (Scope, File, Line, Column));
  Metadata *Ops[] = {File, Scope};
  DEFINE_GETIMPL_STORE(MDLexicalBlock, (Line, Column), Ops);
}

MDLexicalBlockFile *MDLexicalBlockFile::getImpl(LLVMContext &Context,
                                                Metadata *Scope, Metadata *File,
                                                unsigned Discriminator,
                                                StorageType Storage,
                                                bool ShouldCreate) {
  assert(Scope && "Expected scope");
  DEFINE_GETIMPL_LOOKUP(MDLexicalBlockFile, (Scope, File, Discriminator));
  Metadata *Ops[] = {File, Scope};
  DEFINE_GETIMPL_STORE(MDLexicalBlockFile, (Discriminator), Ops);
}

MDNamespace *MDNamespace::getImpl(LLVMContext &Context, Metadata *Scope,
                                  Metadata *File, MDString *Name, unsigned Line,
                                  StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDNamespace, (Scope, File, getString(Name), Line));
  Metadata *Ops[] = {File, Scope, Name};
  DEFINE_GETIMPL_STORE(MDNamespace, (Line), Ops);
}

MDTemplateTypeParameter *MDTemplateTypeParameter::getImpl(LLVMContext &Context,
                                                          MDString *Name,
                                                          Metadata *Type,
                                                          StorageType Storage,
                                                          bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDTemplateTypeParameter, (getString(Name), Type));
  Metadata *Ops[] = {Name, Type};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(MDTemplateTypeParameter, Ops);
}

MDTemplateValueParameter *MDTemplateValueParameter::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *Type,
    Metadata *Value, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDTemplateValueParameter,
                        (Tag, getString(Name), Type, Value));
  Metadata *Ops[] = {Name, Type, Value};
  DEFINE_GETIMPL_STORE(MDTemplateValueParameter, (Tag), Ops);
}

MDGlobalVariable *
MDGlobalVariable::getImpl(LLVMContext &Context, Metadata *Scope, MDString *Name,
                          MDString *LinkageName, Metadata *File, unsigned Line,
                          Metadata *Type, bool IsLocalToUnit, bool IsDefinition,
                          Metadata *Variable,
                          Metadata *StaticDataMemberDeclaration,
                          StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(LinkageName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDGlobalVariable,
                        (Scope, getString(Name), getString(LinkageName), File,
                         Line, Type, IsLocalToUnit, IsDefinition, Variable,
                         StaticDataMemberDeclaration));
  Metadata *Ops[] = {Scope, Name,        File,     Type,
                     Name,  LinkageName, Variable, StaticDataMemberDeclaration};
  DEFINE_GETIMPL_STORE(MDGlobalVariable, (Line, IsLocalToUnit, IsDefinition),
                       Ops);
}

MDLocalVariable *MDLocalVariable::getImpl(LLVMContext &Context, unsigned Tag,
                                          Metadata *Scope, MDString *Name,
                                          Metadata *File, unsigned Line,
                                          Metadata *Type, unsigned Arg,
                                          unsigned Flags, StorageType Storage,
                                          bool ShouldCreate) {
  // Truncate Arg to 8 bits.
  //
  // FIXME: This is gross (and should be changed to an assert or removed), but
  // it matches historical behaviour for now.
  Arg &= (1u << 8) - 1;

  assert(Scope && "Expected scope");
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDLocalVariable, (Tag, Scope, getString(Name), File,
                                          Line, Type, Arg, Flags));
  Metadata *Ops[] = {Scope, Name, File, Type};
  DEFINE_GETIMPL_STORE(MDLocalVariable, (Tag, Line, Arg, Flags), Ops);
}

MDExpression *MDExpression::getImpl(LLVMContext &Context,
                                    ArrayRef<uint64_t> Elements,
                                    StorageType Storage, bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(MDExpression, (Elements));
  DEFINE_GETIMPL_STORE_NO_OPS(MDExpression, (Elements));
}

unsigned MDExpression::ExprOperand::getSize() const {
  switch (getOp()) {
  case dwarf::DW_OP_bit_piece:
    return 3;
  case dwarf::DW_OP_plus:
    return 2;
  default:
    return 1;
  }
}

bool MDExpression::isValid() const {
  for (auto I = expr_op_begin(), E = expr_op_end(); I != E; ++I) {
    // Check that there's space for the operand.
    if (I->get() + I->getSize() > E->get())
      return false;

    // Check that the operand is valid.
    switch (I->getOp()) {
    default:
      return false;
    case dwarf::DW_OP_bit_piece:
      // Piece expressions must be at the end.
      return I->get() + I->getSize() == E->get();
    case dwarf::DW_OP_plus:
    case dwarf::DW_OP_deref:
      break;
    }
  }
  return true;
}

bool MDExpression::isBitPiece() const {
  assert(isValid() && "Expected valid expression");
  if (unsigned N = getNumElements())
    if (N >= 3)
      return getElement(N - 3) == dwarf::DW_OP_bit_piece;
  return false;
}

uint64_t MDExpression::getBitPieceOffset() const {
  assert(isBitPiece() && "Expected bit piece");
  return getElement(getNumElements() - 2);
}

uint64_t MDExpression::getBitPieceSize() const {
  assert(isBitPiece() && "Expected bit piece");
  return getElement(getNumElements() - 1);
}

MDObjCProperty *MDObjCProperty::getImpl(
    LLVMContext &Context, MDString *Name, Metadata *File, unsigned Line,
    MDString *GetterName, MDString *SetterName, unsigned Attributes,
    Metadata *Type, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(GetterName) && "Expected canonical MDString");
  assert(isCanonical(SetterName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDObjCProperty,
                        (getString(Name), File, Line, getString(GetterName),
                         getString(SetterName), Attributes, Type));
  Metadata *Ops[] = {Name, File, GetterName, SetterName, Type};
  DEFINE_GETIMPL_STORE(MDObjCProperty, (Line, Attributes), Ops);
}

MDImportedEntity *MDImportedEntity::getImpl(LLVMContext &Context, unsigned Tag,
                                            Metadata *Scope, Metadata *Entity,
                                            unsigned Line, MDString *Name,
                                            StorageType Storage,
                                            bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(MDImportedEntity,
                        (Tag, Scope, Entity, Line, getString(Name)));
  Metadata *Ops[] = {Scope, Entity, Name};
  DEFINE_GETIMPL_STORE(MDImportedEntity, (Tag, Line), Ops);
}
