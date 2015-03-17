//===- llvm/IR/DebugInfoMetadata.h - Debug info metadata --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Declarations for metadata specific to debug info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFOMETADATA_H
#define LLVM_IR_DEBUGINFOMETADATA_H

#include "llvm/IR/Metadata.h"
#include "llvm/Support/Dwarf.h"

// Helper macros for defining get() overrides.
#define DEFINE_MDNODE_GET_UNPACK_IMPL(...) __VA_ARGS__
#define DEFINE_MDNODE_GET_UNPACK(ARGS) DEFINE_MDNODE_GET_UNPACK_IMPL ARGS
#define DEFINE_MDNODE_GET(CLASS, FORMAL, ARGS)                                 \
  static CLASS *get(LLVMContext &Context, DEFINE_MDNODE_GET_UNPACK(FORMAL)) {  \
    return getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Uniqued);          \
  }                                                                            \
  static CLASS *getIfExists(LLVMContext &Context,                              \
                            DEFINE_MDNODE_GET_UNPACK(FORMAL)) {                \
    return getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Uniqued,           \
                   /* ShouldCreate */ false);                                  \
  }                                                                            \
  static CLASS *getDistinct(LLVMContext &Context,                              \
                            DEFINE_MDNODE_GET_UNPACK(FORMAL)) {                \
    return getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Distinct);         \
  }                                                                            \
  static Temp##CLASS getTemporary(LLVMContext &Context,                        \
                                  DEFINE_MDNODE_GET_UNPACK(FORMAL)) {          \
    return Temp##CLASS(                                                        \
        getImpl(Context, DEFINE_MDNODE_GET_UNPACK(ARGS), Temporary));          \
  }

namespace llvm {

/// \brief Debug location.
///
/// A debug location in source code, used for debug info and otherwise.
class MDLocation : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDLocation(LLVMContext &C, StorageType Storage, unsigned Line,
             unsigned Column, ArrayRef<Metadata *> MDs);
  ~MDLocation() { dropAllReferences(); }

  static MDLocation *getImpl(LLVMContext &Context, unsigned Line,
                             unsigned Column, Metadata *Scope,
                             Metadata *InlinedAt, StorageType Storage,
                             bool ShouldCreate = true);

  TempMDLocation cloneImpl() const {
    return getTemporary(getContext(), getLine(), getColumn(), getScope(),
                        getInlinedAt());
  }

  // Disallow replacing operands.
  void replaceOperandWith(unsigned I, Metadata *New) = delete;

public:
  DEFINE_MDNODE_GET(MDLocation,
                    (unsigned Line, unsigned Column, Metadata *Scope,
                     Metadata *InlinedAt = nullptr),
                    (Line, Column, Scope, InlinedAt))

  /// \brief Return a (temporary) clone of this.
  TempMDLocation clone() const { return cloneImpl(); }

  unsigned getLine() const { return SubclassData32; }
  unsigned getColumn() const { return SubclassData16; }
  Metadata *getScope() const { return getOperand(0); }
  Metadata *getInlinedAt() const {
    if (getNumOperands() == 2)
      return getOperand(1);
    return nullptr;
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLocationKind;
  }
};

/// \brief Tagged DWARF-like metadata node.
///
/// A metadata node with a DWARF tag (i.e., a constant named \c DW_TAG_*,
/// defined in llvm/Support/Dwarf.h).  Called \a DebugNode because it's
/// potentially used for non-DWARF output.
class DebugNode : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

protected:
  DebugNode(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
            ArrayRef<Metadata *> Ops1, ArrayRef<Metadata *> Ops2 = None)
      : MDNode(C, ID, Storage, Ops1, Ops2) {
    assert(Tag < 1u << 16);
    SubclassData16 = Tag;
  }
  ~DebugNode() {}

  template <class Ty> Ty *getOperandAs(unsigned I) const {
    return cast_or_null<Ty>(getOperand(I));
  }

  StringRef getStringOperand(unsigned I) const {
    if (auto *S = getOperandAs<MDString>(I))
      return S->getString();
    return StringRef();
  }

  static MDString *getCanonicalMDString(LLVMContext &Context, StringRef S) {
    if (S.empty())
      return nullptr;
    return MDString::get(Context, S);
  }

public:
  unsigned getTag() const { return SubclassData16; }

  static bool classof(const Metadata *MD) {
    switch (MD->getMetadataID()) {
    default:
      return false;
    case GenericDebugNodeKind:
    case MDSubrangeKind:
    case MDEnumeratorKind:
    case MDBasicTypeKind:
    case MDDerivedTypeKind:
    case MDCompositeTypeKind:
    case MDSubroutineTypeKind:
    case MDFileKind:
    case MDCompileUnitKind:
    case MDSubprogramKind:
    case MDLexicalBlockKind:
    case MDLexicalBlockFileKind:
    case MDNamespaceKind:
    case MDTemplateTypeParameterKind:
    case MDTemplateValueParameterKind:
    case MDGlobalVariableKind:
    case MDLocalVariableKind:
    case MDObjCPropertyKind:
    case MDImportedEntityKind:
      return true;
    }
  }
};

/// \brief Generic tagged DWARF-like metadata node.
///
/// An un-specialized DWARF-like metadata node.  The first operand is a
/// (possibly empty) null-separated \a MDString header that contains arbitrary
/// fields.  The remaining operands are \a dwarf_operands(), and are pointers
/// to other metadata.
class GenericDebugNode : public DebugNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  GenericDebugNode(LLVMContext &C, StorageType Storage, unsigned Hash,
                   unsigned Tag, ArrayRef<Metadata *> Ops1,
                   ArrayRef<Metadata *> Ops2)
      : DebugNode(C, GenericDebugNodeKind, Storage, Tag, Ops1, Ops2) {
    setHash(Hash);
  }
  ~GenericDebugNode() { dropAllReferences(); }

  void setHash(unsigned Hash) { SubclassData32 = Hash; }
  void recalculateHash();

  static GenericDebugNode *getImpl(LLVMContext &Context, unsigned Tag,
                                   StringRef Header,
                                   ArrayRef<Metadata *> DwarfOps,
                                   StorageType Storage,
                                   bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Header),
                   DwarfOps, Storage, ShouldCreate);
  }

  static GenericDebugNode *getImpl(LLVMContext &Context, unsigned Tag,
                                   MDString *Header,
                                   ArrayRef<Metadata *> DwarfOps,
                                   StorageType Storage,
                                   bool ShouldCreate = true);

  TempGenericDebugNode cloneImpl() const {
    return getTemporary(
        getContext(), getTag(), getHeader(),
        SmallVector<Metadata *, 4>(dwarf_op_begin(), dwarf_op_end()));
  }

public:
  unsigned getHash() const { return SubclassData32; }

  DEFINE_MDNODE_GET(GenericDebugNode, (unsigned Tag, StringRef Header,
                                       ArrayRef<Metadata *> DwarfOps),
                    (Tag, Header, DwarfOps))
  DEFINE_MDNODE_GET(GenericDebugNode, (unsigned Tag, MDString *Header,
                                       ArrayRef<Metadata *> DwarfOps),
                    (Tag, Header, DwarfOps))

  /// \brief Return a (temporary) clone of this.
  TempGenericDebugNode clone() const { return cloneImpl(); }

  unsigned getTag() const { return SubclassData16; }
  StringRef getHeader() const { return getStringOperand(0); }

  op_iterator dwarf_op_begin() const { return op_begin() + 1; }
  op_iterator dwarf_op_end() const { return op_end(); }
  op_range dwarf_operands() const {
    return op_range(dwarf_op_begin(), dwarf_op_end());
  }

  unsigned getNumDwarfOperands() const { return getNumOperands() - 1; }
  const MDOperand &getDwarfOperand(unsigned I) const {
    return getOperand(I + 1);
  }
  void replaceDwarfOperandWith(unsigned I, Metadata *New) {
    replaceOperandWith(I + 1, New);
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == GenericDebugNodeKind;
  }
};

/// \brief Array subrange.
///
/// TODO: Merge into node for DW_TAG_array_type, which should have a custom
/// type.
class MDSubrange : public DebugNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  int64_t Count;
  int64_t Lo;

  MDSubrange(LLVMContext &C, StorageType Storage, int64_t Count, int64_t Lo)
      : DebugNode(C, MDSubrangeKind, Storage, dwarf::DW_TAG_subrange_type,
                  None),
        Count(Count), Lo(Lo) {}
  ~MDSubrange() {}

  static MDSubrange *getImpl(LLVMContext &Context, int64_t Count, int64_t Lo,
                             StorageType Storage, bool ShouldCreate = true);

  TempMDSubrange cloneImpl() const {
    return getTemporary(getContext(), getCount(), getLo());
  }

public:
  DEFINE_MDNODE_GET(MDSubrange, (int64_t Count, int64_t Lo = 0), (Count, Lo))

  TempMDSubrange clone() const { return cloneImpl(); }

  int64_t getLo() const { return Lo; }
  int64_t getCount() const { return Count; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDSubrangeKind;
  }
};

/// \brief Enumeration value.
///
/// TODO: Add a pointer to the context (DW_TAG_enumeration_type) once that no
/// longer creates a type cycle.
class MDEnumerator : public DebugNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  int64_t Value;

  MDEnumerator(LLVMContext &C, StorageType Storage, int64_t Value,
               ArrayRef<Metadata *> Ops)
      : DebugNode(C, MDEnumeratorKind, Storage, dwarf::DW_TAG_enumerator, Ops),
        Value(Value) {}
  ~MDEnumerator() {}

  static MDEnumerator *getImpl(LLVMContext &Context, int64_t Value,
                               StringRef Name, StorageType Storage,
                               bool ShouldCreate = true) {
    return getImpl(Context, Value, getCanonicalMDString(Context, Name), Storage,
                   ShouldCreate);
  }
  static MDEnumerator *getImpl(LLVMContext &Context, int64_t Value,
                               MDString *Name, StorageType Storage,
                               bool ShouldCreate = true);

  TempMDEnumerator cloneImpl() const {
    return getTemporary(getContext(), getValue(), getName());
  }

public:
  DEFINE_MDNODE_GET(MDEnumerator, (int64_t Value, StringRef Name),
                    (Value, Name))
  DEFINE_MDNODE_GET(MDEnumerator, (int64_t Value, MDString *Name),
                    (Value, Name))

  TempMDEnumerator clone() const { return cloneImpl(); }

  int64_t getValue() const { return Value; }
  StringRef getName() const { return getStringOperand(0); }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDEnumeratorKind;
  }
};

/// \brief Base class for scope-like contexts.
///
/// Base class for lexical scopes and types (which are also declaration
/// contexts).
///
/// TODO: Separate the concepts of declaration contexts and lexical scopes.
class MDScope : public DebugNode {
protected:
  MDScope(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
          ArrayRef<Metadata *> Ops)
      : DebugNode(C, ID, Storage, Tag, Ops) {}
  ~MDScope() {}

public:
  /// \brief Return the underlying file.
  ///
  /// An \a MDFile is an \a MDScope, but it doesn't point at a separate file
  /// (it\em is the file).  If \c this is an \a MDFile, we need to return \c
  /// this.  Otherwise, return the first operand, which is where all other
  /// subclasses store their file pointer.
  Metadata *getFile() const {
    return isa<MDFile>(this) ? const_cast<MDScope *>(this)
                             : static_cast<Metadata *>(getOperand(0));
  }

  static bool classof(const Metadata *MD) {
    switch (MD->getMetadataID()) {
    default:
      return false;
    case MDBasicTypeKind:
    case MDDerivedTypeKind:
    case MDCompositeTypeKind:
    case MDSubroutineTypeKind:
    case MDFileKind:
    case MDCompileUnitKind:
    case MDSubprogramKind:
    case MDLexicalBlockKind:
    case MDLexicalBlockFileKind:
    case MDNamespaceKind:
      return true;
    }
  }
};

/// \brief Base class for types.
///
/// TODO: Remove the hardcoded name and context, since many types don't use
/// them.
/// TODO: Split up flags.
class MDType : public MDScope {
  unsigned Line;
  unsigned Flags;
  uint64_t SizeInBits;
  uint64_t AlignInBits;
  uint64_t OffsetInBits;

protected:
  MDType(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
         unsigned Line, uint64_t SizeInBits, uint64_t AlignInBits,
         uint64_t OffsetInBits, unsigned Flags, ArrayRef<Metadata *> Ops)
      : MDScope(C, ID, Storage, Tag, Ops), Line(Line), Flags(Flags),
        SizeInBits(SizeInBits), AlignInBits(AlignInBits),
        OffsetInBits(OffsetInBits) {}
  ~MDType() {}

public:
  TempMDType clone() const {
    return TempMDType(cast<MDType>(MDNode::clone().release()));
  }

  unsigned getLine() const { return Line; }
  uint64_t getSizeInBits() const { return SizeInBits; }
  uint64_t getAlignInBits() const { return AlignInBits; }
  uint64_t getOffsetInBits() const { return OffsetInBits; }
  unsigned getFlags() const { return Flags; }

  Metadata *getScope() const { return getOperand(1); }
  StringRef getName() const { return getStringOperand(2); }

  MDString *getRawName() const { return getOperandAs<MDString>(2); }

  void setFlags(unsigned NewFlags) {
    assert(!isUniqued() && "Cannot set flags on uniqued nodes");
    Flags = NewFlags;
  }

  static bool classof(const Metadata *MD) {
    switch (MD->getMetadataID()) {
    default:
      return false;
    case MDBasicTypeKind:
    case MDDerivedTypeKind:
    case MDCompositeTypeKind:
    case MDSubroutineTypeKind:
      return true;
    }
  }
};

/// \brief Basic type.
///
/// TODO: Split out DW_TAG_unspecified_type.
/// TODO: Drop unused accessors.
class MDBasicType : public MDType {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Encoding;

  MDBasicType(LLVMContext &C, StorageType Storage, unsigned Tag,
              uint64_t SizeInBits, uint64_t AlignInBits, unsigned Encoding,
              ArrayRef<Metadata *> Ops)
      : MDType(C, MDBasicTypeKind, Storage, Tag, 0, SizeInBits, AlignInBits, 0,
               0, Ops),
        Encoding(Encoding) {}
  ~MDBasicType() {}

  static MDBasicType *getImpl(LLVMContext &Context, unsigned Tag,
                              StringRef Name, uint64_t SizeInBits,
                              uint64_t AlignInBits, unsigned Encoding,
                              StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Name),
                   SizeInBits, AlignInBits, Encoding, Storage, ShouldCreate);
  }
  static MDBasicType *getImpl(LLVMContext &Context, unsigned Tag,
                              MDString *Name, uint64_t SizeInBits,
                              uint64_t AlignInBits, unsigned Encoding,
                              StorageType Storage, bool ShouldCreate = true);

  TempMDBasicType cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getSizeInBits(),
                        getAlignInBits(), getEncoding());
  }

public:
  DEFINE_MDNODE_GET(MDBasicType, (unsigned Tag, StringRef Name),
                    (Tag, Name, 0, 0, 0))
  DEFINE_MDNODE_GET(MDBasicType,
                    (unsigned Tag, StringRef Name, uint64_t SizeInBits,
                     uint64_t AlignInBits, unsigned Encoding),
                    (Tag, Name, SizeInBits, AlignInBits, Encoding))
  DEFINE_MDNODE_GET(MDBasicType,
                    (unsigned Tag, MDString *Name, uint64_t SizeInBits,
                     uint64_t AlignInBits, unsigned Encoding),
                    (Tag, Name, SizeInBits, AlignInBits, Encoding))

  TempMDBasicType clone() const { return cloneImpl(); }

  unsigned getEncoding() const { return Encoding; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDBasicTypeKind;
  }
};

/// \brief Base class for MDDerivedType and MDCompositeType.
///
/// TODO: Delete; they're not really related.
class MDDerivedTypeBase : public MDType {
protected:
  MDDerivedTypeBase(LLVMContext &C, unsigned ID, StorageType Storage,
                    unsigned Tag, unsigned Line, uint64_t SizeInBits,
                    uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags,
                    ArrayRef<Metadata *> Ops)
      : MDType(C, ID, Storage, Tag, Line, SizeInBits, AlignInBits, OffsetInBits,
               Flags, Ops) {}
  ~MDDerivedTypeBase() {}

public:
  Metadata *getBaseType() const { return getOperand(3); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDDerivedTypeKind ||
           MD->getMetadataID() == MDCompositeTypeKind ||
           MD->getMetadataID() == MDSubroutineTypeKind;
  }
};

/// \brief Derived types.
///
/// This includes qualified types, pointers, references, friends, typedefs, and
/// class members.
///
/// TODO: Split out members (inheritance, fields, methods, etc.).
class MDDerivedType : public MDDerivedTypeBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDDerivedType(LLVMContext &C, StorageType Storage, unsigned Tag,
                unsigned Line, uint64_t SizeInBits, uint64_t AlignInBits,
                uint64_t OffsetInBits, unsigned Flags, ArrayRef<Metadata *> Ops)
      : MDDerivedTypeBase(C, MDDerivedTypeKind, Storage, Tag, Line, SizeInBits,
                          AlignInBits, OffsetInBits, Flags, Ops) {}
  ~MDDerivedType() {}

  static MDDerivedType *getImpl(LLVMContext &Context, unsigned Tag,
                                StringRef Name, Metadata *File, unsigned Line,
                                Metadata *Scope, Metadata *BaseType,
                                uint64_t SizeInBits, uint64_t AlignInBits,
                                uint64_t OffsetInBits, unsigned Flags,
                                Metadata *ExtraData, StorageType Storage,
                                bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Name), File,
                   Line, Scope, BaseType, SizeInBits, AlignInBits, OffsetInBits,
                   Flags, ExtraData, Storage, ShouldCreate);
  }
  static MDDerivedType *getImpl(LLVMContext &Context, unsigned Tag,
                                MDString *Name, Metadata *File, unsigned Line,
                                Metadata *Scope, Metadata *BaseType,
                                uint64_t SizeInBits, uint64_t AlignInBits,
                                uint64_t OffsetInBits, unsigned Flags,
                                Metadata *ExtraData, StorageType Storage,
                                bool ShouldCreate = true);

  TempMDDerivedType cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getFile(), getLine(),
                        getScope(), getBaseType(), getSizeInBits(),
                        getAlignInBits(), getOffsetInBits(), getFlags(),
                        getExtraData());
  }

public:
  DEFINE_MDNODE_GET(MDDerivedType,
                    (unsigned Tag, MDString *Name, Metadata *File,
                     unsigned Line, Metadata *Scope, Metadata *BaseType,
                     uint64_t SizeInBits, uint64_t AlignInBits,
                     uint64_t OffsetInBits, unsigned Flags,
                     Metadata *ExtraData = nullptr),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, ExtraData))
  DEFINE_MDNODE_GET(MDDerivedType,
                    (unsigned Tag, StringRef Name, Metadata *File,
                     unsigned Line, Metadata *Scope, Metadata *BaseType,
                     uint64_t SizeInBits, uint64_t AlignInBits,
                     uint64_t OffsetInBits, unsigned Flags,
                     Metadata *ExtraData = nullptr),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, ExtraData))

  TempMDDerivedType clone() const { return cloneImpl(); }

  /// \brief Get extra data associated with this derived type.
  ///
  /// Class type for pointer-to-members, objective-c property node for ivars,
  /// or global constant wrapper for static members.
  ///
  /// TODO: Separate out types that need this extra operand: pointer-to-member
  /// types and member fields (static members and ivars).
  Metadata *getExtraData() const { return getOperand(4); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDDerivedTypeKind;
  }
};

/// \brief Base class for MDCompositeType and MDSubroutineType.
///
/// TODO: Delete; they're not really related.
class MDCompositeTypeBase : public MDDerivedTypeBase {
  unsigned RuntimeLang;

protected:
  MDCompositeTypeBase(LLVMContext &C, unsigned ID, StorageType Storage,
                      unsigned Tag, unsigned Line, unsigned RuntimeLang,
                      uint64_t SizeInBits, uint64_t AlignInBits,
                      uint64_t OffsetInBits, unsigned Flags,
                      ArrayRef<Metadata *> Ops)
      : MDDerivedTypeBase(C, ID, Storage, Tag, Line, SizeInBits, AlignInBits,
                          OffsetInBits, Flags, Ops),
        RuntimeLang(RuntimeLang) {}
  ~MDCompositeTypeBase() {}

public:
  Metadata *getElements() const { return getOperand(4); }
  Metadata *getVTableHolder() const { return getOperand(5); }
  Metadata *getTemplateParams() const { return getOperand(6); }
  StringRef getIdentifier() const { return getStringOperand(7); }
  unsigned getRuntimeLang() const { return RuntimeLang; }

  MDString *getRawIdentifier() const { return getOperandAs<MDString>(7); }

  /// \brief Replace operands.
  ///
  /// If this \a isUniqued() and not \a isResolved(), on a uniquing collision
  /// this will be RAUW'ed and deleted.  Use a \a TrackingMDRef to keep track
  /// of its movement if necessary.
  /// @{
  void replaceElements(MDTuple *Elements) {
#ifndef NDEBUG
    if (auto *Old = cast_or_null<MDTuple>(getElements()))
      for (const auto &Op : Old->operands())
        assert(std::find(Elements->op_begin(), Elements->op_end(), Op) &&
               "Lost a member during member list replacement");
#endif
    replaceOperandWith(4, Elements);
  }
  void replaceVTableHolder(Metadata *VTableHolder) {
    replaceOperandWith(5, VTableHolder);
  }
  void replaceTemplateParams(MDTuple *TemplateParams) {
    replaceOperandWith(6, TemplateParams);
  }
  /// @}

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDCompositeTypeKind ||
           MD->getMetadataID() == MDSubroutineTypeKind;
  }
};

/// \brief Composite types.
///
/// TODO: Detach from DerivedTypeBase (split out MDEnumType?).
/// TODO: Create a custom, unrelated node for DW_TAG_array_type.
class MDCompositeType : public MDCompositeTypeBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDCompositeType(LLVMContext &C, StorageType Storage, unsigned Tag,
                  unsigned Line, unsigned RuntimeLang, uint64_t SizeInBits,
                  uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags,
                  ArrayRef<Metadata *> Ops)
      : MDCompositeTypeBase(C, MDCompositeTypeKind, Storage, Tag, Line,
                            RuntimeLang, SizeInBits, AlignInBits, OffsetInBits,
                            Flags, Ops) {}
  ~MDCompositeType() {}

  static MDCompositeType *
  getImpl(LLVMContext &Context, unsigned Tag, StringRef Name, Metadata *File,
          unsigned Line, Metadata *Scope, Metadata *BaseType,
          uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
          uint64_t Flags, Metadata *Elements, unsigned RuntimeLang,
          Metadata *VTableHolder, Metadata *TemplateParams,
          StringRef Identifier, StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Name), File,
                   Line, Scope, BaseType, SizeInBits, AlignInBits, OffsetInBits,
                   Flags, Elements, RuntimeLang, VTableHolder, TemplateParams,
                   getCanonicalMDString(Context, Identifier), Storage,
                   ShouldCreate);
  }
  static MDCompositeType *
  getImpl(LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
          unsigned Line, Metadata *Scope, Metadata *BaseType,
          uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
          unsigned Flags, Metadata *Elements, unsigned RuntimeLang,
          Metadata *VTableHolder, Metadata *TemplateParams,
          MDString *Identifier, StorageType Storage, bool ShouldCreate = true);

  TempMDCompositeType cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getFile(), getLine(),
                        getScope(), getBaseType(), getSizeInBits(),
                        getAlignInBits(), getOffsetInBits(), getFlags(),
                        getElements(), getRuntimeLang(), getVTableHolder(),
                        getTemplateParams(), getIdentifier());
  }

public:
  DEFINE_MDNODE_GET(MDCompositeType,
                    (unsigned Tag, StringRef Name, Metadata *File,
                     unsigned Line, Metadata *Scope, Metadata *BaseType,
                     uint64_t SizeInBits, uint64_t AlignInBits,
                     uint64_t OffsetInBits, unsigned Flags, Metadata *Elements,
                     unsigned RuntimeLang, Metadata *VTableHolder,
                     Metadata *TemplateParams = nullptr,
                     StringRef Identifier = ""),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                     VTableHolder, TemplateParams, Identifier))
  DEFINE_MDNODE_GET(MDCompositeType,
                    (unsigned Tag, MDString *Name, Metadata *File,
                     unsigned Line, Metadata *Scope, Metadata *BaseType,
                     uint64_t SizeInBits, uint64_t AlignInBits,
                     uint64_t OffsetInBits, unsigned Flags, Metadata *Elements,
                     unsigned RuntimeLang, Metadata *VTableHolder,
                     Metadata *TemplateParams = nullptr,
                     MDString *Identifier = nullptr),
                    (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                     AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                     VTableHolder, TemplateParams, Identifier))

  TempMDCompositeType clone() const { return cloneImpl(); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDCompositeTypeKind;
  }
};

/// \brief Type array for a subprogram.
///
/// TODO: Detach from CompositeType, and fold the array of types in directly
/// as operands.
class MDSubroutineType : public MDCompositeTypeBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDSubroutineType(LLVMContext &C, StorageType Storage, unsigned Flags,
                   ArrayRef<Metadata *> Ops)
      : MDCompositeTypeBase(C, MDSubroutineTypeKind, Storage,
                            dwarf::DW_TAG_subroutine_type, 0, 0, 0, 0, 0, Flags,
                            Ops) {}
  ~MDSubroutineType() {}

  static MDSubroutineType *getImpl(LLVMContext &Context, unsigned Flags,
                                   Metadata *TypeArray, StorageType Storage,
                                   bool ShouldCreate = true);

  TempMDSubroutineType cloneImpl() const {
    return getTemporary(getContext(), getFlags(), getTypeArray());
  }

public:
  DEFINE_MDNODE_GET(MDSubroutineType, (unsigned Flags, Metadata *TypeArray),
                    (Flags, TypeArray))

  TempMDSubroutineType clone() const { return cloneImpl(); }

  Metadata *getTypeArray() const { return getElements(); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDSubroutineTypeKind;
  }
};

/// \brief File.
///
/// TODO: Merge with directory/file node (including users).
/// TODO: Canonicalize paths on creation.
class MDFile : public MDScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDFile(LLVMContext &C, StorageType Storage, ArrayRef<Metadata *> Ops)
      : MDScope(C, MDFileKind, Storage, dwarf::DW_TAG_file_type, Ops) {}
  ~MDFile() {}

  static MDFile *getImpl(LLVMContext &Context, StringRef Filename,
                         StringRef Directory, StorageType Storage,
                         bool ShouldCreate = true) {
    return getImpl(Context, getCanonicalMDString(Context, Filename),
                   getCanonicalMDString(Context, Directory), Storage,
                   ShouldCreate);
  }
  static MDFile *getImpl(LLVMContext &Context, MDString *Filename,
                         MDString *Directory, StorageType Storage,
                         bool ShouldCreate = true);

  TempMDFile cloneImpl() const {
    return getTemporary(getContext(), getFilename(), getDirectory());
  }

public:
  DEFINE_MDNODE_GET(MDFile, (StringRef Filename, StringRef Directory),
                    (Filename, Directory))
  DEFINE_MDNODE_GET(MDFile, (MDString * Filename, MDString *Directory),
                    (Filename, Directory))

  TempMDFile clone() const { return cloneImpl(); }

  StringRef getFilename() const { return getStringOperand(0); }
  StringRef getDirectory() const { return getStringOperand(1); }

  MDString *getRawFilename() const { return getOperandAs<MDString>(0); }
  MDString *getRawDirectory() const { return getOperandAs<MDString>(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDFileKind;
  }
};

/// \brief Compile unit.
class MDCompileUnit : public MDScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned SourceLanguage;
  bool IsOptimized;
  unsigned RuntimeVersion;
  unsigned EmissionKind;

  MDCompileUnit(LLVMContext &C, StorageType Storage, unsigned SourceLanguage,
                bool IsOptimized, unsigned RuntimeVersion,
                unsigned EmissionKind, ArrayRef<Metadata *> Ops)
      : MDScope(C, MDCompileUnitKind, Storage, dwarf::DW_TAG_compile_unit, Ops),
        SourceLanguage(SourceLanguage), IsOptimized(IsOptimized),
        RuntimeVersion(RuntimeVersion), EmissionKind(EmissionKind) {}
  ~MDCompileUnit() {}

  static MDCompileUnit *
  getImpl(LLVMContext &Context, unsigned SourceLanguage, Metadata *File,
          StringRef Producer, bool IsOptimized, StringRef Flags,
          unsigned RuntimeVersion, StringRef SplitDebugFilename,
          unsigned EmissionKind, Metadata *EnumTypes, Metadata *RetainedTypes,
          Metadata *Subprograms, Metadata *GlobalVariables,
          Metadata *ImportedEntities, StorageType Storage,
          bool ShouldCreate = true) {
    return getImpl(Context, SourceLanguage, File,
                   getCanonicalMDString(Context, Producer), IsOptimized,
                   getCanonicalMDString(Context, Flags), RuntimeVersion,
                   getCanonicalMDString(Context, SplitDebugFilename),
                   EmissionKind, EnumTypes, RetainedTypes, Subprograms,
                   GlobalVariables, ImportedEntities, Storage, ShouldCreate);
  }
  static MDCompileUnit *
  getImpl(LLVMContext &Context, unsigned SourceLanguage, Metadata *File,
          MDString *Producer, bool IsOptimized, MDString *Flags,
          unsigned RuntimeVersion, MDString *SplitDebugFilename,
          unsigned EmissionKind, Metadata *EnumTypes, Metadata *RetainedTypes,
          Metadata *Subprograms, Metadata *GlobalVariables,
          Metadata *ImportedEntities, StorageType Storage,
          bool ShouldCreate = true);

  TempMDCompileUnit cloneImpl() const {
    return getTemporary(
        getContext(), getSourceLanguage(), getFile(), getProducer(),
        isOptimized(), getFlags(), getRuntimeVersion(), getSplitDebugFilename(),
        getEmissionKind(), getEnumTypes(), getRetainedTypes(), getSubprograms(),
        getGlobalVariables(), getImportedEntities());
  }

public:
  DEFINE_MDNODE_GET(MDCompileUnit,
                    (unsigned SourceLanguage, Metadata *File,
                     StringRef Producer, bool IsOptimized, StringRef Flags,
                     unsigned RuntimeVersion, StringRef SplitDebugFilename,
                     unsigned EmissionKind, Metadata *EnumTypes,
                     Metadata *RetainedTypes, Metadata *Subprograms,
                     Metadata *GlobalVariables, Metadata *ImportedEntities),
                    (SourceLanguage, File, Producer, IsOptimized, Flags,
                     RuntimeVersion, SplitDebugFilename, EmissionKind,
                     EnumTypes, RetainedTypes, Subprograms, GlobalVariables,
                     ImportedEntities))
  DEFINE_MDNODE_GET(MDCompileUnit,
                    (unsigned SourceLanguage, Metadata *File,
                     MDString *Producer, bool IsOptimized, MDString *Flags,
                     unsigned RuntimeVersion, MDString *SplitDebugFilename,
                     unsigned EmissionKind, Metadata *EnumTypes,
                     Metadata *RetainedTypes, Metadata *Subprograms,
                     Metadata *GlobalVariables, Metadata *ImportedEntities),
                    (SourceLanguage, File, Producer, IsOptimized, Flags,
                     RuntimeVersion, SplitDebugFilename, EmissionKind,
                     EnumTypes, RetainedTypes, Subprograms, GlobalVariables,
                     ImportedEntities))

  TempMDCompileUnit clone() const { return cloneImpl(); }

  unsigned getSourceLanguage() const { return SourceLanguage; }
  bool isOptimized() const { return IsOptimized; }
  unsigned getRuntimeVersion() const { return RuntimeVersion; }
  unsigned getEmissionKind() const { return EmissionKind; }
  StringRef getProducer() const { return getStringOperand(1); }
  StringRef getFlags() const { return getStringOperand(2); }
  StringRef getSplitDebugFilename() const { return getStringOperand(3); }
  Metadata *getEnumTypes() const { return getOperand(4); }
  Metadata *getRetainedTypes() const { return getOperand(5); }
  Metadata *getSubprograms() const { return getOperand(6); }
  Metadata *getGlobalVariables() const { return getOperand(7); }
  Metadata *getImportedEntities() const { return getOperand(8); }

  MDString *getRawProducer() const { return getOperandAs<MDString>(1); }
  MDString *getRawFlags() const { return getOperandAs<MDString>(2); }
  MDString *getRawSplitDebugFilename() const {
    return getOperandAs<MDString>(3);
  }

  /// \brief Replace arrays.
  ///
  /// If this \a isUniqued() and not \a isResolved(), it will be RAUW'ed and
  /// deleted on a uniquing collision.  In practice, uniquing collisions on \a
  /// MDCompileUnit should be fairly rare.
  /// @{
  void replaceSubprograms(MDTuple *N) { replaceOperandWith(6, N); }
  void replaceGlobalVariables(MDTuple *N) { replaceOperandWith(7, N); }
  /// @}

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDCompileUnitKind;
  }
};

/// \brief Subprogram description.
///
/// TODO: Remove DisplayName.  It's always equal to Name.
/// TODO: Split up flags.
class MDSubprogram : public MDScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;
  unsigned ScopeLine;
  unsigned Virtuality;
  unsigned VirtualIndex;
  unsigned Flags;
  bool IsLocalToUnit;
  bool IsDefinition;
  bool IsOptimized;

  MDSubprogram(LLVMContext &C, StorageType Storage, unsigned Line,
               unsigned ScopeLine, unsigned Virtuality, unsigned VirtualIndex,
               unsigned Flags, bool IsLocalToUnit, bool IsDefinition,
               bool IsOptimized, ArrayRef<Metadata *> Ops)
      : MDScope(C, MDSubprogramKind, Storage, dwarf::DW_TAG_subprogram, Ops),
        Line(Line), ScopeLine(ScopeLine), Virtuality(Virtuality),
        VirtualIndex(VirtualIndex), Flags(Flags), IsLocalToUnit(IsLocalToUnit),
        IsDefinition(IsDefinition), IsOptimized(IsOptimized) {}
  ~MDSubprogram() {}

  static MDSubprogram *
  getImpl(LLVMContext &Context, Metadata *Scope, StringRef Name,
          StringRef LinkageName, Metadata *File, unsigned Line, Metadata *Type,
          bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
          Metadata *ContainingType, unsigned Virtuality, unsigned VirtualIndex,
          unsigned Flags, bool IsOptimized, Metadata *Function,
          Metadata *TemplateParams, Metadata *Declaration, Metadata *Variables,
          StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, LinkageName), File, Line, Type,
                   IsLocalToUnit, IsDefinition, ScopeLine, ContainingType,
                   Virtuality, VirtualIndex, Flags, IsOptimized, Function,
                   TemplateParams, Declaration, Variables, Storage,
                   ShouldCreate);
  }
  static MDSubprogram *
  getImpl(LLVMContext &Context, Metadata *Scope, MDString *Name,
          MDString *LinkageName, Metadata *File, unsigned Line, Metadata *Type,
          bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
          Metadata *ContainingType, unsigned Virtuality, unsigned VirtualIndex,
          unsigned Flags, bool IsOptimized, Metadata *Function,
          Metadata *TemplateParams, Metadata *Declaration, Metadata *Variables,
          StorageType Storage, bool ShouldCreate = true);

  TempMDSubprogram cloneImpl() const {
    return getTemporary(getContext(), getScope(), getName(), getLinkageName(),
                        getFile(), getLine(), getType(), isLocalToUnit(),
                        isDefinition(), getScopeLine(), getContainingType(),
                        getVirtuality(), getVirtualIndex(), getFlags(),
                        isOptimized(), getFunction(), getTemplateParams(),
                        getDeclaration(), getVariables());
  }

public:
  DEFINE_MDNODE_GET(
      MDSubprogram,
      (Metadata * Scope, StringRef Name, StringRef LinkageName, Metadata *File,
       unsigned Line, Metadata *Type, bool IsLocalToUnit, bool IsDefinition,
       unsigned ScopeLine, Metadata *ContainingType, unsigned Virtuality,
       unsigned VirtualIndex, unsigned Flags, bool IsOptimized,
       Metadata *Function = nullptr, Metadata *TemplateParams = nullptr,
       Metadata *Declaration = nullptr, Metadata *Variables = nullptr),
      (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit, IsDefinition,
       ScopeLine, ContainingType, Virtuality, VirtualIndex, Flags, IsOptimized,
       Function, TemplateParams, Declaration, Variables))
  DEFINE_MDNODE_GET(
      MDSubprogram,
      (Metadata * Scope, MDString *Name, MDString *LinkageName, Metadata *File,
       unsigned Line, Metadata *Type, bool IsLocalToUnit, bool IsDefinition,
       unsigned ScopeLine, Metadata *ContainingType, unsigned Virtuality,
       unsigned VirtualIndex, unsigned Flags, bool IsOptimized,
       Metadata *Function = nullptr, Metadata *TemplateParams = nullptr,
       Metadata *Declaration = nullptr, Metadata *Variables = nullptr),
      (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit, IsDefinition,
       ScopeLine, ContainingType, Virtuality, VirtualIndex, Flags, IsOptimized,
       Function, TemplateParams, Declaration, Variables))

  TempMDSubprogram clone() const { return cloneImpl(); }

public:
  unsigned getLine() const { return Line; }
  unsigned getVirtuality() const { return Virtuality; }
  unsigned getVirtualIndex() const { return VirtualIndex; }
  unsigned getScopeLine() const { return ScopeLine; }
  unsigned getFlags() const { return Flags; }
  bool isLocalToUnit() const { return IsLocalToUnit; }
  bool isDefinition() const { return IsDefinition; }
  bool isOptimized() const { return IsOptimized; }

  Metadata *getScope() const { return getOperand(1); }

  StringRef getName() const { return getStringOperand(2); }
  StringRef getDisplayName() const { return getStringOperand(3); }
  StringRef getLinkageName() const { return getStringOperand(4); }

  MDString *getRawName() const { return getOperandAs<MDString>(2); }
  MDString *getRawLinkageName() const { return getOperandAs<MDString>(4); }

  Metadata *getType() const { return getOperand(5); }
  Metadata *getContainingType() const { return getOperand(6); }

  Metadata *getFunction() const { return getOperand(7); }
  Metadata *getTemplateParams() const { return getOperand(8); }
  Metadata *getDeclaration() const { return getOperand(9); }
  Metadata *getVariables() const { return getOperand(10); }

  /// \brief Replace the function.
  ///
  /// If \a isUniqued() and not \a isResolved(), this could node will be
  /// RAUW'ed and deleted out from under the caller.  Use a \a TrackingMDRef if
  /// that's a problem.
  /// @{
  void replaceFunction(Function *F);
  void replaceFunction(ConstantAsMetadata *MD) { replaceOperandWith(7, MD); }
  void replaceFunction(std::nullptr_t) { replaceOperandWith(7, nullptr); }
  /// @}

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDSubprogramKind;
  }
};

class MDLexicalBlockBase : public MDScope {
protected:
  MDLexicalBlockBase(LLVMContext &C, unsigned ID, StorageType Storage,
                     ArrayRef<Metadata *> Ops)
      : MDScope(C, ID, Storage, dwarf::DW_TAG_lexical_block, Ops) {}
  ~MDLexicalBlockBase() {}

public:
  Metadata *getScope() const { return getOperand(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLexicalBlockKind ||
           MD->getMetadataID() == MDLexicalBlockFileKind;
  }
};

class MDLexicalBlock : public MDLexicalBlockBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;
  unsigned Column;

  MDLexicalBlock(LLVMContext &C, StorageType Storage, unsigned Line,
                 unsigned Column, ArrayRef<Metadata *> Ops)
      : MDLexicalBlockBase(C, MDLexicalBlockKind, Storage, Ops), Line(Line),
        Column(Column) {}
  ~MDLexicalBlock() {}

  static MDLexicalBlock *getImpl(LLVMContext &Context, Metadata *Scope,
                                 Metadata *File, unsigned Line, unsigned Column,
                                 StorageType Storage, bool ShouldCreate = true);

  TempMDLexicalBlock cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(), getLine(),
                        getColumn());
  }

public:
  DEFINE_MDNODE_GET(MDLexicalBlock, (Metadata * Scope, Metadata *File,
                                     unsigned Line, unsigned Column),
                    (Scope, File, Line, Column))

  TempMDLexicalBlock clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLexicalBlockKind;
  }
};

class MDLexicalBlockFile : public MDLexicalBlockBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Discriminator;

  MDLexicalBlockFile(LLVMContext &C, StorageType Storage,
                     unsigned Discriminator, ArrayRef<Metadata *> Ops)
      : MDLexicalBlockBase(C, MDLexicalBlockFileKind, Storage, Ops),
        Discriminator(Discriminator) {}
  ~MDLexicalBlockFile() {}

  static MDLexicalBlockFile *getImpl(LLVMContext &Context, Metadata *Scope,
                                     Metadata *File, unsigned Discriminator,
                                     StorageType Storage,
                                     bool ShouldCreate = true);

  TempMDLexicalBlockFile cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(),
                        getDiscriminator());
  }

public:
  DEFINE_MDNODE_GET(MDLexicalBlockFile,
                    (Metadata * Scope, Metadata *File, unsigned Discriminator),
                    (Scope, File, Discriminator))

  TempMDLexicalBlockFile clone() const { return cloneImpl(); }

  unsigned getDiscriminator() const { return Discriminator; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLexicalBlockFileKind;
  }
};

class MDNamespace : public MDScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  MDNamespace(LLVMContext &Context, StorageType Storage, unsigned Line,
              ArrayRef<Metadata *> Ops)
      : MDScope(Context, MDNamespaceKind, Storage, dwarf::DW_TAG_namespace,
                Ops),
        Line(Line) {}
  ~MDNamespace() {}

  static MDNamespace *getImpl(LLVMContext &Context, Metadata *Scope,
                              Metadata *File, StringRef Name, unsigned Line,
                              StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Scope, File, getCanonicalMDString(Context, Name),
                   Line, Storage, ShouldCreate);
  }
  static MDNamespace *getImpl(LLVMContext &Context, Metadata *Scope,
                              Metadata *File, MDString *Name, unsigned Line,
                              StorageType Storage, bool ShouldCreate = true);

  TempMDNamespace cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(), getName(),
                        getLine());
  }

public:
  DEFINE_MDNODE_GET(MDNamespace, (Metadata * Scope, Metadata *File,
                                  StringRef Name, unsigned Line),
                    (Scope, File, Name, Line))
  DEFINE_MDNODE_GET(MDNamespace, (Metadata * Scope, Metadata *File,
                                  MDString *Name, unsigned Line),
                    (Scope, File, Name, Line))

  TempMDNamespace clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  Metadata *getScope() const { return getOperand(1); }
  StringRef getName() const { return getStringOperand(2); }

  MDString *getRawName() const { return getOperandAs<MDString>(2); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDNamespaceKind;
  }
};

/// \brief Base class for template parameters.
class MDTemplateParameter : public DebugNode {
protected:
  MDTemplateParameter(LLVMContext &Context, unsigned ID, StorageType Storage,
                      unsigned Tag, ArrayRef<Metadata *> Ops)
      : DebugNode(Context, ID, Storage, Tag, Ops) {}
  ~MDTemplateParameter() {}

public:
  StringRef getName() const { return getStringOperand(0); }
  Metadata *getType() const { return getOperand(1); }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDTemplateTypeParameterKind ||
           MD->getMetadataID() == MDTemplateValueParameterKind;
  }
};

class MDTemplateTypeParameter : public MDTemplateParameter {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDTemplateTypeParameter(LLVMContext &Context, StorageType Storage,
                          ArrayRef<Metadata *> Ops)
      : MDTemplateParameter(Context, MDTemplateTypeParameterKind, Storage,
                            dwarf::DW_TAG_template_type_parameter, Ops) {}
  ~MDTemplateTypeParameter() {}

  static MDTemplateTypeParameter *getImpl(LLVMContext &Context, StringRef Name,
                                          Metadata *Type, StorageType Storage,
                                          bool ShouldCreate = true) {
    return getImpl(Context, getCanonicalMDString(Context, Name), Type, Storage,
                   ShouldCreate);
  }
  static MDTemplateTypeParameter *getImpl(LLVMContext &Context, MDString *Name,
                                          Metadata *Type, StorageType Storage,
                                          bool ShouldCreate = true);

  TempMDTemplateTypeParameter cloneImpl() const {
    return getTemporary(getContext(), getName(), getType());
  }

public:
  DEFINE_MDNODE_GET(MDTemplateTypeParameter, (StringRef Name, Metadata *Type),
                    (Name, Type))
  DEFINE_MDNODE_GET(MDTemplateTypeParameter, (MDString * Name, Metadata *Type),
                    (Name, Type))

  TempMDTemplateTypeParameter clone() const { return cloneImpl(); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDTemplateTypeParameterKind;
  }
};

class MDTemplateValueParameter : public MDTemplateParameter {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDTemplateValueParameter(LLVMContext &Context, StorageType Storage,
                           unsigned Tag, ArrayRef<Metadata *> Ops)
      : MDTemplateParameter(Context, MDTemplateValueParameterKind, Storage, Tag,
                            Ops) {}
  ~MDTemplateValueParameter() {}

  static MDTemplateValueParameter *getImpl(LLVMContext &Context, unsigned Tag,
                                           StringRef Name, Metadata *Type,
                                           Metadata *Value, StorageType Storage,
                                           bool ShouldCreate = true) {
    return getImpl(Context, Tag, getCanonicalMDString(Context, Name), Type,
                   Value, Storage, ShouldCreate);
  }
  static MDTemplateValueParameter *getImpl(LLVMContext &Context, unsigned Tag,
                                           MDString *Name, Metadata *Type,
                                           Metadata *Value, StorageType Storage,
                                           bool ShouldCreate = true);

  TempMDTemplateValueParameter cloneImpl() const {
    return getTemporary(getContext(), getTag(), getName(), getType(),
                        getValue());
  }

public:
  DEFINE_MDNODE_GET(MDTemplateValueParameter, (unsigned Tag, StringRef Name,
                                               Metadata *Type, Metadata *Value),
                    (Tag, Name, Type, Value))
  DEFINE_MDNODE_GET(MDTemplateValueParameter, (unsigned Tag, MDString *Name,
                                               Metadata *Type, Metadata *Value),
                    (Tag, Name, Type, Value))

  TempMDTemplateValueParameter clone() const { return cloneImpl(); }

  Metadata *getValue() const { return getOperand(2); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDTemplateValueParameterKind;
  }
};

/// \brief Base class for variables.
///
/// TODO: Hardcode to DW_TAG_variable.
class MDVariable : public DebugNode {
  unsigned Line;

protected:
  MDVariable(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
             unsigned Line, ArrayRef<Metadata *> Ops)
      : DebugNode(C, ID, Storage, Tag, Ops), Line(Line) {}
  ~MDVariable() {}

public:
  unsigned getLine() const { return Line; }
  Metadata *getScope() const { return getOperand(0); }
  StringRef getName() const { return getStringOperand(1); }
  Metadata *getFile() const { return getOperand(2); }
  Metadata *getType() const { return getOperand(3); }

  MDString *getRawName() const { return getOperandAs<MDString>(1); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLocalVariableKind ||
           MD->getMetadataID() == MDGlobalVariableKind;
  }
};

/// \brief Global variables.
///
/// TODO: Remove DisplayName.  It's always equal to Name.
class MDGlobalVariable : public MDVariable {
  friend class LLVMContextImpl;
  friend class MDNode;

  bool IsLocalToUnit;
  bool IsDefinition;

  MDGlobalVariable(LLVMContext &C, StorageType Storage, unsigned Line,
                   bool IsLocalToUnit, bool IsDefinition,
                   ArrayRef<Metadata *> Ops)
      : MDVariable(C, MDGlobalVariableKind, Storage, dwarf::DW_TAG_variable,
                   Line, Ops),
        IsLocalToUnit(IsLocalToUnit), IsDefinition(IsDefinition) {}
  ~MDGlobalVariable() {}

  static MDGlobalVariable *
  getImpl(LLVMContext &Context, Metadata *Scope, StringRef Name,
          StringRef LinkageName, Metadata *File, unsigned Line, Metadata *Type,
          bool IsLocalToUnit, bool IsDefinition, Metadata *Variable,
          Metadata *StaticDataMemberDeclaration, StorageType Storage,
          bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, LinkageName), File, Line, Type,
                   IsLocalToUnit, IsDefinition, Variable,
                   StaticDataMemberDeclaration, Storage, ShouldCreate);
  }
  static MDGlobalVariable *
  getImpl(LLVMContext &Context, Metadata *Scope, MDString *Name,
          MDString *LinkageName, Metadata *File, unsigned Line, Metadata *Type,
          bool IsLocalToUnit, bool IsDefinition, Metadata *Variable,
          Metadata *StaticDataMemberDeclaration, StorageType Storage,
          bool ShouldCreate = true);

  TempMDGlobalVariable cloneImpl() const {
    return getTemporary(getContext(), getScope(), getName(), getLinkageName(),
                        getFile(), getLine(), getType(), isLocalToUnit(),
                        isDefinition(), getVariable(),
                        getStaticDataMemberDeclaration());
  }

public:
  DEFINE_MDNODE_GET(MDGlobalVariable,
                    (Metadata * Scope, StringRef Name, StringRef LinkageName,
                     Metadata *File, unsigned Line, Metadata *Type,
                     bool IsLocalToUnit, bool IsDefinition, Metadata *Variable,
                     Metadata *StaticDataMemberDeclaration),
                    (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
                     IsDefinition, Variable, StaticDataMemberDeclaration))
  DEFINE_MDNODE_GET(MDGlobalVariable,
                    (Metadata * Scope, MDString *Name, MDString *LinkageName,
                     Metadata *File, unsigned Line, Metadata *Type,
                     bool IsLocalToUnit, bool IsDefinition, Metadata *Variable,
                     Metadata *StaticDataMemberDeclaration),
                    (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
                     IsDefinition, Variable, StaticDataMemberDeclaration))

  TempMDGlobalVariable clone() const { return cloneImpl(); }

  bool isLocalToUnit() const { return IsLocalToUnit; }
  bool isDefinition() const { return IsDefinition; }
  StringRef getDisplayName() const { return getStringOperand(4); }
  StringRef getLinkageName() const { return getStringOperand(5); }
  Metadata *getVariable() const { return getOperand(6); }
  Metadata *getStaticDataMemberDeclaration() const { return getOperand(7); }

  MDString *getRawLinkageName() const { return getOperandAs<MDString>(5); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDGlobalVariableKind;
  }
};

/// \brief Local variable.
///
/// TODO: Split between arguments and otherwise.
/// TODO: Use \c DW_TAG_variable instead of fake tags.
/// TODO: Split up flags.
class MDLocalVariable : public MDVariable {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Arg;
  unsigned Flags;

  MDLocalVariable(LLVMContext &C, StorageType Storage, unsigned Tag,
                  unsigned Line, unsigned Arg, unsigned Flags,
                  ArrayRef<Metadata *> Ops)
      : MDVariable(C, MDLocalVariableKind, Storage, Tag, Line, Ops), Arg(Arg),
        Flags(Flags) {}
  ~MDLocalVariable() {}

  static MDLocalVariable *getImpl(LLVMContext &Context, unsigned Tag,
                                  Metadata *Scope, StringRef Name,
                                  Metadata *File, unsigned Line, Metadata *Type,
                                  unsigned Arg, unsigned Flags,
                                  Metadata *InlinedAt, StorageType Storage,
                                  bool ShouldCreate = true) {
    return getImpl(Context, Tag, Scope, getCanonicalMDString(Context, Name),
                   File, Line, Type, Arg, Flags, InlinedAt, Storage,
                   ShouldCreate);
  }
  static MDLocalVariable *getImpl(LLVMContext &Context, unsigned Tag,
                                  Metadata *Scope, MDString *Name,
                                  Metadata *File, unsigned Line, Metadata *Type,
                                  unsigned Arg, unsigned Flags,
                                  Metadata *InlinedAt, StorageType Storage,
                                  bool ShouldCreate = true);

  TempMDLocalVariable cloneImpl() const {
    return getTemporary(getContext(), getTag(), getScope(), getName(),
                        getFile(), getLine(), getType(), getArg(), getFlags(),
                        getInlinedAt());
  }

public:
  DEFINE_MDNODE_GET(MDLocalVariable,
                    (unsigned Tag, Metadata *Scope, StringRef Name,
                     Metadata *File, unsigned Line, Metadata *Type,
                     unsigned Arg, unsigned Flags,
                     Metadata *InlinedAt = nullptr),
                    (Tag, Scope, Name, File, Line, Type, Arg, Flags, InlinedAt))
  DEFINE_MDNODE_GET(MDLocalVariable,
                    (unsigned Tag, Metadata *Scope, MDString *Name,
                     Metadata *File, unsigned Line, Metadata *Type,
                     unsigned Arg, unsigned Flags,
                     Metadata *InlinedAt = nullptr),
                    (Tag, Scope, Name, File, Line, Type, Arg, Flags, InlinedAt))

  TempMDLocalVariable clone() const { return cloneImpl(); }

  unsigned getArg() const { return Arg; }
  unsigned getFlags() const { return Flags; }
  Metadata *getInlinedAt() const { return getOperand(4); }

  /// \brief Get an inlined version of this variable.
  ///
  /// Returns a version of this with \a getAlinedAt() set to \c InlinedAt.
  MDLocalVariable *withInline(MDLocation *InlinedAt) const {
    if (InlinedAt == getInlinedAt())
      return const_cast<MDLocalVariable *>(this);
    auto Temp = clone();
    Temp->replaceOperandWith(4, InlinedAt);
    return replaceWithUniqued(std::move(Temp));
  }
  MDLocalVariable *withoutInline() const { return withInline(nullptr); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLocalVariableKind;
  }
};

/// \brief DWARF expression.
///
/// TODO: Co-allocate the expression elements.
/// TODO: Separate from MDNode, or otherwise drop Distinct and Temporary
/// storage types.
class MDExpression : public MDNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  std::vector<uint64_t> Elements;

  MDExpression(LLVMContext &C, StorageType Storage, ArrayRef<uint64_t> Elements)
      : MDNode(C, MDExpressionKind, Storage, None),
        Elements(Elements.begin(), Elements.end()) {}
  ~MDExpression() {}

  static MDExpression *getImpl(LLVMContext &Context,
                               ArrayRef<uint64_t> Elements, StorageType Storage,
                               bool ShouldCreate = true);

  TempMDExpression cloneImpl() const {
    return getTemporary(getContext(), getElements());
  }

public:
  DEFINE_MDNODE_GET(MDExpression, (ArrayRef<uint64_t> Elements), (Elements))

  TempMDExpression clone() const { return cloneImpl(); }

  ArrayRef<uint64_t> getElements() const { return Elements; }

  unsigned getNumElements() const { return Elements.size(); }
  uint64_t getElement(unsigned I) const {
    assert(I < Elements.size() && "Index out of range");
    return Elements[I];
  }

  typedef ArrayRef<uint64_t>::iterator element_iterator;
  element_iterator elements_begin() const { return getElements().begin(); }
  element_iterator elements_end() const { return getElements().end(); }

  /// \brief A lightweight wrapper around an expression operand.
  ///
  /// TODO: Store arguments directly and change \a MDExpression to store a
  /// range of these.
  class ExprOperand {
    const uint64_t *Op;

  public:
    explicit ExprOperand(const uint64_t *Op) : Op(Op) {}

    const uint64_t *get() const { return Op; }

    /// \brief Get the operand code.
    uint64_t getOp() const { return *Op; }

    /// \brief Get an argument to the operand.
    ///
    /// Never returns the operand itself.
    uint64_t getArg(unsigned I) const { return Op[I + 1]; }

    unsigned getNumArgs() const { return getSize() - 1; }

    /// \brief Return the size of the operand.
    ///
    /// Return the number of elements in the operand (1 + args).
    unsigned getSize() const;
  };

  /// \brief An iterator for expression operands.
  class expr_op_iterator
      : public std::iterator<std::input_iterator_tag, ExprOperand> {
    ExprOperand Op;

  public:
    explicit expr_op_iterator(element_iterator I) : Op(I) {}

    element_iterator getBase() const { return Op.get(); }
    const ExprOperand &operator*() const { return Op; }
    const ExprOperand *operator->() const { return &Op; }

    expr_op_iterator &operator++() {
      increment();
      return *this;
    }
    expr_op_iterator operator++(int) {
      expr_op_iterator T(*this);
      increment();
      return T;
    }

    bool operator==(const expr_op_iterator &X) const {
      return getBase() == X.getBase();
    }
    bool operator!=(const expr_op_iterator &X) const {
      return getBase() != X.getBase();
    }

  private:
    void increment() { Op = ExprOperand(getBase() + Op.getSize()); }
  };

  /// \brief Visit the elements via ExprOperand wrappers.
  ///
  /// These range iterators visit elements through \a ExprOperand wrappers.
  /// This is not guaranteed to be a valid range unless \a isValid() gives \c
  /// true.
  ///
  /// \pre \a isValid() gives \c true.
  /// @{
  expr_op_iterator expr_op_begin() const {
    return expr_op_iterator(elements_begin());
  }
  expr_op_iterator expr_op_end() const {
    return expr_op_iterator(elements_end());
  }
  /// @}

  bool isValid() const;

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDExpressionKind;
  }
};

class MDObjCProperty : public DebugNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;
  unsigned Attributes;

  MDObjCProperty(LLVMContext &C, StorageType Storage, unsigned Line,
                 unsigned Attributes, ArrayRef<Metadata *> Ops)
      : DebugNode(C, MDObjCPropertyKind, Storage, dwarf::DW_TAG_APPLE_property,
                  Ops),
        Line(Line), Attributes(Attributes) {}
  ~MDObjCProperty() {}

  static MDObjCProperty *
  getImpl(LLVMContext &Context, StringRef Name, Metadata *File, unsigned Line,
          StringRef GetterName, StringRef SetterName, unsigned Attributes,
          Metadata *Type, StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, getCanonicalMDString(Context, Name), File, Line,
                   getCanonicalMDString(Context, GetterName),
                   getCanonicalMDString(Context, SetterName), Attributes, Type,
                   Storage, ShouldCreate);
  }
  static MDObjCProperty *getImpl(LLVMContext &Context, MDString *Name,
                                 Metadata *File, unsigned Line,
                                 MDString *GetterName, MDString *SetterName,
                                 unsigned Attributes, Metadata *Type,
                                 StorageType Storage, bool ShouldCreate = true);

  TempMDObjCProperty cloneImpl() const {
    return getTemporary(getContext(), getName(), getFile(), getLine(),
                        getGetterName(), getSetterName(), getAttributes(),
                        getType());
  }

public:
  DEFINE_MDNODE_GET(MDObjCProperty,
                    (StringRef Name, Metadata *File, unsigned Line,
                     StringRef GetterName, StringRef SetterName,
                     unsigned Attributes, Metadata *Type),
                    (Name, File, Line, GetterName, SetterName, Attributes,
                     Type))
  DEFINE_MDNODE_GET(MDObjCProperty,
                    (MDString * Name, Metadata *File, unsigned Line,
                     MDString *GetterName, MDString *SetterName,
                     unsigned Attributes, Metadata *Type),
                    (Name, File, Line, GetterName, SetterName, Attributes,
                     Type))

  TempMDObjCProperty clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  unsigned getAttributes() const { return Attributes; }
  StringRef getName() const { return getStringOperand(0); }
  Metadata *getFile() const { return getOperand(1); }
  StringRef getGetterName() const { return getStringOperand(2); }
  StringRef getSetterName() const { return getStringOperand(3); }
  Metadata *getType() const { return getOperand(4); }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }
  MDString *getRawGetterName() const { return getOperandAs<MDString>(2); }
  MDString *getRawSetterName() const { return getOperandAs<MDString>(3); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDObjCPropertyKind;
  }
};

class MDImportedEntity : public DebugNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  MDImportedEntity(LLVMContext &C, StorageType Storage, unsigned Tag,
                   unsigned Line, ArrayRef<Metadata *> Ops)
      : DebugNode(C, MDImportedEntityKind, Storage, Tag, Ops), Line(Line) {}
  ~MDImportedEntity() {}

  static MDImportedEntity *getImpl(LLVMContext &Context, unsigned Tag,
                                   Metadata *Scope, Metadata *Entity,
                                   unsigned Line, StringRef Name,
                                   StorageType Storage,
                                   bool ShouldCreate = true) {
    return getImpl(Context, Tag, Scope, Entity, Line,
                   getCanonicalMDString(Context, Name), Storage, ShouldCreate);
  }
  static MDImportedEntity *getImpl(LLVMContext &Context, unsigned Tag,
                                   Metadata *Scope, Metadata *Entity,
                                   unsigned Line, MDString *Name,
                                   StorageType Storage,
                                   bool ShouldCreate = true);

  TempMDImportedEntity cloneImpl() const {
    return getTemporary(getContext(), getTag(), getScope(), getEntity(),
                        getLine(), getName());
  }

public:
  DEFINE_MDNODE_GET(MDImportedEntity,
                    (unsigned Tag, Metadata *Scope, Metadata *Entity,
                     unsigned Line, StringRef Name = ""),
                    (Tag, Scope, Entity, Line, Name))
  DEFINE_MDNODE_GET(MDImportedEntity,
                    (unsigned Tag, Metadata *Scope, Metadata *Entity,
                     unsigned Line, MDString *Name),
                    (Tag, Scope, Entity, Line, Name))

  TempMDImportedEntity clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  Metadata *getScope() const { return getOperand(0); }
  Metadata *getEntity() const { return getOperand(1); }
  StringRef getName() const { return getStringOperand(2); }

  MDString *getRawName() const { return getOperandAs<MDString>(2); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDImportedEntityKind;
  }
};

} // end namespace llvm

#undef DEFINE_MDNODE_GET_UNPACK_IMPL
#undef DEFINE_MDNODE_GET_UNPACK
#undef DEFINE_MDNODE_GET

#endif
