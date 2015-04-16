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

/// \brief Pointer union between a subclass of DebugNode and MDString.
///
/// \a MDCompositeType can be referenced via an \a MDString unique identifier.
/// This class allows some type safety in the face of that, requiring either a
/// node of a particular type or an \a MDString.
template <class T> class TypedDebugNodeRef {
  const Metadata *MD = nullptr;

public:
  TypedDebugNodeRef() = default;
  TypedDebugNodeRef(std::nullptr_t) {}

  /// \brief Construct from a raw pointer.
  explicit TypedDebugNodeRef(const Metadata *MD) : MD(MD) {
    assert((!MD || isa<MDString>(MD) || isa<T>(MD)) && "Expected valid ref");
  }

  template <class U>
  TypedDebugNodeRef(
      const TypedDebugNodeRef<U> &X,
      typename std::enable_if<std::is_convertible<U *, T *>::value>::type * =
          nullptr)
      : MD(X) {}

  operator Metadata *() const { return const_cast<Metadata *>(MD); }

  bool operator==(const TypedDebugNodeRef<T> &X) const { return MD == X.MD; };
  bool operator!=(const TypedDebugNodeRef<T> &X) const { return MD != X.MD; };

  /// \brief Create a reference.
  ///
  /// Get a reference to \c N, using an \a MDString reference if available.
  static TypedDebugNodeRef get(const T *N);

  template <class MapTy> T *resolve(const MapTy &Map) const {
    if (!MD)
      return nullptr;

    if (auto *Typed = dyn_cast<T>(MD))
      return const_cast<T *>(Typed);

    auto *S = cast<MDString>(MD);
    auto I = Map.find(S);
    assert(I != Map.end() && "Missing identifier in type map");
    return cast<T>(I->second);
  }
};

typedef TypedDebugNodeRef<DebugNode> DebugNodeRef;
typedef TypedDebugNodeRef<MDScope> MDScopeRef;
typedef TypedDebugNodeRef<MDType> MDTypeRef;

class MDTypeRefArray {
  const MDTuple *N = nullptr;

public:
  MDTypeRefArray(const MDTuple *N) : N(N) {}

  explicit operator bool() const { return get(); }
  explicit operator MDTuple *() const { return get(); }

  MDTuple *get() const { return const_cast<MDTuple *>(N); }
  MDTuple *operator->() const { return get(); }
  MDTuple &operator*() const { return *get(); }

  // FIXME: Fix callers and remove condition on N.
  unsigned size() const { return N ? N->getNumOperands() : 0u; }
  MDTypeRef operator[](unsigned I) const { return MDTypeRef(N->getOperand(I)); }

  class iterator : std::iterator<std::input_iterator_tag, MDTypeRef,
                                 std::ptrdiff_t, void, MDTypeRef> {
    MDNode::op_iterator I = nullptr;

  public:
    iterator() = default;
    explicit iterator(MDNode::op_iterator I) : I(I) {}
    MDTypeRef operator*() const { return MDTypeRef(*I); }
    iterator &operator++() {
      ++I;
      return *this;
    }
    iterator operator++(int) {
      iterator Temp(*this);
      ++I;
      return Temp;
    }
    bool operator==(const iterator &X) const { return I == X.I; }
    bool operator!=(const iterator &X) const { return I != X.I; }
  };

  // FIXME: Fix callers and remove condition on N.
  iterator begin() const { return N ? iterator(N->op_begin()) : iterator(); }
  iterator end() const { return N ? iterator(N->op_end()) : iterator(); }
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
  ~DebugNode() = default;

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

  /// \brief Debug info flags.
  ///
  /// The three accessibility flags are mutually exclusive and rolled together
  /// in the first two bits.
  enum DIFlags {
#define HANDLE_DI_FLAG(ID, NAME) Flag##NAME = ID,
#include "llvm/IR/DebugInfoFlags.def"
    FlagAccessibility = FlagPrivate | FlagProtected | FlagPublic
  };

  static unsigned getFlag(StringRef Flag);
  static const char *getFlagString(unsigned Flag);

  /// \brief Split up a flags bitfield.
  ///
  /// Split \c Flags into \c SplitFlags, a vector of its components.  Returns
  /// any remaining (unrecognized) bits.
  static unsigned splitFlags(unsigned Flags,
                             SmallVectorImpl<unsigned> &SplitFlags);

  DebugNodeRef getRef() const { return DebugNodeRef::get(this); }

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

template <class T>
struct simplify_type<const TypedDebugNodeRef<T>> {
  typedef Metadata *SimpleType;
  static SimpleType getSimplifiedValue(const TypedDebugNodeRef<T> &MD) {
    return MD;
  }
};

template <class T>
struct simplify_type<TypedDebugNodeRef<T>>
    : simplify_type<const TypedDebugNodeRef<T>> {};

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
  int64_t LowerBound;

  MDSubrange(LLVMContext &C, StorageType Storage, int64_t Count,
             int64_t LowerBound)
      : DebugNode(C, MDSubrangeKind, Storage, dwarf::DW_TAG_subrange_type,
                  None),
        Count(Count), LowerBound(LowerBound) {}
  ~MDSubrange() = default;

  static MDSubrange *getImpl(LLVMContext &Context, int64_t Count,
                             int64_t LowerBound, StorageType Storage,
                             bool ShouldCreate = true);

  TempMDSubrange cloneImpl() const {
    return getTemporary(getContext(), getCount(), getLowerBound());
  }

public:
  DEFINE_MDNODE_GET(MDSubrange, (int64_t Count, int64_t LowerBound = 0),
                    (Count, LowerBound))

  TempMDSubrange clone() const { return cloneImpl(); }

  int64_t getLowerBound() const { return LowerBound; }
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
  ~MDEnumerator() = default;

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
  ~MDScope() = default;

public:
  MDFile *getFile() const { return cast_or_null<MDFile>(getRawFile()); }

  inline StringRef getFilename() const;
  inline StringRef getDirectory() const;

  StringRef getName() const;
  MDScopeRef getScope() const;

  /// \brief Return the raw underlying file.
  ///
  /// An \a MDFile is an \a MDScope, but it doesn't point at a separate file
  /// (it\em is the file).  If \c this is an \a MDFile, we need to return \c
  /// this.  Otherwise, return the first operand, which is where all other
  /// subclasses store their file pointer.
  Metadata *getRawFile() const {
    return isa<MDFile>(this) ? const_cast<MDScope *>(this)
                             : static_cast<Metadata *>(getOperand(0));
  }

  MDScopeRef getRef() const { return MDScopeRef::get(this); }

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

/// \brief File.
///
/// TODO: Merge with directory/file node (including users).
/// TODO: Canonicalize paths on creation.
class MDFile : public MDScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  MDFile(LLVMContext &C, StorageType Storage, ArrayRef<Metadata *> Ops)
      : MDScope(C, MDFileKind, Storage, dwarf::DW_TAG_file_type, Ops) {}
  ~MDFile() = default;

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

StringRef MDScope::getFilename() const {
  if (auto *F = getFile())
    return F->getFilename();
  return "";
}

StringRef MDScope::getDirectory() const {
  if (auto *F = getFile())
    return F->getDirectory();
  return "";
}

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
  ~MDType() = default;

public:
  TempMDType clone() const {
    return TempMDType(cast<MDType>(MDNode::clone().release()));
  }

  unsigned getLine() const { return Line; }
  uint64_t getSizeInBits() const { return SizeInBits; }
  uint64_t getAlignInBits() const { return AlignInBits; }
  uint64_t getOffsetInBits() const { return OffsetInBits; }
  unsigned getFlags() const { return Flags; }

  MDScopeRef getScope() const { return MDScopeRef(getRawScope()); }
  StringRef getName() const { return getStringOperand(2); }


  Metadata *getRawScope() const { return getOperand(1); }
  MDString *getRawName() const { return getOperandAs<MDString>(2); }

  void setFlags(unsigned NewFlags) {
    assert(!isUniqued() && "Cannot set flags on uniqued nodes");
    Flags = NewFlags;
  }

  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  bool isForwardDecl() const { return getFlags() & FlagFwdDecl; }
  bool isAppleBlockExtension() const { return getFlags() & FlagAppleBlock; }
  bool isBlockByrefStruct() const { return getFlags() & FlagBlockByrefStruct; }
  bool isVirtual() const { return getFlags() & FlagVirtual; }
  bool isArtificial() const { return getFlags() & FlagArtificial; }
  bool isObjectPointer() const { return getFlags() & FlagObjectPointer; }
  bool isObjcClassComplete() const {
    return getFlags() & FlagObjcClassComplete;
  }
  bool isVector() const { return getFlags() & FlagVector; }
  bool isStaticMember() const { return getFlags() & FlagStaticMember; }
  bool isLValueReference() const { return getFlags() & FlagLValueReference; }
  bool isRValueReference() const { return getFlags() & FlagRValueReference; }

  MDTypeRef getRef() const { return MDTypeRef::get(this); }

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

/// \brief Basic type, like 'int' or 'float'.
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
  ~MDBasicType() = default;

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
  ~MDDerivedTypeBase() = default;

public:
  MDTypeRef getBaseType() const { return MDTypeRef(getRawBaseType()); }
  Metadata *getRawBaseType() const { return getOperand(3); }

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
  ~MDDerivedType() = default;

  static MDDerivedType *getImpl(LLVMContext &Context, unsigned Tag,
                                StringRef Name, MDFile *File, unsigned Line,
                                MDScopeRef Scope, MDTypeRef BaseType,
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
                    (unsigned Tag, StringRef Name, MDFile *File, unsigned Line,
                     MDScopeRef Scope, MDTypeRef BaseType, uint64_t SizeInBits,
                     uint64_t AlignInBits, uint64_t OffsetInBits,
                     unsigned Flags, Metadata *ExtraData = nullptr),
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
  Metadata *getExtraData() const { return getRawExtraData(); }
  Metadata *getRawExtraData() const { return getOperand(4); }

  /// \brief Get casted version of extra data.
  /// @{
  MDTypeRef getClassType() const {
    assert(getTag() == dwarf::DW_TAG_ptr_to_member_type);
    return MDTypeRef(getExtraData());
  }
  MDObjCProperty *getObjCProperty() const {
    return dyn_cast_or_null<MDObjCProperty>(getExtraData());
  }
  Constant *getConstant() const {
    assert(getTag() == dwarf::DW_TAG_member && isStaticMember());
    if (auto *C = cast_or_null<ConstantAsMetadata>(getExtraData()))
      return C->getValue();
    return nullptr;
  }
  /// @}

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
  ~MDCompositeTypeBase() = default;

public:
  /// \brief Get the elements of the composite type.
  ///
  /// \note Calling this is only valid for \a MDCompositeType.  This assertion
  /// can be removed once \a MDSubroutineType has been separated from
  /// "composite types".
  DebugNodeArray getElements() const {
    assert(!isa<MDSubroutineType>(this) && "no elements for DISubroutineType");
    return cast_or_null<MDTuple>(getRawElements());
  }
  MDTypeRef getVTableHolder() const { return MDTypeRef(getRawVTableHolder()); }
  MDTemplateParameterArray getTemplateParams() const {
    return cast_or_null<MDTuple>(getRawTemplateParams());
  }
  StringRef getIdentifier() const { return getStringOperand(7); }
  unsigned getRuntimeLang() const { return RuntimeLang; }

  Metadata *getRawElements() const { return getOperand(4); }
  Metadata *getRawVTableHolder() const { return getOperand(5); }
  Metadata *getRawTemplateParams() const { return getOperand(6); }
  MDString *getRawIdentifier() const { return getOperandAs<MDString>(7); }

  /// \brief Replace operands.
  ///
  /// If this \a isUniqued() and not \a isResolved(), on a uniquing collision
  /// this will be RAUW'ed and deleted.  Use a \a TrackingMDRef to keep track
  /// of its movement if necessary.
  /// @{
  void replaceElements(DebugNodeArray Elements) {
#ifndef NDEBUG
    for (DebugNode *Op : getElements())
      assert(std::find(Elements->op_begin(), Elements->op_end(), Op) &&
             "Lost a member during member list replacement");
#endif
    replaceOperandWith(4, Elements.get());
  }
  void replaceVTableHolder(MDTypeRef VTableHolder) {
    replaceOperandWith(5, VTableHolder);
  }
  void replaceTemplateParams(MDTemplateParameterArray TemplateParams) {
    replaceOperandWith(6, TemplateParams.get());
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
  ~MDCompositeType() = default;

  static MDCompositeType *
  getImpl(LLVMContext &Context, unsigned Tag, StringRef Name, Metadata *File,
          unsigned Line, MDScopeRef Scope, MDTypeRef BaseType,
          uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
          uint64_t Flags, DebugNodeArray Elements, unsigned RuntimeLang,
          MDTypeRef VTableHolder, MDTemplateParameterArray TemplateParams,
          StringRef Identifier, StorageType Storage, bool ShouldCreate = true) {
    return getImpl(
        Context, Tag, getCanonicalMDString(Context, Name), File, Line, Scope,
        BaseType, SizeInBits, AlignInBits, OffsetInBits, Flags, Elements.get(),
        RuntimeLang, VTableHolder, TemplateParams.get(),
        getCanonicalMDString(Context, Identifier), Storage, ShouldCreate);
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
                    (unsigned Tag, StringRef Name, MDFile *File, unsigned Line,
                     MDScopeRef Scope, MDTypeRef BaseType, uint64_t SizeInBits,
                     uint64_t AlignInBits, uint64_t OffsetInBits,
                     unsigned Flags, DebugNodeArray Elements,
                     unsigned RuntimeLang, MDTypeRef VTableHolder,
                     MDTemplateParameterArray TemplateParams = nullptr,
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

template <class T> TypedDebugNodeRef<T> TypedDebugNodeRef<T>::get(const T *N) {
  if (N)
    if (auto *Composite = dyn_cast<MDCompositeType>(N))
      if (auto *S = Composite->getRawIdentifier())
        return TypedDebugNodeRef<T>(S);
  return TypedDebugNodeRef<T>(N);
}

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
  ~MDSubroutineType() = default;

  static MDSubroutineType *getImpl(LLVMContext &Context, unsigned Flags,
                                   MDTypeRefArray TypeArray,
                                   StorageType Storage,
                                   bool ShouldCreate = true) {
    return getImpl(Context, Flags, TypeArray.get(), Storage, ShouldCreate);
  }
  static MDSubroutineType *getImpl(LLVMContext &Context, unsigned Flags,
                                   Metadata *TypeArray, StorageType Storage,
                                   bool ShouldCreate = true);

  TempMDSubroutineType cloneImpl() const {
    return getTemporary(getContext(), getFlags(), getTypeArray());
  }

public:
  DEFINE_MDNODE_GET(MDSubroutineType,
                    (unsigned Flags, MDTypeRefArray TypeArray),
                    (Flags, TypeArray))
  DEFINE_MDNODE_GET(MDSubroutineType, (unsigned Flags, Metadata *TypeArray),
                    (Flags, TypeArray))

  TempMDSubroutineType clone() const { return cloneImpl(); }

  MDTypeRefArray getTypeArray() const {
    return cast_or_null<MDTuple>(getRawTypeArray());
  }
  Metadata *getRawTypeArray() const { return getRawElements(); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDSubroutineTypeKind;
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
  ~MDCompileUnit() = default;

  static MDCompileUnit *
  getImpl(LLVMContext &Context, unsigned SourceLanguage, MDFile *File,
          StringRef Producer, bool IsOptimized, StringRef Flags,
          unsigned RuntimeVersion, StringRef SplitDebugFilename,
          unsigned EmissionKind, MDCompositeTypeArray EnumTypes,
          MDTypeArray RetainedTypes, MDSubprogramArray Subprograms,
          MDGlobalVariableArray GlobalVariables,
          MDImportedEntityArray ImportedEntities, StorageType Storage,
          bool ShouldCreate = true) {
    return getImpl(
        Context, SourceLanguage, File, getCanonicalMDString(Context, Producer),
        IsOptimized, getCanonicalMDString(Context, Flags), RuntimeVersion,
        getCanonicalMDString(Context, SplitDebugFilename), EmissionKind,
        EnumTypes.get(), RetainedTypes.get(), Subprograms.get(),
        GlobalVariables.get(), ImportedEntities.get(), Storage, ShouldCreate);
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
                    (unsigned SourceLanguage, MDFile *File, StringRef Producer,
                     bool IsOptimized, StringRef Flags, unsigned RuntimeVersion,
                     StringRef SplitDebugFilename, unsigned EmissionKind,
                     MDCompositeTypeArray EnumTypes, MDTypeArray RetainedTypes,
                     MDSubprogramArray Subprograms,
                     MDGlobalVariableArray GlobalVariables,
                     MDImportedEntityArray ImportedEntities),
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
  MDCompositeTypeArray getEnumTypes() const {
    return cast_or_null<MDTuple>(getRawEnumTypes());
  }
  MDTypeArray getRetainedTypes() const {
    return cast_or_null<MDTuple>(getRawRetainedTypes());
  }
  MDSubprogramArray getSubprograms() const {
    return cast_or_null<MDTuple>(getRawSubprograms());
  }
  MDGlobalVariableArray getGlobalVariables() const {
    return cast_or_null<MDTuple>(getRawGlobalVariables());
  }
  MDImportedEntityArray getImportedEntities() const {
    return cast_or_null<MDTuple>(getRawImportedEntities());
  }

  MDString *getRawProducer() const { return getOperandAs<MDString>(1); }
  MDString *getRawFlags() const { return getOperandAs<MDString>(2); }
  MDString *getRawSplitDebugFilename() const {
    return getOperandAs<MDString>(3);
  }
  Metadata *getRawEnumTypes() const { return getOperand(4); }
  Metadata *getRawRetainedTypes() const { return getOperand(5); }
  Metadata *getRawSubprograms() const { return getOperand(6); }
  Metadata *getRawGlobalVariables() const { return getOperand(7); }
  Metadata *getRawImportedEntities() const { return getOperand(8); }

  /// \brief Replace arrays.
  ///
  /// If this \a isUniqued() and not \a isResolved(), it will be RAUW'ed and
  /// deleted on a uniquing collision.  In practice, uniquing collisions on \a
  /// MDCompileUnit should be fairly rare.
  /// @{
  void replaceSubprograms(MDSubprogramArray N) {
    replaceOperandWith(6, N.get());
  }
  void replaceGlobalVariables(MDGlobalVariableArray N) {
    replaceOperandWith(7, N.get());
  }
  /// @}

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDCompileUnitKind;
  }
};

/// \brief A scope for locals.
///
/// A legal scope for lexical blocks, local variables, and debug info
/// locations.  Subclasses are \a MDSubprogram, \a MDLexicalBlock, and \a
/// MDLexicalBlockFile.
class MDLocalScope : public MDScope {
protected:
  MDLocalScope(LLVMContext &C, unsigned ID, StorageType Storage, unsigned Tag,
               ArrayRef<Metadata *> Ops)
      : MDScope(C, ID, Storage, Tag, Ops) {}
  ~MDLocalScope() = default;

public:
  /// \brief Get the subprogram for this scope.
  ///
  /// Return this if it's an \a MDSubprogram; otherwise, look up the scope
  /// chain.
  MDSubprogram *getSubprogram() const;

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDSubprogramKind ||
           MD->getMetadataID() == MDLexicalBlockKind ||
           MD->getMetadataID() == MDLexicalBlockFileKind;
  }
};

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
  static MDLocation *getImpl(LLVMContext &Context, unsigned Line,
                             unsigned Column, MDLocalScope *Scope,
                             MDLocation *InlinedAt, StorageType Storage,
                             bool ShouldCreate = true) {
    return getImpl(Context, Line, Column, static_cast<Metadata *>(Scope),
                   static_cast<Metadata *>(InlinedAt), Storage, ShouldCreate);
  }

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
  DEFINE_MDNODE_GET(MDLocation,
                    (unsigned Line, unsigned Column, MDLocalScope *Scope,
                     MDLocation *InlinedAt = nullptr),
                    (Line, Column, Scope, InlinedAt))

  /// \brief Return a (temporary) clone of this.
  TempMDLocation clone() const { return cloneImpl(); }

  unsigned getLine() const { return SubclassData32; }
  unsigned getColumn() const { return SubclassData16; }
  MDLocalScope *getScope() const {
    return cast<MDLocalScope>(getRawScope());
  }
  MDLocation *getInlinedAt() const {
    return cast_or_null<MDLocation>(getRawInlinedAt());
  }

  MDFile *getFile() const { return getScope()->getFile(); }
  StringRef getFilename() const { return getScope()->getFilename(); }
  StringRef getDirectory() const { return getScope()->getDirectory(); }

  /// \brief Get the scope where this is inlined.
  ///
  /// Walk through \a getInlinedAt() and return \a getScope() from the deepest
  /// location.
  MDLocalScope *getInlinedAtScope() const {
    if (auto *IA = getInlinedAt())
      return IA->getInlinedAtScope();
    return getScope();
  }

  /// \brief Check whether this can be discriminated from another location.
  ///
  /// Check \c this can be discriminated from \c RHS in a linetable entry.
  /// Scope and inlined-at chains are not recorded in the linetable, so they
  /// cannot be used to distinguish basic blocks.
  ///
  /// The current implementation is weaker than it should be, since it just
  /// checks filename and line.
  ///
  /// FIXME: Add a check for getDiscriminator().
  /// FIXME: Add a check for getColumn().
  /// FIXME: Change the getFilename() check to getFile() (or add one for
  /// getDirectory()).
  bool canDiscriminate(const MDLocation &RHS) const {
    return getFilename() != RHS.getFilename() || getLine() != RHS.getLine();
  }

  /// \brief Get the DWARF discriminator.
  ///
  /// DWARF discriminators distinguish identical file locations between
  /// instructions that are on different basic blocks.
  inline unsigned getDiscriminator() const;

  /// \brief Compute new discriminator in the given context.
  ///
  /// This modifies the \a LLVMContext that \c this is in to increment the next
  /// discriminator for \c this's line/filename combination.
  ///
  /// FIXME: Delete this.  See comments in implementation and at the only call
  /// site in \a AddDiscriminators::runOnFunction().
  unsigned computeNewDiscriminator() const;

  Metadata *getRawScope() const { return getOperand(0); }
  Metadata *getRawInlinedAt() const {
    if (getNumOperands() == 2)
      return getOperand(1);
    return nullptr;
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLocationKind;
  }
};

/// \brief Subprogram description.
///
/// TODO: Remove DisplayName.  It's always equal to Name.
/// TODO: Split up flags.
class MDSubprogram : public MDLocalScope {
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
      : MDLocalScope(C, MDSubprogramKind, Storage, dwarf::DW_TAG_subprogram,
                     Ops),
        Line(Line), ScopeLine(ScopeLine), Virtuality(Virtuality),
        VirtualIndex(VirtualIndex), Flags(Flags), IsLocalToUnit(IsLocalToUnit),
        IsDefinition(IsDefinition), IsOptimized(IsOptimized) {}
  ~MDSubprogram() = default;

  static MDSubprogram *
  getImpl(LLVMContext &Context, MDScopeRef Scope, StringRef Name,
          StringRef LinkageName, MDFile *File, unsigned Line,
          MDSubroutineType *Type, bool IsLocalToUnit, bool IsDefinition,
          unsigned ScopeLine, MDTypeRef ContainingType, unsigned Virtuality,
          unsigned VirtualIndex, unsigned Flags, bool IsOptimized,
          Constant *Function, MDTemplateParameterArray TemplateParams,
          MDSubprogram *Declaration, MDLocalVariableArray Variables,
          StorageType Storage, bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, LinkageName), File, Line, Type,
                   IsLocalToUnit, IsDefinition, ScopeLine, ContainingType,
                   Virtuality, VirtualIndex, Flags, IsOptimized,
                   Function ? ConstantAsMetadata::get(Function) : nullptr,
                   TemplateParams.get(), Declaration, Variables.get(), Storage,
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
                        isOptimized(), getFunctionConstant(),
                        getTemplateParams(), getDeclaration(), getVariables());
  }

public:
  DEFINE_MDNODE_GET(MDSubprogram,
                    (MDScopeRef Scope, StringRef Name, StringRef LinkageName,
                     MDFile *File, unsigned Line, MDSubroutineType *Type,
                     bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
                     MDTypeRef ContainingType, unsigned Virtuality,
                     unsigned VirtualIndex, unsigned Flags, bool IsOptimized,
                     Constant *Function = nullptr,
                     MDTemplateParameterArray TemplateParams = nullptr,
                     MDSubprogram *Declaration = nullptr,
                     MDLocalVariableArray Variables = nullptr),
                    (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
                     IsDefinition, ScopeLine, ContainingType, Virtuality,
                     VirtualIndex, Flags, IsOptimized, Function, TemplateParams,
                     Declaration, Variables))
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

  unsigned isArtificial() const { return getFlags() & FlagArtificial; }
  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  bool isExplicit() const { return getFlags() & FlagExplicit; }
  bool isPrototyped() const { return getFlags() & FlagPrototyped; }

  /// \brief Check if this is reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 reference-qualified non-static
  /// member function (void foo() &).
  unsigned isLValueReference() const {
    return getFlags() & FlagLValueReference;
  }

  /// \brief Check if this is rvalue-reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 rvalue-reference-qualified
  /// non-static member function (void foo() &&).
  unsigned isRValueReference() const {
    return getFlags() & FlagRValueReference;
  }

  MDScopeRef getScope() const { return MDScopeRef(getRawScope()); }

  StringRef getName() const { return getStringOperand(2); }
  StringRef getDisplayName() const { return getStringOperand(3); }
  StringRef getLinkageName() const { return getStringOperand(4); }

  MDString *getRawName() const { return getOperandAs<MDString>(2); }
  MDString *getRawLinkageName() const { return getOperandAs<MDString>(4); }

  MDSubroutineType *getType() const {
    return cast_or_null<MDSubroutineType>(getRawType());
  }
  MDTypeRef getContainingType() const {
    return MDTypeRef(getRawContainingType());
  }

  Constant *getFunctionConstant() const {
    if (auto *C = cast_or_null<ConstantAsMetadata>(getRawFunction()))
      return C->getValue();
    return nullptr;
  }
  MDTemplateParameterArray getTemplateParams() const {
    return cast_or_null<MDTuple>(getRawTemplateParams());
  }
  MDSubprogram *getDeclaration() const {
    return cast_or_null<MDSubprogram>(getRawDeclaration());
  }
  MDLocalVariableArray getVariables() const {
    return cast_or_null<MDTuple>(getRawVariables());
  }

  Metadata *getRawScope() const { return getOperand(1); }
  Metadata *getRawType() const { return getOperand(5); }
  Metadata *getRawContainingType() const { return getOperand(6); }
  Metadata *getRawFunction() const { return getOperand(7); }
  Metadata *getRawTemplateParams() const { return getOperand(8); }
  Metadata *getRawDeclaration() const { return getOperand(9); }
  Metadata *getRawVariables() const { return getOperand(10); }

  /// \brief Get a pointer to the function this subprogram describes.
  ///
  /// This dyn_casts \a getFunctionConstant() to \a Function.
  ///
  /// FIXME: Should this be looking through bitcasts?
  Function *getFunction() const;

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

  /// \brief Check if this subprogram decribes the given function.
  ///
  /// FIXME: Should this be looking through bitcasts?
  bool describes(const Function *F) const;

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDSubprogramKind;
  }
};

class MDLexicalBlockBase : public MDLocalScope {
protected:
  MDLexicalBlockBase(LLVMContext &C, unsigned ID, StorageType Storage,
                     ArrayRef<Metadata *> Ops)
      : MDLocalScope(C, ID, Storage, dwarf::DW_TAG_lexical_block, Ops) {}
  ~MDLexicalBlockBase() = default;

public:
  MDLocalScope *getScope() const { return cast<MDLocalScope>(getRawScope()); }

  Metadata *getRawScope() const { return getOperand(1); }

  /// \brief Forwarding accessors to LexicalBlock.
  ///
  /// TODO: Remove these and update code to use \a MDLexicalBlock directly.
  /// @{
  inline unsigned getLine() const;
  inline unsigned getColumn() const;
  /// @}
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
  ~MDLexicalBlock() = default;

  static MDLexicalBlock *getImpl(LLVMContext &Context, MDLocalScope *Scope,
                                 MDFile *File, unsigned Line, unsigned Column,
                                 StorageType Storage,
                                 bool ShouldCreate = true) {
    return getImpl(Context, static_cast<Metadata *>(Scope),
                   static_cast<Metadata *>(File), Line, Column, Storage,
                   ShouldCreate);
  }

  static MDLexicalBlock *getImpl(LLVMContext &Context, Metadata *Scope,
                                 Metadata *File, unsigned Line, unsigned Column,
                                 StorageType Storage, bool ShouldCreate = true);

  TempMDLexicalBlock cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(), getLine(),
                        getColumn());
  }

public:
  DEFINE_MDNODE_GET(MDLexicalBlock, (MDLocalScope * Scope, MDFile *File,
                                     unsigned Line, unsigned Column),
                    (Scope, File, Line, Column))
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

unsigned MDLexicalBlockBase::getLine() const {
  if (auto *N = dyn_cast<MDLexicalBlock>(this))
    return N->getLine();
  return 0;
}

unsigned MDLexicalBlockBase::getColumn() const {
  if (auto *N = dyn_cast<MDLexicalBlock>(this))
    return N->getColumn();
  return 0;
}

class MDLexicalBlockFile : public MDLexicalBlockBase {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Discriminator;

  MDLexicalBlockFile(LLVMContext &C, StorageType Storage,
                     unsigned Discriminator, ArrayRef<Metadata *> Ops)
      : MDLexicalBlockBase(C, MDLexicalBlockFileKind, Storage, Ops),
        Discriminator(Discriminator) {}
  ~MDLexicalBlockFile() = default;

  static MDLexicalBlockFile *getImpl(LLVMContext &Context, MDLocalScope *Scope,
                                     MDFile *File, unsigned Discriminator,
                                     StorageType Storage,
                                     bool ShouldCreate = true) {
    return getImpl(Context, static_cast<Metadata *>(Scope),
                   static_cast<Metadata *>(File), Discriminator, Storage,
                   ShouldCreate);
  }

  static MDLexicalBlockFile *getImpl(LLVMContext &Context, Metadata *Scope,
                                     Metadata *File, unsigned Discriminator,
                                     StorageType Storage,
                                     bool ShouldCreate = true);

  TempMDLexicalBlockFile cloneImpl() const {
    return getTemporary(getContext(), getScope(), getFile(),
                        getDiscriminator());
  }

public:
  DEFINE_MDNODE_GET(MDLexicalBlockFile, (MDLocalScope * Scope, MDFile *File,
                                         unsigned Discriminator),
                    (Scope, File, Discriminator))
  DEFINE_MDNODE_GET(MDLexicalBlockFile,
                    (Metadata * Scope, Metadata *File, unsigned Discriminator),
                    (Scope, File, Discriminator))

  TempMDLexicalBlockFile clone() const { return cloneImpl(); }

  // TODO: Remove these once they're gone from MDLexicalBlockBase.
  unsigned getLine() const = delete;
  unsigned getColumn() const = delete;

  unsigned getDiscriminator() const { return Discriminator; }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLexicalBlockFileKind;
  }
};

unsigned MDLocation::getDiscriminator() const {
  if (auto *F = dyn_cast<MDLexicalBlockFile>(getScope()))
    return F->getDiscriminator();
  return 0;
}

class MDNamespace : public MDScope {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  MDNamespace(LLVMContext &Context, StorageType Storage, unsigned Line,
              ArrayRef<Metadata *> Ops)
      : MDScope(Context, MDNamespaceKind, Storage, dwarf::DW_TAG_namespace,
                Ops),
        Line(Line) {}
  ~MDNamespace() = default;

  static MDNamespace *getImpl(LLVMContext &Context, MDScope *Scope,
                              MDFile *File, StringRef Name, unsigned Line,
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
  DEFINE_MDNODE_GET(MDNamespace, (MDScope * Scope, MDFile *File, StringRef Name,
                                  unsigned Line),
                    (Scope, File, Name, Line))
  DEFINE_MDNODE_GET(MDNamespace, (Metadata * Scope, Metadata *File,
                                  MDString *Name, unsigned Line),
                    (Scope, File, Name, Line))

  TempMDNamespace clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  MDScope *getScope() const { return cast_or_null<MDScope>(getRawScope()); }
  StringRef getName() const { return getStringOperand(2); }

  Metadata *getRawScope() const { return getOperand(1); }
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
  ~MDTemplateParameter() = default;

public:
  StringRef getName() const { return getStringOperand(0); }
  MDTypeRef getType() const { return MDTypeRef(getRawType()); }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }
  Metadata *getRawType() const { return getOperand(1); }

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
  ~MDTemplateTypeParameter() = default;

  static MDTemplateTypeParameter *getImpl(LLVMContext &Context, StringRef Name,
                                          MDTypeRef Type, StorageType Storage,
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
  DEFINE_MDNODE_GET(MDTemplateTypeParameter, (StringRef Name, MDTypeRef Type),
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
  ~MDTemplateValueParameter() = default;

  static MDTemplateValueParameter *getImpl(LLVMContext &Context, unsigned Tag,
                                           StringRef Name, MDTypeRef Type,
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
                                               MDTypeRef Type, Metadata *Value),
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
  ~MDVariable() = default;

public:
  unsigned getLine() const { return Line; }
  MDScope *getScope() const { return cast_or_null<MDScope>(getRawScope()); }
  StringRef getName() const { return getStringOperand(1); }
  MDFile *getFile() const { return cast_or_null<MDFile>(getRawFile()); }
  MDTypeRef getType() const { return MDTypeRef(getRawType()); }

  StringRef getFilename() const {
    if (auto *F = getFile())
      return F->getFilename();
    return "";
  }
  StringRef getDirectory() const {
    if (auto *F = getFile())
      return F->getDirectory();
    return "";
  }

  Metadata *getRawScope() const { return getOperand(0); }
  MDString *getRawName() const { return getOperandAs<MDString>(1); }
  Metadata *getRawFile() const { return getOperand(2); }
  Metadata *getRawType() const { return getOperand(3); }

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
  ~MDGlobalVariable() = default;

  static MDGlobalVariable *
  getImpl(LLVMContext &Context, MDScope *Scope, StringRef Name,
          StringRef LinkageName, MDFile *File, unsigned Line, MDTypeRef Type,
          bool IsLocalToUnit, bool IsDefinition, Constant *Variable,
          MDDerivedType *StaticDataMemberDeclaration, StorageType Storage,
          bool ShouldCreate = true) {
    return getImpl(Context, Scope, getCanonicalMDString(Context, Name),
                   getCanonicalMDString(Context, LinkageName), File, Line, Type,
                   IsLocalToUnit, IsDefinition,
                   Variable ? ConstantAsMetadata::get(Variable) : nullptr,
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
                    (MDScope * Scope, StringRef Name, StringRef LinkageName,
                     MDFile *File, unsigned Line, MDTypeRef Type,
                     bool IsLocalToUnit, bool IsDefinition, Constant *Variable,
                     MDDerivedType *StaticDataMemberDeclaration),
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
  Constant *getVariable() const {
    if (auto *C = cast_or_null<ConstantAsMetadata>(getRawVariable()))
      return dyn_cast<Constant>(C->getValue());
    return nullptr;
  }
  MDDerivedType *getStaticDataMemberDeclaration() const {
    return cast_or_null<MDDerivedType>(getRawStaticDataMemberDeclaration());
  }

  MDString *getRawLinkageName() const { return getOperandAs<MDString>(5); }
  Metadata *getRawVariable() const { return getOperand(6); }
  Metadata *getRawStaticDataMemberDeclaration() const { return getOperand(7); }

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
  ~MDLocalVariable() = default;

  static MDLocalVariable *getImpl(LLVMContext &Context, unsigned Tag,
                                  MDScope *Scope, StringRef Name, MDFile *File,
                                  unsigned Line, MDTypeRef Type, unsigned Arg,
                                  unsigned Flags, StorageType Storage,
                                  bool ShouldCreate = true) {
    return getImpl(Context, Tag, Scope, getCanonicalMDString(Context, Name),
                   File, Line, Type, Arg, Flags, Storage, ShouldCreate);
  }
  static MDLocalVariable *
  getImpl(LLVMContext &Context, unsigned Tag, Metadata *Scope, MDString *Name,
          Metadata *File, unsigned Line, Metadata *Type, unsigned Arg,
          unsigned Flags, StorageType Storage, bool ShouldCreate = true);

  TempMDLocalVariable cloneImpl() const {
    return getTemporary(getContext(), getTag(), getScope(), getName(),
                        getFile(), getLine(), getType(), getArg(), getFlags());
  }

public:
  DEFINE_MDNODE_GET(MDLocalVariable,
                    (unsigned Tag, MDLocalScope *Scope, StringRef Name,
                     MDFile *File, unsigned Line, MDTypeRef Type, unsigned Arg,
                     unsigned Flags),
                    (Tag, Scope, Name, File, Line, Type, Arg, Flags))
  DEFINE_MDNODE_GET(MDLocalVariable,
                    (unsigned Tag, Metadata *Scope, MDString *Name,
                     Metadata *File, unsigned Line, Metadata *Type,
                     unsigned Arg, unsigned Flags),
                    (Tag, Scope, Name, File, Line, Type, Arg, Flags))

  TempMDLocalVariable clone() const { return cloneImpl(); }

  /// \brief Get the local scope for this variable.
  ///
  /// Variables must be defined in a local scope.
  MDLocalScope *getScope() const {
    return cast<MDLocalScope>(MDVariable::getScope());
  }

  unsigned getArg() const { return Arg; }
  unsigned getFlags() const { return Flags; }

  bool isArtificial() const { return getFlags() & FlagArtificial; }
  bool isObjectPointer() const { return getFlags() & FlagObjectPointer; }

  /// \brief Check that a location is valid for this variable.
  ///
  /// Check that \c DL exists, is in the same subprogram, and has the same
  /// inlined-at location as \c this.  (Otherwise, it's not a valid attachemnt
  /// to a \a DbgInfoIntrinsic.)
  bool isValidLocationForIntrinsic(const MDLocation *DL) const {
    return DL && getScope()->getSubprogram() == DL->getScope()->getSubprogram();
  }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDLocalVariableKind;
  }
};

/// \brief DWARF expression.
///
/// This is (almost) a DWARF expression that modifies the location of a
/// variable or (or the location of a single piece of a variable).
///
/// FIXME: Instead of DW_OP_plus taking an argument, this should use DW_OP_const
/// and have DW_OP_plus consume the topmost elements on the stack.
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
  ~MDExpression() = default;

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

  /// \brief Return whether this is a piece of an aggregate variable.
  bool isBitPiece() const;

  /// \brief Return the offset of this piece in bits.
  uint64_t getBitPieceOffset() const;

  /// \brief Return the size of this piece in bits.
  uint64_t getBitPieceSize() const;

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

    /// \brief Get the next iterator.
    ///
    /// \a std::next() doesn't work because this is technically an
    /// input_iterator, but it's a perfectly valid operation.  This is an
    /// accessor to provide the same functionality.
    expr_op_iterator getNext() const { return ++expr_op_iterator(*this); }

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
  ~MDObjCProperty() = default;

  static MDObjCProperty *
  getImpl(LLVMContext &Context, StringRef Name, MDFile *File, unsigned Line,
          StringRef GetterName, StringRef SetterName, unsigned Attributes,
          MDType *Type, StorageType Storage, bool ShouldCreate = true) {
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
                    (StringRef Name, MDFile *File, unsigned Line,
                     StringRef GetterName, StringRef SetterName,
                     unsigned Attributes, MDType *Type),
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
  MDFile *getFile() const { return cast_or_null<MDFile>(getRawFile()); }
  StringRef getGetterName() const { return getStringOperand(2); }
  StringRef getSetterName() const { return getStringOperand(3); }

  /// \brief Get the type.
  ///
  /// \note Objective-C doesn't have an ODR, so there is no benefit in storing
  /// the type as a DITypeRef here.
  MDType *getType() const { return cast_or_null<MDType>(getRawType()); }

  StringRef getFilename() const {
    if (auto *F = getFile())
      return F->getFilename();
    return "";
  }
  StringRef getDirectory() const {
    if (auto *F = getFile())
      return F->getDirectory();
    return "";
  }

  MDString *getRawName() const { return getOperandAs<MDString>(0); }
  Metadata *getRawFile() const { return getOperand(1); }
  MDString *getRawGetterName() const { return getOperandAs<MDString>(2); }
  MDString *getRawSetterName() const { return getOperandAs<MDString>(3); }
  Metadata *getRawType() const { return getOperand(4); }

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDObjCPropertyKind;
  }
};

/// \brief An imported module (C++ using directive or similar).
class MDImportedEntity : public DebugNode {
  friend class LLVMContextImpl;
  friend class MDNode;

  unsigned Line;

  MDImportedEntity(LLVMContext &C, StorageType Storage, unsigned Tag,
                   unsigned Line, ArrayRef<Metadata *> Ops)
      : DebugNode(C, MDImportedEntityKind, Storage, Tag, Ops), Line(Line) {}
  ~MDImportedEntity() = default;

  static MDImportedEntity *getImpl(LLVMContext &Context, unsigned Tag,
                                   MDScope *Scope, DebugNodeRef Entity,
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
                    (unsigned Tag, MDScope *Scope, DebugNodeRef Entity,
                     unsigned Line, StringRef Name = ""),
                    (Tag, Scope, Entity, Line, Name))
  DEFINE_MDNODE_GET(MDImportedEntity,
                    (unsigned Tag, Metadata *Scope, Metadata *Entity,
                     unsigned Line, MDString *Name),
                    (Tag, Scope, Entity, Line, Name))

  TempMDImportedEntity clone() const { return cloneImpl(); }

  unsigned getLine() const { return Line; }
  MDScope *getScope() const { return cast_or_null<MDScope>(getRawScope()); }
  DebugNodeRef getEntity() const { return DebugNodeRef(getRawEntity()); }
  StringRef getName() const { return getStringOperand(2); }

  Metadata *getRawScope() const { return getOperand(0); }
  Metadata *getRawEntity() const { return getOperand(1); }
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
