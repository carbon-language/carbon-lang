//===- DebugInfo.h - Debug Information Helpers ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a bunch of datatypes that are useful for creating and
// walking debug info in LLVM IR form. They essentially provide wrappers around
// the information in the global variables that's needed when constructing the
// DWARF information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFO_H
#define LLVM_IR_DEBUGINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include <iterator>

namespace llvm {
class BasicBlock;
class Constant;
class Function;
class GlobalVariable;
class Module;
class Type;
class Value;
class DbgDeclareInst;
class DbgValueInst;
class Instruction;
class Metadata;
class MDNode;
class MDString;
class NamedMDNode;
class LLVMContext;
class raw_ostream;

class DIFile;
class DISubprogram;
class DILexicalBlock;
class DILexicalBlockFile;
class DIVariable;
class DIType;
class DIScope;
class DIObjCProperty;

/// \brief Maps from type identifier to the actual MDNode.
typedef DenseMap<const MDString *, MDNode *> DITypeIdentifierMap;

class DIHeaderFieldIterator
    : public std::iterator<std::input_iterator_tag, StringRef, std::ptrdiff_t,
                           const StringRef *, StringRef> {
  StringRef Header;
  StringRef Current;

public:
  DIHeaderFieldIterator() {}
  explicit DIHeaderFieldIterator(StringRef Header)
      : Header(Header), Current(Header.slice(0, Header.find('\0'))) {}
  StringRef operator*() const { return Current; }
  const StringRef *operator->() const { return &Current; }
  DIHeaderFieldIterator &operator++() {
    increment();
    return *this;
  }
  DIHeaderFieldIterator operator++(int) {
    DIHeaderFieldIterator X(*this);
    increment();
    return X;
  }
  bool operator==(const DIHeaderFieldIterator &X) const {
    return Current.data() == X.Current.data();
  }
  bool operator!=(const DIHeaderFieldIterator &X) const {
    return !(*this == X);
  }

  StringRef getHeader() const { return Header; }
  StringRef getCurrent() const { return Current; }
  StringRef getPrefix() const {
    if (Current.begin() == Header.begin())
      return StringRef();
    return Header.slice(0, Current.begin() - Header.begin() - 1);
  }
  StringRef getSuffix() const {
    if (Current.end() == Header.end())
      return StringRef();
    return Header.slice(Current.end() - Header.begin() + 1, StringRef::npos);
  }

  /// \brief Get the current field as a number.
  ///
  /// Convert the current field into a number.  Return \c 0 on error.
  template <class T> T getNumber() const {
    T Int;
    if (getCurrent().getAsInteger(0, Int))
      return 0;
    return Int;
  }

private:
  void increment() {
    assert(Current.data() != nullptr && "Cannot increment past the end");
    StringRef Suffix = getSuffix();
    Current = Suffix.slice(0, Suffix.find('\0'));
  }
};

/// \brief A thin wraper around MDNode to access encoded debug info.
///
/// This should not be stored in a container, because the underlying MDNode may
/// change in certain situations.
class DIDescriptor {
  // Befriends DIRef so DIRef can befriend the protected member
  // function: getFieldAs<DIRef>.
  template <typename T> friend class DIRef;

public:
  /// \brief Accessibility flags.
  ///
  /// The three accessibility flags are mutually exclusive and rolled together
  /// in the first two bits.
  enum {
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

protected:
  const MDNode *DbgNode;

  StringRef getStringField(unsigned Elt) const;
  unsigned getUnsignedField(unsigned Elt) const {
    return (unsigned)getUInt64Field(Elt);
  }
  uint64_t getUInt64Field(unsigned Elt) const;
  int64_t getInt64Field(unsigned Elt) const;
  DIDescriptor getDescriptorField(unsigned Elt) const;

  template <typename DescTy> DescTy getFieldAs(unsigned Elt) const {
    return DescTy(getDescriptorField(Elt));
  }

  GlobalVariable *getGlobalVariableField(unsigned Elt) const;
  Constant *getConstantField(unsigned Elt) const;
  Function *getFunctionField(unsigned Elt) const;

public:
  explicit DIDescriptor(const MDNode *N = nullptr) : DbgNode(N) {}

  bool Verify() const;

  MDNode *get() const { return const_cast<MDNode *>(DbgNode); }
  operator MDNode *() const { return get(); }
  MDNode *operator->() const { return get(); }

  // An explicit operator bool so that we can do testing of DI values
  // easily.
  // FIXME: This operator bool isn't actually protecting anything at the
  // moment due to the conversion operator above making DIDescriptor nodes
  // implicitly convertable to bool.
  explicit operator bool() const { return DbgNode != nullptr; }

  bool operator==(DIDescriptor Other) const { return DbgNode == Other.DbgNode; }
  bool operator!=(DIDescriptor Other) const { return !operator==(Other); }

  StringRef getHeader() const { return getStringField(0); }

  size_t getNumHeaderFields() const {
    return std::distance(DIHeaderFieldIterator(getHeader()),
                         DIHeaderFieldIterator());
  }

  DIHeaderFieldIterator header_begin() const {
    return DIHeaderFieldIterator(getHeader());
  }
  DIHeaderFieldIterator header_end() const { return DIHeaderFieldIterator(); }

  DIHeaderFieldIterator getHeaderIterator(unsigned Index) const {
    // Since callers expect an empty string for out-of-range accesses, we can't
    // use std::advance() here.
    for (auto I = header_begin(), E = header_end(); I != E; ++I, --Index)
      if (!Index)
        return I;
    return header_end();
  }

  StringRef getHeaderField(unsigned Index) const {
    return *getHeaderIterator(Index);
  }

  template <class T> T getHeaderFieldAs(unsigned Index) const {
    return getHeaderIterator(Index).getNumber<T>();
  }

  uint16_t getTag() const {
    if (auto *N = dyn_cast_or_null<DebugNode>(get()))
      return N->getTag();
    return 0;
  }

  bool isDerivedType() const { return get() && isa<MDDerivedTypeBase>(get()); }
  bool isCompositeType() const {
    return get() && isa<MDCompositeTypeBase>(get());
  }
  bool isSubroutineType() const {
    return get() && isa<MDSubroutineType>(get());
  }
  bool isBasicType() const { return get() && isa<MDBasicType>(get()); }
  bool isVariable() const { return get() && isa<MDLocalVariable>(get()); }
  bool isSubprogram() const { return get() && isa<MDSubprogram>(get()); }
  bool isGlobalVariable() const {
    return get() && isa<MDGlobalVariable>(get());
  }
  bool isScope() const { return get() && isa<MDScope>(get()); }
  bool isFile() const { return get() && isa<MDFile>(get()); }
  bool isCompileUnit() const { return get() && isa<MDCompileUnit>(get()); }
  bool isNameSpace() const{ return get() && isa<MDNamespace>(get()); }
  bool isLexicalBlockFile() const {
    return get() && isa<MDLexicalBlockFile>(get());
  }
  bool isLexicalBlock() const {
    return get() && isa<MDLexicalBlockBase>(get());
  }
  bool isSubrange() const { return get() && isa<MDSubrange>(get()); }
  bool isEnumerator() const { return get() && isa<MDEnumerator>(get()); }
  bool isType() const { return get() && isa<MDType>(get()); }
  bool isTemplateTypeParameter() const {
    return get() && isa<MDTemplateTypeParameter>(get());
  }
  bool isTemplateValueParameter() const {
    return get() && isa<MDTemplateValueParameter>(get());
  }
  bool isObjCProperty() const { return get() && isa<MDObjCProperty>(get()); }
  bool isImportedEntity() const {
    return get() && isa<MDImportedEntity>(get());
  }
  bool isExpression() const { return get() && isa<MDExpression>(get()); }

  void print(raw_ostream &OS) const;
  void dump() const;

  /// \brief Replace all uses of debug info referenced by this descriptor.
  void replaceAllUsesWith(LLVMContext &VMContext, DIDescriptor D);
  void replaceAllUsesWith(MDNode *D);
};

#define RETURN_FROM_RAW(VALID, DEFAULT)                                        \
  do {                                                                         \
    if (auto *N = getRaw())                                                    \
      return VALID;                                                            \
    return DEFAULT;                                                            \
  } while (false)
#define RETURN_DESCRIPTOR_FROM_RAW(DESC, VALID)                                \
  do {                                                                         \
    if (auto *N = getRaw())                                                    \
      return DESC(dyn_cast_or_null<MDNode>(VALID));                            \
    return DESC(static_cast<const MDNode *>(nullptr));                         \
  } while (false)
#define RETURN_REF_FROM_RAW(REF, VALID)                                        \
  do {                                                                         \
    if (auto *N = getRaw())                                                    \
      return REF::get(VALID);                                                  \
    return REF::get(nullptr);                                                  \
  } while (false)

/// \brief This is used to represent ranges, for array bounds.
class DISubrange : public DIDescriptor {
  MDSubrange *getRaw() const { return dyn_cast_or_null<MDSubrange>(get()); }

public:
  explicit DISubrange(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DISubrange(const MDSubrange *N) : DIDescriptor(N) {}

  int64_t getLo() const { RETURN_FROM_RAW(N->getLo(), 0); }
  int64_t getCount() const { RETURN_FROM_RAW(N->getCount(), 0); }
  bool Verify() const;
};

/// \brief This descriptor holds an array of nodes with type T.
template <typename T> class DITypedArray : public DIDescriptor {
public:
  explicit DITypedArray(const MDNode *N = nullptr) : DIDescriptor(N) {}
  unsigned getNumElements() const {
    return DbgNode ? DbgNode->getNumOperands() : 0;
  }
  T getElement(unsigned Idx) const { return getFieldAs<T>(Idx); }
};

typedef DITypedArray<DIDescriptor> DIArray;

/// \brief A wrapper for an enumerator (e.g. X and Y in 'enum {X,Y}').
///
/// FIXME: it seems strange that this doesn't have either a reference to the
/// type/precision or a file/line pair for location info.
class DIEnumerator : public DIDescriptor {
  MDEnumerator *getRaw() const { return dyn_cast_or_null<MDEnumerator>(get()); }

public:
  explicit DIEnumerator(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIEnumerator(const MDEnumerator *N) : DIDescriptor(N) {}

  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  int64_t getEnumValue() const { RETURN_FROM_RAW(N->getValue(), 0); }
  bool Verify() const;
};

template <typename T> class DIRef;
typedef DIRef<DIDescriptor> DIDescriptorRef;
typedef DIRef<DIScope> DIScopeRef;
typedef DIRef<DIType> DITypeRef;
typedef DITypedArray<DITypeRef> DITypeArray;

/// \brief A base class for various scopes.
///
/// Although, implementation-wise, DIScope is the parent class of most
/// other DIxxx classes, including DIType and its descendants, most of
/// DIScope's descendants are not a substitutable subtype of
/// DIScope. The DIDescriptor::isScope() method only is true for
/// DIScopes that are scopes in the strict lexical scope sense
/// (DICompileUnit, DISubprogram, etc.), but not for, e.g., a DIType.
class DIScope : public DIDescriptor {
protected:
  MDScope *getRaw() const { return dyn_cast_or_null<MDScope>(get()); }

public:
  explicit DIScope(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIScope(const MDScope *N) : DIDescriptor(N) {}

  /// \brief Get the parent scope.
  ///
  /// Gets the parent scope for this scope node or returns a default
  /// constructed scope.
  DIScopeRef getContext() const;
  /// \brief Get the scope name.
  ///
  /// If the scope node has a name, return that, else return an empty string.
  StringRef getName() const;
  StringRef getFilename() const;
  StringRef getDirectory() const;

  /// \brief Generate a reference to this DIScope.
  ///
  /// Uses the type identifier instead of the actual MDNode if possible, to
  /// help type uniquing.
  DIScopeRef getRef() const;
};

/// \brief Represents reference to a DIDescriptor.
///
/// Abstracts over direct and identifier-based metadata references.
template <typename T> class DIRef {
  template <typename DescTy>
  friend DescTy DIDescriptor::getFieldAs(unsigned Elt) const;
  friend DIScopeRef DIScope::getContext() const;
  friend DIScopeRef DIScope::getRef() const;
  friend class DIType;

  /// \brief Val can be either a MDNode or a MDString.
  ///
  /// In the latter, MDString specifies the type identifier.
  const Metadata *Val;
  explicit DIRef(const Metadata *V);

public:
  T resolve(const DITypeIdentifierMap &Map) const;
  StringRef getName() const;
  operator Metadata *() const { return const_cast<Metadata *>(Val); }

  static DIRef get(const Metadata *MD) { return DIRef(MD); }
};

template <typename T>
T DIRef<T>::resolve(const DITypeIdentifierMap &Map) const {
  if (!Val)
    return T();

  if (const MDNode *MD = dyn_cast<MDNode>(Val))
    return T(MD);

  const MDString *MS = cast<MDString>(Val);
  // Find the corresponding MDNode.
  DITypeIdentifierMap::const_iterator Iter = Map.find(MS);
  assert(Iter != Map.end() && "Identifier not in the type map?");
  assert(DIDescriptor(Iter->second).isType() &&
         "MDNode in DITypeIdentifierMap should be a DIType.");
  return T(Iter->second);
}

template <typename T> StringRef DIRef<T>::getName() const {
  if (!Val)
    return StringRef();

  if (const MDNode *MD = dyn_cast<MDNode>(Val))
    return T(MD).getName();

  const MDString *MS = cast<MDString>(Val);
  return MS->getString();
}

/// \brief Handle fields that are references to DIDescriptors.
template <>
DIDescriptorRef DIDescriptor::getFieldAs<DIDescriptorRef>(unsigned Elt) const;
/// \brief Specialize DIRef constructor for DIDescriptorRef.
template <> DIRef<DIDescriptor>::DIRef(const Metadata *V);

/// \brief Handle fields that are references to DIScopes.
template <> DIScopeRef DIDescriptor::getFieldAs<DIScopeRef>(unsigned Elt) const;
/// \brief Specialize DIRef constructor for DIScopeRef.
template <> DIRef<DIScope>::DIRef(const Metadata *V);

/// \brief Handle fields that are references to DITypes.
template <> DITypeRef DIDescriptor::getFieldAs<DITypeRef>(unsigned Elt) const;
/// \brief Specialize DIRef constructor for DITypeRef.
template <> DIRef<DIType>::DIRef(const Metadata *V);

/// \brief This is a wrapper for a type.
///
/// FIXME: Types should be factored much better so that CV qualifiers and
/// others do not require a huge and empty descriptor full of zeros.
class DIType : public DIScope {
  MDType *getRaw() const { return dyn_cast_or_null<MDType>(get()); }

public:
  explicit DIType(const MDNode *N = nullptr) : DIScope(N) {}
  DIType(const MDType *N) : DIScope(N) {}

  operator DITypeRef() const {
    assert(isType() &&
           "constructing DITypeRef from an MDNode that is not a type");
    return DITypeRef(&*getRef());
  }

  bool Verify() const;

  DIScopeRef getContext() const {
    RETURN_REF_FROM_RAW(DIScopeRef, N->getScope());
  }
  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }
  uint64_t getSizeInBits() const { RETURN_FROM_RAW(N->getSizeInBits(), 0); }
  uint64_t getAlignInBits() const { RETURN_FROM_RAW(N->getAlignInBits(), 0); }
  // FIXME: Offset is only used for DW_TAG_member nodes.  Making every type
  // carry this is just plain insane.
  uint64_t getOffsetInBits() const { RETURN_FROM_RAW(N->getOffsetInBits(), 0); }
  unsigned getFlags() const { RETURN_FROM_RAW(N->getFlags(), 0); }
  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  bool isForwardDecl() const { return (getFlags() & FlagFwdDecl) != 0; }
  bool isAppleBlockExtension() const {
    return (getFlags() & FlagAppleBlock) != 0;
  }
  bool isBlockByrefStruct() const {
    return (getFlags() & FlagBlockByrefStruct) != 0;
  }
  bool isVirtual() const { return (getFlags() & FlagVirtual) != 0; }
  bool isArtificial() const { return (getFlags() & FlagArtificial) != 0; }
  bool isObjectPointer() const { return (getFlags() & FlagObjectPointer) != 0; }
  bool isObjcClassComplete() const {
    return (getFlags() & FlagObjcClassComplete) != 0;
  }
  bool isVector() const { return (getFlags() & FlagVector) != 0; }
  bool isStaticMember() const { return (getFlags() & FlagStaticMember) != 0; }
  bool isLValueReference() const {
    return (getFlags() & FlagLValueReference) != 0;
  }
  bool isRValueReference() const {
    return (getFlags() & FlagRValueReference) != 0;
  }
  bool isValid() const { return DbgNode && isType(); }
};

/// \brief A basic type, like 'int' or 'float'.
class DIBasicType : public DIType {
  MDBasicType *getRaw() const { return dyn_cast_or_null<MDBasicType>(get()); }

public:
  explicit DIBasicType(const MDNode *N = nullptr) : DIType(N) {}
  DIBasicType(const MDBasicType *N) : DIType(N) {}

  unsigned getEncoding() const { RETURN_FROM_RAW(N->getEncoding(), 0); }

  bool Verify() const;
};

/// \brief A simple derived type
///
/// Like a const qualified type, a typedef, a pointer or reference, et cetera.
/// Or, a data member of a class/struct/union.
class DIDerivedType : public DIType {
  MDDerivedTypeBase *getRaw() const {
    return dyn_cast_or_null<MDDerivedTypeBase>(get());
  }

public:
  explicit DIDerivedType(const MDNode *N = nullptr) : DIType(N) {}
  DIDerivedType(const MDDerivedTypeBase *N) : DIType(N) {}

  DITypeRef getTypeDerivedFrom() const {
    RETURN_REF_FROM_RAW(DITypeRef, N->getBaseType());
  }

  /// \brief Return property node, if this ivar is associated with one.
  MDNode *getObjCProperty() const {
    if (auto *N = dyn_cast_or_null<MDDerivedType>(get()))
      return dyn_cast_or_null<MDNode>(N->getExtraData());
    return nullptr;
  }

  DITypeRef getClassType() const {
    assert(getTag() == dwarf::DW_TAG_ptr_to_member_type);
    if (auto *N = dyn_cast_or_null<MDDerivedType>(get()))
      return DITypeRef::get(N->getExtraData());
    return DITypeRef::get(nullptr);
  }

  Constant *getConstant() const {
    assert((getTag() == dwarf::DW_TAG_member) && isStaticMember());
    if (auto *N = dyn_cast_or_null<MDDerivedType>(get()))
      if (auto *C = dyn_cast_or_null<ConstantAsMetadata>(N->getExtraData()))
        return C->getValue();

    return nullptr;
  }

  bool Verify() const;
};

/// \brief Types that refer to multiple other types.
///
/// This descriptor holds a type that can refer to multiple other types, like a
/// function or struct.
///
/// DICompositeType is derived from DIDerivedType because some
/// composite types (such as enums) can be derived from basic types
// FIXME: Make this derive from DIType directly & just store the
// base type in a single DIType field.
class DICompositeType : public DIDerivedType {
  friend class DIBuilder;

  /// \brief Set the array of member DITypes.
  void setArraysHelper(MDNode *Elements, MDNode *TParams);

  MDCompositeTypeBase *getRaw() const {
    return dyn_cast_or_null<MDCompositeTypeBase>(get());
  }

public:
  explicit DICompositeType(const MDNode *N = nullptr) : DIDerivedType(N) {}
  DICompositeType(const MDCompositeTypeBase *N) : DIDerivedType(N) {}

  DIArray getElements() const {
    assert(!isSubroutineType() && "no elements for DISubroutineType");
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getElements());
  }

private:
  template <typename T>
  void setArrays(DITypedArray<T> Elements, DIArray TParams = DIArray()) {
    assert(
        (!TParams || DbgNode->getNumOperands() == 8) &&
        "If you're setting the template parameters this should include a slot "
        "for that!");
    setArraysHelper(Elements, TParams);
  }

public:
  unsigned getRunTimeLang() const { RETURN_FROM_RAW(N->getRuntimeLang(), 0); }
  DITypeRef getContainingType() const {
    RETURN_REF_FROM_RAW(DITypeRef, N->getVTableHolder());
  }

private:
  /// \brief Set the containing type.
  void setContainingType(DICompositeType ContainingType);

public:
  DIArray getTemplateParams() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getTemplateParams());
  }
  MDString *getIdentifier() const {
    RETURN_FROM_RAW(N->getRawIdentifier(), nullptr);
  }

  bool Verify() const;
};

class DISubroutineType : public DICompositeType {
  MDSubroutineType *getRaw() const {
    return dyn_cast_or_null<MDSubroutineType>(get());
  }

public:
  explicit DISubroutineType(const MDNode *N = nullptr) : DICompositeType(N) {}
  DISubroutineType(const MDSubroutineType *N) : DICompositeType(N) {}

  DITypedArray<DITypeRef> getTypeArray() const {
    RETURN_DESCRIPTOR_FROM_RAW(DITypedArray<DITypeRef>, N->getTypeArray());
  }
};

/// \brief This is a wrapper for a file.
class DIFile : public DIScope {
  MDFile *getRaw() const { return dyn_cast_or_null<MDFile>(get()); }

public:
  explicit DIFile(const MDNode *N = nullptr) : DIScope(N) {}
  DIFile(const MDFile *N) : DIScope(N) {}

  /// \brief Retrieve the MDNode for the directory/file pair.
  MDNode *getFileNode() const { return get(); }
  bool Verify() const;
};

/// \brief A wrapper for a compile unit.
class DICompileUnit : public DIScope {
  MDCompileUnit *getRaw() const {
    return dyn_cast_or_null<MDCompileUnit>(get());
  }

public:
  explicit DICompileUnit(const MDNode *N = nullptr) : DIScope(N) {}
  DICompileUnit(const MDCompileUnit *N) : DIScope(N) {}

  dwarf::SourceLanguage getLanguage() const {
    RETURN_FROM_RAW(static_cast<dwarf::SourceLanguage>(N->getSourceLanguage()),
                    static_cast<dwarf::SourceLanguage>(0));
  }
  StringRef getProducer() const { RETURN_FROM_RAW(N->getProducer(), ""); }
  bool isOptimized() const { RETURN_FROM_RAW(N->isOptimized(), false); }
  StringRef getFlags() const { RETURN_FROM_RAW(N->getFlags(), ""); }
  unsigned getRunTimeVersion() const {
    RETURN_FROM_RAW(N->getRuntimeVersion(), 0);
  }

  DIArray getEnumTypes() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getEnumTypes());
  }
  DIArray getRetainedTypes() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getRetainedTypes());
  }
  DIArray getSubprograms() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getSubprograms());
  }
  DIArray getGlobalVariables() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getGlobalVariables());
  }
  DIArray getImportedEntities() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getImportedEntities());
  }

  void replaceSubprograms(DIArray Subprograms);
  void replaceGlobalVariables(DIArray GlobalVariables);

  StringRef getSplitDebugFilename() const {
    RETURN_FROM_RAW(N->getSplitDebugFilename(), "");
  }
  unsigned getEmissionKind() const { RETURN_FROM_RAW(N->getEmissionKind(), 0); }

  bool Verify() const;
};

/// \brief This is a wrapper for a subprogram (e.g. a function).
class DISubprogram : public DIScope {
  MDSubprogram *getRaw() const { return dyn_cast_or_null<MDSubprogram>(get()); }

public:
  explicit DISubprogram(const MDNode *N = nullptr) : DIScope(N) {}
  DISubprogram(const MDSubprogram *N) : DIScope(N) {}

  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  StringRef getDisplayName() const { RETURN_FROM_RAW(N->getDisplayName(), ""); }
  StringRef getLinkageName() const { RETURN_FROM_RAW(N->getLinkageName(), ""); }
  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }

  /// \brief Check if this is local (like 'static' in C).
  unsigned isLocalToUnit() const { RETURN_FROM_RAW(N->isLocalToUnit(), 0); }
  unsigned isDefinition() const { RETURN_FROM_RAW(N->isDefinition(), 0); }

  unsigned getVirtuality() const { RETURN_FROM_RAW(N->getVirtuality(), 0); }
  unsigned getVirtualIndex() const { RETURN_FROM_RAW(N->getVirtualIndex(), 0); }

  unsigned getFlags() const { RETURN_FROM_RAW(N->getFlags(), 0); }

  unsigned isOptimized() const { RETURN_FROM_RAW(N->isOptimized(), 0); }

  /// \brief Get the beginning of the scope of the function (not the name).
  unsigned getScopeLineNumber() const { RETURN_FROM_RAW(N->getScopeLine(), 0); }

  DIScopeRef getContext() const {
    RETURN_REF_FROM_RAW(DIScopeRef, N->getScope());
  }
  DISubroutineType getType() const {
    RETURN_DESCRIPTOR_FROM_RAW(DISubroutineType, N->getType());
  }

  DITypeRef getContainingType() const {
    RETURN_REF_FROM_RAW(DITypeRef, N->getContainingType());
  }

  bool Verify() const;

  /// \brief Check if this provides debugging information for the function F.
  bool describes(const Function *F);

  Function *getFunction() const;

  void replaceFunction(Function *F) {
    if (auto *N = getRaw())
      N->replaceFunction(F);
  }
  DIArray getTemplateParams() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getTemplateParams());
  }
  DISubprogram getFunctionDeclaration() const {
    RETURN_DESCRIPTOR_FROM_RAW(DISubprogram, N->getDeclaration());
  }
  MDNode *getVariablesNodes() const { return getVariables(); }
  DIArray getVariables() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getVariables());
  }

  unsigned isArtificial() const { return (getFlags() & FlagArtificial) != 0; }
  /// \brief Check for the "private" access specifier.
  bool isPrivate() const {
    return (getFlags() & FlagAccessibility) == FlagPrivate;
  }
  /// \brief Check for the "protected" access specifier.
  bool isProtected() const {
    return (getFlags() & FlagAccessibility) == FlagProtected;
  }
  /// \brief Check for the "public" access specifier.
  bool isPublic() const {
    return (getFlags() & FlagAccessibility) == FlagPublic;
  }
  /// \brief Check for "explicit".
  bool isExplicit() const { return (getFlags() & FlagExplicit) != 0; }
  /// \brief Check if this is prototyped.
  bool isPrototyped() const { return (getFlags() & FlagPrototyped) != 0; }

  /// \brief Check if this is reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 reference-qualified non-static
  /// member function (void foo() &).
  unsigned isLValueReference() const {
    return (getFlags() & FlagLValueReference) != 0;
  }

  /// \brief Check if this is rvalue-reference-qualified.
  ///
  /// Return true if this subprogram is a C++11 rvalue-reference-qualified
  /// non-static member function (void foo() &&).
  unsigned isRValueReference() const {
    return (getFlags() & FlagRValueReference) != 0;
  }
};

/// \brief This is a wrapper for a lexical block.
class DILexicalBlock : public DIScope {
  MDLexicalBlockBase *getRaw() const {
    return dyn_cast_or_null<MDLexicalBlockBase>(get());
  }

public:
  explicit DILexicalBlock(const MDNode *N = nullptr) : DIScope(N) {}
  DILexicalBlock(const MDLexicalBlock *N) : DIScope(N) {}

  DIScope getContext() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIScope, N->getScope());
  }
  unsigned getLineNumber() const {
    if (auto *N = dyn_cast_or_null<MDLexicalBlock>(get()))
      return N->getLine();
    return 0;
  }
  unsigned getColumnNumber() const {
    if (auto *N = dyn_cast_or_null<MDLexicalBlock>(get()))
      return N->getColumn();
    return 0;
  }
  bool Verify() const;
};

/// \brief This is a wrapper for a lexical block with a filename change.
class DILexicalBlockFile : public DIScope {
  MDLexicalBlockFile *getRaw() const {
    return dyn_cast_or_null<MDLexicalBlockFile>(get());
  }

public:
  explicit DILexicalBlockFile(const MDNode *N = nullptr) : DIScope(N) {}
  DILexicalBlockFile(const MDLexicalBlockFile *N) : DIScope(N) {}

  DIScope getContext() const {
    // FIXME: This logic is horrible.  getScope() returns a DILexicalBlock, but
    // then we check if it's a subprogram?  WHAT?!?
    if (getScope().isSubprogram())
      return getScope();
    return getScope().getContext();
  }
  unsigned getLineNumber() const { return getScope().getLineNumber(); }
  unsigned getColumnNumber() const { return getScope().getColumnNumber(); }
  DILexicalBlock getScope() const {
    RETURN_DESCRIPTOR_FROM_RAW(DILexicalBlock, N->getScope());
  }
  unsigned getDiscriminator() const {
    RETURN_FROM_RAW(N->getDiscriminator(), 0);
  }
  bool Verify() const;
};

/// \brief A wrapper for a C++ style name space.
class DINameSpace : public DIScope {
  MDNamespace *getRaw() const { return dyn_cast_or_null<MDNamespace>(get()); }

public:
  explicit DINameSpace(const MDNode *N = nullptr) : DIScope(N) {}
  DINameSpace(const MDNamespace *N) : DIScope(N) {}

  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }
  DIScope getContext() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIScope, N->getScope());
  }
  bool Verify() const;
};

/// \brief This is a wrapper for template type parameter.
class DITemplateTypeParameter : public DIDescriptor {
  MDTemplateTypeParameter *getRaw() const {
    return dyn_cast_or_null<MDTemplateTypeParameter>(get());
  }

public:
  explicit DITemplateTypeParameter(const MDNode *N = nullptr)
      : DIDescriptor(N) {}
  DITemplateTypeParameter(const MDTemplateTypeParameter *N) : DIDescriptor(N) {}

  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }

  DITypeRef getType() const { RETURN_REF_FROM_RAW(DITypeRef, N->getType()); }
  bool Verify() const;
};

/// \brief This is a wrapper for template value parameter.
class DITemplateValueParameter : public DIDescriptor {
  MDTemplateValueParameter *getRaw() const {
    return dyn_cast_or_null<MDTemplateValueParameter>(get());
  }

public:
  explicit DITemplateValueParameter(const MDNode *N = nullptr)
      : DIDescriptor(N) {}
  DITemplateValueParameter(const MDTemplateValueParameter *N)
      : DIDescriptor(N) {}

  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  DITypeRef getType() const { RETURN_REF_FROM_RAW(DITypeRef, N->getType()); }
  Metadata *getValue() const { RETURN_FROM_RAW(N->getValue(), nullptr); }
  bool Verify() const;
};

/// \brief This is a wrapper for a global variable.
class DIGlobalVariable : public DIDescriptor {
  MDGlobalVariable *getRaw() const {
    return dyn_cast_or_null<MDGlobalVariable>(get());
  }

  DIFile getFile() const { RETURN_DESCRIPTOR_FROM_RAW(DIFile, N->getFile()); }

public:
  explicit DIGlobalVariable(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIGlobalVariable(const MDGlobalVariable *N) : DIDescriptor(N) {}

  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  StringRef getDisplayName() const { RETURN_FROM_RAW(N->getDisplayName(), ""); }
  StringRef getLinkageName() const { RETURN_FROM_RAW(N->getLinkageName(), ""); }
  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }
  unsigned isLocalToUnit() const { RETURN_FROM_RAW(N->isLocalToUnit(), 0); }
  unsigned isDefinition() const { RETURN_FROM_RAW(N->isDefinition(), 0); }

  DIScope getContext() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIScope, N->getScope());
  }
  StringRef getFilename() const { return getFile().getFilename(); }
  StringRef getDirectory() const { return getFile().getDirectory(); }
  DITypeRef getType() const { RETURN_REF_FROM_RAW(DITypeRef, N->getType()); }

  GlobalVariable *getGlobal() const;
  Constant *getConstant() const {
    if (auto *N = getRaw())
      if (auto *C = dyn_cast_or_null<ConstantAsMetadata>(N->getVariable()))
        return C->getValue();
    return nullptr;
  }
  DIDerivedType getStaticDataMemberDeclaration() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIDerivedType,
                               N->getStaticDataMemberDeclaration());
  }

  bool Verify() const;
};

/// \brief This is a wrapper for a variable (e.g. parameter, local, global etc).
class DIVariable : public DIDescriptor {
  MDLocalVariable *getRaw() const {
    return dyn_cast_or_null<MDLocalVariable>(get());
  }

  unsigned getFlags() const { RETURN_FROM_RAW(N->getFlags(), 0); }

public:
  explicit DIVariable(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIVariable(const MDLocalVariable *N) : DIDescriptor(N) {}

  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }
  unsigned getArgNumber() const { RETURN_FROM_RAW(N->getArg(), 0); }

  DIScope getContext() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIScope, N->getScope());
  }
  DIFile getFile() const { RETURN_DESCRIPTOR_FROM_RAW(DIFile, N->getFile()); }
  DITypeRef getType() const { RETURN_REF_FROM_RAW(DITypeRef, N->getType()); }

  /// \brief Return true if this variable is marked as "artificial".
  bool isArtificial() const {
    return (getFlags() & FlagArtificial) != 0;
  }

  bool isObjectPointer() const {
    return (getFlags() & FlagObjectPointer) != 0;
  }

  /// \brief If this variable is inlined then return inline location.
  MDNode *getInlinedAt() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIDescriptor, N->getInlinedAt());
  }

  bool Verify() const;

  /// \brief Check if this is a "__block" variable (Apple Blocks).
  bool isBlockByrefVariable(const DITypeIdentifierMap &Map) const {
    return (getType().resolve(Map)).isBlockByrefStruct();
  }

  /// \brief Check if this is an inlined function argument.
  bool isInlinedFnArgument(const Function *CurFn);

  /// \brief Return the size reported by the variable's type.
  unsigned getSizeInBits(const DITypeIdentifierMap &Map);

  void printExtendedName(raw_ostream &OS) const;
};

/// \brief A complex location expression in postfix notation.
///
/// This is (almost) a DWARF expression that modifies the location of a
/// variable or (or the location of a single piece of a variable).
///
/// FIXME: Instead of DW_OP_plus taking an argument, this should use DW_OP_const
/// and have DW_OP_plus consume the topmost elements on the stack.
class DIExpression : public DIDescriptor {
public:
  explicit DIExpression(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIExpression(const MDExpression *N) : DIDescriptor(N) {}

  MDExpression *get() const {
    return cast_or_null<MDExpression>(DIDescriptor::get());
  }
  operator MDExpression *() const { return get(); }
  MDExpression *operator->() const { return get(); }

  // Don't call this.  Call isValid() directly.
  bool Verify() const = delete;

  /// \brief Return the number of elements in the complex expression.
  unsigned getNumElements() const { return get()->getNumElements(); }

  /// \brief return the Idx'th complex address element.
  uint64_t getElement(unsigned I) const { return get()->getElement(I); }

  /// \brief Return whether this is a piece of an aggregate variable.
  bool isBitPiece() const;
  /// \brief Return the offset of this piece in bits.
  uint64_t getBitPieceOffset() const;
  /// \brief Return the size of this piece in bits.
  uint64_t getBitPieceSize() const;

  class iterator;
  /// \brief A lightweight wrapper around an element of a DIExpression.
  class Operand {
    friend class iterator;
    MDExpression::element_iterator I;
    Operand() {}
    Operand(MDExpression::element_iterator I) : I(I) {}
  public:
    /// \brief Operands such as DW_OP_piece have explicit (non-stack) arguments.
    /// Argument 0 is the operand itself.
    uint64_t getArg(unsigned N) const {
      MDExpression::element_iterator In = I;
      std::advance(In, N);
      return *In;
    }
    operator uint64_t () const { return *I; }
    /// \brief Returns underlying MDExpression::element_iterator.
    const MDExpression::element_iterator &getBase() const { return I; }
    /// \brief Returns the next operand.
    iterator getNext() const;
  };

  /// \brief An iterator for DIExpression elements.
  class iterator : public std::iterator<std::input_iterator_tag, StringRef,
                                        unsigned, const Operand*, Operand> {
    friend class Operand;
    MDExpression::element_iterator I;
    Operand Tmp;

  public:
    iterator(MDExpression::element_iterator I) : I(I) {}
    const Operand &operator*() { return Tmp = Operand(I); }
    const Operand *operator->() { return &(Tmp = Operand(I)); }
    iterator &operator++() {
      increment();
      return *this;
    }
    iterator operator++(int) {
      iterator X(*this);
      increment();
      return X;
    }
    bool operator==(const iterator &X) const { return I == X.I; }
    bool operator!=(const iterator &X) const { return !(*this == X); }

  private:
    void increment() {
      switch (**this) {
      case dwarf::DW_OP_bit_piece: std::advance(I, 3); break;
      case dwarf::DW_OP_plus:      std::advance(I, 2); break;
      case dwarf::DW_OP_deref:     std::advance(I, 1); break;
      default:
        llvm_unreachable("unsupported operand");
      }
    }
  };

  iterator begin() const { return get()->elements_begin(); }
  iterator end() const { return get()->elements_end(); }
};

/// \brief This object holds location information.
///
/// This object is not associated with any DWARF tag.
class DILocation : public DIDescriptor {
  MDLocation *getRaw() const { return dyn_cast_or_null<MDLocation>(get()); }

public:
  explicit DILocation(const MDNode *N) : DIDescriptor(N) {}

  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }
  unsigned getColumnNumber() const { RETURN_FROM_RAW(N->getColumn(), 0); }
  DIScope getScope() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIScope, N->getScope());
  }
  DILocation getOrigLocation() const {
    RETURN_DESCRIPTOR_FROM_RAW(DILocation, N->getInlinedAt());
  }
  StringRef getFilename() const { return getScope().getFilename(); }
  StringRef getDirectory() const { return getScope().getDirectory(); }
  bool Verify() const;
  bool atSameLineAs(const DILocation &Other) const {
    return (getLineNumber() == Other.getLineNumber() &&
            getFilename() == Other.getFilename());
  }
  /// \brief Get the DWAF discriminator.
  ///
  /// DWARF discriminators are used to distinguish identical file locations for
  /// instructions that are on different basic blocks. If two instructions are
  /// inside the same lexical block and are in different basic blocks, we
  /// create a new lexical block with identical location as the original but
  /// with a different discriminator value
  /// (lib/Transforms/Util/AddDiscriminators.cpp for details).
  unsigned getDiscriminator() const {
    // Since discriminators are associated with lexical blocks, make
    // sure this location is a lexical block before retrieving its
    // value.
    return getScope().isLexicalBlockFile()
               ? DILexicalBlockFile(
                     cast<MDNode>(cast<MDLocation>(DbgNode)->getScope()))
                     .getDiscriminator()
               : 0;
  }

  /// \brief Generate a new discriminator value for this location.
  unsigned computeNewDiscriminator(LLVMContext &Ctx);

  /// \brief Return a copy of this location with a different scope.
  DILocation copyWithNewScope(LLVMContext &Ctx, DILexicalBlockFile NewScope);
};

class DIObjCProperty : public DIDescriptor {
  MDObjCProperty *getRaw() const {
    return dyn_cast_or_null<MDObjCProperty>(get());
  }

public:
  explicit DIObjCProperty(const MDNode *N) : DIDescriptor(N) {}
  DIObjCProperty(const MDObjCProperty *N) : DIDescriptor(N) {}

  StringRef getObjCPropertyName() const { RETURN_FROM_RAW(N->getName(), ""); }
  DIFile getFile() const { RETURN_DESCRIPTOR_FROM_RAW(DIFile, N->getFile()); }
  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }

  StringRef getObjCPropertyGetterName() const {
    RETURN_FROM_RAW(N->getGetterName(), "");
  }
  StringRef getObjCPropertySetterName() const {
    RETURN_FROM_RAW(N->getSetterName(), "");
  }
  unsigned getAttributes() const { RETURN_FROM_RAW(N->getAttributes(), 0); }
  bool isReadOnlyObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_readonly) != 0;
  }
  bool isReadWriteObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_readwrite) != 0;
  }
  bool isAssignObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_assign) != 0;
  }
  bool isRetainObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_retain) != 0;
  }
  bool isCopyObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_copy) != 0;
  }
  bool isNonAtomicObjCProperty() const {
    return (getAttributes() & dwarf::DW_APPLE_PROPERTY_nonatomic) != 0;
  }

  /// \brief Get the type.
  ///
  /// \note Objective-C doesn't have an ODR, so there is no benefit in storing
  /// the type as a DITypeRef here.
  DIType getType() const { RETURN_DESCRIPTOR_FROM_RAW(DIType, N->getType()); }

  bool Verify() const;
};

/// \brief An imported module (C++ using directive or similar).
class DIImportedEntity : public DIDescriptor {
  MDImportedEntity *getRaw() const {
    return dyn_cast_or_null<MDImportedEntity>(get());
  }

public:
  DIImportedEntity() = default;
  explicit DIImportedEntity(const MDNode *N) : DIDescriptor(N) {}
  DIImportedEntity(const MDImportedEntity *N) : DIDescriptor(N) {}

  DIScope getContext() const {
    RETURN_DESCRIPTOR_FROM_RAW(DIScope, N->getScope());
  }
  DIDescriptorRef getEntity() const {
    RETURN_REF_FROM_RAW(DIDescriptorRef, N->getEntity());
  }
  unsigned getLineNumber() const { RETURN_FROM_RAW(N->getLine(), 0); }
  StringRef getName() const { RETURN_FROM_RAW(N->getName(), ""); }
  bool Verify() const;
};

#undef RETURN_FROM_RAW
#undef RETURN_DESCRIPTOR_FROM_RAW
#undef RETURN_REF_FROM_RAW

/// \brief Find subprogram that is enclosing this scope.
DISubprogram getDISubprogram(const MDNode *Scope);

/// \brief Find debug info for a given function.
/// \returns a valid DISubprogram, if found. Otherwise, it returns an empty
/// DISubprogram.
DISubprogram getDISubprogram(const Function *F);

/// \brief Find underlying composite type.
DICompositeType getDICompositeType(DIType T);

/// \brief Create a new inlined variable based on current variable.
///
/// @param DV            Current Variable.
/// @param InlinedScope  Location at current variable is inlined.
DIVariable createInlinedVariable(MDNode *DV, MDNode *InlinedScope,
                                 LLVMContext &VMContext);

/// \brief Remove inlined scope from the variable.
DIVariable cleanseInlinedVariable(MDNode *DV, LLVMContext &VMContext);

/// \brief Generate map by visiting all retained types.
DITypeIdentifierMap generateDITypeIdentifierMap(const NamedMDNode *CU_Nodes);

/// \brief Strip debug info in the module if it exists.
///
/// To do this, we remove all calls to the debugger intrinsics and any named
/// metadata for debugging. We also remove debug locations for instructions.
/// Return true if module is modified.
bool StripDebugInfo(Module &M);

/// \brief Return Debug Info Metadata Version by checking module flags.
unsigned getDebugMetadataVersionFromModule(const Module &M);

/// \brief Utility to find all debug info in a module.
///
/// DebugInfoFinder tries to list all debug info MDNodes used in a module. To
/// list debug info MDNodes used by an instruction, DebugInfoFinder uses
/// processDeclare, processValue and processLocation to handle DbgDeclareInst,
/// DbgValueInst and DbgLoc attached to instructions. processModule will go
/// through all DICompileUnits in llvm.dbg.cu and list debug info MDNodes
/// used by the CUs.
class DebugInfoFinder {
public:
  DebugInfoFinder() : TypeMapInitialized(false) {}

  /// \brief Process entire module and collect debug info anchors.
  void processModule(const Module &M);

  /// \brief Process DbgDeclareInst.
  void processDeclare(const Module &M, const DbgDeclareInst *DDI);
  /// \brief Process DbgValueInst.
  void processValue(const Module &M, const DbgValueInst *DVI);
  /// \brief Process DILocation.
  void processLocation(const Module &M, DILocation Loc);

  /// \brief Process DIExpression.
  void processExpression(DIExpression Expr);

  /// \brief Clear all lists.
  void reset();

private:
  void InitializeTypeMap(const Module &M);

  void processType(DIType DT);
  void processSubprogram(DISubprogram SP);
  void processScope(DIScope Scope);
  bool addCompileUnit(DICompileUnit CU);
  bool addGlobalVariable(DIGlobalVariable DIG);
  bool addSubprogram(DISubprogram SP);
  bool addType(DIType DT);
  bool addScope(DIScope Scope);

public:
  typedef SmallVectorImpl<DICompileUnit>::const_iterator compile_unit_iterator;
  typedef SmallVectorImpl<DISubprogram>::const_iterator subprogram_iterator;
  typedef SmallVectorImpl<DIGlobalVariable>::const_iterator
      global_variable_iterator;
  typedef SmallVectorImpl<DIType>::const_iterator type_iterator;
  typedef SmallVectorImpl<DIScope>::const_iterator scope_iterator;

  iterator_range<compile_unit_iterator> compile_units() const {
    return iterator_range<compile_unit_iterator>(CUs.begin(), CUs.end());
  }

  iterator_range<subprogram_iterator> subprograms() const {
    return iterator_range<subprogram_iterator>(SPs.begin(), SPs.end());
  }

  iterator_range<global_variable_iterator> global_variables() const {
    return iterator_range<global_variable_iterator>(GVs.begin(), GVs.end());
  }

  iterator_range<type_iterator> types() const {
    return iterator_range<type_iterator>(TYs.begin(), TYs.end());
  }

  iterator_range<scope_iterator> scopes() const {
    return iterator_range<scope_iterator>(Scopes.begin(), Scopes.end());
  }

  unsigned compile_unit_count() const { return CUs.size(); }
  unsigned global_variable_count() const { return GVs.size(); }
  unsigned subprogram_count() const { return SPs.size(); }
  unsigned type_count() const { return TYs.size(); }
  unsigned scope_count() const { return Scopes.size(); }

private:
  SmallVector<DICompileUnit, 8> CUs;
  SmallVector<DISubprogram, 8> SPs;
  SmallVector<DIGlobalVariable, 8> GVs;
  SmallVector<DIType, 8> TYs;
  SmallVector<DIScope, 8> Scopes;
  SmallPtrSet<MDNode *, 64> NodesSeen;
  DITypeIdentifierMap TypeIdentifierMap;

  /// \brief Specify if TypeIdentifierMap is initialized.
  bool TypeMapInitialized;
};

DenseMap<const Function *, DISubprogram> makeSubprogramMap(const Module &M);

} // end namespace llvm

#endif
