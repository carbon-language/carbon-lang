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

/// \brief A thin wraper around MDNode to access encoded debug info.
///
/// This should not be stored in a container, because the underlying MDNode may
/// change in certain situations.
class DIDescriptor {
  // Befriends DIRef so DIRef can befriend the protected member
  // function: getFieldAs<DIRef>.
  template <typename T> friend class DIRef;

public:
  /// \brief Duplicated debug info flags.
  ///
  /// \see DebugNode::DIFlags.
  enum {
#define HANDLE_DI_FLAG(ID, NAME) Flag##NAME = DebugNode::Flag##NAME,
#include "llvm/IR/DebugInfoFlags.def"
    FlagAccessibility = DebugNode::FlagAccessibility
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

  DIDescriptor getDescriptorField(unsigned Elt) const;
  template <typename DescTy> DescTy getFieldAs(unsigned Elt) const {
    return DescTy(getDescriptorField(Elt));
  }

public:
  explicit DIDescriptor(const MDNode *N = nullptr) : DbgNode(N) {}

  MDNode *get() const { return const_cast<MDNode *>(DbgNode); }
  operator MDNode *() const { return get(); }
  MDNode *operator->() const { return get(); }
  MDNode &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  // An explicit operator bool so that we can do testing of DI values
  // easily.
  // FIXME: This operator bool isn't actually protecting anything at the
  // moment due to the conversion operator above making DIDescriptor nodes
  // implicitly convertable to bool.
  explicit operator bool() const { return DbgNode != nullptr; }

  bool operator==(DIDescriptor Other) const { return DbgNode == Other.DbgNode; }
  bool operator!=(DIDescriptor Other) const { return !operator==(Other); }

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

/// \brief This is used to represent ranges, for array bounds.
class DISubrange : public DIDescriptor {
public:
  explicit DISubrange(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DISubrange(const MDSubrange *N) : DIDescriptor(N) {}

  MDSubrange *get() const {
    return cast_or_null<MDSubrange>(DIDescriptor::get());
  }
  operator MDSubrange *() const { return get(); }
  MDSubrange *operator->() const { return get(); }
  MDSubrange &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  int64_t getLo() const { return get()->getLo(); }
  int64_t getCount() const { return get()->getCount(); }
};

/// \brief This descriptor holds an array of nodes with type T.
template <typename T> class DITypedArray : public DIDescriptor {
public:
  explicit DITypedArray(const MDNode *N = nullptr) : DIDescriptor(N) {}
  operator MDTuple *() const {
    return const_cast<MDTuple *>(cast_or_null<MDTuple>(DbgNode));
  }
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
public:
  explicit DIEnumerator(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIEnumerator(const MDEnumerator *N) : DIDescriptor(N) {}

  MDEnumerator *get() const {
    return cast_or_null<MDEnumerator>(DIDescriptor::get());
  }
  operator MDEnumerator *() const { return get(); }
  MDEnumerator *operator->() const { return get(); }
  MDEnumerator &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getName() const { return get()->getName(); }
  int64_t getEnumValue() const { return get()->getValue(); }
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
public:
  explicit DIScope(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIScope(const MDScope *N) : DIDescriptor(N) {}

  MDScope *get() const { return cast_or_null<MDScope>(DIDescriptor::get()); }
  operator MDScope *() const { return get(); }
  MDScope *operator->() const { return get(); }
  MDScope &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

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
  template <class U>
  DIRef(const TypedDebugNodeRef<U> &Ref,
        typename std::enable_if<std::is_convertible<U *, T>::value>::type * =
            nullptr)
      : Val(Ref) {}

  T resolve(const DITypeIdentifierMap &Map) const;
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
public:
  explicit DIType(const MDNode *N = nullptr) : DIScope(N) {}
  DIType(const MDType *N) : DIScope(N) {}

  MDType *get() const { return cast_or_null<MDType>(DIDescriptor::get()); }
  operator MDType *() const { return get(); }
  MDType *operator->() const { return get(); }
  MDType &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  DIScopeRef getContext() const { return DIScopeRef::get(get()->getScope()); }
  StringRef getName() const { return get()->getName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  uint64_t getSizeInBits() const { return get()->getSizeInBits(); }
  uint64_t getAlignInBits() const { return get()->getAlignInBits(); }
  // FIXME: Offset is only used for DW_TAG_member nodes.  Making every type
  // carry this is just plain insane.
  uint64_t getOffsetInBits() const { return get()->getOffsetInBits(); }
  unsigned getFlags() const { return get()->getFlags(); }
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
public:
  explicit DIBasicType(const MDNode *N = nullptr) : DIType(N) {}
  DIBasicType(const MDBasicType *N) : DIType(N) {}

  MDBasicType *get() const {
    return cast_or_null<MDBasicType>(DIDescriptor::get());
  }
  operator MDBasicType *() const { return get(); }
  MDBasicType *operator->() const { return get(); }
  MDBasicType &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  unsigned getEncoding() const { return get()->getEncoding(); }
};

/// \brief A simple derived type
///
/// Like a const qualified type, a typedef, a pointer or reference, et cetera.
/// Or, a data member of a class/struct/union.
class DIDerivedType : public DIType {
public:
  explicit DIDerivedType(const MDNode *N = nullptr) : DIType(N) {}
  DIDerivedType(const MDDerivedTypeBase *N) : DIType(N) {}

  MDDerivedTypeBase *get() const {
    return cast_or_null<MDDerivedTypeBase>(DIDescriptor::get());
  }
  operator MDDerivedTypeBase *() const { return get(); }
  MDDerivedTypeBase *operator->() const { return get(); }
  MDDerivedTypeBase &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  DITypeRef getTypeDerivedFrom() const {
    return DITypeRef::get(get()->getBaseType());
  }

  /// \brief Return property node, if this ivar is associated with one.
  MDNode *getObjCProperty() const {
    if (auto *N = dyn_cast<MDDerivedType>(get()))
      return dyn_cast_or_null<MDNode>(N->getExtraData());
    return nullptr;
  }

  DITypeRef getClassType() const {
    assert(getTag() == dwarf::DW_TAG_ptr_to_member_type);
    if (auto *N = dyn_cast<MDDerivedType>(get()))
      return DITypeRef::get(N->getExtraData());
    return DITypeRef::get(nullptr);
  }

  Constant *getConstant() const {
    assert((getTag() == dwarf::DW_TAG_member) && isStaticMember());
    if (auto *N = dyn_cast<MDDerivedType>(get()))
      if (auto *C = dyn_cast_or_null<ConstantAsMetadata>(N->getExtraData()))
        return C->getValue();

    return nullptr;
  }
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

public:
  explicit DICompositeType(const MDNode *N = nullptr) : DIDerivedType(N) {}
  DICompositeType(const MDCompositeTypeBase *N) : DIDerivedType(N) {}

  MDCompositeTypeBase *get() const {
    return cast_or_null<MDCompositeTypeBase>(DIDescriptor::get());
  }
  operator MDCompositeTypeBase *() const { return get(); }
  MDCompositeTypeBase *operator->() const { return get(); }
  MDCompositeTypeBase &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  DIArray getElements() const {
    assert(!isSubroutineType() && "no elements for DISubroutineType");
    return DIArray(get()->getElements());
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
  unsigned getRunTimeLang() const { return get()->getRuntimeLang(); }
  DITypeRef getContainingType() const {
    return DITypeRef::get(get()->getVTableHolder());
  }

private:
  /// \brief Set the containing type.
  void setContainingType(DICompositeType ContainingType);

public:
  DIArray getTemplateParams() const {
    return DIArray(get()->getTemplateParams());
  }
  MDString *getIdentifier() const { return get()->getRawIdentifier(); }
};

class DISubroutineType : public DICompositeType {
public:
  explicit DISubroutineType(const MDNode *N = nullptr) : DICompositeType(N) {}
  DISubroutineType(const MDSubroutineType *N) : DICompositeType(N) {}

  MDSubroutineType *get() const {
    return cast_or_null<MDSubroutineType>(DIDescriptor::get());
  }
  operator MDSubroutineType *() const { return get(); }
  MDSubroutineType *operator->() const { return get(); }
  MDSubroutineType &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  DITypedArray<DITypeRef> getTypeArray() const {
    return DITypedArray<DITypeRef>(get()->getTypeArray());
  }
};

/// \brief This is a wrapper for a file.
class DIFile : public DIScope {
public:
  explicit DIFile(const MDNode *N = nullptr) : DIScope(N) {}
  DIFile(const MDFile *N) : DIScope(N) {}

  MDFile *get() const { return cast_or_null<MDFile>(DIDescriptor::get()); }
  operator MDFile *() const { return get(); }
  MDFile *operator->() const { return get(); }
  MDFile &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  /// \brief Retrieve the MDNode for the directory/file pair.
  MDNode *getFileNode() const { return get(); }
};

/// \brief A wrapper for a compile unit.
class DICompileUnit : public DIScope {
public:
  explicit DICompileUnit(const MDNode *N = nullptr) : DIScope(N) {}
  DICompileUnit(const MDCompileUnit *N) : DIScope(N) {}

  MDCompileUnit *get() const {
    return cast_or_null<MDCompileUnit>(DIDescriptor::get());
  }
  operator MDCompileUnit *() const { return get(); }
  MDCompileUnit *operator->() const { return get(); }
  MDCompileUnit &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  dwarf::SourceLanguage getLanguage() const {
    return static_cast<dwarf::SourceLanguage>(get()->getSourceLanguage());
  }
  StringRef getProducer() const { return get()->getProducer(); }
  bool isOptimized() const { return get()->isOptimized(); }
  StringRef getFlags() const { return get()->getFlags(); }
  unsigned getRunTimeVersion() const { return get()->getRuntimeVersion(); }

  DIArray getEnumTypes() const { return DIArray(get()->getEnumTypes()); }
  DIArray getRetainedTypes() const {
    return DIArray(get()->getRetainedTypes());
  }
  DIArray getSubprograms() const { return DIArray(get()->getSubprograms()); }
  DIArray getGlobalVariables() const {
    return DIArray(get()->getGlobalVariables());
  }
  DIArray getImportedEntities() const {
    return DIArray(get()->getImportedEntities());
  }

  void replaceSubprograms(DIArray Subprograms);
  void replaceGlobalVariables(DIArray GlobalVariables);

  StringRef getSplitDebugFilename() const {
    return get()->getSplitDebugFilename();
  }
  unsigned getEmissionKind() const { return get()->getEmissionKind(); }
};

/// \brief This is a wrapper for a subprogram (e.g. a function).
class DISubprogram : public DIScope {
public:
  explicit DISubprogram(const MDNode *N = nullptr) : DIScope(N) {}
  DISubprogram(const MDSubprogram *N) : DIScope(N) {}

  MDSubprogram *get() const {
    return cast_or_null<MDSubprogram>(DIDescriptor::get());
  }
  operator MDSubprogram *() const { return get(); }
  MDSubprogram *operator->() const { return get(); }
  MDSubprogram &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getName() const { return get()->getName(); }
  StringRef getDisplayName() const { return get()->getDisplayName(); }
  StringRef getLinkageName() const { return get()->getLinkageName(); }
  unsigned getLineNumber() const { return get()->getLine(); }

  /// \brief Check if this is local (like 'static' in C).
  unsigned isLocalToUnit() const { return get()->isLocalToUnit(); }
  unsigned isDefinition() const { return get()->isDefinition(); }

  unsigned getVirtuality() const { return get()->getVirtuality(); }
  unsigned getVirtualIndex() const { return get()->getVirtualIndex(); }

  unsigned getFlags() const { return get()->getFlags(); }

  unsigned isOptimized() const { return get()->isOptimized(); }

  /// \brief Get the beginning of the scope of the function (not the name).
  unsigned getScopeLineNumber() const { return get()->getScopeLine(); }

  DIScopeRef getContext() const { return DIScopeRef::get(get()->getScope()); }
  DISubroutineType getType() const {
    return DISubroutineType(get()->getType());
  }

  DITypeRef getContainingType() const {
    return DITypeRef::get(get()->getContainingType());
  }

  /// \brief Check if this provides debugging information for the function F.
  bool describes(const Function *F);

  Function *getFunction() const;

  void replaceFunction(Function *F) {
    if (auto *N = get())
      N->replaceFunction(F);
  }
  DIArray getTemplateParams() const {
    return DIArray(get()->getTemplateParams());
  }
  DISubprogram getFunctionDeclaration() const {
    return DISubprogram(get()->getDeclaration());
  }
  MDNode *getVariablesNodes() const { return getVariables(); }
  DIArray getVariables() const { return DIArray(get()->getVariables()); }

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
public:
  explicit DILexicalBlock(const MDNode *N = nullptr) : DIScope(N) {}
  DILexicalBlock(const MDLexicalBlock *N) : DIScope(N) {}

  MDLexicalBlockBase *get() const {
    return cast_or_null<MDLexicalBlockBase>(DIDescriptor::get());
  }
  operator MDLexicalBlockBase *() const { return get(); }
  MDLexicalBlockBase *operator->() const { return get(); }
  MDLexicalBlockBase &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  DIScope getContext() const { return DIScope(get()->getScope()); }
  unsigned getLineNumber() const {
    if (auto *N = dyn_cast<MDLexicalBlock>(get()))
      return N->getLine();
    return 0;
  }
  unsigned getColumnNumber() const {
    if (auto *N = dyn_cast<MDLexicalBlock>(get()))
      return N->getColumn();
    return 0;
  }
};

/// \brief This is a wrapper for a lexical block with a filename change.
class DILexicalBlockFile : public DIScope {
public:
  explicit DILexicalBlockFile(const MDNode *N = nullptr) : DIScope(N) {}
  DILexicalBlockFile(const MDLexicalBlockFile *N) : DIScope(N) {}

  MDLexicalBlockFile *get() const {
    return cast_or_null<MDLexicalBlockFile>(DIDescriptor::get());
  }
  operator MDLexicalBlockFile *() const { return get(); }
  MDLexicalBlockFile *operator->() const { return get(); }
  MDLexicalBlockFile &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  DIScope getContext() const { return getScope(); }
  unsigned getLineNumber() const { return getScope().getLineNumber(); }
  unsigned getColumnNumber() const { return getScope().getColumnNumber(); }
  DILexicalBlock getScope() const { return DILexicalBlock(get()->getScope()); }
  unsigned getDiscriminator() const { return get()->getDiscriminator(); }
};

/// \brief A wrapper for a C++ style name space.
class DINameSpace : public DIScope {
public:
  explicit DINameSpace(const MDNode *N = nullptr) : DIScope(N) {}
  DINameSpace(const MDNamespace *N) : DIScope(N) {}

  MDNamespace *get() const {
    return cast_or_null<MDNamespace>(DIDescriptor::get());
  }
  operator MDNamespace *() const { return get(); }
  MDNamespace *operator->() const { return get(); }
  MDNamespace &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getName() const { return get()->getName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  DIScope getContext() const { return DIScope(get()->getScope()); }
};

/// \brief This is a wrapper for template type parameter.
class DITemplateTypeParameter : public DIDescriptor {
public:
  explicit DITemplateTypeParameter(const MDNode *N = nullptr)
      : DIDescriptor(N) {}
  DITemplateTypeParameter(const MDTemplateTypeParameter *N) : DIDescriptor(N) {}

  MDTemplateTypeParameter *get() const {
    return cast_or_null<MDTemplateTypeParameter>(DIDescriptor::get());
  }
  operator MDTemplateTypeParameter *() const { return get(); }
  MDTemplateTypeParameter *operator->() const { return get(); }
  MDTemplateTypeParameter &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getName() const { return get()->getName(); }

  DITypeRef getType() const { return DITypeRef::get(get()->getType()); }
};

/// \brief This is a wrapper for template value parameter.
class DITemplateValueParameter : public DIDescriptor {
public:
  explicit DITemplateValueParameter(const MDNode *N = nullptr)
      : DIDescriptor(N) {}
  DITemplateValueParameter(const MDTemplateValueParameter *N)
      : DIDescriptor(N) {}

  MDTemplateValueParameter *get() const {
    return cast_or_null<MDTemplateValueParameter>(DIDescriptor::get());
  }
  operator MDTemplateValueParameter *() const { return get(); }
  MDTemplateValueParameter *operator->() const { return get(); }
  MDTemplateValueParameter &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getName() const { return get()->getName(); }
  DITypeRef getType() const { return DITypeRef::get(get()->getType()); }
  Metadata *getValue() const { return get()->getValue(); }
};

/// \brief This is a wrapper for a global variable.
class DIGlobalVariable : public DIDescriptor {
  DIFile getFile() const { return DIFile(get()->getFile()); }

public:
  explicit DIGlobalVariable(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIGlobalVariable(const MDGlobalVariable *N) : DIDescriptor(N) {}

  MDGlobalVariable *get() const {
    return cast_or_null<MDGlobalVariable>(DIDescriptor::get());
  }
  operator MDGlobalVariable *() const { return get(); }
  MDGlobalVariable *operator->() const { return get(); }
  MDGlobalVariable &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getName() const { return get()->getName(); }
  StringRef getDisplayName() const { return get()->getDisplayName(); }
  StringRef getLinkageName() const { return get()->getLinkageName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  unsigned isLocalToUnit() const { return get()->isLocalToUnit(); }
  unsigned isDefinition() const { return get()->isDefinition(); }

  DIScope getContext() const { return DIScope(get()->getScope()); }
  StringRef getFilename() const { return getFile().getFilename(); }
  StringRef getDirectory() const { return getFile().getDirectory(); }
  DITypeRef getType() const { return DITypeRef::get(get()->getType()); }

  GlobalVariable *getGlobal() const;
  Constant *getConstant() const {
    if (auto *N = get())
      if (auto *C = dyn_cast_or_null<ConstantAsMetadata>(N->getVariable()))
        return C->getValue();
    return nullptr;
  }
  DIDerivedType getStaticDataMemberDeclaration() const {
    return DIDerivedType(get()->getStaticDataMemberDeclaration());
  }
};

/// \brief This is a wrapper for a variable (e.g. parameter, local, global etc).
class DIVariable : public DIDescriptor {
  unsigned getFlags() const { return get()->getFlags(); }

public:
  explicit DIVariable(const MDNode *N = nullptr) : DIDescriptor(N) {}
  DIVariable(const MDLocalVariable *N) : DIDescriptor(N) {}

  MDLocalVariable *get() const {
    return cast_or_null<MDLocalVariable>(DIDescriptor::get());
  }
  operator MDLocalVariable *() const { return get(); }
  MDLocalVariable *operator->() const { return get(); }
  MDLocalVariable &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getName() const { return get()->getName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  unsigned getArgNumber() const { return get()->getArg(); }

  DIScope getContext() const { return DIScope(get()->getScope()); }
  DIFile getFile() const { return DIFile(get()->getFile()); }
  DITypeRef getType() const { return DITypeRef::get(get()->getType()); }

  /// \brief Return true if this variable is marked as "artificial".
  bool isArtificial() const {
    return (getFlags() & FlagArtificial) != 0;
  }

  bool isObjectPointer() const {
    return (getFlags() & FlagObjectPointer) != 0;
  }

  /// \brief If this variable is inlined then return inline location.
  MDNode *getInlinedAt() const { return DIDescriptor(get()->getInlinedAt()); }

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
  MDExpression &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

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
public:
  explicit DILocation(const MDNode *N) : DIDescriptor(N) {}
  DILocation(const MDLocation *N) : DIDescriptor(N) {}

  MDLocation *get() const {
    return cast_or_null<MDLocation>(DIDescriptor::get());
  }
  operator MDLocation *() const { return get(); }
  MDLocation *operator->() const { return get(); }
  MDLocation &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  unsigned getLineNumber() const { return get()->getLine(); }
  unsigned getColumnNumber() const { return get()->getColumn(); }
  DIScope getScope() const { return DIScope(get()->getScope()); }
  DILocation getOrigLocation() const {
    return DILocation(get()->getInlinedAt());
  }
  StringRef getFilename() const { return getScope().getFilename(); }
  StringRef getDirectory() const { return getScope().getDirectory(); }
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
public:
  explicit DIObjCProperty(const MDNode *N) : DIDescriptor(N) {}
  DIObjCProperty(const MDObjCProperty *N) : DIDescriptor(N) {}

  MDObjCProperty *get() const {
    return cast_or_null<MDObjCProperty>(DIDescriptor::get());
  }
  operator MDObjCProperty *() const { return get(); }
  MDObjCProperty *operator->() const { return get(); }
  MDObjCProperty &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  StringRef getObjCPropertyName() const { return get()->getName(); }
  DIFile getFile() const { return DIFile(get()->getFile()); }
  unsigned getLineNumber() const { return get()->getLine(); }

  StringRef getObjCPropertyGetterName() const { return get()->getGetterName(); }
  StringRef getObjCPropertySetterName() const { return get()->getSetterName(); }
  unsigned getAttributes() const { return get()->getAttributes(); }
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
  DIType getType() const { return DIType(get()->getType()); }
};

/// \brief An imported module (C++ using directive or similar).
class DIImportedEntity : public DIDescriptor {
public:
  DIImportedEntity() = default;
  explicit DIImportedEntity(const MDNode *N) : DIDescriptor(N) {}
  DIImportedEntity(const MDImportedEntity *N) : DIDescriptor(N) {}

  MDImportedEntity *get() const {
    return cast_or_null<MDImportedEntity>(DIDescriptor::get());
  }
  operator MDImportedEntity *() const { return get(); }
  MDImportedEntity *operator->() const { return get(); }
  MDImportedEntity &operator*() const {
    assert(get() && "Expected valid pointer");
    return *get();
  }

  DIScope getContext() const { return DIScope(get()->getScope()); }
  DIDescriptorRef getEntity() const {
    return DIDescriptorRef::get(get()->getEntity());
  }
  unsigned getLineNumber() const { return get()->getLine(); }
  StringRef getName() const { return get()->getName(); }
};

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
bool stripDebugInfo(Function &F);

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
