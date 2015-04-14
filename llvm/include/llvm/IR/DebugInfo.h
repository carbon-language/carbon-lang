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
public:
  /// \brief Duplicated debug info flags.
  ///
  /// \see DebugNode::DIFlags.
  enum {
#define HANDLE_DI_FLAG(ID, NAME) Flag##NAME = DebugNode::Flag##NAME,
#include "llvm/IR/DebugInfoFlags.def"
    FlagAccessibility = DebugNode::FlagAccessibility
  };

protected:
  const MDNode *DbgNode;

public:
  explicit DIDescriptor(const MDNode *N = nullptr) : DbgNode(N) {}
  DIDescriptor(const DebugNode *N) : DbgNode(N) {}

  MDNode *get() const { return const_cast<MDNode *>(DbgNode); }
  operator MDNode *() const { return get(); }
  MDNode *operator->() const { return get(); }
  MDNode &operator*() const { return *get(); }

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

  void print(raw_ostream &OS) const;
  void dump() const;
};

#define DECLARE_SIMPLIFY_DESCRIPTOR(DESC)                                      \
  class DESC;                                                                  \
  template <> struct simplify_type<const DESC>;                                \
  template <> struct simplify_type<DESC>;
DECLARE_SIMPLIFY_DESCRIPTOR(DIDescriptor)
DECLARE_SIMPLIFY_DESCRIPTOR(DISubrange)
DECLARE_SIMPLIFY_DESCRIPTOR(DIEnumerator)
DECLARE_SIMPLIFY_DESCRIPTOR(DIScope)
DECLARE_SIMPLIFY_DESCRIPTOR(DIType)
DECLARE_SIMPLIFY_DESCRIPTOR(DIBasicType)
DECLARE_SIMPLIFY_DESCRIPTOR(DIDerivedType)
DECLARE_SIMPLIFY_DESCRIPTOR(DICompositeType)
DECLARE_SIMPLIFY_DESCRIPTOR(DISubroutineType)
DECLARE_SIMPLIFY_DESCRIPTOR(DIFile)
DECLARE_SIMPLIFY_DESCRIPTOR(DICompileUnit)
DECLARE_SIMPLIFY_DESCRIPTOR(DISubprogram)
DECLARE_SIMPLIFY_DESCRIPTOR(DILexicalBlock)
DECLARE_SIMPLIFY_DESCRIPTOR(DILexicalBlockFile)
DECLARE_SIMPLIFY_DESCRIPTOR(DINameSpace)
DECLARE_SIMPLIFY_DESCRIPTOR(DITemplateTypeParameter)
DECLARE_SIMPLIFY_DESCRIPTOR(DITemplateValueParameter)
DECLARE_SIMPLIFY_DESCRIPTOR(DIGlobalVariable)
DECLARE_SIMPLIFY_DESCRIPTOR(DIVariable)
DECLARE_SIMPLIFY_DESCRIPTOR(DIExpression)
DECLARE_SIMPLIFY_DESCRIPTOR(DILocation)
DECLARE_SIMPLIFY_DESCRIPTOR(DIObjCProperty)
DECLARE_SIMPLIFY_DESCRIPTOR(DIImportedEntity)
#undef DECLARE_SIMPLIFY_DESCRIPTOR

typedef DebugNodeArray DIArray;
typedef MDTypeRefArray DITypeArray;

/// \brief This is used to represent ranges, for array bounds.
class DISubrange : public DIDescriptor {
public:
  DISubrange() = default;
  DISubrange(const MDSubrange *N) : DIDescriptor(N) {}

  MDSubrange *get() const {
    return cast_or_null<MDSubrange>(DIDescriptor::get());
  }
  operator MDSubrange *() const { return get(); }
  MDSubrange *operator->() const { return get(); }
  MDSubrange &operator*() const { return *get(); }

  int64_t getLo() const { return get()->getLowerBound(); }
  int64_t getCount() const { return get()->getCount(); }
};

/// \brief A wrapper for an enumerator (e.g. X and Y in 'enum {X,Y}').
///
/// FIXME: it seems strange that this doesn't have either a reference to the
/// type/precision or a file/line pair for location info.
class DIEnumerator : public DIDescriptor {
public:
  DIEnumerator() = default;
  DIEnumerator(const MDEnumerator *N) : DIDescriptor(N) {}

  MDEnumerator *get() const {
    return cast_or_null<MDEnumerator>(DIDescriptor::get());
  }
  operator MDEnumerator *() const { return get(); }
  MDEnumerator *operator->() const { return get(); }
  MDEnumerator &operator*() const { return *get(); }

  StringRef getName() const { return get()->getName(); }
  int64_t getEnumValue() const { return get()->getValue(); }
};

template <typename T> class DIRef;
typedef DIRef<DIDescriptor> DIDescriptorRef;
typedef DIRef<DIScope> DIScopeRef;
typedef DIRef<DIType> DITypeRef;

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
  DIScope() = default;
  DIScope(const MDScope *N) : DIDescriptor(N) {}

  MDScope *get() const { return cast_or_null<MDScope>(DIDescriptor::get()); }
  operator MDScope *() const { return get(); }
  MDScope *operator->() const { return get(); }
  MDScope &operator*() const { return *get(); }

  inline DIScopeRef getContext() const;
  StringRef getName() const { return get()->getName(); }
  StringRef getFilename() const { return get()->getFilename(); }
  StringRef getDirectory() const { return get()->getDirectory(); }

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
  /// \brief Val can be either a MDNode or a MDString.
  ///
  /// In the latter, MDString specifies the type identifier.
  const Metadata *Val;

public:
  template <class U>
  DIRef(const TypedDebugNodeRef<U> &Ref,
        typename std::enable_if<std::is_convertible<U *, T>::value>::type * =
            nullptr)
      : Val(Ref) {}

  T resolve(const DITypeIdentifierMap &Map) const;
  operator Metadata *() const { return const_cast<Metadata *>(Val); }
};

template <>
DIDescriptor DIRef<DIDescriptor>::resolve(const DITypeIdentifierMap &Map) const;
template <>
DIScope DIRef<DIScope>::resolve(const DITypeIdentifierMap &Map) const;
template <> DIType DIRef<DIType>::resolve(const DITypeIdentifierMap &Map) const;

DIScopeRef DIScope::getContext() const { return get()->getScope(); }

/// \brief This is a wrapper for a type.
///
/// FIXME: Types should be factored much better so that CV qualifiers and
/// others do not require a huge and empty descriptor full of zeros.
class DIType : public DIScope {
public:
  DIType() = default;
  DIType(const MDType *N) : DIScope(N) {}

  MDType *get() const { return cast_or_null<MDType>(DIDescriptor::get()); }
  operator MDType *() const { return get(); }
  MDType *operator->() const { return get(); }
  MDType &operator*() const { return *get(); }

  DIScopeRef getContext() const { return get()->getScope(); }
  StringRef getName() const { return get()->getName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  uint64_t getSizeInBits() const { return get()->getSizeInBits(); }
  uint64_t getAlignInBits() const { return get()->getAlignInBits(); }
  // FIXME: Offset is only used for DW_TAG_member nodes.  Making every type
  // carry this is just plain insane.
  uint64_t getOffsetInBits() const { return get()->getOffsetInBits(); }
  unsigned getFlags() const { return get()->getFlags(); }

  bool isPrivate() const { return get()->isPrivate(); }
  bool isProtected() const { return get()->isProtected(); }
  bool isPublic() const { return get()->isPublic(); }
  bool isForwardDecl() const { return get()->isForwardDecl(); }
  bool isAppleBlockExtension() const { return get()->isAppleBlockExtension(); }
  bool isBlockByrefStruct() const { return get()->isBlockByrefStruct(); }
  bool isVirtual() const { return get()->isVirtual(); }
  bool isArtificial() const { return get()->isArtificial(); }
  bool isObjectPointer() const { return get()->isObjectPointer(); }
  bool isObjcClassComplete() const { return get()->isObjcClassComplete(); }
  bool isVector() const { return get()->isVector(); }
  bool isStaticMember() const { return get()->isStaticMember(); }
  bool isLValueReference() const { return get()->isLValueReference(); }
  bool isRValueReference() const { return get()->isRValueReference(); }

  bool isValid() const { return DbgNode && isa<MDType>(*this); }
};

/// \brief A basic type, like 'int' or 'float'.
class DIBasicType : public DIType {
public:
  DIBasicType() = default;
  DIBasicType(const MDBasicType *N) : DIType(N) {}

  MDBasicType *get() const {
    return cast_or_null<MDBasicType>(DIDescriptor::get());
  }
  operator MDBasicType *() const { return get(); }
  MDBasicType *operator->() const { return get(); }
  MDBasicType &operator*() const { return *get(); }

  unsigned getEncoding() const { return get()->getEncoding(); }
};

/// \brief A simple derived type
///
/// Like a const qualified type, a typedef, a pointer or reference, et cetera.
/// Or, a data member of a class/struct/union.
class DIDerivedType : public DIType {
public:
  DIDerivedType() = default;
  DIDerivedType(const MDDerivedTypeBase *N) : DIType(N) {}

  MDDerivedTypeBase *get() const {
    return cast_or_null<MDDerivedTypeBase>(DIDescriptor::get());
  }
  operator MDDerivedTypeBase *() const { return get(); }
  MDDerivedTypeBase *operator->() const { return get(); }
  MDDerivedTypeBase &operator*() const { return *get(); }

  DITypeRef getTypeDerivedFrom() const { return get()->getBaseType(); }

  /// \brief Return property node, if this ivar is associated with one.
  MDObjCProperty *getObjCProperty() const {
    return cast<MDDerivedType>(get())->getObjCProperty();
  }

  DITypeRef getClassType() const {
    return cast<MDDerivedType>(get())->getClassType();
  }

  Constant *getConstant() const {
    return cast<MDDerivedType>(get())->getConstant();
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

public:
  DICompositeType() = default;
  DICompositeType(const MDCompositeTypeBase *N) : DIDerivedType(N) {}

  MDCompositeTypeBase *get() const {
    return cast_or_null<MDCompositeTypeBase>(DIDescriptor::get());
  }
  operator MDCompositeTypeBase *() const { return get(); }
  MDCompositeTypeBase *operator->() const { return get(); }
  MDCompositeTypeBase &operator*() const { return *get(); }

  DIArray getElements() const { return get()->getElements(); }

  unsigned getRunTimeLang() const { return get()->getRuntimeLang(); }
  DITypeRef getContainingType() const { return get()->getVTableHolder(); }

  DIArray getTemplateParams() const { return get()->getTemplateParams(); }
  MDString *getIdentifier() const { return get()->getRawIdentifier(); }
};

class DISubroutineType : public DICompositeType {
public:
  DISubroutineType() = default;
  DISubroutineType(const MDSubroutineType *N) : DICompositeType(N) {}

  MDSubroutineType *get() const {
    return cast_or_null<MDSubroutineType>(DIDescriptor::get());
  }
  operator MDSubroutineType *() const { return get(); }
  MDSubroutineType *operator->() const { return get(); }
  MDSubroutineType &operator*() const { return *get(); }

  MDTypeRefArray getTypeArray() const { return get()->getTypeArray(); }
};

/// \brief This is a wrapper for a file.
class DIFile : public DIScope {
public:
  DIFile() = default;
  DIFile(const MDFile *N) : DIScope(N) {}

  MDFile *get() const { return cast_or_null<MDFile>(DIDescriptor::get()); }
  operator MDFile *() const { return get(); }
  MDFile *operator->() const { return get(); }
  MDFile &operator*() const { return *get(); }

  /// \brief Retrieve the MDNode for the directory/file pair.
  MDNode *getFileNode() const { return get(); }
};

/// \brief A wrapper for a compile unit.
class DICompileUnit : public DIScope {
public:
  DICompileUnit() = default;
  DICompileUnit(const MDCompileUnit *N) : DIScope(N) {}

  MDCompileUnit *get() const {
    return cast_or_null<MDCompileUnit>(DIDescriptor::get());
  }
  operator MDCompileUnit *() const { return get(); }
  MDCompileUnit *operator->() const { return get(); }
  MDCompileUnit &operator*() const { return *get(); }

  dwarf::SourceLanguage getLanguage() const {
    return static_cast<dwarf::SourceLanguage>(get()->getSourceLanguage());
  }
  StringRef getProducer() const { return get()->getProducer(); }
  bool isOptimized() const { return get()->isOptimized(); }
  StringRef getFlags() const { return get()->getFlags(); }
  unsigned getRunTimeVersion() const { return get()->getRuntimeVersion(); }

  DIArray getEnumTypes() const { return get()->getEnumTypes(); }
  DIArray getRetainedTypes() const { return get()->getRetainedTypes(); }
  DIArray getSubprograms() const { return get()->getSubprograms(); }
  DIArray getGlobalVariables() const { return get()->getGlobalVariables(); }
  DIArray getImportedEntities() const { return get()->getImportedEntities(); }

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
  DISubprogram() = default;
  DISubprogram(const MDSubprogram *N) : DIScope(N) {}

  MDSubprogram *get() const {
    return cast_or_null<MDSubprogram>(DIDescriptor::get());
  }
  operator MDSubprogram *() const { return get(); }
  MDSubprogram *operator->() const { return get(); }
  MDSubprogram &operator*() const { return *get(); }

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

  DIScopeRef getContext() const { return get()->getScope(); }
  DISubroutineType getType() const { return get()->getType(); }

  DITypeRef getContainingType() const { return get()->getContainingType(); }

  /// \brief Check if this provides debugging information for the function F.
  bool describes(const Function *F) const { return get()->describes(F); }

  Function *getFunction() const { return get()->getFunction(); }

  void replaceFunction(Function *F) { get()->replaceFunction(F); }
  DIArray getTemplateParams() const { return get()->getTemplateParams(); }
  DISubprogram getFunctionDeclaration() const {
    return get()->getDeclaration();
  }
  DIArray getVariables() const { return DIArray(get()->getVariables()); }

  unsigned isArtificial() const { return get()->isArtificial(); }
  bool isPrivate() const { return get()->isPrivate(); }
  bool isProtected() const { return get()->isProtected(); }
  bool isPublic() const { return get()->isPublic(); }
  bool isExplicit() const { return get()->isExplicit(); }
  bool isPrototyped() const { return get()->isPrototyped(); }
  unsigned isLValueReference() const { return get()->isLValueReference(); }
  unsigned isRValueReference() const { return get()->isRValueReference(); }
};

/// \brief This is a wrapper for a lexical block.
class DILexicalBlock : public DIScope {
public:
  DILexicalBlock() = default;
  DILexicalBlock(const MDLexicalBlockBase *N) : DIScope(N) {}

  MDLexicalBlockBase *get() const {
    return cast_or_null<MDLexicalBlockBase>(DIDescriptor::get());
  }
  operator MDLexicalBlockBase *() const { return get(); }
  MDLexicalBlockBase *operator->() const { return get(); }
  MDLexicalBlockBase &operator*() const { return *get(); }

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
  DILexicalBlockFile() = default;
  DILexicalBlockFile(const MDLexicalBlockFile *N) : DIScope(N) {}

  MDLexicalBlockFile *get() const {
    return cast_or_null<MDLexicalBlockFile>(DIDescriptor::get());
  }
  operator MDLexicalBlockFile *() const { return get(); }
  MDLexicalBlockFile *operator->() const { return get(); }
  MDLexicalBlockFile &operator*() const { return *get(); }

  DIScope getContext() const { return get()->getScope(); }
  unsigned getDiscriminator() const { return get()->getDiscriminator(); }
};

/// \brief A wrapper for a C++ style name space.
class DINameSpace : public DIScope {
public:
  DINameSpace() = default;
  DINameSpace(const MDNamespace *N) : DIScope(N) {}

  MDNamespace *get() const {
    return cast_or_null<MDNamespace>(DIDescriptor::get());
  }
  operator MDNamespace *() const { return get(); }
  MDNamespace *operator->() const { return get(); }
  MDNamespace &operator*() const { return *get(); }

  StringRef getName() const { return get()->getName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  DIScope getContext() const { return DIScope(get()->getScope()); }
};

/// \brief This is a wrapper for template type parameter.
class DITemplateTypeParameter : public DIDescriptor {
public:
  DITemplateTypeParameter() = default;
  DITemplateTypeParameter(const MDTemplateTypeParameter *N) : DIDescriptor(N) {}

  MDTemplateTypeParameter *get() const {
    return cast_or_null<MDTemplateTypeParameter>(DIDescriptor::get());
  }
  operator MDTemplateTypeParameter *() const { return get(); }
  MDTemplateTypeParameter *operator->() const { return get(); }
  MDTemplateTypeParameter &operator*() const { return *get(); }

  StringRef getName() const { return get()->getName(); }
  DITypeRef getType() const { return get()->getType(); }
};

/// \brief This is a wrapper for template value parameter.
class DITemplateValueParameter : public DIDescriptor {
public:
  DITemplateValueParameter() = default;
  DITemplateValueParameter(const MDTemplateValueParameter *N)
      : DIDescriptor(N) {}

  MDTemplateValueParameter *get() const {
    return cast_or_null<MDTemplateValueParameter>(DIDescriptor::get());
  }
  operator MDTemplateValueParameter *() const { return get(); }
  MDTemplateValueParameter *operator->() const { return get(); }
  MDTemplateValueParameter &operator*() const { return *get(); }

  StringRef getName() const { return get()->getName(); }
  DITypeRef getType() const { return get()->getType(); }
  Metadata *getValue() const { return get()->getValue(); }
};

/// \brief This is a wrapper for a global variable.
class DIGlobalVariable : public DIDescriptor {
  DIFile getFile() const { return DIFile(get()->getFile()); }

public:
  DIGlobalVariable() = default;
  DIGlobalVariable(const MDGlobalVariable *N) : DIDescriptor(N) {}

  MDGlobalVariable *get() const {
    return cast_or_null<MDGlobalVariable>(DIDescriptor::get());
  }
  operator MDGlobalVariable *() const { return get(); }
  MDGlobalVariable *operator->() const { return get(); }
  MDGlobalVariable &operator*() const { return *get(); }

  StringRef getName() const { return get()->getName(); }
  StringRef getDisplayName() const { return get()->getDisplayName(); }
  StringRef getLinkageName() const { return get()->getLinkageName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  unsigned isLocalToUnit() const { return get()->isLocalToUnit(); }
  unsigned isDefinition() const { return get()->isDefinition(); }

  DIScope getContext() const { return get()->getScope(); }
  StringRef getFilename() const { return get()->getFilename(); }
  StringRef getDirectory() const { return get()->getDirectory(); }
  DITypeRef getType() const { return get()->getType(); }

  Constant *getConstant() const { return get()->getVariable(); }
  DIDerivedType getStaticDataMemberDeclaration() const {
    return get()->getStaticDataMemberDeclaration();
  }
};

/// \brief This is a wrapper for a variable (e.g. parameter, local, global etc).
class DIVariable : public DIDescriptor {
  unsigned getFlags() const { return get()->getFlags(); }

public:
  DIVariable() = default;
  DIVariable(const MDLocalVariable *N) : DIDescriptor(N) {}

  MDLocalVariable *get() const {
    return cast_or_null<MDLocalVariable>(DIDescriptor::get());
  }
  operator MDLocalVariable *() const { return get(); }
  MDLocalVariable *operator->() const { return get(); }
  MDLocalVariable &operator*() const { return *get(); }

  StringRef getName() const { return get()->getName(); }
  unsigned getLineNumber() const { return get()->getLine(); }
  unsigned getArgNumber() const { return get()->getArg(); }

  DIScope getContext() const { return get()->getScope(); }
  DIFile getFile() const { return get()->getFile(); }
  DITypeRef getType() const { return get()->getType(); }

  bool isArtificial() const { return get()->isArtificial(); }
  bool isObjectPointer() const { return get()->isObjectPointer(); }

  /// \brief If this variable is inlined then return inline location.
  MDNode *getInlinedAt() const { return get()->getInlinedAt(); }
};

class DIExpression {
  MDExpression *N;

public:
  DIExpression(const MDExpression *N = nullptr)
      : N(const_cast<MDExpression *>(N)) {}

  operator MDExpression *() const { return N; }
  MDExpression *operator->() const { return N; }
  MDExpression &operator*() const { return *N; }
};

class DILocation {
  MDLocation *N;

public:
  DILocation(const MDLocation *N = nullptr) : N(const_cast<MDLocation *>(N)) {}

  operator MDLocation *() const { return N; }
  MDLocation *operator->() const { return N; }
  MDLocation &operator*() const { return *N; }
};

class DIObjCProperty {
  MDObjCProperty *N;

public:
  DIObjCProperty(const MDObjCProperty *N = nullptr)
      : N(const_cast<MDObjCProperty *>(N)) {}

  operator MDObjCProperty *() const { return N; }
  MDObjCProperty *operator->() const { return N; }
  MDObjCProperty &operator*() const { return *N; }
};

class DIImportedEntity {
  MDImportedEntity *N;

public:
  DIImportedEntity(const MDImportedEntity *N = nullptr)
      : N(const_cast<MDImportedEntity *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator MDImportedEntity *() const { return N; }
  MDImportedEntity *operator->() const { return N; }
  MDImportedEntity &operator*() const { return *N; }
};

#define SIMPLIFY_DESCRIPTOR(DESC)                                              \
  template <> struct simplify_type<const DESC> {                               \
    typedef Metadata *SimpleType;                                              \
    static SimpleType getSimplifiedValue(const DESC &DI) { return DI; }        \
  };                                                                           \
  template <> struct simplify_type<DESC> : simplify_type<const DESC> {};
SIMPLIFY_DESCRIPTOR(DIDescriptor)
SIMPLIFY_DESCRIPTOR(DISubrange)
SIMPLIFY_DESCRIPTOR(DIEnumerator)
SIMPLIFY_DESCRIPTOR(DIScope)
SIMPLIFY_DESCRIPTOR(DIType)
SIMPLIFY_DESCRIPTOR(DIBasicType)
SIMPLIFY_DESCRIPTOR(DIDerivedType)
SIMPLIFY_DESCRIPTOR(DICompositeType)
SIMPLIFY_DESCRIPTOR(DISubroutineType)
SIMPLIFY_DESCRIPTOR(DIFile)
SIMPLIFY_DESCRIPTOR(DICompileUnit)
SIMPLIFY_DESCRIPTOR(DISubprogram)
SIMPLIFY_DESCRIPTOR(DILexicalBlock)
SIMPLIFY_DESCRIPTOR(DILexicalBlockFile)
SIMPLIFY_DESCRIPTOR(DINameSpace)
SIMPLIFY_DESCRIPTOR(DITemplateTypeParameter)
SIMPLIFY_DESCRIPTOR(DITemplateValueParameter)
SIMPLIFY_DESCRIPTOR(DIGlobalVariable)
SIMPLIFY_DESCRIPTOR(DIVariable)
SIMPLIFY_DESCRIPTOR(DIExpression)
SIMPLIFY_DESCRIPTOR(DILocation)
SIMPLIFY_DESCRIPTOR(DIObjCProperty)
SIMPLIFY_DESCRIPTOR(DIImportedEntity)
#undef SIMPLIFY_DESCRIPTOR

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
