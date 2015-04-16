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

class DIDescriptor {
  MDNode *N;

public:
  DIDescriptor(const MDNode *N = nullptr) : N(const_cast<MDNode *>(N)) {}

  operator MDNode *() const { return N; }
  MDNode *operator->() const { return N; }
  MDNode &operator*() const { return *N; }
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
typedef DebugNodeRef DIDescriptorRef;
typedef MDScopeRef DIScopeRef;
typedef MDTypeRef DITypeRef;

class DISubrange {
  MDSubrange *N;

public:
  DISubrange(const MDSubrange *N = nullptr) : N(const_cast<MDSubrange *>(N)) {}

  operator MDSubrange *() const { return N; }
  MDSubrange *operator->() const { return N; }
  MDSubrange &operator*() const { return *N; }
};

class DIEnumerator {
  MDEnumerator *N;

public:
  DIEnumerator(const MDEnumerator *N = nullptr)
      : N(const_cast<MDEnumerator *>(N)) {}

  operator MDEnumerator *() const { return N; }
  MDEnumerator *operator->() const { return N; }
  MDEnumerator &operator*() const { return *N; }
};

class DIScope {
  MDScope *N;

public:
  DIScope(const MDScope *N = nullptr) : N(const_cast<MDScope *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator MDScope *() const { return N; }
  MDScope *operator->() const { return N; }
  MDScope &operator*() const { return *N; }
};

class DIType {
  MDType *N;

public:
  DIType(const MDType *N = nullptr) : N(const_cast<MDType *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIScope() const { return N; }
  operator MDType *() const { return N; }
  MDType *operator->() const { return N; }
  MDType &operator*() const { return *N; }
};

class DIBasicType {
  MDBasicType *N;

public:
  DIBasicType(const MDBasicType *N = nullptr)
      : N(const_cast<MDBasicType *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIType() const { return N; }
  operator MDBasicType *() const { return N; }
  MDBasicType *operator->() const { return N; }
  MDBasicType &operator*() const { return *N; }
};

class DIDerivedType {
  MDDerivedTypeBase *N;

public:
  DIDerivedType(const MDDerivedTypeBase *N = nullptr)
      : N(const_cast<MDDerivedTypeBase *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIType() const { return N; }
  operator MDDerivedTypeBase *() const { return N; }
  MDDerivedTypeBase *operator->() const { return N; }
  MDDerivedTypeBase &operator*() const { return *N; }
};

class DICompositeType {
  MDCompositeTypeBase *N;

public:
  DICompositeType(const MDCompositeTypeBase *N = nullptr)
      : N(const_cast<MDCompositeTypeBase *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIType() const { return N; }
  operator MDCompositeTypeBase *() const { return N; }
  MDCompositeTypeBase *operator->() const { return N; }
  MDCompositeTypeBase &operator*() const { return *N; }
};

class DISubroutineType {
  MDSubroutineType *N;

public:
  DISubroutineType(const MDSubroutineType *N = nullptr)
      : N(const_cast<MDSubroutineType *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIType() const { return N; }
  operator DICompositeType() const { return N; }
  operator MDSubroutineType *() const { return N; }
  MDSubroutineType *operator->() const { return N; }
  MDSubroutineType &operator*() const { return *N; }
};

class DIFile {
  MDFile *N;

public:
  DIFile(const MDFile *N = nullptr) : N(const_cast<MDFile *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIScope() const { return N; }
  operator MDFile *() const { return N; }
  MDFile *operator->() const { return N; }
  MDFile &operator*() const { return *N; }
};

class DICompileUnit {
  MDCompileUnit *N;

public:
  DICompileUnit(const MDCompileUnit *N = nullptr)
      : N(const_cast<MDCompileUnit *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIScope() const { return N; }
  operator MDCompileUnit *() const { return N; }
  MDCompileUnit *operator->() const { return N; }
  MDCompileUnit &operator*() const { return *N; }
};

class DISubprogram {
  MDSubprogram *N;

public:
  DISubprogram(const MDSubprogram *N = nullptr)
      : N(const_cast<MDSubprogram *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIScope() const { return N; }
  operator MDSubprogram *() const { return N; }
  MDSubprogram *operator->() const { return N; }
  MDSubprogram &operator*() const { return *N; }
};

class DILexicalBlock {
  MDLexicalBlockBase *N;

public:
  DILexicalBlock(const MDLexicalBlockBase *N = nullptr)
      : N(const_cast<MDLexicalBlockBase *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator MDLexicalBlockBase *() const { return N; }
  MDLexicalBlockBase *operator->() const { return N; }
  MDLexicalBlockBase &operator*() const { return *N; }
};

class DILexicalBlockFile {
  MDLexicalBlockFile *N;

public:
  DILexicalBlockFile(const MDLexicalBlockFile *N = nullptr)
      : N(const_cast<MDLexicalBlockFile *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator MDLexicalBlockFile *() const { return N; }
  MDLexicalBlockFile *operator->() const { return N; }
  MDLexicalBlockFile &operator*() const { return *N; }
};

class DINameSpace {
  MDNamespace *N;

public:
  DINameSpace(const MDNamespace *N = nullptr)
      : N(const_cast<MDNamespace *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator DIScope() const { return N; }
  operator MDNamespace *() const { return N; }
  MDNamespace *operator->() const { return N; }
  MDNamespace &operator*() const { return *N; }
};

class DITemplateTypeParameter {
  MDTemplateTypeParameter *N;

public:
  DITemplateTypeParameter(const MDTemplateTypeParameter *N = nullptr)
      : N(const_cast<MDTemplateTypeParameter *>(N)) {}

  operator MDTemplateTypeParameter *() const { return N; }
  MDTemplateTypeParameter *operator->() const { return N; }
  MDTemplateTypeParameter &operator*() const { return *N; }
};

class DITemplateValueParameter {
  MDTemplateValueParameter *N;

public:
  DITemplateValueParameter(const MDTemplateValueParameter *N = nullptr)
      : N(const_cast<MDTemplateValueParameter *>(N)) {}

  operator MDTemplateValueParameter *() const { return N; }
  MDTemplateValueParameter *operator->() const { return N; }
  MDTemplateValueParameter &operator*() const { return *N; }
};

class DIGlobalVariable {
  MDGlobalVariable *N;

public:
  DIGlobalVariable(const MDGlobalVariable *N = nullptr)
      : N(const_cast<MDGlobalVariable *>(N)) {}

  operator DIDescriptor() const { return N; }
  operator MDGlobalVariable *() const { return N; }
  MDGlobalVariable *operator->() const { return N; }
  MDGlobalVariable &operator*() const { return *N; }
};

class DIVariable {
  MDLocalVariable *N;

public:
  DIVariable(const MDLocalVariable *N = nullptr)
      : N(const_cast<MDLocalVariable *>(N)) {}

  operator MDLocalVariable *() const { return N; }
  MDLocalVariable *operator->() const { return N; }
  MDLocalVariable &operator*() const { return *N; }
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
