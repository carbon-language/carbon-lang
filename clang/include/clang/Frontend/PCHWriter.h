//===--- PCHWriter.h - Precompiled Headers Writer ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHWriter class, which writes a precompiled
//  header containing a serialized representation of a translation
//  unit.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_FRONTEND_PCH_WRITER_H
#define LLVM_CLANG_FRONTEND_PCH_WRITER_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Frontend/PCHBitCodes.h"
#include "clang/Frontend/PCHDeserializationListener.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <queue>

namespace llvm {
  class APFloat;
  class APInt;
  class BitstreamWriter;
}

namespace clang {

class ASTContext;
class NestedNameSpecifier;
class CXXBaseSpecifier;
class CXXBaseOrMemberInitializer;
class LabelStmt;
class MacroDefinition;
class MemorizeStatCalls;
class PCHReader;
class Preprocessor;
class Sema;
class SourceManager;
class SwitchCase;
class TargetInfo;

/// A structure for putting "fast"-unqualified QualTypes into a
/// DenseMap.  This uses the standard pointer hash function.
struct UnsafeQualTypeDenseMapInfo {
  static inline bool isEqual(QualType A, QualType B) { return A == B; }
  static inline QualType getEmptyKey() {
    return QualType::getFromOpaquePtr((void*) 1);
  }
  static inline QualType getTombstoneKey() {
    return QualType::getFromOpaquePtr((void*) 2);
  }
  static inline unsigned getHashValue(QualType T) {
    assert(!T.getLocalFastQualifiers() && 
           "hash invalid for types with fast quals");
    uintptr_t v = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
    return (unsigned(v) >> 4) ^ (unsigned(v) >> 9);
  }
};

/// \brief Writes a precompiled header containing the contents of a
/// translation unit.
///
/// The PCHWriter class produces a bitstream containing the serialized
/// representation of a given abstract syntax tree and its supporting
/// data structures. This bitstream can be de-serialized via an
/// instance of the PCHReader class.
class PCHWriter : public PCHDeserializationListener {
public:
  typedef llvm::SmallVector<uint64_t, 64> RecordData;

private:
  /// \brief The bitstream writer used to emit this precompiled header.
  llvm::BitstreamWriter &Stream;

  /// \brief The reader of existing PCH files, if we're chaining.
  PCHReader *Chain;

  /// \brief Stores a declaration or a type to be written to the PCH file.
  class DeclOrType {
  public:
    DeclOrType(Decl *D) : Stored(D), IsType(false) { }
    DeclOrType(QualType T) : Stored(T.getAsOpaquePtr()), IsType(true) { }
    
    bool isType() const { return IsType; }
    bool isDecl() const { return !IsType; }
    
    QualType getType() const {
      assert(isType() && "Not a type!");
      return QualType::getFromOpaquePtr(Stored);
    }
    
    Decl *getDecl() const {
      assert(isDecl() && "Not a decl!");
      return static_cast<Decl *>(Stored);
    }
    
  private:
    void *Stored;
    bool IsType;
  };

  /// \brief The declarations and types to emit.
  std::queue<DeclOrType> DeclTypesToEmit;

  /// \brief The first ID number we can use for our own declarations.
  pch::DeclID FirstDeclID;

  /// \brief The decl ID that will be assigned to the next new decl.
  pch::DeclID NextDeclID;

  /// \brief Map that provides the ID numbers of each declaration within
  /// the output stream, as well as those deserialized from a chained PCH.
  ///
  /// The ID numbers of declarations are consecutive (in order of
  /// discovery) and start at 2. 1 is reserved for the translation
  /// unit, while 0 is reserved for NULL.
  llvm::DenseMap<const Decl *, pch::DeclID> DeclIDs;

  /// \brief Offset of each declaration in the bitstream, indexed by
  /// the declaration's ID.
  std::vector<uint32_t> DeclOffsets;

  /// \brief The first ID number we can use for our own types.
  pch::TypeID FirstTypeID;

  /// \brief The type ID that will be assigned to the next new type.
  pch::TypeID NextTypeID;

  /// \brief Map that provides the ID numbers of each type within the
  /// output stream, plus those deserialized from a chained PCH.
  ///
  /// The ID numbers of types are consecutive (in order of discovery)
  /// and start at 1. 0 is reserved for NULL. When types are actually
  /// stored in the stream, the ID number is shifted by 2 bits to
  /// allow for the const/volatile qualifiers.
  ///
  /// Keys in the map never have const/volatile qualifiers.
  llvm::DenseMap<QualType, pch::TypeID, UnsafeQualTypeDenseMapInfo> TypeIDs;

  /// \brief Offset of each type in the bitstream, indexed by
  /// the type's ID.
  std::vector<uint32_t> TypeOffsets;

  /// \brief The first ID number we can use for our own identifiers.
  pch::IdentID FirstIdentID;

  /// \brief The identifier ID that will be assigned to the next new identifier.
  pch::IdentID NextIdentID;

  /// \brief Map that provides the ID numbers of each identifier in
  /// the output stream.
  ///
  /// The ID numbers for identifiers are consecutive (in order of
  /// discovery), starting at 1. An ID of zero refers to a NULL
  /// IdentifierInfo.
  llvm::DenseMap<const IdentifierInfo *, pch::IdentID> IdentifierIDs;

  /// \brief Offsets of each of the identifier IDs into the identifier
  /// table.
  std::vector<uint32_t> IdentifierOffsets;

  /// \brief Map that provides the ID numbers of each Selector.
  llvm::DenseMap<Selector, pch::SelectorID> SelectorIDs;

  /// \brief Offset of each selector within the method pool/selector
  /// table, indexed by the Selector ID (-1).
  std::vector<uint32_t> SelectorOffsets;

  /// \brief A vector of all Selectors (ordered by ID).
  std::vector<Selector> SelVector;

  /// \brief Offsets of each of the macro identifiers into the
  /// bitstream.
  ///
  /// For each identifier that is associated with a macro, this map
  /// provides the offset into the bitstream where that macro is
  /// defined.
  llvm::DenseMap<const IdentifierInfo *, uint64_t> MacroOffsets;

  /// \brief Mapping from macro definitions (as they occur in the preprocessing
  /// record) to the index into the macro definitions table.
  llvm::DenseMap<const MacroDefinition *, pch::IdentID> MacroDefinitions;
  
  /// \brief Mapping from the macro definition indices in \c MacroDefinitions
  /// to the corresponding offsets within the preprocessor block.
  std::vector<uint32_t> MacroDefinitionOffsets;
  
  /// \brief Declarations encountered that might be external
  /// definitions.
  ///
  /// We keep track of external definitions (as well as tentative
  /// definitions) as we are emitting declarations to the PCH
  /// file. The PCH file contains a separate record for these external
  /// definitions, which are provided to the AST consumer by the PCH
  /// reader. This is behavior is required to properly cope with,
  /// e.g., tentative variable definitions that occur within
  /// headers. The declarations themselves are stored as declaration
  /// IDs, since they will be written out to an EXTERNAL_DEFINITIONS
  /// record.
  llvm::SmallVector<uint64_t, 16> ExternalDefinitions;

  /// \brief Statements that we've encountered while serializing a
  /// declaration or type.
  llvm::SmallVector<Stmt *, 16> StmtsToEmit;
  
  /// \brief Statements collection to use for PCHWriter::AddStmt().
  /// It will point to StmtsToEmit unless it is overriden. 
  llvm::SmallVector<Stmt *, 16> *CollectedStmts;

  /// \brief Mapping from SwitchCase statements to IDs.
  std::map<SwitchCase *, unsigned> SwitchCaseIDs;

  /// \brief Mapping from LabelStmt statements to IDs.
  std::map<LabelStmt *, unsigned> LabelIDs;

  /// \brief The number of statements written to the PCH file.
  unsigned NumStatements;

  /// \brief The number of macros written to the PCH file.
  unsigned NumMacros;

  /// \brief The number of lexical declcontexts written to the PCH
  /// file.
  unsigned NumLexicalDeclContexts;

  /// \brief The number of visible declcontexts written to the PCH
  /// file.
  unsigned NumVisibleDeclContexts;

  /// \brief Write the given subexpression to the bitstream.
  void WriteSubStmt(Stmt *S);

  void WriteBlockInfoBlock();
  void WriteMetadata(ASTContext &Context, const char *isysroot);
  void WriteLanguageOptions(const LangOptions &LangOpts);
  void WriteStatCache(MemorizeStatCalls &StatCalls);
  void WriteSourceManagerBlock(SourceManager &SourceMgr,
                               const Preprocessor &PP,
                               const char* isysroot);
  void WritePreprocessor(const Preprocessor &PP);
  void WriteType(QualType T);
  uint64_t WriteDeclContextLexicalBlock(ASTContext &Context, DeclContext *DC);
  uint64_t WriteDeclContextVisibleBlock(ASTContext &Context, DeclContext *DC);
  void WriteTypeDeclOffsets();
  void WriteMethodPool(Sema &SemaRef);
  void WriteReferencedSelectorsPool(Sema &SemaRef);
  void WriteIdentifierTable(Preprocessor &PP);
  void WriteAttributeRecord(const Attr *Attr);

  unsigned ParmVarDeclAbbrev;
  unsigned DeclContextLexicalAbbrev;
  void WriteDeclsBlockAbbrevs();
  void WriteDecl(ASTContext &Context, Decl *D);

  void WritePCHCore(Sema &SemaRef, MemorizeStatCalls *StatCalls,
                    const char* isysroot);
  void WritePCHChain(Sema &SemaRef, MemorizeStatCalls *StatCalls,
                     const char* isysroot);
  
public:
  /// \brief Create a new precompiled header writer that outputs to
  /// the given bitstream.
  PCHWriter(llvm::BitstreamWriter &Stream);

  /// \brief Write a precompiled header for the given semantic analysis.
  ///
  /// \param SemaRef a reference to the semantic analysis object that processed
  /// the AST to be written into the precompiled header.
  ///
  /// \param StatCalls the object that cached all of the stat() calls made while
  /// searching for source files and headers.
  ///
  /// \param isysroot if non-NULL, write a relocatable PCH file whose headers
  /// are relative to the given system root.
  ///
  /// \param PPRec Record of the preprocessing actions that occurred while
  /// preprocessing this file, e.g., macro instantiations
  void WritePCH(Sema &SemaRef, MemorizeStatCalls *StatCalls,
                const char* isysroot);

  /// \brief Emit a source location.
  void AddSourceLocation(SourceLocation Loc, RecordData &Record);

  /// \brief Emit a source range.
  void AddSourceRange(SourceRange Range, RecordData &Record);
  
  /// \brief Emit an integral value.
  void AddAPInt(const llvm::APInt &Value, RecordData &Record);

  /// \brief Emit a signed integral value.
  void AddAPSInt(const llvm::APSInt &Value, RecordData &Record);

  /// \brief Emit a floating-point value.
  void AddAPFloat(const llvm::APFloat &Value, RecordData &Record);

  /// \brief Emit a reference to an identifier.
  void AddIdentifierRef(const IdentifierInfo *II, RecordData &Record);

  /// \brief Emit a Selector (which is a smart pointer reference).
  void AddSelectorRef(Selector, RecordData &Record);

  /// \brief Emit a CXXTemporary.
  void AddCXXTemporary(const CXXTemporary *Temp, RecordData &Record);
  
  /// \brief Get the unique number used to refer to the given
  /// identifier.
  pch::IdentID getIdentifierRef(const IdentifierInfo *II);

  /// \brief Retrieve the offset of the macro definition for the given
  /// identifier.
  ///
  /// The identifier must refer to a macro.
  uint64_t getMacroOffset(const IdentifierInfo *II) {
    assert(MacroOffsets.find(II) != MacroOffsets.end() &&
           "Identifier does not name a macro");
    return MacroOffsets[II];
  }

  /// \brief Retrieve the ID number corresponding to the given macro 
  /// definition.
  pch::IdentID getMacroDefinitionID(MacroDefinition *MD);
  
  /// \brief Emit a reference to a type.
  void AddTypeRef(QualType T, RecordData &Record);

  /// \brief Emits a reference to a declarator info.
  void AddTypeSourceInfo(TypeSourceInfo *TInfo, RecordData &Record);

  /// \brief Emits a template argument location info.
  void AddTemplateArgumentLocInfo(TemplateArgument::ArgKind Kind,
                                  const TemplateArgumentLocInfo &Arg,
                                  RecordData &Record);

  /// \brief Emits a template argument location.
  void AddTemplateArgumentLoc(const TemplateArgumentLoc &Arg,
                              RecordData &Record);

  /// \brief Emit a reference to a declaration.
  void AddDeclRef(const Decl *D, RecordData &Record);

  /// \brief Force a declaration to be emitted and get its ID.
  pch::DeclID GetDeclRef(const Decl *D);

  /// \brief Determine the declaration ID of an already-emitted
  /// declaration.
  pch::DeclID getDeclID(const Decl *D);

  /// \brief Emit a declaration name.
  void AddDeclarationName(DeclarationName Name, RecordData &Record);

  /// \brief Emit a nested name specifier.
  void AddNestedNameSpecifier(NestedNameSpecifier *NNS, RecordData &Record);
  
  /// \brief Emit a template name.
  void AddTemplateName(TemplateName Name, RecordData &Record);

  /// \brief Emit a template argument.
  void AddTemplateArgument(const TemplateArgument &Arg, RecordData &Record);

  /// \brief Emit a template parameter list.
  void AddTemplateParameterList(const TemplateParameterList *TemplateParams,
                                RecordData &Record);

  /// \brief Emit a template argument list.
  void AddTemplateArgumentList(const TemplateArgumentList *TemplateArgs,
                                RecordData &Record);

  /// \brief Emit a UnresolvedSet structure.
  void AddUnresolvedSet(const UnresolvedSetImpl &Set, RecordData &Record);

  /// brief Emit a C++ base specifier.
  void AddCXXBaseSpecifier(const CXXBaseSpecifier &Base, RecordData &Record);

  /// \brief Add a string to the given record.
  void AddString(const std::string &Str, RecordData &Record);

  /// \brief Note that the identifier II occurs at the given offset
  /// within the identifier table.
  void SetIdentifierOffset(const IdentifierInfo *II, uint32_t Offset);

  /// \brief Note that the selector Sel occurs at the given offset
  /// within the method pool/selector table.
  void SetSelectorOffset(Selector Sel, uint32_t Offset);

  /// \brief Add the given statement or expression to the queue of
  /// statements to emit.
  ///
  /// This routine should be used when emitting types and declarations
  /// that have expressions as part of their formulation. Once the
  /// type or declaration has been written, call FlushStmts() to write
  /// the corresponding statements just after the type or
  /// declaration.
  void AddStmt(Stmt *S) {
      CollectedStmts->push_back(S);
  }

  /// \brief Flush all of the statements and expressions that have
  /// been added to the queue via AddStmt().
  void FlushStmts();

  /// \brief Record an ID for the given switch-case statement.
  unsigned RecordSwitchCaseID(SwitchCase *S);

  /// \brief Retrieve the ID for the given switch-case statement.
  unsigned getSwitchCaseID(SwitchCase *S);

  /// \brief Retrieve the ID for the given label statement, which may
  /// or may not have been emitted yet.
  unsigned GetLabelID(LabelStmt *S);

  unsigned getParmVarDeclAbbrev() const { return ParmVarDeclAbbrev; }

  bool hasChain() const { return Chain; }

  // PCHDeserializationListener implementation
  void SetReader(PCHReader *Reader);
  void IdentifierRead(pch::IdentID ID, IdentifierInfo *II);
  void TypeRead(pch::TypeID ID, QualType T);
  void DeclRead(pch::DeclID ID, const Decl *D);
};

} // end namespace clang

#endif
