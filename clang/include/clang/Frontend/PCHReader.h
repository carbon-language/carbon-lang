//===--- PCHReader.h - Precompiled Headers Reader ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHReader class, which reads a precompiled header.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_FRONTEND_PCH_READER_H
#define LLVM_CLANG_FRONTEND_PCH_READER_H

#include "clang/Frontend/PCHBitCodes.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/DataTypes.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {

class AddrLabelExpr;
class ASTContext;
class Attr;
class Decl;
class DeclContext;
class GotoStmt;
class LabelStmt;
class Preprocessor;
class SwitchCase;

/// \brief Reads a precompiled head containing the contents of a
/// translation unit.
///
/// The PCHReader class reads a bitstream (produced by the PCHWriter
/// class) containing the serialized representation of a given
/// abstract syntax tree and its supporting data structures. An
/// instance of the PCHReader can be attached to an ASTContext object,
/// which will provide access to the contents of the PCH file.
///
/// The PCH reader provides lazy de-serialization of declarations, as
/// required when traversing the AST. Only those AST nodes that are
/// actually required will be de-serialized.
class PCHReader : public ExternalASTSource {
public:
  enum PCHReadResult { Success, Failure, IgnorePCH };

private:
  /// \brief The preprocessor that will be loading the source file.
  Preprocessor &PP;

  /// \brief The AST context into which we'll read the PCH file.
  ASTContext &Context;

  /// \brief The bitstream reader from which we'll read the PCH file.
  llvm::BitstreamReader Stream;

  /// \brief The file name of the PCH file.
  std::string FileName;

  /// \brief The memory buffer that stores the data associated with
  /// this PCH file.
  llvm::OwningPtr<llvm::MemoryBuffer> Buffer;

  /// \brief Offset of each type within the bitstream, indexed by the
  /// type ID, or the representation of a Type*.
  llvm::SmallVector<uint64_t, 16> TypeOffsets;

  /// \brief Whether the type with a given index has already been loaded.
  /// 
  /// When the bit at a given index I is true, then TypeOffsets[I] is
  /// the already-loaded Type*. Otherwise, TypeOffsets[I] is the
  /// location of the type's record in the PCH file.
  ///
  /// FIXME: We can probably eliminate this, e.g., by bitmangling the
  /// values in TypeOffsets.
  std::vector<bool> TypeAlreadyLoaded;

  /// \brief Offset of each declaration within the bitstream, indexed
  /// by the declaration ID.
  llvm::SmallVector<uint64_t, 16> DeclOffsets;

  /// \brief Whether the declaration with a given index has already
  /// been loaded.
  ///
  /// When the bit at the given index I is true, then DeclOffsets[I]
  /// is the already-loaded Decl*. Otherwise, DeclOffsets[I] is the
  /// location of the declaration's record in the PCH file.
  ///
  /// FIXME: We can probably eliminate this, e.g., by bitmangling the
  /// values in DeclOffsets.
  std::vector<bool> DeclAlreadyLoaded;

  typedef llvm::DenseMap<const DeclContext *, std::pair<uint64_t, uint64_t> >
    DeclContextOffsetsMap;

  /// \brief Offsets of the lexical and visible declarations for each
  /// DeclContext.
  DeclContextOffsetsMap DeclContextOffsets;

  /// \brief String data for the identifiers in the PCH file.
  const char *IdentifierTable;

  /// \brief String data for identifiers, indexed by the identifier ID
  /// minus one.
  ///
  /// Each element in this array is either an offset into
  /// IdentifierTable that contains the string data (if the lowest bit
  /// is set) or is an IdentifierInfo* that has already been resolved.
  llvm::SmallVector<uint64_t, 16> IdentifierData;

  /// \brief The set of external definitions stored in the the PCH
  /// file.
  llvm::SmallVector<uint64_t, 16> ExternalDefinitions;

  /// \brief Mapping from switch-case IDs in the PCH file to
  /// switch-case statements.
  std::map<unsigned, SwitchCase *> SwitchCaseStmts;

  /// \brief Mapping from label statement IDs in the PCH file to label
  /// statements.
  std::map<unsigned, LabelStmt *> LabelStmts;

  /// \brief Mapping from label IDs to the set of "goto" statements
  /// that point to that label before the label itself has been
  /// de-serialized.
  std::multimap<unsigned, GotoStmt *> UnresolvedGotoStmts;

  /// \brief Mapping from label IDs to the set of address label
  /// expressions that point to that label before the label itself has
  /// been de-serialized.
  std::multimap<unsigned, AddrLabelExpr *> UnresolvedAddrLabelExprs;

  /// \brief The number of statements (and expressions) de-serialized
  /// from the PCH file.
  unsigned NumStatementsRead;

  /// \brief The total number of statements (and expressions) stored
  /// in the PCH file.
  unsigned TotalNumStatements;

  PCHReadResult ReadPCHBlock();
  bool CheckPredefinesBuffer(const char *PCHPredef, 
                             unsigned PCHPredefLen,
                             FileID PCHBufferID);
  PCHReadResult ReadSourceManagerBlock();
  bool ReadPreprocessorBlock();

  bool ParseLanguageOptions(const llvm::SmallVectorImpl<uint64_t> &Record);
  QualType ReadTypeRecord(uint64_t Offset);
  void LoadedDecl(unsigned Index, Decl *D);
  Decl *ReadDeclRecord(uint64_t Offset, unsigned Index);

  PCHReader(const PCHReader&); // do not implement
  PCHReader &operator=(const PCHReader &); // do not implement

public:
  typedef llvm::SmallVector<uint64_t, 64> RecordData;

  PCHReader(Preprocessor &PP, ASTContext &Context) 
    : PP(PP), Context(Context), IdentifierTable(0), NumStatementsRead(0) { }

  ~PCHReader() {}

  PCHReadResult ReadPCH(const std::string &FileName);

  /// \brief Resolve a type ID into a type, potentially building a new
  /// type.
  virtual QualType GetType(pch::TypeID ID);

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  virtual Decl *GetDecl(pch::DeclID ID);

  /// \brief Read all of the declarations lexically stored in a
  /// declaration context.
  ///
  /// \param DC The declaration context whose declarations will be
  /// read.
  ///
  /// \param Decls Vector that will contain the declarations loaded
  /// from the external source. The caller is responsible for merging
  /// these declarations with any declarations already stored in the
  /// declaration context.
  ///
  /// \returns true if there was an error while reading the
  /// declarations for this declaration context.
  virtual bool ReadDeclsLexicallyInContext(DeclContext *DC,
                                      llvm::SmallVectorImpl<unsigned> &Decls);

  /// \brief Read all of the declarations visible from a declaration
  /// context.
  ///
  /// \param DC The declaration context whose visible declarations
  /// will be read.
  ///
  /// \param Decls A vector of visible declaration structures,
  /// providing the mapping from each name visible in the declaration
  /// context to the declaration IDs of declarations with that name.
  ///
  /// \returns true if there was an error while reading the
  /// declarations for this declaration context.
  ///
  /// FIXME: Using this intermediate data structure results in an
  /// extraneous copying of the data. Could we pass in a reference to
  /// the StoredDeclsMap instead?
  virtual bool ReadDeclsVisibleInContext(DeclContext *DC,
                       llvm::SmallVectorImpl<VisibleDeclaration> & Decls);

  /// \brief Function that will be invoked when we begin parsing a new
  /// translation unit involving this external AST source.
  ///
  /// This function will provide all of the external definitions to
  /// the ASTConsumer.
  virtual void StartTranslationUnit(ASTConsumer *Consumer);

  /// \brief Print some statistics about PCH usage.
  virtual void PrintStats();

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(unsigned DiagID);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  IdentifierInfo *DecodeIdentifierInfo(unsigned Idx);
  
  IdentifierInfo *GetIdentifierInfo(const RecordData &Record, unsigned &Idx) {
    return DecodeIdentifierInfo(Record[Idx++]);
  }
  DeclarationName ReadDeclarationName(const RecordData &Record, unsigned &Idx);

  /// \brief Read an integral value
  llvm::APInt ReadAPInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a signed integral value
  llvm::APSInt ReadAPSInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a floating-point value
  llvm::APFloat ReadAPFloat(const RecordData &Record, unsigned &Idx);

  // \brief Read a string
  std::string ReadString(const RecordData &Record, unsigned &Idx);

  /// \brief Reads attributes from the current stream position.
  Attr *ReadAttributes();

  /// \brief Reads an expression from the current stream position.
  Expr *ReadExpr();

  /// \brief Reads a statement from the current stream position.
  Stmt *ReadStmt();

  /// \brief Retrieve the AST context that this PCH reader
  /// supplements.
  ASTContext &getContext() { return Context; }

  /// \brief Record that the given ID maps to the given switch-case
  /// statement.
  void RecordSwitchCaseID(SwitchCase *SC, unsigned ID);

  /// \brief Retrieve the switch-case statement with the given ID.
  SwitchCase *getSwitchCaseWithID(unsigned ID);

  /// \brief Record that the given label statement has been
  /// deserialized and has the given ID.
  void RecordLabelStmt(LabelStmt *S, unsigned ID);

  /// \brief Set the label of the given statement to the label
  /// identified by ID.
  ///
  /// Depending on the order in which the label and other statements
  /// referencing that label occur, this operation may complete
  /// immediately (updating the statement) or it may queue the
  /// statement to be back-patched later.
  void SetLabelOf(GotoStmt *S, unsigned ID);

  /// \brief Set the label of the given expression to the label
  /// identified by ID.
  ///
  /// Depending on the order in which the label and other statements
  /// referencing that label occur, this operation may complete
  /// immediately (updating the statement) or it may queue the
  /// statement to be back-patched later.
  void SetLabelOf(AddrLabelExpr *S, unsigned ID);
};

} // end namespace clang

#endif
