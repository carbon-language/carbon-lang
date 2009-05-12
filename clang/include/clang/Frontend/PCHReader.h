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
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
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
class ASTConsumer;
class ASTContext;
class Attr;
class Decl;
class DeclContext;
class GotoStmt;
class LabelStmt;
class NamedDecl;
class Preprocessor;
class Sema;
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
class PCHReader 
  : public ExternalSemaSource, 
    public IdentifierInfoLookup,
    public ExternalIdentifierLookup,
    public ExternalSLocEntrySource {
public:
  enum PCHReadResult { Success, Failure, IgnorePCH };

private:
  /// \brief The semantic analysis object that will be processing the
  /// PCH file and the translation unit that uses it.
  Sema *SemaObj;

  /// \brief The preprocessor that will be loading the source file.
  Preprocessor &PP;

  /// \brief The AST context into which we'll read the PCH file.
  ASTContext *Context;

  /// \brief The AST consumer.
  ASTConsumer *Consumer;

  /// \brief The bitstream reader from which we'll read the PCH file.
  llvm::BitstreamReader StreamFile;
  llvm::BitstreamCursor Stream;

  /// DeclsCursor - This is a cursor to the start of the DECLS_BLOCK block.  It
  /// has read all the abbreviations at the start of the block and is ready to
  /// jump around with these in context.
  llvm::BitstreamCursor DeclsCursor;

  /// \brief The file name of the PCH file.
  std::string FileName;

  /// \brief The memory buffer that stores the data associated with
  /// this PCH file.
  llvm::OwningPtr<llvm::MemoryBuffer> Buffer;

  /// \brief Offset type for all of the source location entries in the
  /// PCH file.
  const uint32_t *SLocOffsets;

  /// \brief The number of source location entries in the PCH file.
  unsigned TotalNumSLocEntries;

  /// \brief Cursor used to read source location entries.
  llvm::BitstreamCursor SLocEntryCursor;

  /// \brief Offset of each type within the bitstream, indexed by the
  /// type ID, or the representation of a Type*.
  const uint32_t *TypeOffsets;

  /// \brief Types that have already been loaded from the PCH file.
  /// 
  /// When the pointer at index I is non-NULL, the type with 
  /// ID = (I + 1) << 3 has already been loaded from the PCH file.
  std::vector<Type *> TypesLoaded;

  /// \brief Offset of each declaration within the bitstream, indexed
  /// by the declaration ID (-1).
  const uint32_t *DeclOffsets;

  /// \brief Declarations that have already been loaded from the PCH file.
  ///
  /// When the pointer at index I is non-NULL, the declaration with ID
  /// = I + 1 has already been loaded.
  std::vector<Decl *> DeclsLoaded;

  typedef llvm::DenseMap<const DeclContext *, std::pair<uint64_t, uint64_t> >
    DeclContextOffsetsMap;

  /// \brief Offsets of the lexical and visible declarations for each
  /// DeclContext.
  DeclContextOffsetsMap DeclContextOffsets;

  /// \brief Actual data for the on-disk hash table.
  ///
  // This pointer points into a memory buffer, where the on-disk hash
  // table for identifiers actually lives.
  const char *IdentifierTableData;

  /// \brief A pointer to an on-disk hash table of opaque type
  /// IdentifierHashTable.
  void *IdentifierLookupTable;

  /// \brief Offsets into the identifier table data.
  ///
  /// This array is indexed by the identifier ID (-1), and provides
  /// the offset into IdentifierTableData where the string data is
  /// stored.
  const uint32_t *IdentifierOffsets;

  /// \brief A vector containing identifiers that have already been
  /// loaded.
  ///
  /// If the pointer at index I is non-NULL, then it refers to the
  /// IdentifierInfo for the identifier with ID=I+1 that has already
  /// been loaded.
  std::vector<IdentifierInfo *> IdentifiersLoaded;

  /// \brief A pointer to an on-disk hash table of opaque type
  /// PCHMethodPoolLookupTable.
  ///
  /// This hash table provides the instance and factory methods
  /// associated with every selector known in the PCH file.
  void *MethodPoolLookupTable;

  /// \brief A pointer to the character data that comprises the method
  /// pool.
  ///
  /// The SelectorOffsets table refers into this memory.
  const unsigned char *MethodPoolLookupTableData;

  /// \brief The number of selectors stored in the method pool itself.
  unsigned TotalSelectorsInMethodPool;

  /// \brief Offsets into the method pool lookup table's data array
  /// where each selector resides.
  const uint32_t *SelectorOffsets;

  /// \brief The total number of selectors stored in the PCH file.
  unsigned TotalNumSelectors;

  /// \brief A vector containing selectors that have already been loaded. 
  ///
  /// This vector is indexed by the Selector ID (-1). NULL selector
  /// entries indicate that the particular selector ID has not yet
  /// been loaded.
  llvm::SmallVector<Selector, 16> SelectorsLoaded;

  /// \brief The set of external definitions stored in the the PCH
  /// file.
  llvm::SmallVector<uint64_t, 16> ExternalDefinitions;

  /// \brief The set of tentative definitions stored in the the PCH
  /// file.
  llvm::SmallVector<uint64_t, 16> TentativeDefinitions;

  /// \brief The set of locally-scoped external declarations stored in
  /// the the PCH file.
  llvm::SmallVector<uint64_t, 16> LocallyScopedExternalDecls;

  /// \brief The set of ext_vector type declarations stored in the the
  /// PCH file.
  llvm::SmallVector<uint64_t, 4> ExtVectorDecls;

  /// \brief The set of Objective-C category definitions stored in the
  /// the PCH file.
  llvm::SmallVector<uint64_t, 4> ObjCCategoryImpls;

  /// \brief The original file name that was used to build the PCH
  /// file.
  std::string OriginalFileName;

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

  /// \brief The number of stat() calls that hit/missed the stat
  /// cache.
  unsigned NumStatHits, NumStatMisses;

  /// \brief The number of source location entries de-serialized from
  /// the PCH file.
  unsigned NumSLocEntriesRead;

  /// \brief The number of statements (and expressions) de-serialized
  /// from the PCH file.
  unsigned NumStatementsRead;

  /// \brief The total number of statements (and expressions) stored
  /// in the PCH file.
  unsigned TotalNumStatements;

  /// \brief The number of macros de-serialized from the PCH file.
  unsigned NumMacrosRead;

  /// \brief The number of method pool entries that have been read.
  unsigned NumMethodPoolSelectorsRead;

  /// \brief The number of times we have looked into the global method
  /// pool and not found anything.
  unsigned NumMethodPoolMisses;

  /// \brief The total number of macros stored in the PCH file.
  unsigned TotalNumMacros;

  /// Number of lexical decl contexts read/total.
  unsigned NumLexicalDeclContextsRead, TotalLexicalDeclContexts;

  /// Number of visible decl contexts read/total.
  unsigned NumVisibleDeclContextsRead, TotalVisibleDeclContexts;

  /// \brief FIXME: document!
  llvm::SmallVector<uint64_t, 4> SpecialTypes;

  /// \brief Contains declarations and definitions that will be
  /// "interesting" to the ASTConsumer, when we get that AST consumer.
  ///
  /// "Interesting" declarations are those that have data that may
  /// need to be emitted, such as inline function definitions or
  /// Objective-C protocols.
  llvm::SmallVector<Decl *, 16> InterestingDecls;

  /// \brief The file ID for the predefines buffer in the PCH file.
  FileID PCHPredefinesBufferID;

  /// \brief Pointer to the beginning of the predefines buffer in the
  /// PCH file.
  const char *PCHPredefines;

  /// \brief Length of the predefines buffer in the PCH file.
  unsigned PCHPredefinesLen;

  /// \brief Suggested contents of the predefines buffer, after this
  /// PCH file has been processed.
  ///
  /// In most cases, this string will be empty, because the predefines
  /// buffer computed to build the PCH file will be identical to the
  /// predefines buffer computed from the command line. However, when
  /// there are differences that the PCH reader can work around, this
  /// predefines buffer may contain additional definitions.
  std::string SuggestedPredefines;
  
  PCHReadResult ReadPCHBlock();
  bool CheckPredefinesBuffer(const char *PCHPredef, 
                             unsigned PCHPredefLen,
                             FileID PCHBufferID);
  PCHReadResult ReadSourceManagerBlock();
  PCHReadResult ReadSLocEntryRecord(unsigned ID);
  
  bool ParseLanguageOptions(const llvm::SmallVectorImpl<uint64_t> &Record);
  QualType ReadTypeRecord(uint64_t Offset);
  void LoadedDecl(unsigned Index, Decl *D);
  Decl *ReadDeclRecord(uint64_t Offset, unsigned Index);

  /// \brief Produce an error diagnostic and return true.
  ///
  /// This routine should only be used for fatal errors that have to
  /// do with non-routine failures (e.g., corrupted PCH file).
  bool Error(const char *Msg);

  PCHReader(const PCHReader&); // do not implement
  PCHReader &operator=(const PCHReader &); // do not implement

public:
  typedef llvm::SmallVector<uint64_t, 64> RecordData;

  explicit PCHReader(Preprocessor &PP, ASTContext *Context);
  ~PCHReader();

  /// \brief Load the precompiled header designated by the given file
  /// name.
  PCHReadResult ReadPCH(const std::string &FileName);

  /// \brief Retrieve the name of the original source file name 
  const std::string &getOriginalSourceFile() { return OriginalFileName; }

  /// \brief Retrieve the name of the original source file name
  /// directly from the PCH file, without actually loading the PCH
  /// file.
  static std::string getOriginalSourceFile(const std::string &PCHFileName);

  /// \brief Returns the suggested contents of the predefines buffer,
  /// which contains a (typically-empty) subset of the predefines
  /// build prior to including the precompiled header.
  const std::string &getSuggestedPredefines() { return SuggestedPredefines; }

  /// \brief Resolve a type ID into a type, potentially building a new
  /// type.
  virtual QualType GetType(pch::TypeID ID);

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  virtual Decl *GetDecl(pch::DeclID ID);

  /// \brief Resolve the offset of a statement into a statement.
  ///
  /// This operation will read a new statement from the external
  /// source each time it is called, and is meant to be used via a
  /// LazyOffsetPtr (which is used by Decls for the body of functions, etc).
  virtual Stmt *GetDeclStmt(uint64_t Offset);

  /// ReadBlockAbbrevs - Enter a subblock of the specified BlockID with the
  /// specified cursor.  Read the abbreviations that are at the top of the block
  /// and then leave the cursor pointing into the block.
  bool ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor, unsigned BlockID);

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
                                 llvm::SmallVectorImpl<pch::DeclID> &Decls);

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

  /// \brief Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  virtual void InitializeSema(Sema &S);

  /// \brief Retrieve the IdentifierInfo for the named identifier.
  ///
  /// This routine builds a new IdentifierInfo for the given
  /// identifier. If any declarations with this name are visible from
  /// translation unit scope, their declarations will be deserialized
  /// and introduced into the declaration chain of the
  /// identifier. FIXME: if this identifier names a macro, deserialize
  /// the macro.
  virtual IdentifierInfo* get(const char *NameStart, const char *NameEnd);

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  ///
  /// \returns a pair of Objective-C methods lists containing the
  /// instance and factory methods, respectively, with this selector.
  virtual std::pair<ObjCMethodList, ObjCMethodList> 
    ReadMethodPool(Selector Sel);

  void SetIdentifierInfo(unsigned ID, IdentifierInfo *II);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(unsigned DiagID);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  IdentifierInfo *DecodeIdentifierInfo(unsigned Idx);
  
  IdentifierInfo *GetIdentifierInfo(const RecordData &Record, unsigned &Idx) {
    return DecodeIdentifierInfo(Record[Idx++]);
  }
  
  virtual IdentifierInfo *GetIdentifier(unsigned ID) {
    return DecodeIdentifierInfo(ID);
  }

  /// \brief Read the source location entry with index ID.
  virtual void ReadSLocEntry(unsigned ID);

  Selector DecodeSelector(unsigned Idx);
  
  Selector GetSelector(const RecordData &Record, unsigned &Idx) {
    return DecodeSelector(Record[Idx++]);
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

  /// \brief ReadDeclExpr - Reads an expression from the current decl cursor.
  Expr *ReadDeclExpr();
      
  /// \brief ReadTypeExpr - Reads an expression from the current type cursor.
  Expr *ReadTypeExpr();

  /// \brief Reads a statement from the specified cursor.
  Stmt *ReadStmt(llvm::BitstreamCursor &Cursor);
      
  /// \brief Read a statement from the current DeclCursor.
  Stmt *ReadDeclStmt() {
    return ReadStmt(DeclsCursor);
  }

  /// \brief Reads the macro record located at the given offset.
  void ReadMacroRecord(uint64_t Offset);

  /// \brief Retrieve the AST context that this PCH reader
  /// supplements.
  ASTContext *getContext() { return Context; }

  // \brief Contains declarations that were loaded before we have
  // access to a Sema object.
  llvm::SmallVector<NamedDecl *, 16> PreloadedDecls;

  /// \brief Retrieve the semantic analysis object used to analyze the
  /// translation unit in which the precompiled header is being
  /// imported.
  Sema *getSema() { return SemaObj; }

  /// \brief Retrieve the stream that this PCH reader is reading from.
  llvm::BitstreamCursor &getStream() { return Stream; }
  llvm::BitstreamCursor &getDeclsCursor() { return DeclsCursor; }

  /// \brief Retrieve the identifier table associated with the
  /// preprocessor.
  IdentifierTable &getIdentifierTable();

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

/// \brief Helper class that saves the current stream position and
/// then restores it when destroyed.
struct SavedStreamPosition {
  explicit SavedStreamPosition(llvm::BitstreamCursor &Cursor)
  : Cursor(Cursor), Offset(Cursor.GetCurrentBitNo()) { }
  
  ~SavedStreamPosition() {
    Cursor.JumpToBit(Offset);
  }
  
private:
  llvm::BitstreamCursor &Cursor;
  uint64_t Offset;
};
  
} // end namespace clang

#endif
