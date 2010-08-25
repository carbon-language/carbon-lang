//===--- ASTReader.h - AST File Reader --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTReader class, which reads AST files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_AST_READER_H
#define LLVM_CLANG_FRONTEND_AST_READER_H

#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/System/DataTypes.h"
#include <deque>
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
class NestedNameSpecifier;
class CXXBaseSpecifier;
class CXXBaseOrMemberInitializer;
class GotoStmt;
class LabelStmt;
class MacroDefinition;
class NamedDecl;
class ASTDeserializationListener;
class Preprocessor;
class Sema;
class SwitchCase;
class ASTReader;
class ASTDeclReader;
struct HeaderFileInfo;

struct PCHPredefinesBlock {
  /// \brief The file ID for this predefines buffer in a PCH file.
  FileID BufferID;

  /// \brief This predefines buffer in a PCH file.
  llvm::StringRef Data;
};
typedef llvm::SmallVector<PCHPredefinesBlock, 2> PCHPredefinesBlocks;

/// \brief Abstract interface for callback invocations by the ASTReader.
///
/// While reading an AST file, the ASTReader will call the methods of the
/// listener to pass on specific information. Some of the listener methods can
/// return true to indicate to the ASTReader that the information (and
/// consequently the AST file) is invalid.
class ASTReaderListener {
public:
  virtual ~ASTReaderListener();

  /// \brief Receives the language options.
  ///
  /// \returns true to indicate the options are invalid or false otherwise.
  virtual bool ReadLanguageOptions(const LangOptions &LangOpts) {
    return false;
  }

  /// \brief Receives the target triple.
  ///
  /// \returns true to indicate the target triple is invalid or false otherwise.
  virtual bool ReadTargetTriple(llvm::StringRef Triple) {
    return false;
  }

  /// \brief Receives the contents of the predefines buffer.
  ///
  /// \param Buffers Information about the predefines buffers.
  ///
  /// \param OriginalFileName The original file name for the AST file, which
  /// will appear as an entry in the predefines buffer.
  ///
  /// \param SuggestedPredefines If necessary, additional definitions are added
  /// here.
  ///
  /// \returns true to indicate the predefines are invalid or false otherwise.
  virtual bool ReadPredefinesBuffer(const PCHPredefinesBlocks &Buffers,
                                    llvm::StringRef OriginalFileName,
                                    std::string &SuggestedPredefines) {
    return false;
  }

  /// \brief Receives a HeaderFileInfo entry.
  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID) {}

  /// \brief Receives __COUNTER__ value.
  virtual void ReadCounter(unsigned Value) {}
};

/// \brief ASTReaderListener implementation to validate the information of
/// the PCH file against an initialized Preprocessor.
class PCHValidator : public ASTReaderListener {
  Preprocessor &PP;
  ASTReader &Reader;

  unsigned NumHeaderInfos;

public:
  PCHValidator(Preprocessor &PP, ASTReader &Reader)
    : PP(PP), Reader(Reader), NumHeaderInfos(0) {}

  virtual bool ReadLanguageOptions(const LangOptions &LangOpts);
  virtual bool ReadTargetTriple(llvm::StringRef Triple);
  virtual bool ReadPredefinesBuffer(const PCHPredefinesBlocks &Buffers,
                                    llvm::StringRef OriginalFileName,
                                    std::string &SuggestedPredefines);
  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID);
  virtual void ReadCounter(unsigned Value);

private:
  void Error(const char *Msg);
};

/// \brief Reads an AST files chain containing the contents of a translation
/// unit.
///
/// The ASTReader class reads bitstreams (produced by the ASTWriter
/// class) containing the serialized representation of a given
/// abstract syntax tree and its supporting data structures. An
/// instance of the ASTReader can be attached to an ASTContext object,
/// which will provide access to the contents of the AST files.
///
/// The AST reader provides lazy de-serialization of declarations, as
/// required when traversing the AST. Only those AST nodes that are
/// actually required will be de-serialized.
class ASTReader
  : public ExternalPreprocessorSource,
    public ExternalPreprocessingRecordSource,
    public ExternalSemaSource,
    public IdentifierInfoLookup,
    public ExternalIdentifierLookup,
    public ExternalSLocEntrySource {
public:
  enum ASTReadResult { Success, Failure, IgnorePCH };
  friend class PCHValidator;
  friend class ASTDeclReader;
private:
  /// \brief The receiver of some callbacks invoked by ASTReader.
  llvm::OwningPtr<ASTReaderListener> Listener;

  /// \brief The receiver of deserialization events.
  ASTDeserializationListener *DeserializationListener;

  SourceManager &SourceMgr;
  FileManager &FileMgr;
  Diagnostic &Diags;

  /// \brief The semantic analysis object that will be processing the
  /// AST files and the translation unit that uses it.
  Sema *SemaObj;

  /// \brief The preprocessor that will be loading the source file.
  Preprocessor *PP;

  /// \brief The AST context into which we'll read the AST files.
  ASTContext *Context;
      
  /// \brief The AST consumer.
  ASTConsumer *Consumer;

  /// \brief Information that is needed for every file in the chain.
  struct PerFileData {
    PerFileData();
    ~PerFileData();

    /// \brief The AST stat cache installed for this file, if any.
    ///
    /// The dynamic type of this stat cache is always ASTStatCache
    void *StatCache;

    /// \brief The bitstream reader from which we'll read the AST file.
    llvm::BitstreamReader StreamFile;
    llvm::BitstreamCursor Stream;

    /// \brief The size of this file, in bits.
    uint64_t SizeInBits;

    /// \brief The cursor to the start of the preprocessor block, which stores
    /// all of the macro definitions.
    llvm::BitstreamCursor MacroCursor;
      
    /// DeclsCursor - This is a cursor to the start of the DECLS_BLOCK block. It
    /// has read all the abbreviations at the start of the block and is ready to
    /// jump around with these in context.
    llvm::BitstreamCursor DeclsCursor;

    /// \brief The file name of the AST file.
    std::string FileName;

    /// \brief The memory buffer that stores the data associated with
    /// this AST file.
    llvm::OwningPtr<llvm::MemoryBuffer> Buffer;

    /// \brief Cursor used to read source location entries.
    llvm::BitstreamCursor SLocEntryCursor;

    /// \brief The number of source location entries in this AST file.
    unsigned LocalNumSLocEntries;

    /// \brief Offsets for all of the source location entries in the
    /// AST file.
    const uint32_t *SLocOffsets;

    /// \brief The number of types in this AST file.
    unsigned LocalNumTypes;

    /// \brief Offset of each type within the bitstream, indexed by the
    /// type ID, or the representation of a Type*.
    const uint32_t *TypeOffsets;

    /// \brief The number of declarations in this AST file.
    unsigned LocalNumDecls;

    /// \brief Offset of each declaration within the bitstream, indexed
    /// by the declaration ID (-1).
    const uint32_t *DeclOffsets;

    /// \brief The number of identifiers in this AST file.
    unsigned LocalNumIdentifiers;

    /// \brief Offsets into the identifier table data.
    ///
    /// This array is indexed by the identifier ID (-1), and provides
    /// the offset into IdentifierTableData where the string data is
    /// stored.
    const uint32_t *IdentifierOffsets;

    /// \brief Actual data for the on-disk hash table.
    ///
    // This pointer points into a memory buffer, where the on-disk hash
    // table for identifiers actually lives.
    const char *IdentifierTableData;

    /// \brief A pointer to an on-disk hash table of opaque type
    /// IdentifierHashTable.
    void *IdentifierLookupTable;

    /// \brief The number of macro definitions in this file.
    unsigned LocalNumMacroDefinitions;

    /// \brief Offsets of all of the macro definitions in the preprocessing
    /// record in the AST file.
    const uint32_t *MacroDefinitionOffsets;
      
    /// \brief The number of preallocated preprocessing entities in the
    /// preprocessing record.
    unsigned NumPreallocatedPreprocessingEntities;

    /// \brief A pointer to an on-disk hash table of opaque type
    /// ASTSelectorLookupTable.
    ///
    /// This hash table provides the IDs of all selectors, and the associated
    /// instance and factory methods.
    void *SelectorLookupTable;

    /// \brief A pointer to the character data that comprises the selector table
    ///
    /// The SelectorOffsets table refers into this memory.
    const unsigned char *SelectorLookupTableData;

    /// \brief Offsets into the method pool lookup table's data array
    /// where each selector resides.
    const uint32_t *SelectorOffsets;

    /// \brief The number of selectors new to this file.
    ///
    /// This is the number of entries in SelectorOffsets.
    unsigned LocalNumSelectors;
  };

  /// \brief The chain of AST files. The first entry is the one named by the
  /// user, the last one is the one that doesn't depend on anything further.
  /// That is, the entry I was created with -include-pch I+1.
  llvm::SmallVector<PerFileData*, 2> Chain;

  /// \brief Types that have already been loaded from the chain.
  ///
  /// When the pointer at index I is non-NULL, the type with
  /// ID = (I + 1) << FastQual::Width has already been loaded
  std::vector<QualType> TypesLoaded;

  /// \brief Map that provides the ID numbers of each type within the
  /// output stream, plus those deserialized from a chained PCH.
  ///
  /// The ID numbers of types are consecutive (in order of discovery)
  /// and start at 1. 0 is reserved for NULL. When types are actually
  /// stored in the stream, the ID number is shifted by 2 bits to
  /// allow for the const/volatile qualifiers.
  ///
  /// Keys in the map never have const/volatile qualifiers.
  serialization::TypeIdxMap TypeIdxs;

  /// \brief Declarations that have already been loaded from the chain.
  ///
  /// When the pointer at index I is non-NULL, the declaration with ID
  /// = I + 1 has already been loaded.
  std::vector<Decl *> DeclsLoaded;

  typedef llvm::DenseMap<serialization::DeclID,
                         std::pair<PerFileData *, uint64_t> >
      DeclReplacementMap;
  /// \brief Declarations that have been replaced in a later file in the chain.
  DeclReplacementMap ReplacedDecls;

  /// \brief Information about the contents of a DeclContext.
  struct DeclContextInfo {
    void *NameLookupTableData; // a ASTDeclContextNameLookupTable.
    const serialization::DeclID *LexicalDecls;
    unsigned NumLexicalDecls;
  };
  // In a full chain, there could be multiple updates to every decl context,
  // so this is a vector. However, typically a chain is only two elements long,
  // with only one file containing updates, so there will be only one update
  // per decl context.
  typedef llvm::SmallVector<DeclContextInfo, 1> DeclContextInfos;
  typedef llvm::DenseMap<const DeclContext *, DeclContextInfos>
      DeclContextOffsetsMap;
  // Updates for visible decls can occur for other contexts than just the
  // TU, and when we read those update records, the actual context will not
  // be available yet (unless it's the TU), so have this pending map using the
  // ID as a key. It will be realized when the context is actually loaded.
  typedef llvm::SmallVector<void *, 1> DeclContextVisibleUpdates;
  typedef llvm::DenseMap<serialization::DeclID, DeclContextVisibleUpdates>
      DeclContextVisibleUpdatesPending;

  /// \brief Offsets of the lexical and visible declarations for each
  /// DeclContext.
  DeclContextOffsetsMap DeclContextOffsets;

  /// \brief Updates to the visible declarations of declaration contexts that
  /// haven't been loaded yet.
  DeclContextVisibleUpdatesPending PendingVisibleUpdates;

  typedef llvm::DenseMap<serialization::DeclID, serialization::DeclID>
      FirstLatestDeclIDMap;
  /// \brief Map of first declarations from a chained PCH that point to the
  /// most recent declarations in another AST file.
  FirstLatestDeclIDMap FirstLatestDeclIDs;

  typedef llvm::SmallVector<serialization::DeclID, 4>
      AdditionalTemplateSpecializations;
  typedef llvm::DenseMap<serialization::DeclID,
                         AdditionalTemplateSpecializations>
      AdditionalTemplateSpecializationsMap;

  /// \brief Additional specializations (including partial) of templates that
  /// were introduced after the template was serialized.
  AdditionalTemplateSpecializationsMap AdditionalTemplateSpecializationsPending;

  /// \brief Read the records that describe the contents of declcontexts.
  bool ReadDeclContextStorage(llvm::BitstreamCursor &Cursor,
                              const std::pair<uint64_t, uint64_t> &Offsets,
                              DeclContextInfo &Info);

  /// \brief A vector containing identifiers that have already been
  /// loaded.
  ///
  /// If the pointer at index I is non-NULL, then it refers to the
  /// IdentifierInfo for the identifier with ID=I+1 that has already
  /// been loaded.
  std::vector<IdentifierInfo *> IdentifiersLoaded;

  /// \brief A vector containing selectors that have already been loaded.
  ///
  /// This vector is indexed by the Selector ID (-1). NULL selector
  /// entries indicate that the particular selector ID has not yet
  /// been loaded.
  llvm::SmallVector<Selector, 16> SelectorsLoaded;

  /// \brief Method selectors used in a @selector expression. Used for
  /// implementation of -Wselector.
  llvm::SmallVector<long long unsigned int,64u> ReferencedSelectorsData;
      
  /// \brief The macro definitions we have already loaded.
  llvm::SmallVector<MacroDefinition *, 16> MacroDefinitionsLoaded;
      
  /// \brief The set of external definitions stored in the the chain.
  llvm::SmallVector<uint64_t, 16> ExternalDefinitions;

  /// \brief The set of tentative definitions stored in the the chain.
  llvm::SmallVector<uint64_t, 16> TentativeDefinitions;
      
  /// \brief The set of unused file scoped decls stored in the the chain.
  llvm::SmallVector<uint64_t, 16> UnusedFileScopedDecls;
  
  /// \brief The set of weak undeclared identifiers stored in the chain.
  llvm::SmallVector<uint64_t, 64> WeakUndeclaredIdentifiers;

  /// \brief The set of locally-scoped external declarations stored in the chain
  llvm::SmallVector<uint64_t, 16> LocallyScopedExternalDecls;

  /// \brief The set of ext_vector type declarations stored in the the chain.
  llvm::SmallVector<uint64_t, 4> ExtVectorDecls;

  /// \brief The set of VTable uses of CXXRecordDecls stored in the chain.
  llvm::SmallVector<uint64_t, 64> VTableUses;

  /// \brief The set of dynamic CXXRecord declarations stored in the chain.
  llvm::SmallVector<uint64_t, 16> DynamicClasses;

  /// \brief The set of pending implicit instantiations stored in the chain.
  llvm::SmallVector<uint64_t, 64> PendingImplicitInstantiations;

  /// \brief The set of Sema declaration references stored in the chain.
  llvm::SmallVector<uint64_t, 4> SemaDeclRefs;

  /// \brief The original file name that was used to build the primary AST file,
  /// which may have been modified for relocatable-pch support.
  std::string OriginalFileName;

  /// \brief The actual original file name that was used to build the primary
  /// AST file.
  std::string ActualOriginalFileName;

  /// \brief Whether this precompiled header is a relocatable PCH file.
  bool RelocatablePCH;

  /// \brief The system include root to be used when loading the
  /// precompiled header.
  const char *isysroot;

  /// \brief Whether to disable the normal validation performed on precompiled
  /// headers when they are loaded.
  bool DisableValidation;
      
  /// \brief Mapping from switch-case IDs in the chain to switch-case statements
  std::map<unsigned, SwitchCase *> SwitchCaseStmts;

  /// \brief Mapping from label statement IDs in the chain to label statements.
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

  /// \brief The number of source location entries in the chain.
  unsigned TotalNumSLocEntries;

  /// \brief The number of statements (and expressions) de-serialized
  /// from the chain.
  unsigned NumStatementsRead;

  /// \brief The total number of statements (and expressions) stored
  /// in the chain.
  unsigned TotalNumStatements;

  /// \brief The number of macros de-serialized from the chain.
  unsigned NumMacrosRead;

  /// \brief The total number of macros stored in the chain.
  unsigned TotalNumMacros;

  /// \brief The number of selectors that have been read.
  unsigned NumSelectorsRead;

  /// \brief The number of method pool entries that have been read.
  unsigned NumMethodPoolEntriesRead;

  /// \brief The number of times we have looked up a selector in the method
  /// pool and not found anything interesting.
  unsigned NumMethodPoolMisses;

  /// \brief The total number of method pool entries in the selector table.
  unsigned TotalNumMethodPoolEntries;

  /// Number of lexical decl contexts read/total.
  unsigned NumLexicalDeclContextsRead, TotalLexicalDeclContexts;

  /// Number of visible decl contexts read/total.
  unsigned NumVisibleDeclContextsRead, TotalVisibleDeclContexts;
  
  /// \brief Number of Decl/types that are currently deserializing.
  unsigned NumCurrentElementsDeserializing;

  /// \brief An IdentifierInfo that has been loaded but whose top-level
  /// declarations of the same name have not (yet) been loaded.
  struct PendingIdentifierInfo {
    IdentifierInfo *II;
    llvm::SmallVector<uint32_t, 4> DeclIDs;
  };

  /// \brief The set of identifiers that were read while the AST reader was
  /// (recursively) loading declarations.
  ///
  /// The declarations on the identifier chain for these identifiers will be
  /// loaded once the recursive loading has completed.
  std::deque<PendingIdentifierInfo> PendingIdentifierInfos;

  /// \brief FIXME: document!
  llvm::SmallVector<uint64_t, 16> SpecialTypes;

  /// \brief Contains declarations and definitions that will be
  /// "interesting" to the ASTConsumer, when we get that AST consumer.
  ///
  /// "Interesting" declarations are those that have data that may
  /// need to be emitted, such as inline function definitions or
  /// Objective-C protocols.
  std::deque<Decl *> InterestingDecls;

  /// \brief When reading a Stmt tree, Stmt operands are placed in this stack.
  llvm::SmallVector<Stmt *, 16> StmtStack;

  /// \brief What kind of records we are reading.
  enum ReadingKind {
    Read_Decl, Read_Type, Read_Stmt
  };

  /// \brief What kind of records we are reading. 
  ReadingKind ReadingKind;

  /// \brief RAII object to change the reading kind.
  class ReadingKindTracker {
    ASTReader &Reader;
    enum ReadingKind PrevKind;

    ReadingKindTracker(const ReadingKindTracker&); // do not implement
    ReadingKindTracker &operator=(const ReadingKindTracker&);// do not implement

  public:
    ReadingKindTracker(enum ReadingKind newKind, ASTReader &reader)
      : Reader(reader), PrevKind(Reader.ReadingKind) {
      Reader.ReadingKind = newKind;
    }

    ~ReadingKindTracker() { Reader.ReadingKind = PrevKind; }
  };

  /// \brief All predefines buffers in the chain, to be treated as if
  /// concatenated.
  PCHPredefinesBlocks PCHPredefinesBuffers;

  /// \brief Suggested contents of the predefines buffer, after this
  /// PCH file has been processed.
  ///
  /// In most cases, this string will be empty, because the predefines
  /// buffer computed to build the PCH file will be identical to the
  /// predefines buffer computed from the command line. However, when
  /// there are differences that the PCH reader can work around, this
  /// predefines buffer may contain additional definitions.
  std::string SuggestedPredefines;

  /// \brief Reads a statement from the specified cursor.
  Stmt *ReadStmtFromStream(llvm::BitstreamCursor &Cursor);

  void MaybeAddSystemRootToFilename(std::string &Filename);

  ASTReadResult ReadASTCore(llvm::StringRef FileName);
  ASTReadResult ReadASTBlock(PerFileData &F);
  bool CheckPredefinesBuffers();
  bool ParseLineTable(llvm::SmallVectorImpl<uint64_t> &Record);
  ASTReadResult ReadSourceManagerBlock(PerFileData &F);
  ASTReadResult ReadSLocEntryRecord(unsigned ID);
  llvm::BitstreamCursor &SLocCursorForID(unsigned ID);
  bool ParseLanguageOptions(const llvm::SmallVectorImpl<uint64_t> &Record);

  typedef std::pair<llvm::BitstreamCursor *, uint64_t> RecordLocation;

  QualType ReadTypeRecord(unsigned Index);
  RecordLocation TypeCursorForIndex(unsigned Index);
  void LoadedDecl(unsigned Index, Decl *D);
  Decl *ReadDeclRecord(unsigned Index, serialization::DeclID ID);
  RecordLocation DeclCursorForIndex(unsigned Index, serialization::DeclID ID);

  void PassInterestingDeclsToConsumer();

  /// \brief Produce an error diagnostic and return true.
  ///
  /// This routine should only be used for fatal errors that have to
  /// do with non-routine failures (e.g., corrupted AST file).
  void Error(const char *Msg);

  ASTReader(const ASTReader&); // do not implement
  ASTReader &operator=(const ASTReader &); // do not implement
public:
  typedef llvm::SmallVector<uint64_t, 64> RecordData;

  /// \brief Load the AST file and validate its contents against the given
  /// Preprocessor.
  ///
  /// \param PP the preprocessor associated with the context in which this
  /// precompiled header will be loaded.
  ///
  /// \param Context the AST context that this precompiled header will be
  /// loaded into.
  ///
  /// \param isysroot If non-NULL, the system include path specified by the
  /// user. This is only used with relocatable PCH files. If non-NULL,
  /// a relocatable PCH file will use the default path "/".
  ///
  /// \param DisableValidation If true, the AST reader will suppress most
  /// of its regular consistency checking, allowing the use of precompiled
  /// headers that cannot be determined to be compatible.
  ASTReader(Preprocessor &PP, ASTContext *Context, const char *isysroot = 0,
            bool DisableValidation = false);

  /// \brief Load the AST file without using any pre-initialized Preprocessor.
  ///
  /// The necessary information to initialize a Preprocessor later can be
  /// obtained by setting a ASTReaderListener.
  ///
  /// \param SourceMgr the source manager into which the AST file will be loaded
  ///
  /// \param FileMgr the file manager into which the AST file will be loaded.
  ///
  /// \param Diags the diagnostics system to use for reporting errors and
  /// warnings relevant to loading the AST file.
  ///
  /// \param isysroot If non-NULL, the system include path specified by the
  /// user. This is only used with relocatable PCH files. If non-NULL,
  /// a relocatable PCH file will use the default path "/".
  ///
  /// \param DisableValidation If true, the AST reader will suppress most
  /// of its regular consistency checking, allowing the use of precompiled
  /// headers that cannot be determined to be compatible.
      ASTReader(SourceManager &SourceMgr, FileManager &FileMgr,
            Diagnostic &Diags, const char *isysroot = 0,
            bool DisableValidation = false);
  ~ASTReader();

  /// \brief Load the precompiled header designated by the given file
  /// name.
  ASTReadResult ReadAST(const std::string &FileName);

  /// \brief Set the AST callbacks listener.
  void setListener(ASTReaderListener *listener) {
    Listener.reset(listener);
  }

  /// \brief Set the AST deserialization listener.
  void setDeserializationListener(ASTDeserializationListener *Listener);

  /// \brief Set the Preprocessor to use.
  void setPreprocessor(Preprocessor &pp);

  /// \brief Sets and initializes the given Context.
  void InitializeContext(ASTContext &Context);

  /// \brief Retrieve the name of the named (primary) AST file
  const std::string &getFileName() const { return Chain[0]->FileName; }

  /// \brief Retrieve the name of the original source file name
  const std::string &getOriginalSourceFile() { return OriginalFileName; }

  /// \brief Retrieve the name of the original source file name directly from
  /// the AST file, without actually loading the AST file.
  static std::string getOriginalSourceFile(const std::string &ASTFileName,
                                           Diagnostic &Diags);

  /// \brief Returns the suggested contents of the predefines buffer,
  /// which contains a (typically-empty) subset of the predefines
  /// build prior to including the precompiled header.
  const std::string &getSuggestedPredefines() { return SuggestedPredefines; }
      
  /// \brief Read preprocessed entities into the 
  virtual void ReadPreprocessedEntities();

  /// \brief Returns the number of source locations found in the chain.
  unsigned getTotalNumSLocs() const {
    return TotalNumSLocEntries;
  }

  /// \brief Returns the number of identifiers found in the chain.
  unsigned getTotalNumIdentifiers() const {
    return static_cast<unsigned>(IdentifiersLoaded.size());
  }

  /// \brief Returns the number of types found in the chain.
  unsigned getTotalNumTypes() const {
    return static_cast<unsigned>(TypesLoaded.size());
  }

  /// \brief Returns the number of declarations found in the chain.
  unsigned getTotalNumDecls() const {
    return static_cast<unsigned>(DeclsLoaded.size());
  }

  /// \brief Returns the number of selectors found in the chain.
  unsigned getTotalNumSelectors() const {
    return static_cast<unsigned>(SelectorsLoaded.size());
  }

  /// \brief Reads a TemplateArgumentLocInfo appropriate for the
  /// given TemplateArgument kind.
  TemplateArgumentLocInfo
  GetTemplateArgumentLocInfo(TemplateArgument::ArgKind Kind,
                             llvm::BitstreamCursor &DeclsCursor,
                             const RecordData &Record, unsigned &Idx);

  /// \brief Reads a TemplateArgumentLoc.
  TemplateArgumentLoc
  ReadTemplateArgumentLoc(llvm::BitstreamCursor &DeclsCursor,
                          const RecordData &Record, unsigned &Idx);

  /// \brief Reads a declarator info from the given record.
  TypeSourceInfo *GetTypeSourceInfo(llvm::BitstreamCursor &DeclsCursor,
                                    const RecordData &Record, unsigned &Idx);

  /// \brief Resolve and return the translation unit declaration.
  TranslationUnitDecl *GetTranslationUnitDecl();

  /// \brief Resolve a type ID into a type, potentially building a new
  /// type.
  QualType GetType(serialization::TypeID ID);

  /// \brief Returns the type ID associated with the given type.
  /// If the type didn't come from the AST file the ID that is returned is
  /// marked as "doesn't exist in AST".
  serialization::TypeID GetTypeID(QualType T) const;

  /// \brief Returns the type index associated with the given type.
  /// If the type didn't come from the AST file the index that is returned is
  /// marked as "doesn't exist in AST".
  serialization::TypeIdx GetTypeIdx(QualType T) const;

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  Decl *GetDecl(serialization::DeclID ID);
  virtual Decl *GetExternalDecl(uint32_t ID);

  /// \brief Resolve the offset of a statement into a statement.
  ///
  /// This operation will read a new statement from the external
  /// source each time it is called, and is meant to be used via a
  /// LazyOffsetPtr (which is used by Decls for the body of functions, etc).
  virtual Stmt *GetExternalDeclStmt(uint64_t Offset);

  /// ReadBlockAbbrevs - Enter a subblock of the specified BlockID with the
  /// specified cursor.  Read the abbreviations that are at the top of the block
  /// and then leave the cursor pointing into the block.
  bool ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor, unsigned BlockID);

  /// \brief Finds all the visible declarations with a given name.
  /// The current implementation of this method just loads the entire
  /// lookup table as unmaterialized references.
  virtual DeclContext::lookup_result
  FindExternalVisibleDeclsByName(const DeclContext *DC,
                                 DeclarationName Name);

  virtual void MaterializeVisibleDecls(const DeclContext *DC);

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
  virtual bool FindExternalLexicalDecls(const DeclContext *DC,
                                        llvm::SmallVectorImpl<Decl*> &Decls);

  /// \brief Notify ASTReader that we started deserialization of
  /// a decl or type so until FinishedDeserializing is called there may be
  /// decls that are initializing. Must be paired with FinishedDeserializing.
  virtual void StartedDeserializing() { ++NumCurrentElementsDeserializing; }

  /// \brief Notify ASTReader that we finished the deserialization of
  /// a decl or type. Must be paired with StartedDeserializing.
  virtual void FinishedDeserializing();

  /// \brief Function that will be invoked when we begin parsing a new
  /// translation unit involving this external AST source.
  ///
  /// This function will provide all of the external definitions to
  /// the ASTConsumer.
  virtual void StartTranslationUnit(ASTConsumer *Consumer);

  /// \brief Print some statistics about AST usage.
  virtual void PrintStats();

  /// \brief Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  virtual void InitializeSema(Sema &S);

  /// \brief Inform the semantic consumer that Sema is no longer available.
  virtual void ForgetSema() { SemaObj = 0; }

  /// \brief Retrieve the IdentifierInfo for the named identifier.
  ///
  /// This routine builds a new IdentifierInfo for the given identifier. If any
  /// declarations with this name are visible from translation unit scope, their
  /// declarations will be deserialized and introduced into the declaration
  /// chain of the identifier.
  virtual IdentifierInfo *get(const char *NameStart, const char *NameEnd);
  IdentifierInfo *get(llvm::StringRef Name) {
    return get(Name.begin(), Name.end());
  }

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  ///
  /// \returns a pair of Objective-C methods lists containing the
  /// instance and factory methods, respectively, with this selector.
  virtual std::pair<ObjCMethodList, ObjCMethodList>
    ReadMethodPool(Selector Sel);

  /// \brief Load a selector from disk, registering its ID if it exists.
  void LoadSelector(Selector Sel);

  void SetIdentifierInfo(unsigned ID, IdentifierInfo *II);
  void SetGloballyVisibleDecls(IdentifierInfo *II,
                               const llvm::SmallVectorImpl<uint32_t> &DeclIDs,
                               bool Nonrecursive = false);

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

  virtual Selector GetExternalSelector(uint32_t ID);
  uint32_t GetNumExternalSelectors();

  Selector GetSelector(const RecordData &Record, unsigned &Idx) {
    return DecodeSelector(Record[Idx++]);
  }

  /// \brief Read a declaration name.
  DeclarationName ReadDeclarationName(const RecordData &Record, unsigned &Idx);

  NestedNameSpecifier *ReadNestedNameSpecifier(const RecordData &Record,
                                               unsigned &Idx);

  /// \brief Read a template name.
  TemplateName ReadTemplateName(const RecordData &Record, unsigned &Idx);

  /// \brief Read a template argument.
  TemplateArgument ReadTemplateArgument(llvm::BitstreamCursor &DeclsCursor,
                                        const RecordData &Record,unsigned &Idx);
  
  /// \brief Read a template parameter list.
  TemplateParameterList *ReadTemplateParameterList(const RecordData &Record,
                                                   unsigned &Idx);
  
  /// \brief Read a template argument array.
  void
  ReadTemplateArgumentList(llvm::SmallVector<TemplateArgument, 8> &TemplArgs,
                           llvm::BitstreamCursor &DeclsCursor,
                           const RecordData &Record, unsigned &Idx);

  /// \brief Read a UnresolvedSet structure.
  void ReadUnresolvedSet(UnresolvedSetImpl &Set,
                         const RecordData &Record, unsigned &Idx);

  /// \brief Read a C++ base specifier.
  CXXBaseSpecifier ReadCXXBaseSpecifier(llvm::BitstreamCursor &DeclsCursor,
                                        const RecordData &Record,unsigned &Idx);

  /// \brief Read a CXXBaseOrMemberInitializer array.
  std::pair<CXXBaseOrMemberInitializer **, unsigned>
  ReadCXXBaseOrMemberInitializers(llvm::BitstreamCursor &DeclsCursor,
                                  const RecordData &Record, unsigned &Idx);

  /// \brief Read a source location.
  SourceLocation ReadSourceLocation(const RecordData &Record, unsigned& Idx) {
    return SourceLocation::getFromRawEncoding(Record[Idx++]);
  }

  /// \brief Read a source range.
  SourceRange ReadSourceRange(const RecordData &Record, unsigned& Idx);

  /// \brief Read an integral value
  llvm::APInt ReadAPInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a signed integral value
  llvm::APSInt ReadAPSInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a floating-point value
  llvm::APFloat ReadAPFloat(const RecordData &Record, unsigned &Idx);

  // \brief Read a string
  std::string ReadString(const RecordData &Record, unsigned &Idx);

  CXXTemporary *ReadCXXTemporary(const RecordData &Record, unsigned &Idx);
      
  /// \brief Reads attributes from the current stream position.
  void ReadAttributes(llvm::BitstreamCursor &DeclsCursor, AttrVec &Attrs);

  /// \brief Reads a statement.
  Stmt *ReadStmt(llvm::BitstreamCursor &Cursor);

  /// \brief Reads an expression.
  Expr *ReadExpr(llvm::BitstreamCursor &Cursor);

  /// \brief Reads a sub-statement operand during statement reading.
  Stmt *ReadSubStmt() {
    assert(ReadingKind == Read_Stmt &&
           "Should be called only during statement reading!");
    // Subexpressions are stored from last to first, so the next Stmt we need
    // is at the back of the stack.
    assert(!StmtStack.empty() && "Read too many sub statements!");
    return StmtStack.pop_back_val();
  }

  /// \brief Reads a sub-expression operand during statement reading.
  Expr *ReadSubExpr();

  /// \brief Reads the macro record located at the given offset.
  void ReadMacroRecord(llvm::BitstreamCursor &Stream, uint64_t Offset);

  /// \brief Read the set of macros defined by this external macro source.
  virtual void ReadDefinedMacros();

  /// \brief Retrieve the macro definition with the given ID.
  MacroDefinition *getMacroDefinition(serialization::IdentID ID);

  /// \brief Retrieve the AST context that this AST reader supplements.
  ASTContext *getContext() { return Context; }

  // \brief Contains declarations that were loaded before we have
  // access to a Sema object.
  llvm::SmallVector<NamedDecl *, 16> PreloadedDecls;

  /// \brief Retrieve the semantic analysis object used to analyze the
  /// translation unit in which the precompiled header is being
  /// imported.
  Sema *getSema() { return SemaObj; }

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

inline void PCHValidator::Error(const char *Msg) {
  Reader.Error(Msg);
}

} // end namespace clang

#endif
