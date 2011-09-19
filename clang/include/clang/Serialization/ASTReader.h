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
#include "clang/Serialization/ContinuousRangeMap.h"
#include "clang/Serialization/Module.h"
#include "clang/Serialization/ModuleManager.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/DataTypes.h"
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
class ASTIdentifierIterator;
class ASTUnit; // FIXME: Layering violation and egregious hack.
class Attr;
class Decl;
class DeclContext;
class NestedNameSpecifier;
class CXXBaseSpecifier;
class CXXConstructorDecl;
class CXXCtorInitializer;
class GotoStmt;
class MacroDefinition;
class NamedDecl;
class OpaqueValueExpr;
class Preprocessor;
class Sema;
class SwitchCase;
class ASTDeserializationListener;
class ASTWriter;
class ASTReader;
class ASTDeclReader;
class ASTStmtReader;
class TypeLocReader;
struct HeaderFileInfo;
class VersionTuple;

struct PCHPredefinesBlock {
  /// \brief The file ID for this predefines buffer in a PCH file.
  FileID BufferID;

  /// \brief This predefines buffer in a PCH file.
  StringRef Data;
};
typedef SmallVector<PCHPredefinesBlock, 2> PCHPredefinesBlocks;

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
  virtual bool ReadTargetTriple(StringRef Triple) {
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
                                    StringRef OriginalFileName,
                                    std::string &SuggestedPredefines,
                                    FileManager &FileMgr) {
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
  virtual bool ReadTargetTriple(StringRef Triple);
  virtual bool ReadPredefinesBuffer(const PCHPredefinesBlocks &Buffers,
                                    StringRef OriginalFileName,
                                    std::string &SuggestedPredefines,
                                    FileManager &FileMgr);
  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID);
  virtual void ReadCounter(unsigned Value);

private:
  void Error(const char *Msg);
};

namespace serialization {    

class ReadMethodPoolVisitor;
  
namespace reader {
  class ASTIdentifierLookupTrait;
}
  
} // end namespace serialization
  
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
    public ExternalHeaderFileInfoSource,
    public ExternalSemaSource,
    public IdentifierInfoLookup,
    public ExternalIdentifierLookup,
    public ExternalSLocEntrySource 
{
public:
  enum ASTReadResult { Success, Failure, IgnorePCH };
  /// \brief Types of AST files.
  friend class PCHValidator;
  friend class ASTDeclReader;
  friend class ASTStmtReader;
  friend class ASTIdentifierIterator;
  friend class serialization::reader::ASTIdentifierLookupTrait;
  friend class TypeLocReader;
  friend class ASTWriter;
  friend class ASTUnit; // ASTUnit needs to remap source locations.
  friend class serialization::ReadMethodPoolVisitor;
  
  typedef serialization::Module Module;
  typedef serialization::ModuleKind ModuleKind;
  typedef serialization::ModuleManager ModuleManager;
  
  typedef ModuleManager::ModuleIterator ModuleIterator;
  typedef ModuleManager::ModuleConstIterator ModuleConstIterator;
  typedef ModuleManager::ModuleReverseIterator ModuleReverseIterator;

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
  Preprocessor &PP;

  /// \brief The AST context into which we'll read the AST files.
  ASTContext &Context;
      
  /// \brief The AST consumer.
  ASTConsumer *Consumer;

  /// \brief The module manager which manages modules and their dependencies
  ModuleManager ModuleMgr;

  /// \brief A map of global bit offsets to the module that stores entities
  /// at those bit offsets.
  ContinuousRangeMap<uint64_t, Module*, 4> GlobalBitOffsetsMap;

  /// \brief A map of negated SLocEntryIDs to the modules containing them.
  ContinuousRangeMap<unsigned, Module*, 64> GlobalSLocEntryMap;

  typedef ContinuousRangeMap<int, Module*, 64> GlobalSLocOffsetMapType;
  
  /// \brief A map of negated SourceLocation offsets to the modules containing
  /// them.
  GlobalSLocOffsetMapType GlobalSLocOffsetMap;
  
  /// \brief Types that have already been loaded from the chain.
  ///
  /// When the pointer at index I is non-NULL, the type with
  /// ID = (I + 1) << FastQual::Width has already been loaded
  std::vector<QualType> TypesLoaded;

  typedef ContinuousRangeMap<serialization::TypeID, Module *, 4>
    GlobalTypeMapType;

  /// \brief Mapping from global type IDs to the module in which the
  /// type resides along with the offset that should be added to the
  /// global type ID to produce a local ID.
  GlobalTypeMapType GlobalTypeMap;

  /// \brief Declarations that have already been loaded from the chain.
  ///
  /// When the pointer at index I is non-NULL, the declaration with ID
  /// = I + 1 has already been loaded.
  std::vector<Decl *> DeclsLoaded;

  typedef ContinuousRangeMap<serialization::DeclID, Module *, 4> 
    GlobalDeclMapType;
  
  /// \brief Mapping from global declaration IDs to the module in which the
  /// declaration resides.
  GlobalDeclMapType GlobalDeclMap;
  
  typedef std::pair<Module *, uint64_t> FileOffset;
  typedef SmallVector<FileOffset, 2> FileOffsetsTy;
  typedef llvm::DenseMap<serialization::DeclID, FileOffsetsTy>
      DeclUpdateOffsetsMap;
  
  /// \brief Declarations that have modifications residing in a later file
  /// in the chain.
  DeclUpdateOffsetsMap DeclUpdateOffsets;

  typedef llvm::DenseMap<serialization::DeclID,
                         std::pair<Module *, uint64_t> >
      DeclReplacementMap;
  /// \brief Declarations that have been replaced in a later file in the chain.
  DeclReplacementMap ReplacedDecls;

  // Updates for visible decls can occur for other contexts than just the
  // TU, and when we read those update records, the actual context will not
  // be available yet (unless it's the TU), so have this pending map using the
  // ID as a key. It will be realized when the context is actually loaded.
  typedef SmallVector<std::pair<void *, Module*>, 1> DeclContextVisibleUpdates;
  typedef llvm::DenseMap<serialization::DeclID, DeclContextVisibleUpdates>
      DeclContextVisibleUpdatesPending;

  /// \brief Updates to the visible declarations of declaration contexts that
  /// haven't been loaded yet.
  DeclContextVisibleUpdatesPending PendingVisibleUpdates;

  typedef SmallVector<CXXRecordDecl *, 4> ForwardRefs;
  typedef llvm::DenseMap<const CXXRecordDecl *, ForwardRefs>
      PendingForwardRefsMap;
  /// \brief Forward references that have a definition but the definition decl
  /// is still initializing. When the definition gets read it will update
  /// the DefinitionData pointer of all pending references.
  PendingForwardRefsMap PendingForwardRefs;

  typedef llvm::DenseMap<serialization::DeclID, serialization::DeclID>
      FirstLatestDeclIDMap;
  /// \brief Map of first declarations from a chained PCH that point to the
  /// most recent declarations in another AST file.
  FirstLatestDeclIDMap FirstLatestDeclIDs;

  /// \brief Set of ObjC interfaces that have categories chained to them in
  /// other modules.
  llvm::DenseSet<serialization::GlobalDeclID> ObjCChainedCategoriesInterfaces;

  /// \brief Read the records that describe the contents of declcontexts.
  bool ReadDeclContextStorage(Module &M, 
                              llvm::BitstreamCursor &Cursor,
                              const std::pair<uint64_t, uint64_t> &Offsets,
                              serialization::DeclContextInfo &Info);

  /// \brief A vector containing identifiers that have already been
  /// loaded.
  ///
  /// If the pointer at index I is non-NULL, then it refers to the
  /// IdentifierInfo for the identifier with ID=I+1 that has already
  /// been loaded.
  std::vector<IdentifierInfo *> IdentifiersLoaded;

  typedef ContinuousRangeMap<serialization::IdentID, Module *, 4> 
    GlobalIdentifierMapType;
  
  /// \brief Mapping from global identifer IDs to the module in which the
  /// identifier resides along with the offset that should be added to the
  /// global identifier ID to produce a local ID.
  GlobalIdentifierMapType GlobalIdentifierMap;

  /// \brief A vector containing selectors that have already been loaded.
  ///
  /// This vector is indexed by the Selector ID (-1). NULL selector
  /// entries indicate that the particular selector ID has not yet
  /// been loaded.
  SmallVector<Selector, 16> SelectorsLoaded;

  typedef ContinuousRangeMap<serialization::SelectorID, Module *, 4> 
    GlobalSelectorMapType;
  
  /// \brief Mapping from global selector IDs to the module in which the
  /// selector resides along with the offset that should be added to the
  /// global selector ID to produce a local ID.
  GlobalSelectorMapType GlobalSelectorMap;

  /// \brief Mapping from identifiers that represent macros whose definitions
  /// have not yet been deserialized to the global offset where the macro
  /// record resides.
  llvm::DenseMap<IdentifierInfo *, uint64_t> UnreadMacroRecordOffsets;

  typedef ContinuousRangeMap<unsigned, Module *, 4> 
    GlobalPreprocessedEntityMapType;
  
  /// \brief Mapping from global preprocessing entity IDs to the module in
  /// which the preprocessed entity resides along with the offset that should be
  /// added to the global preprocessing entitiy ID to produce a local ID.
  GlobalPreprocessedEntityMapType GlobalPreprocessedEntityMap;
  
  /// \name CodeGen-relevant special data
  /// \brief Fields containing data that is relevant to CodeGen.
  //@{

  /// \brief The IDs of all declarations that fulfill the criteria of
  /// "interesting" decls.
  ///
  /// This contains the data loaded from all EXTERNAL_DEFINITIONS blocks in the
  /// chain. The referenced declarations are deserialized and passed to the
  /// consumer eagerly.
  SmallVector<uint64_t, 16> ExternalDefinitions;

  /// \brief The IDs of all tentative definitions stored in the the chain.
  ///
  /// Sema keeps track of all tentative definitions in a TU because it has to
  /// complete them and pass them on to CodeGen. Thus, tentative definitions in
  /// the PCH chain must be eagerly deserialized.
  SmallVector<uint64_t, 16> TentativeDefinitions;

  /// \brief The IDs of all CXXRecordDecls stored in the chain whose VTables are
  /// used.
  ///
  /// CodeGen has to emit VTables for these records, so they have to be eagerly
  /// deserialized.
  SmallVector<uint64_t, 64> VTableUses;

  /// \brief A snapshot of the pending instantiations in the chain.
  ///
  /// This record tracks the instantiations that Sema has to perform at the
  /// end of the TU. It consists of a pair of values for every pending
  /// instantiation where the first value is the ID of the decl and the second
  /// is the instantiation location.
  SmallVector<uint64_t, 64> PendingInstantiations;

  //@}

  /// \name Diagnostic-relevant special data
  /// \brief Fields containing data that is used for generating diagnostics
  //@{

  /// \brief A snapshot of Sema's unused file-scoped variable tracking, for
  /// generating warnings.
  SmallVector<uint64_t, 16> UnusedFileScopedDecls;

  /// \brief A list of all the delegating constructors we've seen, to diagnose
  /// cycles.
  SmallVector<uint64_t, 4> DelegatingCtorDecls;
  
  /// \brief Method selectors used in a @selector expression. Used for
  /// implementation of -Wselector.
  SmallVector<uint64_t, 64> ReferencedSelectorsData;

  /// \brief A snapshot of Sema's weak undeclared identifier tracking, for
  /// generating warnings.
  SmallVector<uint64_t, 64> WeakUndeclaredIdentifiers;

  /// \brief The IDs of type aliases for ext_vectors that exist in the chain.
  ///
  /// Used by Sema for finding sugared names for ext_vectors in diagnostics.
  SmallVector<uint64_t, 4> ExtVectorDecls;

  //@}

  /// \name Sema-relevant special data
  /// \brief Fields containing data that is used for semantic analysis
  //@{

  /// \brief The IDs of all locally scoped external decls in the chain.
  ///
  /// Sema tracks these to validate that the types are consistent across all
  /// local external declarations.
  SmallVector<uint64_t, 16> LocallyScopedExternalDecls;

  /// \brief The IDs of all dynamic class declarations in the chain.
  ///
  /// Sema tracks these because it checks for the key functions being defined
  /// at the end of the TU, in which case it directs CodeGen to emit the VTable.
  SmallVector<uint64_t, 16> DynamicClasses;

  /// \brief The IDs of the declarations Sema stores directly.
  ///
  /// Sema tracks a few important decls, such as namespace std, directly.
  SmallVector<uint64_t, 4> SemaDeclRefs;

  /// \brief The IDs of the types ASTContext stores directly.
  ///
  /// The AST context tracks a few important types, such as va_list, directly.
  SmallVector<uint64_t, 16> SpecialTypes;

  /// \brief The IDs of CUDA-specific declarations ASTContext stores directly.
  ///
  /// The AST context tracks a few important decls, currently cudaConfigureCall,
  /// directly.
  SmallVector<uint64_t, 2> CUDASpecialDeclRefs;

  /// \brief The floating point pragma option settings.
  SmallVector<uint64_t, 1> FPPragmaOptions;

  /// \brief The OpenCL extension settings.
  SmallVector<uint64_t, 1> OpenCLExtensions;

  /// \brief A list of the namespaces we've seen.
  SmallVector<uint64_t, 4> KnownNamespaces;

  //@}

  /// \brief The original file name that was used to build the primary AST file,
  /// which may have been modified for relocatable-pch support.
  std::string OriginalFileName;

  /// \brief The actual original file name that was used to build the primary
  /// AST file.
  std::string ActualOriginalFileName;

  /// \brief The file ID for the original file that was used to build the
  /// primary AST file.
  FileID OriginalFileID;
  
  /// \brief The directory that the PCH was originally created in. Used to
  /// allow resolving headers even after headers+PCH was moved to a new path.
  std::string OriginalDir;

  /// \brief The directory that the PCH we are reading is stored in.
  std::string CurrentDir;

  /// \brief Whether this precompiled header is a relocatable PCH file.
  bool RelocatablePCH;

  /// \brief The system include root to be used when loading the
  /// precompiled header.
  std::string isysroot;

  /// \brief Whether to disable the normal validation performed on precompiled
  /// headers when they are loaded.
  bool DisableValidation;
      
  /// \brief Whether to disable the use of stat caches in AST files.
  bool DisableStatCache;

  /// \brief Mapping from switch-case IDs in the chain to switch-case statements
  ///
  /// Statements usually don't have IDs, but switch cases need them, so that the
  /// switch statement can refer to them.
  std::map<unsigned, SwitchCase *> SwitchCaseStmts;

  /// \brief Mapping from opaque value IDs to OpaqueValueExprs.
  std::map<unsigned, OpaqueValueExpr*> OpaqueValueExprs;

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
  
  /// Total size of modules, in bits, currently loaded
  uint64_t TotalModulesSizeInBits;

  /// \brief Number of Decl/types that are currently deserializing.
  unsigned NumCurrentElementsDeserializing;

  /// Number of CXX base specifiers currently loaded
  unsigned NumCXXBaseSpecifiersLoaded;

  /// \brief An IdentifierInfo that has been loaded but whose top-level
  /// declarations of the same name have not (yet) been loaded.
  struct PendingIdentifierInfo {
    IdentifierInfo *II;
    SmallVector<uint32_t, 4> DeclIDs;
  };

  /// \brief The set of identifiers that were read while the AST reader was
  /// (recursively) loading declarations.
  ///
  /// The declarations on the identifier chain for these identifiers will be
  /// loaded once the recursive loading has completed.
  std::deque<PendingIdentifierInfo> PendingIdentifierInfos;

  /// \brief Contains declarations and definitions that will be
  /// "interesting" to the ASTConsumer, when we get that AST consumer.
  ///
  /// "Interesting" declarations are those that have data that may
  /// need to be emitted, such as inline function definitions or
  /// Objective-C protocols.
  std::deque<Decl *> InterestingDecls;

  /// \brief We delay loading of the previous declaration chain to avoid
  /// deeply nested calls when there are many redeclarations.
  std::deque<std::pair<Decl *, serialization::DeclID> > PendingPreviousDecls;

  /// \brief Ready to load the previous declaration of the given Decl.
  void loadAndAttachPreviousDecl(Decl *D, serialization::DeclID ID);

  /// \brief When reading a Stmt tree, Stmt operands are placed in this stack.
  SmallVector<Stmt *, 16> StmtStack;

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
  Stmt *ReadStmtFromStream(Module &F);

  /// \brief Get a FileEntry out of stored-in-PCH filename, making sure we take
  /// into account all the necessary relocations.
  const FileEntry *getFileEntry(StringRef filename);

  void MaybeAddSystemRootToFilename(std::string &Filename);

  ASTReadResult ReadASTCore(StringRef FileName, ModuleKind Type,
                            Module *ImportedBy);
  ASTReadResult ReadASTBlock(Module &F);
  bool CheckPredefinesBuffers();
  bool ParseLineTable(Module &F, SmallVectorImpl<uint64_t> &Record);
  ASTReadResult ReadSourceManagerBlock(Module &F);
  ASTReadResult ReadSLocEntryRecord(int ID);
  llvm::BitstreamCursor &SLocCursorForID(int ID);
  SourceLocation getImportLocation(Module *F);
  bool ParseLanguageOptions(const SmallVectorImpl<uint64_t> &Record);

  struct RecordLocation {
    RecordLocation(Module *M, uint64_t O)
      : F(M), Offset(O) {}
    Module *F;
    uint64_t Offset;
  };

  QualType readTypeRecord(unsigned Index);
  RecordLocation TypeCursorForIndex(unsigned Index);
  void LoadedDecl(unsigned Index, Decl *D);
  Decl *ReadDeclRecord(serialization::DeclID ID);
  RecordLocation DeclCursorForID(serialization::DeclID ID);
  void loadDeclUpdateRecords(serialization::DeclID ID, Decl *D);
  void loadObjCChainedCategories(serialization::GlobalDeclID ID,
                                 ObjCInterfaceDecl *D);
  
  RecordLocation getLocalBitOffset(uint64_t GlobalOffset);
  uint64_t getGlobalBitOffset(Module &M, uint32_t LocalOffset);

  /// \brief Returns the first preprocessed entity ID that ends after \arg BLoc.
  serialization::PreprocessedEntityID
    findBeginPreprocessedEntity(SourceLocation BLoc) const;

  /// \brief Returns the first preprocessed entity ID that begins after \arg ELoc.
  serialization::PreprocessedEntityID
    findEndPreprocessedEntity(SourceLocation ELoc) const;

  /// \brief \arg SLocMapI points at a chunk of a module that contains no
  /// preprocessed entities or the entities it contains are not the ones we are
  /// looking for. Find the next module that contains entities and return the ID
  /// of the first entry.
  serialization::PreprocessedEntityID
    findNextPreprocessedEntity(
                        GlobalSLocOffsetMapType::const_iterator SLocMapI) const;

  void PassInterestingDeclsToConsumer();

  /// \brief Produce an error diagnostic and return true.
  ///
  /// This routine should only be used for fatal errors that have to
  /// do with non-routine failures (e.g., corrupted AST file).
  void Error(StringRef Msg);
  void Error(unsigned DiagID, StringRef Arg1 = StringRef(),
             StringRef Arg2 = StringRef());

  ASTReader(const ASTReader&); // do not implement
  ASTReader &operator=(const ASTReader &); // do not implement
public:
  typedef SmallVector<uint64_t, 64> RecordData;

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
  ///
  /// \param DisableStatCache If true, the AST reader will ignore the
  /// stat cache in the AST files. This performance pessimization can
  /// help when an AST file is being used in cases where the
  /// underlying files in the file system may have changed, but
  /// parsing should still continue.
  ASTReader(Preprocessor &PP, ASTContext &Context, StringRef isysroot = "",
            bool DisableValidation = false, bool DisableStatCache = false);

  ~ASTReader();

  SourceManager &getSourceManager() const { return SourceMgr; }
  
  /// \brief Load the AST file designated by the given file name.
  ASTReadResult ReadAST(const std::string &FileName, ModuleKind Type);

  /// \brief Checks that no file that is stored in PCH is out-of-sync with
  /// the actual file in the file system.
  ASTReadResult validateFileEntries(Module &M);

  /// \brief Set the AST callbacks listener.
  void setListener(ASTReaderListener *listener) {
    Listener.reset(listener);
  }

  /// \brief Set the AST deserialization listener.
  void setDeserializationListener(ASTDeserializationListener *Listener);

  /// \brief Initializes the ASTContext
  void InitializeContext();

  /// \brief Add in-memory (virtual file) buffer.
  void addInMemoryBuffer(StringRef &FileName, llvm::MemoryBuffer *Buffer) {
    ModuleMgr.addInMemoryBuffer(FileName, Buffer);
  }

  /// \brief Retrieve the module manager.
  ModuleManager &getModuleManager() { return ModuleMgr; }

  /// \brief Retrieve the preprocessor.
  Preprocessor &getPreprocessor() const { return PP; }
  
  /// \brief Retrieve the name of the original source file name
  const std::string &getOriginalSourceFile() { return OriginalFileName; }

  /// \brief Retrieve the name of the original source file name directly from
  /// the AST file, without actually loading the AST file.
  static std::string getOriginalSourceFile(const std::string &ASTFileName,
                                           FileManager &FileMgr,
                                           Diagnostic &Diags);

  /// \brief Returns the suggested contents of the predefines buffer,
  /// which contains a (typically-empty) subset of the predefines
  /// build prior to including the precompiled header.
  const std::string &getSuggestedPredefines() { return SuggestedPredefines; }

  /// \brief Read a preallocated preprocessed entity from the external source.
  ///
  /// \returns null if an error occurred that prevented the preprocessed
  /// entity from being loaded.
  virtual PreprocessedEntity *ReadPreprocessedEntity(unsigned Index);

  /// \brief Returns a pair of [Begin, End) indices of preallocated
  /// preprocessed entities that \arg Range encompasses.
  virtual std::pair<unsigned, unsigned>
      findPreprocessedEntitiesInRange(SourceRange Range);

  /// \brief Read the header file information for the given file entry.
  virtual HeaderFileInfo GetHeaderFileInfo(const FileEntry *FE);

  void ReadPragmaDiagnosticMappings(Diagnostic &Diag);

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

  /// \brief Returns the number of preprocessed entities known to the AST
  /// reader.
  unsigned getTotalNumPreprocessedEntities() const {
    unsigned Result = 0;
    for (ModuleConstIterator I = ModuleMgr.begin(),
        E = ModuleMgr.end(); I != E; ++I) {
      Result += (*I)->NumPreprocessedEntities;
    }
    
    return Result;
  }
      
  /// \brief Returns the number of C++ base specifiers found in the chain.
  unsigned getTotalNumCXXBaseSpecifiers() const {
    return NumCXXBaseSpecifiersLoaded;
  }
      
  /// \brief Reads a TemplateArgumentLocInfo appropriate for the
  /// given TemplateArgument kind.
  TemplateArgumentLocInfo
  GetTemplateArgumentLocInfo(Module &F, TemplateArgument::ArgKind Kind,
                             const RecordData &Record, unsigned &Idx);

  /// \brief Reads a TemplateArgumentLoc.
  TemplateArgumentLoc
  ReadTemplateArgumentLoc(Module &F,
                          const RecordData &Record, unsigned &Idx);

  /// \brief Reads a declarator info from the given record.
  TypeSourceInfo *GetTypeSourceInfo(Module &F,
                                    const RecordData &Record, unsigned &Idx);

  /// \brief Resolve a type ID into a type, potentially building a new
  /// type.
  QualType GetType(serialization::TypeID ID);

  /// \brief Resolve a local type ID within a given AST file into a type.
  QualType getLocalType(Module &F, unsigned LocalID);
  
  /// \brief Map a local type ID within a given AST file into a global type ID.
  serialization::TypeID getGlobalTypeID(Module &F, unsigned LocalID) const;
  
  /// \brief Read a type from the current position in the given record, which 
  /// was read from the given AST file.
  QualType readType(Module &F, const RecordData &Record, unsigned &Idx) {
    if (Idx >= Record.size())
      return QualType();
    
    return getLocalType(F, Record[Idx++]);
  }
  
  /// \brief Map from a local declaration ID within a given module to a 
  /// global declaration ID.
  serialization::DeclID getGlobalDeclID(Module &F, unsigned LocalID) const;

  /// \brief Returns true if global DeclID \arg ID originated from module
  /// \arg M.
  bool isDeclIDFromModule(serialization::GlobalDeclID ID, Module &M) const;
  
  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  Decl *GetDecl(serialization::DeclID ID);
  virtual Decl *GetExternalDecl(uint32_t ID);

  /// \brief Reads a declaration with the given local ID in the given module.
  Decl *GetLocalDecl(Module &F, uint32_t LocalID) {
    return GetDecl(getGlobalDeclID(F, LocalID));
  }

  /// \brief Reads a declaration with the given local ID in the given module.
  ///
  /// \returns The requested declaration, casted to the given return type.
  template<typename T>
  T *GetLocalDeclAs(Module &F, uint32_t LocalID) {
    return cast_or_null<T>(GetLocalDecl(F, LocalID));
  }

  /// \brief Reads a declaration ID from the given position in a record in the 
  /// given module.
  ///
  /// \returns The declaration ID read from the record, adjusted to a global ID.
  serialization::DeclID ReadDeclID(Module &F, const RecordData &Record,
                                   unsigned &Idx);
  
  /// \brief Reads a declaration from the given position in a record in the
  /// given module.
  Decl *ReadDecl(Module &F, const RecordData &R, unsigned &I) {
    return GetDecl(ReadDeclID(F, R, I));
  }
  
  /// \brief Reads a declaration from the given position in a record in the
  /// given module.
  ///
  /// \returns The declaration read from this location, casted to the given
  /// result type.
  template<typename T>
  T *ReadDeclAs(Module &F, const RecordData &R, unsigned &I) {
    return cast_or_null<T>(GetDecl(ReadDeclID(F, R, I)));
  }

  /// \brief Read a CXXBaseSpecifiers ID form the given record and
  /// return its global bit offset.
  uint64_t readCXXBaseSpecifiers(Module &M, const RecordData &Record, 
                                 unsigned &Idx);
      
  virtual CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset);
      
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
  virtual ExternalLoadResult FindExternalLexicalDecls(const DeclContext *DC,
                                        bool (*isKindWeWant)(Decl::Kind),
                                        SmallVectorImpl<Decl*> &Decls);

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

  /// \brief Dump information about the AST reader to standard error.
  void dump();
  
  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  virtual void getMemoryBufferSizes(MemoryBufferSizes &sizes) const;

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
  IdentifierInfo *get(StringRef Name) {
    return get(Name.begin(), Name.end());
  }

  /// \brief Retrieve an iterator into the set of all identifiers
  /// in all loaded AST files.
  virtual IdentifierIterator *getIdentifiers() const;

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  ///
  /// \returns a pair of Objective-C methods lists containing the
  /// instance and factory methods, respectively, with this selector.
  virtual std::pair<ObjCMethodList, ObjCMethodList>
    ReadMethodPool(Selector Sel);

  /// \brief Load the set of namespaces that are known to the external source,
  /// which will be used during typo correction.
  virtual void ReadKnownNamespaces(
                           SmallVectorImpl<NamespaceDecl *> &Namespaces);

  virtual void ReadTentativeDefinitions(
                 SmallVectorImpl<VarDecl *> &TentativeDefs);

  virtual void ReadUnusedFileScopedDecls(
                 SmallVectorImpl<const DeclaratorDecl *> &Decls);

  virtual void ReadDelegatingConstructors(
                 SmallVectorImpl<CXXConstructorDecl *> &Decls);

  virtual void ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl *> &Decls);

  virtual void ReadDynamicClasses(SmallVectorImpl<CXXRecordDecl *> &Decls);

  virtual void ReadLocallyScopedExternalDecls(
                 SmallVectorImpl<NamedDecl *> &Decls);
  
  virtual void ReadReferencedSelectors(
                 SmallVectorImpl<std::pair<Selector, SourceLocation> > &Sels);

  virtual void ReadWeakUndeclaredIdentifiers(
                 SmallVectorImpl<std::pair<IdentifierInfo *, WeakInfo> > &WI);

  virtual void ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables);

  virtual void ReadPendingInstantiations(
                 SmallVectorImpl<std::pair<ValueDecl *, 
                                           SourceLocation> > &Pending);

  /// \brief Load a selector from disk, registering its ID if it exists.
  void LoadSelector(Selector Sel);

  void SetIdentifierInfo(unsigned ID, IdentifierInfo *II);
  void SetGloballyVisibleDecls(IdentifierInfo *II,
                               const SmallVectorImpl<uint32_t> &DeclIDs,
                               bool Nonrecursive = false);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(unsigned DiagID);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  IdentifierInfo *DecodeIdentifierInfo(serialization::IdentifierID ID);

  IdentifierInfo *GetIdentifierInfo(Module &M, const RecordData &Record, 
                                    unsigned &Idx) {
    return DecodeIdentifierInfo(getGlobalIdentifierID(M, Record[Idx++]));
  }

  virtual IdentifierInfo *GetIdentifier(serialization::IdentifierID ID) {
    return DecodeIdentifierInfo(ID);
  }

  IdentifierInfo *getLocalIdentifier(Module &M, unsigned LocalID);
  
  serialization::IdentifierID getGlobalIdentifierID(Module &M, 
                                                    unsigned LocalID);
                                 
  /// \brief Read the source location entry with index ID.
  virtual bool ReadSLocEntry(int ID);

  /// \brief Retrieve a selector from the given module with its local ID
  /// number.
  Selector getLocalSelector(Module &M, unsigned LocalID);

  Selector DecodeSelector(serialization::SelectorID Idx);

  virtual Selector GetExternalSelector(serialization::SelectorID ID);
  uint32_t GetNumExternalSelectors();

  Selector ReadSelector(Module &M, const RecordData &Record, unsigned &Idx) {
    return getLocalSelector(M, Record[Idx++]);
  }
  
  /// \brief Retrieve the global selector ID that corresponds to this
  /// the local selector ID in a given module.
  serialization::SelectorID getGlobalSelectorID(Module &F, 
                                                unsigned LocalID) const;

  /// \brief Read a declaration name.
  DeclarationName ReadDeclarationName(Module &F, 
                                      const RecordData &Record, unsigned &Idx);
  void ReadDeclarationNameLoc(Module &F,
                              DeclarationNameLoc &DNLoc, DeclarationName Name,
                              const RecordData &Record, unsigned &Idx);
  void ReadDeclarationNameInfo(Module &F, DeclarationNameInfo &NameInfo,
                               const RecordData &Record, unsigned &Idx);

  void ReadQualifierInfo(Module &F, QualifierInfo &Info,
                         const RecordData &Record, unsigned &Idx);

  NestedNameSpecifier *ReadNestedNameSpecifier(Module &F,
                                               const RecordData &Record,
                                               unsigned &Idx);

  NestedNameSpecifierLoc ReadNestedNameSpecifierLoc(Module &F, 
                                                    const RecordData &Record,
                                                    unsigned &Idx);

  /// \brief Read a template name.
  TemplateName ReadTemplateName(Module &F, const RecordData &Record, 
                                unsigned &Idx);

  /// \brief Read a template argument.
  TemplateArgument ReadTemplateArgument(Module &F,
                                        const RecordData &Record,unsigned &Idx);
  
  /// \brief Read a template parameter list.
  TemplateParameterList *ReadTemplateParameterList(Module &F,
                                                   const RecordData &Record,
                                                   unsigned &Idx);
  
  /// \brief Read a template argument array.
  void
  ReadTemplateArgumentList(SmallVector<TemplateArgument, 8> &TemplArgs,
                           Module &F, const RecordData &Record,
                           unsigned &Idx);

  /// \brief Read a UnresolvedSet structure.
  void ReadUnresolvedSet(Module &F, UnresolvedSetImpl &Set,
                         const RecordData &Record, unsigned &Idx);

  /// \brief Read a C++ base specifier.
  CXXBaseSpecifier ReadCXXBaseSpecifier(Module &F,
                                        const RecordData &Record,unsigned &Idx);

  /// \brief Read a CXXCtorInitializer array.
  std::pair<CXXCtorInitializer **, unsigned>
  ReadCXXCtorInitializers(Module &F, const RecordData &Record,
                          unsigned &Idx);

  /// \brief Read a source location from raw form.
  SourceLocation ReadSourceLocation(Module &Module, unsigned Raw) const {
    SourceLocation Loc = SourceLocation::getFromRawEncoding(Raw);
    assert(Module.SLocRemap.find(Loc.getOffset()) != Module.SLocRemap.end() &&
           "Cannot find offset to remap.");
    int Remap = Module.SLocRemap.find(Loc.getOffset())->second;
    return Loc.getLocWithOffset(Remap);
  }

  /// \brief Read a source location.
  SourceLocation ReadSourceLocation(Module &Module,
                                    const RecordData &Record, unsigned& Idx) {
    return ReadSourceLocation(Module, Record[Idx++]);
  }

  /// \brief Read a source range.
  SourceRange ReadSourceRange(Module &F,
                              const RecordData &Record, unsigned& Idx);

  /// \brief Read an integral value
  llvm::APInt ReadAPInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a signed integral value
  llvm::APSInt ReadAPSInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a floating-point value
  llvm::APFloat ReadAPFloat(const RecordData &Record, unsigned &Idx);

  // \brief Read a string
  std::string ReadString(const RecordData &Record, unsigned &Idx);

  /// \brief Read a version tuple.
  VersionTuple ReadVersionTuple(const RecordData &Record, unsigned &Idx);

  CXXTemporary *ReadCXXTemporary(Module &F, const RecordData &Record, 
                                 unsigned &Idx);
      
  /// \brief Reads attributes from the current stream position.
  void ReadAttributes(Module &F, AttrVec &Attrs,
                      const RecordData &Record, unsigned &Idx);

  /// \brief Reads a statement.
  Stmt *ReadStmt(Module &F);

  /// \brief Reads an expression.
  Expr *ReadExpr(Module &F);

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
  void ReadMacroRecord(Module &F, uint64_t Offset);

  /// \brief Reads the preprocessed entity located at the current stream
  /// position.
  PreprocessedEntity *LoadPreprocessedEntity(Module &F);
      
  /// \brief Determine the global preprocessed entity ID that corresponds to
  /// the given local ID within the given module.
  serialization::PreprocessedEntityID 
  getGlobalPreprocessedEntityID(Module &M, unsigned LocalID) const;
  
  /// \brief Note that the identifier is a macro whose record will be loaded
  /// from the given AST file at the given (file-local) offset.
  void SetIdentifierIsMacro(IdentifierInfo *II, Module &F,
                            uint64_t Offset);
      
  /// \brief Read the set of macros defined by this external macro source.
  virtual void ReadDefinedMacros();

  /// \brief Read the macro definition for this identifier.
  virtual void LoadMacroDefinition(IdentifierInfo *II);

  /// \brief Read the macro definition corresponding to this iterator
  /// into the unread macro record offsets table.
  void LoadMacroDefinition(
                     llvm::DenseMap<IdentifierInfo *, uint64_t>::iterator Pos);
  
  /// \brief Retrieve the AST context that this AST reader supplements.
  ASTContext &getContext() { return Context; }

  // \brief Contains declarations that were loaded before we have
  // access to a Sema object.
  SmallVector<NamedDecl *, 16> PreloadedDecls;

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

  void ClearSwitchCaseIDs();
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
