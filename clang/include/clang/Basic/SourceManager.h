//===--- SourceManager.h - Track and cache source files ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SourceManager interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SOURCEMANAGER_H
#define LLVM_CLANG_SOURCEMANAGER_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include <map>
#include <vector>
#include <cassert>

namespace clang {

class Diagnostic;
class SourceManager;
class FileManager;
class FileEntry;
class LineTableInfo;
class LangOptions;
class ASTWriter;
class ASTReader;

/// There are three different types of locations in a file: a spelling
/// location, an expansion location, and a presumed location.
///
/// Given an example of:
/// #define min(x, y) x < y ? x : y
///
/// and then later on a use of min:
/// #line 17
/// return min(a, b);
///
/// The expansion location is the line in the source code where the macro
/// was expanded (the return statement), the spelling location is the
/// location in the source where the macro was originally defined,
/// and the presumed location is where the line directive states that
/// the line is 17, or any other line.

/// SrcMgr - Public enums and private classes that are part of the
/// SourceManager implementation.
///
namespace SrcMgr {
  /// CharacteristicKind - This is used to represent whether a file or directory
  /// holds normal user code, system code, or system code which is implicitly
  /// 'extern "C"' in C++ mode.  Entire directories can be tagged with this
  /// (this is maintained by DirectoryLookup and friends) as can specific
  /// FileInfos when a #pragma system_header is seen or various other cases.
  ///
  enum CharacteristicKind {
    C_User, C_System, C_ExternCSystem
  };

  /// ContentCache - One instance of this struct is kept for every file
  /// loaded or used.  This object owns the MemoryBuffer object.
  class ContentCache {
    enum CCFlags {
      /// \brief Whether the buffer is invalid.
      InvalidFlag = 0x01,
      /// \brief Whether the buffer should not be freed on destruction.
      DoNotFreeFlag = 0x02
    };

    /// Buffer - The actual buffer containing the characters from the input
    /// file.  This is owned by the ContentCache object.
    /// The bits indicate indicates whether the buffer is invalid.
    mutable llvm::PointerIntPair<const llvm::MemoryBuffer *, 2> Buffer;

  public:
    /// Reference to the file entry representing this ContentCache.
    /// This reference does not own the FileEntry object.
    /// It is possible for this to be NULL if
    /// the ContentCache encapsulates an imaginary text buffer.
    const FileEntry *OrigEntry;

    /// \brief References the file which the contents were actually loaded from.
    /// Can be different from 'Entry' if we overridden the contents of one file
    /// with the contents of another file.
    const FileEntry *ContentsEntry;

    /// SourceLineCache - A bump pointer allocated array of offsets for each
    /// source line.  This is lazily computed.  This is owned by the
    /// SourceManager BumpPointerAllocator object.
    unsigned *SourceLineCache;

    /// NumLines - The number of lines in this ContentCache.  This is only valid
    /// if SourceLineCache is non-null.
    unsigned NumLines;

    /// \brief Lazily computed map of macro argument chunks to their expanded
    /// source location.
    typedef std::map<unsigned, SourceLocation> MacroArgsMap;
    MacroArgsMap *MacroArgsCache;

    /// getBuffer - Returns the memory buffer for the associated content.
    ///
    /// \param Diag Object through which diagnostics will be emitted if the
    /// buffer cannot be retrieved.
    ///
    /// \param Loc If specified, is the location that invalid file diagnostics
    ///     will be emitted at.
    ///
    /// \param Invalid If non-NULL, will be set \c true if an error occurred.
    const llvm::MemoryBuffer *getBuffer(Diagnostic &Diag,
                                        const SourceManager &SM,
                                        SourceLocation Loc = SourceLocation(),
                                        bool *Invalid = 0) const;

    /// getSize - Returns the size of the content encapsulated by this
    ///  ContentCache. This can be the size of the source file or the size of an
    ///  arbitrary scratch buffer.  If the ContentCache encapsulates a source
    ///  file this size is retrieved from the file's FileEntry.
    unsigned getSize() const;

    /// getSizeBytesMapped - Returns the number of bytes actually mapped for
    /// this ContentCache. This can be 0 if the MemBuffer was not actually
    /// expanded.
    unsigned getSizeBytesMapped() const;

    /// Returns the kind of memory used to back the memory buffer for
    /// this content cache.  This is used for performance analysis.
    llvm::MemoryBuffer::BufferKind getMemoryBufferKind() const;

    void setBuffer(const llvm::MemoryBuffer *B) {
      assert(!Buffer.getPointer() && "MemoryBuffer already set.");
      Buffer.setPointer(B);
      Buffer.setInt(false);
    }

    /// \brief Get the underlying buffer, returning NULL if the buffer is not
    /// yet available.
    const llvm::MemoryBuffer *getRawBuffer() const {
      return Buffer.getPointer();
    }

    /// \brief Replace the existing buffer (which will be deleted)
    /// with the given buffer.
    void replaceBuffer(const llvm::MemoryBuffer *B, bool DoNotFree = false);

    /// \brief Determine whether the buffer itself is invalid.
    bool isBufferInvalid() const {
      return Buffer.getInt() & InvalidFlag;
    }

    /// \brief Determine whether the buffer should be freed.
    bool shouldFreeBuffer() const {
      return (Buffer.getInt() & DoNotFreeFlag) == 0;
    }

    ContentCache(const FileEntry *Ent = 0)
      : Buffer(0, false), OrigEntry(Ent), ContentsEntry(Ent),
        SourceLineCache(0), NumLines(0), MacroArgsCache(0) {}

    ContentCache(const FileEntry *Ent, const FileEntry *contentEnt)
      : Buffer(0, false), OrigEntry(Ent), ContentsEntry(contentEnt),
        SourceLineCache(0), NumLines(0), MacroArgsCache(0) {}

    ~ContentCache();

    /// The copy ctor does not allow copies where source object has either
    ///  a non-NULL Buffer or SourceLineCache.  Ownership of allocated memory
    ///  is not transferred, so this is a logical error.
    ContentCache(const ContentCache &RHS)
      : Buffer(0, false), SourceLineCache(0), MacroArgsCache(0)
    {
      OrigEntry = RHS.OrigEntry;
      ContentsEntry = RHS.ContentsEntry;

      assert (RHS.Buffer.getPointer() == 0 && RHS.SourceLineCache == 0 &&
              RHS.MacroArgsCache == 0
              && "Passed ContentCache object cannot own a buffer.");

      NumLines = RHS.NumLines;
    }

  private:
    // Disable assignments.
    ContentCache &operator=(const ContentCache& RHS);
  };

  /// FileInfo - Information about a FileID, basically just the logical file
  /// that it represents and include stack information.
  ///
  /// Each FileInfo has include stack information, indicating where it came
  /// from. This information encodes the #include chain that a token was
  /// expanded from. The main include file has an invalid IncludeLoc.
  ///
  /// FileInfos contain a "ContentCache *", with the contents of the file.
  ///
  class FileInfo {
    /// IncludeLoc - The location of the #include that brought in this file.
    /// This is an invalid SLOC for the main file (top of the #include chain).
    unsigned IncludeLoc;  // Really a SourceLocation

    /// \brief Number of FileIDs (files and macros) that were created during
    /// preprocessing of this #include, including this SLocEntry.
    /// Zero means the preprocessor didn't provide such info for this SLocEntry.
    unsigned NumCreatedFIDs;

    /// Data - This contains the ContentCache* and the bits indicating the
    /// characteristic of the file and whether it has #line info, all bitmangled
    /// together.
    uintptr_t Data;

    friend class clang::SourceManager;
    friend class clang::ASTWriter;
    friend class clang::ASTReader;
  public:
    /// get - Return a FileInfo object.
    static FileInfo get(SourceLocation IL, const ContentCache *Con,
                        CharacteristicKind FileCharacter) {
      FileInfo X;
      X.IncludeLoc = IL.getRawEncoding();
      X.NumCreatedFIDs = 0;
      X.Data = (uintptr_t)Con;
      assert((X.Data & 7) == 0 &&"ContentCache pointer insufficiently aligned");
      assert((unsigned)FileCharacter < 4 && "invalid file character");
      X.Data |= (unsigned)FileCharacter;
      return X;
    }

    SourceLocation getIncludeLoc() const {
      return SourceLocation::getFromRawEncoding(IncludeLoc);
    }
    const ContentCache* getContentCache() const {
      return reinterpret_cast<const ContentCache*>(Data & ~7UL);
    }

    /// getCharacteristic - Return whether this is a system header or not.
    CharacteristicKind getFileCharacteristic() const {
      return (CharacteristicKind)(Data & 3);
    }

    /// hasLineDirectives - Return true if this FileID has #line directives in
    /// it.
    bool hasLineDirectives() const { return (Data & 4) != 0; }

    /// setHasLineDirectives - Set the flag that indicates that this FileID has
    /// line table entries associated with it.
    void setHasLineDirectives() {
      Data |= 4;
    }
  };

  /// ExpansionInfo - Each ExpansionInfo encodes the expansion location - where
  /// the token was ultimately expanded, and the SpellingLoc - where the actual
  /// character data for the token came from.
  class ExpansionInfo {
    // Really these are all SourceLocations.

    /// SpellingLoc - Where the spelling for the token can be found.
    unsigned SpellingLoc;

    /// ExpansionLocStart/ExpansionLocEnd - In a macro expansion, these
    /// indicate the start and end of the expansion. In object-like macros,
    /// these will be the same. In a function-like macro expansion, the start
    /// will be the identifier and the end will be the ')'. Finally, in
    /// macro-argument instantitions, the end will be 'SourceLocation()', an
    /// invalid location.
    unsigned ExpansionLocStart, ExpansionLocEnd;

  public:
    SourceLocation getSpellingLoc() const {
      return SourceLocation::getFromRawEncoding(SpellingLoc);
    }
    SourceLocation getExpansionLocStart() const {
      return SourceLocation::getFromRawEncoding(ExpansionLocStart);
    }
    SourceLocation getExpansionLocEnd() const {
      SourceLocation EndLoc =
        SourceLocation::getFromRawEncoding(ExpansionLocEnd);
      return EndLoc.isInvalid() ? getExpansionLocStart() : EndLoc;
    }

    std::pair<SourceLocation,SourceLocation> getExpansionLocRange() const {
      return std::make_pair(getExpansionLocStart(), getExpansionLocEnd());
    }

    bool isMacroArgExpansion() const {
      // Note that this needs to return false for default constructed objects.
      return getExpansionLocStart().isValid() &&
        SourceLocation::getFromRawEncoding(ExpansionLocEnd).isInvalid();
    }

    /// create - Return a ExpansionInfo for an expansion. Start and End specify
    /// the expansion range (where the macro is expanded), and SpellingLoc
    /// specifies the spelling location (where the characters from the token
    /// come from). All three can refer to normal File SLocs or expansion
    /// locations.
    static ExpansionInfo create(SourceLocation SpellingLoc,
                                SourceLocation Start, SourceLocation End) {
      ExpansionInfo X;
      X.SpellingLoc = SpellingLoc.getRawEncoding();
      X.ExpansionLocStart = Start.getRawEncoding();
      X.ExpansionLocEnd = End.getRawEncoding();
      return X;
    }

    /// createForMacroArg - Return a special ExpansionInfo for the expansion of
    /// a macro argument into a function-like macro's body. ExpansionLoc
    /// specifies the expansion location (where the macro is expanded). This
    /// doesn't need to be a range because a macro is always expanded at
    /// a macro parameter reference, and macro parameters are always exactly
    /// one token. SpellingLoc specifies the spelling location (where the
    /// characters from the token come from). ExpansionLoc and SpellingLoc can
    /// both refer to normal File SLocs or expansion locations.
    ///
    /// Given the code:
    /// \code
    ///   #define F(x) f(x)
    ///   F(42);
    /// \endcode
    ///
    /// When expanding '\c F(42)', the '\c x' would call this with an
    /// SpellingLoc pointing at '\c 42' anad an ExpansionLoc pointing at its
    /// location in the definition of '\c F'.
    static ExpansionInfo createForMacroArg(SourceLocation SpellingLoc,
                                           SourceLocation ExpansionLoc) {
      // We store an intentionally invalid source location for the end of the
      // expansion range to mark that this is a macro argument ion rather than
      // a normal one.
      return create(SpellingLoc, ExpansionLoc, SourceLocation());
    }
  };

  /// SLocEntry - This is a discriminated union of FileInfo and
  /// ExpansionInfo.  SourceManager keeps an array of these objects, and
  /// they are uniquely identified by the FileID datatype.
  class SLocEntry {
    unsigned Offset;   // low bit is set for expansion info.
    union {
      FileInfo File;
      ExpansionInfo Expansion;
    };
  public:
    unsigned getOffset() const { return Offset >> 1; }

    bool isExpansion() const { return Offset & 1; }
    bool isFile() const { return !isExpansion(); }

    const FileInfo &getFile() const {
      assert(isFile() && "Not a file SLocEntry!");
      return File;
    }

    const ExpansionInfo &getExpansion() const {
      assert(isExpansion() && "Not a macro expansion SLocEntry!");
      return Expansion;
    }

    static SLocEntry get(unsigned Offset, const FileInfo &FI) {
      SLocEntry E;
      E.Offset = Offset << 1;
      E.File = FI;
      return E;
    }

    static SLocEntry get(unsigned Offset, const ExpansionInfo &Expansion) {
      SLocEntry E;
      E.Offset = (Offset << 1) | 1;
      E.Expansion = Expansion;
      return E;
    }
  };
}  // end SrcMgr namespace.

/// \brief External source of source location entries.
class ExternalSLocEntrySource {
public:
  virtual ~ExternalSLocEntrySource();

  /// \brief Read the source location entry with index ID, which will always be
  /// less than -1.
  ///
  /// \returns true if an error occurred that prevented the source-location
  /// entry from being loaded.
  virtual bool ReadSLocEntry(int ID) = 0;
};


/// IsBeforeInTranslationUnitCache - This class holds the cache used by
/// isBeforeInTranslationUnit.  The cache structure is complex enough to be
/// worth breaking out of SourceManager.
class IsBeforeInTranslationUnitCache {
  /// L/R QueryFID - These are the FID's of the cached query.  If these match up
  /// with a subsequent query, the result can be reused.
  FileID LQueryFID, RQueryFID;

  /// \brief True if LQueryFID was created before RQueryFID. This is used
  /// to compare macro expansion locations.
  bool IsLQFIDBeforeRQFID;

  /// CommonFID - This is the file found in common between the two #include
  /// traces.  It is the nearest common ancestor of the #include tree.
  FileID CommonFID;

  /// L/R CommonOffset - This is the offset of the previous query in CommonFID.
  /// Usually, this represents the location of the #include for QueryFID, but if
  /// LQueryFID is a parent of RQueryFID (or vise versa) then these can be a
  /// random token in the parent.
  unsigned LCommonOffset, RCommonOffset;
public:

  /// isCacheValid - Return true if the currently cached values match up with
  /// the specified LHS/RHS query.  If not, we can't use the cache.
  bool isCacheValid(FileID LHS, FileID RHS) const {
    return LQueryFID == LHS && RQueryFID == RHS;
  }

  /// getCachedResult - If the cache is valid, compute the result given the
  /// specified offsets in the LHS/RHS FID's.
  bool getCachedResult(unsigned LOffset, unsigned ROffset) const {
    // If one of the query files is the common file, use the offset.  Otherwise,
    // use the #include loc in the common file.
    if (LQueryFID != CommonFID) LOffset = LCommonOffset;
    if (RQueryFID != CommonFID) ROffset = RCommonOffset;

    // It is common for multiple macro expansions to be "included" from the same
    // location (expansion location), in which case use the order of the FileIDs
    // to determine which came first. This will also take care the case where
    // one of the locations points at the inclusion/expansion point of the other
    // in which case its FileID will come before the other.
    if (LOffset == ROffset &&
        (LQueryFID != CommonFID || RQueryFID != CommonFID))
      return IsLQFIDBeforeRQFID;

    return LOffset < ROffset;
  }

  // Set up a new query.
  void setQueryFIDs(FileID LHS, FileID RHS, bool isLFIDBeforeRFID) {
    assert(LHS != RHS);
    LQueryFID = LHS;
    RQueryFID = RHS;
    IsLQFIDBeforeRQFID = isLFIDBeforeRFID;
  }

  void clear() {
    LQueryFID = RQueryFID = FileID();
    IsLQFIDBeforeRQFID = false;
  }

  void setCommonLoc(FileID commonFID, unsigned lCommonOffset,
                    unsigned rCommonOffset) {
    CommonFID = commonFID;
    LCommonOffset = lCommonOffset;
    RCommonOffset = rCommonOffset;
  }

};

/// \brief This class handles loading and caching of source files into memory.
///
/// This object owns the MemoryBuffer objects for all of the loaded
/// files and assigns unique FileID's for each unique #include chain.
///
/// The SourceManager can be queried for information about SourceLocation
/// objects, turning them into either spelling or expansion locations. Spelling
/// locations represent where the bytes corresponding to a token came from and
/// expansion locations represent where the location is in the user's view. In
/// the case of a macro expansion, for example, the spelling location indicates
/// where the expanded token came from and the expansion location specifies
/// where it was expanded.
class SourceManager : public llvm::RefCountedBase<SourceManager> {
  /// \brief Diagnostic object.
  Diagnostic &Diag;

  FileManager &FileMgr;

  mutable llvm::BumpPtrAllocator ContentCacheAlloc;

  /// FileInfos - Memoized information about all of the files tracked by this
  /// SourceManager.  This set allows us to merge ContentCache entries based
  /// on their FileEntry*.  All ContentCache objects will thus have unique,
  /// non-null, FileEntry pointers.
  llvm::DenseMap<const FileEntry*, SrcMgr::ContentCache*> FileInfos;

  /// \brief True if the ContentCache for files that are overriden by other
  /// files, should report the original file name. Defaults to true.
  bool OverridenFilesKeepOriginalName;

  /// \brief Files that have been overriden with the contents from another file.
  llvm::DenseMap<const FileEntry *, const FileEntry *> OverriddenFiles;

  /// MemBufferInfos - Information about various memory buffers that we have
  /// read in.  All FileEntry* within the stored ContentCache objects are NULL,
  /// as they do not refer to a file.
  std::vector<SrcMgr::ContentCache*> MemBufferInfos;

  /// \brief The table of SLocEntries that are local to this module.
  ///
  /// Positive FileIDs are indexes into this table. Entry 0 indicates an invalid
  /// expansion.
  std::vector<SrcMgr::SLocEntry> LocalSLocEntryTable;

  /// \brief The table of SLocEntries that are loaded from other modules.
  ///
  /// Negative FileIDs are indexes into this table. To get from ID to an index,
  /// use (-ID - 2).
  std::vector<SrcMgr::SLocEntry> LoadedSLocEntryTable;

  /// \brief The starting offset of the next local SLocEntry.
  ///
  /// This is LocalSLocEntryTable.back().Offset + the size of that entry.
  unsigned NextLocalOffset;

  /// \brief The starting offset of the latest batch of loaded SLocEntries.
  ///
  /// This is LoadedSLocEntryTable.back().Offset, except that that entry might
  /// not have been loaded, so that value would be unknown.
  unsigned CurrentLoadedOffset;

  /// \brief The highest possible offset is 2^31-1, so CurrentLoadedOffset
  /// starts at 2^31.
  static const unsigned MaxLoadedOffset = 1U << 31U;

  /// \brief A bitmap that indicates whether the entries of LoadedSLocEntryTable
  /// have already been loaded from the external source.
  ///
  /// Same indexing as LoadedSLocEntryTable.
  std::vector<bool> SLocEntryLoaded;

  /// \brief An external source for source location entries.
  ExternalSLocEntrySource *ExternalSLocEntries;

  /// LastFileIDLookup - This is a one-entry cache to speed up getFileID.
  /// LastFileIDLookup records the last FileID looked up or created, because it
  /// is very common to look up many tokens from the same file.
  mutable FileID LastFileIDLookup;

  /// LineTable - This holds information for #line directives.  It is referenced
  /// by indices from SLocEntryTable.
  LineTableInfo *LineTable;

  /// LastLineNo - These ivars serve as a cache used in the getLineNumber
  /// method which is used to speedup getLineNumber calls to nearby locations.
  mutable FileID LastLineNoFileIDQuery;
  mutable SrcMgr::ContentCache *LastLineNoContentCache;
  mutable unsigned LastLineNoFilePos;
  mutable unsigned LastLineNoResult;

  /// MainFileID - The file ID for the main source file of the translation unit.
  FileID MainFileID;

  // Statistics for -print-stats.
  mutable unsigned NumLinearScans, NumBinaryProbes;

  // Cache results for the isBeforeInTranslationUnit method.
  mutable IsBeforeInTranslationUnitCache IsBeforeInTUCache;

  // Cache for the "fake" buffer used for error-recovery purposes.
  mutable llvm::MemoryBuffer *FakeBufferForRecovery;

  // SourceManager doesn't support copy construction.
  explicit SourceManager(const SourceManager&);
  void operator=(const SourceManager&);
public:
  SourceManager(Diagnostic &Diag, FileManager &FileMgr);
  ~SourceManager();

  void clearIDTables();

  Diagnostic &getDiagnostics() const { return Diag; }

  FileManager &getFileManager() const { return FileMgr; }

  /// \brief Set true if the SourceManager should report the original file name
  /// for contents of files that were overriden by other files.Defaults to true.
  void setOverridenFilesKeepOriginalName(bool value) {
    OverridenFilesKeepOriginalName = value;
  }

  /// createMainFileIDForMembuffer - Create the FileID for a memory buffer
  ///  that will represent the FileID for the main source.  One example
  ///  of when this would be used is when the main source is read from STDIN.
  FileID createMainFileIDForMemBuffer(const llvm::MemoryBuffer *Buffer) {
    assert(MainFileID.isInvalid() && "MainFileID already set!");
    MainFileID = createFileIDForMemBuffer(Buffer);
    return MainFileID;
  }

  //===--------------------------------------------------------------------===//
  // MainFileID creation and querying methods.
  //===--------------------------------------------------------------------===//

  /// getMainFileID - Returns the FileID of the main source file.
  FileID getMainFileID() const { return MainFileID; }

  /// createMainFileID - Create the FileID for the main source file.
  FileID createMainFileID(const FileEntry *SourceFile) {
    assert(MainFileID.isInvalid() && "MainFileID already set!");
    MainFileID = createFileID(SourceFile, SourceLocation(), SrcMgr::C_User);
    return MainFileID;
  }

  /// \brief Set the file ID for the precompiled preamble, which is also the
  /// main file.
  void SetPreambleFileID(FileID Preamble) {
    assert(MainFileID.isInvalid() && "MainFileID already set!");
    MainFileID = Preamble;
  }

  //===--------------------------------------------------------------------===//
  // Methods to create new FileID's and macro expansions.
  //===--------------------------------------------------------------------===//

  /// createFileID - Create a new FileID that represents the specified file
  /// being #included from the specified IncludePosition.  This translates NULL
  /// into standard input.
  FileID createFileID(const FileEntry *SourceFile, SourceLocation IncludePos,
                      SrcMgr::CharacteristicKind FileCharacter,
                      int LoadedID = 0, unsigned LoadedOffset = 0) {
    const SrcMgr::ContentCache *IR = getOrCreateContentCache(SourceFile);
    assert(IR && "getOrCreateContentCache() cannot return NULL");
    return createFileID(IR, IncludePos, FileCharacter, LoadedID, LoadedOffset);
  }

  /// createFileIDForMemBuffer - Create a new FileID that represents the
  /// specified memory buffer.  This does no caching of the buffer and takes
  /// ownership of the MemoryBuffer, so only pass a MemoryBuffer to this once.
  FileID createFileIDForMemBuffer(const llvm::MemoryBuffer *Buffer,
                                  int LoadedID = 0, unsigned LoadedOffset = 0) {
    return createFileID(createMemBufferContentCache(Buffer), SourceLocation(),
                        SrcMgr::C_User, LoadedID, LoadedOffset);
  }

  /// createMacroArgExpansionLoc - Return a new SourceLocation that encodes the
  /// fact that a token from SpellingLoc should actually be referenced from
  /// ExpansionLoc, and that it represents the expansion of a macro argument
  /// into the function-like macro body.
  SourceLocation createMacroArgExpansionLoc(SourceLocation Loc,
                                            SourceLocation ExpansionLoc,
                                            unsigned TokLength);

  /// createExpansionLoc - Return a new SourceLocation that encodes the fact
  /// that a token from SpellingLoc should actually be referenced from
  /// ExpansionLoc.
  SourceLocation createExpansionLoc(SourceLocation Loc,
                                    SourceLocation ExpansionLocStart,
                                    SourceLocation ExpansionLocEnd,
                                    unsigned TokLength,
                                    int LoadedID = 0,
                                    unsigned LoadedOffset = 0);

  /// \brief Retrieve the memory buffer associated with the given file.
  ///
  /// \param Invalid If non-NULL, will be set \c true if an error
  /// occurs while retrieving the memory buffer.
  const llvm::MemoryBuffer *getMemoryBufferForFile(const FileEntry *File,
                                                   bool *Invalid = 0);

  /// \brief Override the contents of the given source file by providing an
  /// already-allocated buffer.
  ///
  /// \param SourceFile the source file whose contents will be overriden.
  ///
  /// \param Buffer the memory buffer whose contents will be used as the
  /// data in the given source file.
  ///
  /// \param DoNotFree If true, then the buffer will not be freed when the
  /// source manager is destroyed.
  void overrideFileContents(const FileEntry *SourceFile,
                            const llvm::MemoryBuffer *Buffer,
                            bool DoNotFree = false);

  /// \brief Override the the given source file with another one.
  ///
  /// \param SourceFile the source file which will be overriden.
  ///
  /// \param NewFile the file whose contents will be used as the
  /// data instead of the contents of the given source file.
  void overrideFileContents(const FileEntry *SourceFile,
                            const FileEntry *NewFile);

  //===--------------------------------------------------------------------===//
  // FileID manipulation methods.
  //===--------------------------------------------------------------------===//

  /// getBuffer - Return the buffer for the specified FileID. If there is an
  /// error opening this buffer the first time, this manufactures a temporary
  /// buffer and returns a non-empty error string.
  const llvm::MemoryBuffer *getBuffer(FileID FID, SourceLocation Loc,
                                      bool *Invalid = 0) const {
    bool MyInvalid = false;
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &MyInvalid);
    if (MyInvalid || !Entry.isFile()) {
      if (Invalid)
        *Invalid = true;

      return getFakeBufferForRecovery();
    }

    return Entry.getFile().getContentCache()->getBuffer(Diag, *this, Loc,
                                                        Invalid);
  }

  const llvm::MemoryBuffer *getBuffer(FileID FID, bool *Invalid = 0) const {
    bool MyInvalid = false;
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &MyInvalid);
    if (MyInvalid || !Entry.isFile()) {
      if (Invalid)
        *Invalid = true;

      return getFakeBufferForRecovery();
    }

    return Entry.getFile().getContentCache()->getBuffer(Diag, *this,
                                                        SourceLocation(),
                                                        Invalid);
  }

  /// getFileEntryForID - Returns the FileEntry record for the provided FileID.
  const FileEntry *getFileEntryForID(FileID FID) const {
    bool MyInvalid = false;
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &MyInvalid);
    if (MyInvalid || !Entry.isFile())
      return 0;

    return Entry.getFile().getContentCache()->OrigEntry;
  }

  /// Returns the FileEntry record for the provided SLocEntry.
  const FileEntry *getFileEntryForSLocEntry(const SrcMgr::SLocEntry &sloc) const
  {
    return sloc.getFile().getContentCache()->OrigEntry;
  }

  /// getBufferData - Return a StringRef to the source buffer data for the
  /// specified FileID.
  ///
  /// \param FID The file ID whose contents will be returned.
  /// \param Invalid If non-NULL, will be set true if an error occurred.
  StringRef getBufferData(FileID FID, bool *Invalid = 0) const;

  /// \brief Get the number of FileIDs (files and macros) that were created
  /// during preprocessing of \arg FID, including it.
  unsigned getNumCreatedFIDsForFileID(FileID FID) const {
    bool Invalid = false;
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
    if (Invalid || !Entry.isFile())
      return 0;

    return Entry.getFile().NumCreatedFIDs;
  }

  /// \brief Set the number of FileIDs (files and macros) that were created
  /// during preprocessing of \arg FID, including it.
  void setNumCreatedFIDsForFileID(FileID FID, unsigned NumFIDs) const {
    bool Invalid = false;
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
    if (Invalid || !Entry.isFile())
      return;

    assert(Entry.getFile().NumCreatedFIDs == 0 && "Already set!");
    const_cast<SrcMgr::FileInfo &>(Entry.getFile()).NumCreatedFIDs = NumFIDs;
  }

  //===--------------------------------------------------------------------===//
  // SourceLocation manipulation methods.
  //===--------------------------------------------------------------------===//

  /// getFileID - Return the FileID for a SourceLocation.  This is a very
  /// hot method that is used for all SourceManager queries that start with a
  /// SourceLocation object.  It is responsible for finding the entry in
  /// SLocEntryTable which contains the specified location.
  ///
  FileID getFileID(SourceLocation SpellingLoc) const {
    unsigned SLocOffset = SpellingLoc.getOffset();

    // If our one-entry cache covers this offset, just return it.
    if (isOffsetInFileID(LastFileIDLookup, SLocOffset))
      return LastFileIDLookup;

    return getFileIDSlow(SLocOffset);
  }

  /// getLocForStartOfFile - Return the source location corresponding to the
  /// first byte of the specified file.
  SourceLocation getLocForStartOfFile(FileID FID) const {
    bool Invalid = false;
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
    if (Invalid || !Entry.isFile())
      return SourceLocation();

    unsigned FileOffset = Entry.getOffset();
    return SourceLocation::getFileLoc(FileOffset);
  }

  /// \brief Returns the include location if \arg FID is a #include'd file
  /// otherwise it returns an invalid location.
  SourceLocation getIncludeLoc(FileID FID) const {
    bool Invalid = false;
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
    if (Invalid || !Entry.isFile())
      return SourceLocation();

    return Entry.getFile().getIncludeLoc();
  }

  /// getExpansionLoc - Given a SourceLocation object, return the expansion
  /// location referenced by the ID.
  SourceLocation getExpansionLoc(SourceLocation Loc) const {
    // Handle the non-mapped case inline, defer to out of line code to handle
    // expansions.
    if (Loc.isFileID()) return Loc;
    return getExpansionLocSlowCase(Loc);
  }

  /// getImmediateExpansionRange - Loc is required to be an expansion location.
  /// Return the start/end of the expansion information.
  std::pair<SourceLocation,SourceLocation>
  getImmediateExpansionRange(SourceLocation Loc) const;

  /// getExpansionRange - Given a SourceLocation object, return the range of
  /// tokens covered by the expansion the ultimate file.
  std::pair<SourceLocation,SourceLocation>
  getExpansionRange(SourceLocation Loc) const;


  /// getSpellingLoc - Given a SourceLocation object, return the spelling
  /// location referenced by the ID.  This is the place where the characters
  /// that make up the lexed token can be found.
  SourceLocation getSpellingLoc(SourceLocation Loc) const {
    // Handle the non-mapped case inline, defer to out of line code to handle
    // expansions.
    if (Loc.isFileID()) return Loc;
    return getSpellingLocSlowCase(Loc);
  }

  /// getImmediateSpellingLoc - Given a SourceLocation object, return the
  /// spelling location referenced by the ID.  This is the first level down
  /// towards the place where the characters that make up the lexed token can be
  /// found.  This should not generally be used by clients.
  SourceLocation getImmediateSpellingLoc(SourceLocation Loc) const;

  /// getDecomposedLoc - Decompose the specified location into a raw FileID +
  /// Offset pair.  The first element is the FileID, the second is the
  /// offset from the start of the buffer of the location.
  std::pair<FileID, unsigned> getDecomposedLoc(SourceLocation Loc) const {
    FileID FID = getFileID(Loc);
    return std::make_pair(FID, Loc.getOffset()-getSLocEntry(FID).getOffset());
  }

  /// getDecomposedExpansionLoc - Decompose the specified location into a raw
  /// FileID + Offset pair. If the location is an expansion record, walk
  /// through it until we find the final location expanded.
  std::pair<FileID, unsigned>
  getDecomposedExpansionLoc(SourceLocation Loc) const {
    FileID FID = getFileID(Loc);
    const SrcMgr::SLocEntry *E = &getSLocEntry(FID);

    unsigned Offset = Loc.getOffset()-E->getOffset();
    if (Loc.isFileID())
      return std::make_pair(FID, Offset);

    return getDecomposedExpansionLocSlowCase(E);
  }

  /// getDecomposedSpellingLoc - Decompose the specified location into a raw
  /// FileID + Offset pair.  If the location is an expansion record, walk
  /// through it until we find its spelling record.
  std::pair<FileID, unsigned>
  getDecomposedSpellingLoc(SourceLocation Loc) const {
    FileID FID = getFileID(Loc);
    const SrcMgr::SLocEntry *E = &getSLocEntry(FID);

    unsigned Offset = Loc.getOffset()-E->getOffset();
    if (Loc.isFileID())
      return std::make_pair(FID, Offset);
    return getDecomposedSpellingLocSlowCase(E, Offset);
  }

  /// getFileOffset - This method returns the offset from the start
  /// of the file that the specified SourceLocation represents. This is not very
  /// meaningful for a macro ID.
  unsigned getFileOffset(SourceLocation SpellingLoc) const {
    return getDecomposedLoc(SpellingLoc).second;
  }

  /// isMacroArgExpansion - This method tests whether the given source location
  /// represents a macro argument's expansion into the function-like macro
  /// definition. Such source locations only appear inside of the expansion
  /// locations representing where a particular function-like macro was
  /// expanded.
  bool isMacroArgExpansion(SourceLocation Loc) const;

  /// \brief Returns true if \arg Loc is inside the [\arg Start, +\arg Length)
  /// chunk of the source location address space.
  /// If it's true and \arg RelativeOffset is non-null, it will be set to the
  /// relative offset of \arg Loc inside the chunk.
  bool isInSLocAddrSpace(SourceLocation Loc,
                         SourceLocation Start, unsigned Length,
                         unsigned *RelativeOffset = 0) const {
    assert(((Start.getOffset() < NextLocalOffset &&
               Start.getOffset()+Length <= NextLocalOffset) ||
            (Start.getOffset() >= CurrentLoadedOffset &&
                Start.getOffset()+Length < MaxLoadedOffset)) &&
           "Chunk is not valid SLoc address space");
    unsigned LocOffs = Loc.getOffset();
    unsigned BeginOffs = Start.getOffset();
    unsigned EndOffs = BeginOffs + Length;
    if (LocOffs >= BeginOffs && LocOffs < EndOffs) {
      if (RelativeOffset)
        *RelativeOffset = LocOffs - BeginOffs;
      return true;
    }

    return false;
  }

  /// \brief Return true if both \arg LHS and \arg RHS are in the local source
  /// location address space or the loaded one. If it's true and
  /// \arg RelativeOffset is non-null, it will be set to the offset of \arg RHS
  /// relative to \arg LHS.
  bool isInSameSLocAddrSpace(SourceLocation LHS, SourceLocation RHS,
                             int *RelativeOffset) const {
    unsigned LHSOffs = LHS.getOffset(), RHSOffs = RHS.getOffset();
    bool LHSLoaded = LHSOffs >= CurrentLoadedOffset;
    bool RHSLoaded = RHSOffs >= CurrentLoadedOffset;

    if (LHSLoaded == RHSLoaded) {
      if (RelativeOffset)
        *RelativeOffset = RHSOffs - LHSOffs;
      return true;
    }

    return false;
  }

  //===--------------------------------------------------------------------===//
  // Queries about the code at a SourceLocation.
  //===--------------------------------------------------------------------===//

  /// getCharacterData - Return a pointer to the start of the specified location
  /// in the appropriate spelling MemoryBuffer.
  ///
  /// \param Invalid If non-NULL, will be set \c true if an error occurs.
  const char *getCharacterData(SourceLocation SL, bool *Invalid = 0) const;

  /// getColumnNumber - Return the column # for the specified file position.
  /// This is significantly cheaper to compute than the line number.  This
  /// returns zero if the column number isn't known.  This may only be called
  /// on a file sloc, so you must choose a spelling or expansion location
  /// before calling this method.
  unsigned getColumnNumber(FileID FID, unsigned FilePos,
                           bool *Invalid = 0) const;
  unsigned getSpellingColumnNumber(SourceLocation Loc, bool *Invalid = 0) const;
  unsigned getExpansionColumnNumber(SourceLocation Loc,
                                    bool *Invalid = 0) const;
  unsigned getPresumedColumnNumber(SourceLocation Loc, bool *Invalid = 0) const;


  /// getLineNumber - Given a SourceLocation, return the spelling line number
  /// for the position indicated.  This requires building and caching a table of
  /// line offsets for the MemoryBuffer, so this is not cheap: use only when
  /// about to emit a diagnostic.
  unsigned getLineNumber(FileID FID, unsigned FilePos, bool *Invalid = 0) const;
  unsigned getSpellingLineNumber(SourceLocation Loc, bool *Invalid = 0) const;
  unsigned getExpansionLineNumber(SourceLocation Loc, bool *Invalid = 0) const;
  unsigned getPresumedLineNumber(SourceLocation Loc, bool *Invalid = 0) const;

  /// Return the filename or buffer identifier of the buffer the location is in.
  /// Note that this name does not respect #line directives.  Use getPresumedLoc
  /// for normal clients.
  const char *getBufferName(SourceLocation Loc, bool *Invalid = 0) const;

  /// getFileCharacteristic - return the file characteristic of the specified
  /// source location, indicating whether this is a normal file, a system
  /// header, or an "implicit extern C" system header.
  ///
  /// This state can be modified with flags on GNU linemarker directives like:
  ///   # 4 "foo.h" 3
  /// which changes all source locations in the current file after that to be
  /// considered to be from a system header.
  SrcMgr::CharacteristicKind getFileCharacteristic(SourceLocation Loc) const;

  /// getPresumedLoc - This method returns the "presumed" location of a
  /// SourceLocation specifies.  A "presumed location" can be modified by #line
  /// or GNU line marker directives.  This provides a view on the data that a
  /// user should see in diagnostics, for example.
  ///
  /// Note that a presumed location is always given as the expansion point of
  /// an expansion location, not at the spelling location.
  ///
  /// \returns The presumed location of the specified SourceLocation. If the
  /// presumed location cannot be calculate (e.g., because \p Loc is invalid
  /// or the file containing \p Loc has changed on disk), returns an invalid
  /// presumed location.
  PresumedLoc getPresumedLoc(SourceLocation Loc) const;

  /// isFromSameFile - Returns true if both SourceLocations correspond to
  ///  the same file.
  bool isFromSameFile(SourceLocation Loc1, SourceLocation Loc2) const {
    return getFileID(Loc1) == getFileID(Loc2);
  }

  /// isFromMainFile - Returns true if the file of provided SourceLocation is
  ///   the main file.
  bool isFromMainFile(SourceLocation Loc) const {
    return getFileID(Loc) == getMainFileID();
  }

  /// isInSystemHeader - Returns if a SourceLocation is in a system header.
  bool isInSystemHeader(SourceLocation Loc) const {
    return getFileCharacteristic(Loc) != SrcMgr::C_User;
  }

  /// isInExternCSystemHeader - Returns if a SourceLocation is in an "extern C"
  /// system header.
  bool isInExternCSystemHeader(SourceLocation Loc) const {
    return getFileCharacteristic(Loc) == SrcMgr::C_ExternCSystem;
  }

  /// \brief The size of the SLocEnty that \arg FID represents.
  unsigned getFileIDSize(FileID FID) const;

  /// \brief Given a specific FileID, returns true if \arg Loc is inside that
  /// FileID chunk and sets relative offset (offset of \arg Loc from beginning
  /// of FileID) to \arg relativeOffset.
  bool isInFileID(SourceLocation Loc, FileID FID,
                  unsigned *RelativeOffset = 0) const {
    unsigned Offs = Loc.getOffset();
    if (isOffsetInFileID(FID, Offs)) {
      if (RelativeOffset)
        *RelativeOffset = Offs - getSLocEntry(FID).getOffset();
      return true;
    }

    return false;
  }

  //===--------------------------------------------------------------------===//
  // Line Table Manipulation Routines
  //===--------------------------------------------------------------------===//

  /// getLineTableFilenameID - Return the uniqued ID for the specified filename.
  ///
  unsigned getLineTableFilenameID(StringRef Str);

  /// AddLineNote - Add a line note to the line table for the FileID and offset
  /// specified by Loc.  If FilenameID is -1, it is considered to be
  /// unspecified.
  void AddLineNote(SourceLocation Loc, unsigned LineNo, int FilenameID);
  void AddLineNote(SourceLocation Loc, unsigned LineNo, int FilenameID,
                   bool IsFileEntry, bool IsFileExit,
                   bool IsSystemHeader, bool IsExternCHeader);

  /// \brief Determine if the source manager has a line table.
  bool hasLineTable() const { return LineTable != 0; }

  /// \brief Retrieve the stored line table.
  LineTableInfo &getLineTable();

  //===--------------------------------------------------------------------===//
  // Queries for performance analysis.
  //===--------------------------------------------------------------------===//

  /// Return the total amount of physical memory allocated by the
  /// ContentCache allocator.
  size_t getContentCacheSize() const {
    return ContentCacheAlloc.getTotalMemory();
  }

  struct MemoryBufferSizes {
    const size_t malloc_bytes;
    const size_t mmap_bytes;

    MemoryBufferSizes(size_t malloc_bytes, size_t mmap_bytes)
      : malloc_bytes(malloc_bytes), mmap_bytes(mmap_bytes) {}
  };

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  MemoryBufferSizes getMemoryBufferSizes() const;

  // Return the amount of memory used for various side tables and
  // data structures in the SourceManager.
  size_t getDataStructureSizes() const;

  //===--------------------------------------------------------------------===//
  // Other miscellaneous methods.
  //===--------------------------------------------------------------------===//

  /// \brief Get the source location for the given file:line:col triplet.
  ///
  /// If the source file is included multiple times, the source location will
  /// be based upon the first inclusion.
  ///
  /// If the location points inside a function macro argument, the returned
  /// location will be the macro location in which the argument was expanded.
  /// \sa getMacroArgExpandedLocation
  SourceLocation getLocation(const FileEntry *SourceFile,
                             unsigned Line, unsigned Col) {
    SourceLocation Loc = translateFileLineCol(SourceFile, Line, Col);
    return getMacroArgExpandedLocation(Loc);
  }

  /// \brief Get the source location for the given file:line:col triplet.
  ///
  /// If the source file is included multiple times, the source location will
  /// be based upon the first inclusion.
  SourceLocation translateFileLineCol(const FileEntry *SourceFile,
                                      unsigned Line, unsigned Col);

  /// \brief If \arg Loc points inside a function macro argument, the returned
  /// location will be the macro location in which the argument was expanded.
  /// If a macro argument is used multiple times, the expanded location will
  /// be at the first expansion of the argument.
  /// e.g.
  ///   MY_MACRO(foo);
  ///             ^
  /// Passing a file location pointing at 'foo', will yield a macro location
  /// where 'foo' was expanded into.
  SourceLocation getMacroArgExpandedLocation(SourceLocation Loc);

  /// \brief Determines the order of 2 source locations in the translation unit.
  ///
  /// \returns true if LHS source location comes before RHS, false otherwise.
  bool isBeforeInTranslationUnit(SourceLocation LHS, SourceLocation RHS) const;

  /// \brief Comparison function class.
  class LocBeforeThanCompare : public std::binary_function<SourceLocation,
                                                         SourceLocation, bool> {
    SourceManager &SM;

  public:
    explicit LocBeforeThanCompare(SourceManager &SM) : SM(SM) { }

    bool operator()(SourceLocation LHS, SourceLocation RHS) const {
      return SM.isBeforeInTranslationUnit(LHS, RHS);
    }
  };

  /// \brief Determines the order of 2 source locations in the "source location
  /// address space".
  bool isBeforeInSLocAddrSpace(SourceLocation LHS, SourceLocation RHS) const {
    return isBeforeInSLocAddrSpace(LHS, RHS.getOffset());
  }

  /// \brief Determines the order of a source location and a source location
  /// offset in the "source location address space".
  ///
  /// Note that we always consider source locations loaded from
  bool isBeforeInSLocAddrSpace(SourceLocation LHS, unsigned RHS) const {
    unsigned LHSOffset = LHS.getOffset();
    bool LHSLoaded = LHSOffset >= CurrentLoadedOffset;
    bool RHSLoaded = RHS >= CurrentLoadedOffset;
    if (LHSLoaded == RHSLoaded)
      return LHSOffset < RHS;

    return LHSLoaded;
  }

  // Iterators over FileInfos.
  typedef llvm::DenseMap<const FileEntry*, SrcMgr::ContentCache*>
      ::const_iterator fileinfo_iterator;
  fileinfo_iterator fileinfo_begin() const { return FileInfos.begin(); }
  fileinfo_iterator fileinfo_end() const { return FileInfos.end(); }
  bool hasFileInfo(const FileEntry *File) const {
    return FileInfos.find(File) != FileInfos.end();
  }

  /// PrintStats - Print statistics to stderr.
  ///
  void PrintStats() const;

  /// \brief Get the number of local SLocEntries we have.
  unsigned local_sloc_entry_size() const { return LocalSLocEntryTable.size(); }

  /// \brief Get a local SLocEntry. This is exposed for indexing.
  const SrcMgr::SLocEntry &getLocalSLocEntry(unsigned Index,
                                             bool *Invalid = 0) const {
    assert(Index < LocalSLocEntryTable.size() && "Invalid index");
    return LocalSLocEntryTable[Index];
  }

  /// \brief Get the number of loaded SLocEntries we have.
  unsigned loaded_sloc_entry_size() const { return LoadedSLocEntryTable.size();}

  /// \brief Get a loaded SLocEntry. This is exposed for indexing.
  const SrcMgr::SLocEntry &getLoadedSLocEntry(unsigned Index, bool *Invalid=0) const {
    assert(Index < LoadedSLocEntryTable.size() && "Invalid index");
    if (!SLocEntryLoaded[Index])
      ExternalSLocEntries->ReadSLocEntry(-(static_cast<int>(Index) + 2));
    return LoadedSLocEntryTable[Index];
  }

  const SrcMgr::SLocEntry &getSLocEntry(FileID FID, bool *Invalid = 0) const {
    return getSLocEntryByID(FID.ID);
  }

  unsigned getNextLocalOffset() const { return NextLocalOffset; }

  void setExternalSLocEntrySource(ExternalSLocEntrySource *Source) {
    assert(LoadedSLocEntryTable.empty() &&
           "Invalidating existing loaded entries");
    ExternalSLocEntries = Source;
  }

  /// \brief Allocate a number of loaded SLocEntries, which will be actually
  /// loaded on demand from the external source.
  ///
  /// NumSLocEntries will be allocated, which occupy a total of TotalSize space
  /// in the global source view. The lowest ID and the base offset of the
  /// entries will be returned.
  std::pair<int, unsigned>
  AllocateLoadedSLocEntries(unsigned NumSLocEntries, unsigned TotalSize);

  /// \brief Returns true if \arg Loc came from a PCH/Module.
  bool isLoadedSourceLocation(SourceLocation Loc) const {
    return Loc.getOffset() >= CurrentLoadedOffset;
  }

  /// \brief Returns true if \arg Loc did not come from a PCH/Module.
  bool isLocalSourceLocation(SourceLocation Loc) const {
    return Loc.getOffset() < NextLocalOffset;
  }

private:
  const llvm::MemoryBuffer *getFakeBufferForRecovery() const;

  /// \brief Get the entry with the given unwrapped FileID.
  const SrcMgr::SLocEntry &getSLocEntryByID(int ID) const {
    assert(ID != -1 && "Using FileID sentinel value");
    if (ID < 0)
      return getLoadedSLocEntryByID(ID);
    return getLocalSLocEntry(static_cast<unsigned>(ID));
  }

  const SrcMgr::SLocEntry &getLoadedSLocEntryByID(int ID) const {
    return getLoadedSLocEntry(static_cast<unsigned>(-ID - 2));
  }

  /// createExpansionLoc - Implements the common elements of storing an
  /// expansion info struct into the SLocEntry table and producing a source
  /// location that refers to it.
  SourceLocation createExpansionLocImpl(const SrcMgr::ExpansionInfo &Expansion,
                                        unsigned TokLength,
                                        int LoadedID = 0,
                                        unsigned LoadedOffset = 0);

  /// isOffsetInFileID - Return true if the specified FileID contains the
  /// specified SourceLocation offset.  This is a very hot method.
  inline bool isOffsetInFileID(FileID FID, unsigned SLocOffset) const {
    const SrcMgr::SLocEntry &Entry = getSLocEntry(FID);
    // If the entry is after the offset, it can't contain it.
    if (SLocOffset < Entry.getOffset()) return false;

    // If this is the very last entry then it does.
    if (FID.ID == -2)
      return true;

    // If it is the last local entry, then it does if the location is local.
    if (static_cast<unsigned>(FID.ID+1) == LocalSLocEntryTable.size()) {
      return SLocOffset < NextLocalOffset;
    }

    // Otherwise, the entry after it has to not include it. This works for both
    // local and loaded entries.
    return SLocOffset < getSLocEntry(FileID::get(FID.ID+1)).getOffset();
  }

  /// createFileID - Create a new fileID for the specified ContentCache and
  ///  include position.  This works regardless of whether the ContentCache
  ///  corresponds to a file or some other input source.
  FileID createFileID(const SrcMgr::ContentCache* File,
                      SourceLocation IncludePos,
                      SrcMgr::CharacteristicKind DirCharacter,
                      int LoadedID, unsigned LoadedOffset);

  const SrcMgr::ContentCache *
    getOrCreateContentCache(const FileEntry *SourceFile);

  /// createMemBufferContentCache - Create a new ContentCache for the specified
  ///  memory buffer.
  const SrcMgr::ContentCache*
  createMemBufferContentCache(const llvm::MemoryBuffer *Buf);

  FileID getFileIDSlow(unsigned SLocOffset) const;
  FileID getFileIDLocal(unsigned SLocOffset) const;
  FileID getFileIDLoaded(unsigned SLocOffset) const;

  SourceLocation getExpansionLocSlowCase(SourceLocation Loc) const;
  SourceLocation getSpellingLocSlowCase(SourceLocation Loc) const;

  std::pair<FileID, unsigned>
  getDecomposedExpansionLocSlowCase(const SrcMgr::SLocEntry *E) const;
  std::pair<FileID, unsigned>
  getDecomposedSpellingLocSlowCase(const SrcMgr::SLocEntry *E,
                                   unsigned Offset) const;
  void computeMacroArgsCache(SrcMgr::ContentCache *Content, FileID FID);

  friend class ASTReader;
  friend class ASTWriter;
};


}  // end namespace clang

#endif
