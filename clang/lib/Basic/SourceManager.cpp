//===--- SourceManager.cpp - Track and cache source files -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SourceManager interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Path.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
#include "llvm/Support/Streams.h"
#include <algorithm>
using namespace clang;
using namespace SrcMgr;
using llvm::MemoryBuffer;

//===--------------------------------------------------------------------===//
// SourceManager Helper Classes
//===--------------------------------------------------------------------===//

ContentCache::~ContentCache() {
  delete Buffer;
  delete [] SourceLineCache;
}

/// getSizeBytesMapped - Returns the number of bytes actually mapped for
///  this ContentCache.  This can be 0 if the MemBuffer was not actually
///  instantiated.
unsigned ContentCache::getSizeBytesMapped() const {
  return Buffer ? Buffer->getBufferSize() : 0;
}

/// getSize - Returns the size of the content encapsulated by this ContentCache.
///  This can be the size of the source file or the size of an arbitrary
///  scratch buffer.  If the ContentCache encapsulates a source file, that
///  file is not lazily brought in from disk to satisfy this query.
unsigned ContentCache::getSize() const {
  return Entry ? Entry->getSize() : Buffer->getBufferSize();
}

const llvm::MemoryBuffer *ContentCache::getBuffer() const {  
  // Lazily create the Buffer for ContentCaches that wrap files.
  if (!Buffer && Entry) {
    // FIXME: Should we support a way to not have to do this check over
    //   and over if we cannot open the file?
    Buffer = MemoryBuffer::getFile(Entry->getName(), 0, Entry->getSize());
  }
  return Buffer;
}

//===--------------------------------------------------------------------===//
// Line Table Implementation
//===--------------------------------------------------------------------===//

namespace clang {
/// LineTableInfo - This class is used to hold and unique data used to
/// represent #line information.
class LineTableInfo {
  /// FilenameIDs - This map is used to assign unique IDs to filenames in
  /// #line directives.  This allows us to unique the filenames that
  /// frequently reoccur and reference them with indices.  FilenameIDs holds
  /// the mapping from string -> ID, and FilenamesByID holds the mapping of ID
  /// to string.
  llvm::StringMap<unsigned, llvm::BumpPtrAllocator> FilenameIDs;
  std::vector<llvm::StringMapEntry<unsigned>*> FilenamesByID;
public:
  LineTableInfo() {
  }
  
  void clear() {
    FilenameIDs.clear();
    FilenamesByID.clear();
  }
  
  ~LineTableInfo() {}
  
  unsigned getLineTableFilenameID(const char *Ptr, unsigned Len);

};
} // namespace clang




unsigned LineTableInfo::getLineTableFilenameID(const char *Ptr, unsigned Len) {
  // Look up the filename in the string table, returning the pre-existing value
  // if it exists.
  llvm::StringMapEntry<unsigned> &Entry = 
    FilenameIDs.GetOrCreateValue(Ptr, Ptr+Len, ~0U);
  if (Entry.getValue() != ~0U)
    return Entry.getValue();
  
  // Otherwise, assign this the next available ID.
  Entry.setValue(FilenamesByID.size());
  FilenamesByID.push_back(&Entry);
  return FilenamesByID.size()-1;
}

/// getLineTableFilenameID - Return the uniqued ID for the specified filename.
/// 
unsigned SourceManager::getLineTableFilenameID(const char *Ptr, unsigned Len) {
  if (LineTable == 0)
    LineTable = new LineTableInfo();
  return LineTable->getLineTableFilenameID(Ptr, Len);
}


//===--------------------------------------------------------------------===//
// Private 'Create' methods.
//===--------------------------------------------------------------------===//

SourceManager::~SourceManager() {
  delete LineTable;
}

void SourceManager::clearIDTables() {
  MainFileID = FileID();
  SLocEntryTable.clear();
  LastLineNoFileIDQuery = FileID();
  LastLineNoContentCache = 0;
  LastFileIDLookup = FileID();
  
  if (LineTable)
    LineTable->clear();
  
  // Use up FileID #0 as an invalid instantiation.
  NextOffset = 0;
  createInstantiationLoc(SourceLocation(), SourceLocation(), 1);
}

/// getOrCreateContentCache - Create or return a cached ContentCache for the
/// specified file.
const ContentCache *
SourceManager::getOrCreateContentCache(const FileEntry *FileEnt) {
  assert(FileEnt && "Didn't specify a file entry to use?");
  
  // Do we already have information about this file?
  std::set<ContentCache>::iterator I = 
    FileInfos.lower_bound(ContentCache(FileEnt));
  
  if (I != FileInfos.end() && I->Entry == FileEnt)
    return &*I;
  
  // Nope, create a new Cache entry.
  ContentCache& Entry = const_cast<ContentCache&>(*FileInfos.insert(I,FileEnt));
  Entry.SourceLineCache = 0;
  Entry.NumLines = 0;
  return &Entry;
}


/// createMemBufferContentCache - Create a new ContentCache for the specified
///  memory buffer.  This does no caching.
const ContentCache*
SourceManager::createMemBufferContentCache(const MemoryBuffer *Buffer) {
  // Add a new ContentCache to the MemBufferInfos list and return it.  We
  // must default construct the object first that the instance actually
  // stored within MemBufferInfos actually owns the Buffer, and not any
  // temporary we would use in the call to "push_back".
  MemBufferInfos.push_back(ContentCache());
  ContentCache& Entry = const_cast<ContentCache&>(MemBufferInfos.back());
  Entry.setBuffer(Buffer);
  return &Entry;
}

//===----------------------------------------------------------------------===//
// Methods to create new FileID's and instantiations.
//===----------------------------------------------------------------------===//

/// createFileID - Create a new fileID for the specified ContentCache and
/// include position.  This works regardless of whether the ContentCache
/// corresponds to a file or some other input source.
FileID SourceManager::createFileID(const ContentCache *File,
                                   SourceLocation IncludePos,
                                   SrcMgr::CharacteristicKind FileCharacter) {
  SLocEntryTable.push_back(SLocEntry::get(NextOffset, 
                                          FileInfo::get(IncludePos, File,
                                                        FileCharacter)));
  unsigned FileSize = File->getSize();
  assert(NextOffset+FileSize+1 > NextOffset && "Ran out of source locations!");
  NextOffset += FileSize+1;
  
  // Set LastFileIDLookup to the newly created file.  The next getFileID call is
  // almost guaranteed to be from that file.
  return LastFileIDLookup = FileID::get(SLocEntryTable.size()-1);
}

/// createInstantiationLoc - Return a new SourceLocation that encodes the fact
/// that a token from SpellingLoc should actually be referenced from
/// InstantiationLoc.
SourceLocation SourceManager::createInstantiationLoc(SourceLocation SpellingLoc,
                                                     SourceLocation InstantLoc,
                                                     unsigned TokLength) {
  // The specified source location may be a mapped location, due to a macro
  // instantiation or #line directive.  Strip off this information to find out
  // where the characters are actually located.
  SpellingLoc = getSpellingLoc(SpellingLoc);
  
  // Resolve InstantLoc down to a real instantiation location.
  InstantLoc = getInstantiationLoc(InstantLoc);

  SLocEntryTable.push_back(SLocEntry::get(NextOffset, 
                                          InstantiationInfo::get(InstantLoc,
                                                                 SpellingLoc)));
  assert(NextOffset+TokLength+1 > NextOffset && "Ran out of source locations!");
  NextOffset += TokLength+1;
  return SourceLocation::getMacroLoc(NextOffset-(TokLength+1));
}

/// getBufferData - Return a pointer to the start and end of the source buffer
/// data for the specified FileID.
std::pair<const char*, const char*>
SourceManager::getBufferData(FileID FID) const {
  const llvm::MemoryBuffer *Buf = getBuffer(FID);
  return std::make_pair(Buf->getBufferStart(), Buf->getBufferEnd());
}


//===--------------------------------------------------------------------===//
// SourceLocation manipulation methods.
//===--------------------------------------------------------------------===//

/// getFileIDSlow - Return the FileID for a SourceLocation.  This is a very hot
/// method that is used for all SourceManager queries that start with a
/// SourceLocation object.  It is responsible for finding the entry in
/// SLocEntryTable which contains the specified location.
///
FileID SourceManager::getFileIDSlow(unsigned SLocOffset) const {
  assert(SLocOffset && "Invalid FileID");
  
  // After the first and second level caches, I see two common sorts of
  // behavior: 1) a lot of searched FileID's are "near" the cached file location
  // or are "near" the cached instantiation location.  2) others are just
  // completely random and may be a very long way away.
  //
  // To handle this, we do a linear search for up to 8 steps to catch #1 quickly
  // then we fall back to a less cache efficient, but more scalable, binary
  // search to find the location.
  
  // See if this is near the file point - worst case we start scanning from the
  // most newly created FileID.
  std::vector<SrcMgr::SLocEntry>::const_iterator I;
  
  if (SLocEntryTable[LastFileIDLookup.ID].getOffset() < SLocOffset) {
    // Neither loc prunes our search.
    I = SLocEntryTable.end();
  } else {
    // Perhaps it is near the file point.
    I = SLocEntryTable.begin()+LastFileIDLookup.ID;
  }

  // Find the FileID that contains this.  "I" is an iterator that points to a
  // FileID whose offset is known to be larger than SLocOffset.
  unsigned NumProbes = 0;
  while (1) {
    --I;
    if (I->getOffset() <= SLocOffset) {
#if 0
      printf("lin %d -> %d [%s] %d %d\n", SLocOffset,
             I-SLocEntryTable.begin(),
             I->isInstantiation() ? "inst" : "file",
             LastFileIDLookup.ID,  int(SLocEntryTable.end()-I));
#endif
      FileID Res = FileID::get(I-SLocEntryTable.begin());
      
      // If this isn't an instantiation, remember it.  We have good locality
      // across FileID lookups.
      if (!I->isInstantiation())
        LastFileIDLookup = Res;
      NumLinearScans += NumProbes+1;
      return Res;
    }
    if (++NumProbes == 8)
      break;
  }
  
  // Convert "I" back into an index.  We know that it is an entry whose index is
  // larger than the offset we are looking for.
  unsigned GreaterIndex = I-SLocEntryTable.begin();
  // LessIndex - This is the lower bound of the range that we're searching.
  // We know that the offset corresponding to the FileID is is less than
  // SLocOffset.
  unsigned LessIndex = 0;
  NumProbes = 0;
  while (1) {
    unsigned MiddleIndex = (GreaterIndex-LessIndex)/2+LessIndex;
    unsigned MidOffset = SLocEntryTable[MiddleIndex].getOffset();
    
    ++NumProbes;
    
    // If the offset of the midpoint is too large, chop the high side of the
    // range to the midpoint.
    if (MidOffset > SLocOffset) {
      GreaterIndex = MiddleIndex;
      continue;
    }
    
    // If the middle index contains the value, succeed and return.
    if (isOffsetInFileID(FileID::get(MiddleIndex), SLocOffset)) {
#if 0
      printf("bin %d -> %d [%s] %d %d\n", SLocOffset,
             I-SLocEntryTable.begin(),
             I->isInstantiation() ? "inst" : "file",
             LastFileIDLookup.ID, int(SLocEntryTable.end()-I));
#endif
      FileID Res = FileID::get(MiddleIndex);

      // If this isn't an instantiation, remember it.  We have good locality
      // across FileID lookups.
      if (!I->isInstantiation())
        LastFileIDLookup = Res;
      NumBinaryProbes += NumProbes;
      return Res;
    }
    
    // Otherwise, move the low-side up to the middle index.
    LessIndex = MiddleIndex;
  }
}

std::pair<FileID, unsigned>
SourceManager::getDecomposedInstantiationLocSlowCase(const SrcMgr::SLocEntry *E,
                                                     unsigned Offset) const {
  // If this is an instantiation record, walk through all the instantiation
  // points.
  FileID FID;
  SourceLocation Loc;
  do {
    Loc = E->getInstantiation().getInstantiationLoc();
    
    FID = getFileID(Loc);
    E = &getSLocEntry(FID);
    Offset += Loc.getOffset()-E->getOffset();
  } while (!Loc.isFileID());
  
  return std::make_pair(FID, Offset);
}

std::pair<FileID, unsigned>
SourceManager::getDecomposedSpellingLocSlowCase(const SrcMgr::SLocEntry *E,
                                                unsigned Offset) const {
  // If this is an instantiation record, walk through all the instantiation
  // points.
  FileID FID;
  SourceLocation Loc;
  do {
    Loc = E->getInstantiation().getSpellingLoc();
    
    FID = getFileID(Loc);
    E = &getSLocEntry(FID);
    Offset += Loc.getOffset()-E->getOffset();
  } while (!Loc.isFileID());
  
  return std::make_pair(FID, Offset);
}


//===----------------------------------------------------------------------===//
// Queries about the code at a SourceLocation.
//===----------------------------------------------------------------------===//

/// getCharacterData - Return a pointer to the start of the specified location
/// in the appropriate MemoryBuffer.
const char *SourceManager::getCharacterData(SourceLocation SL) const {
  // Note that this is a hot function in the getSpelling() path, which is
  // heavily used by -E mode.
  std::pair<FileID, unsigned> LocInfo = getDecomposedSpellingLoc(SL);
  
  // Note that calling 'getBuffer()' may lazily page in a source file.
  return getSLocEntry(LocInfo.first).getFile().getContentCache()
              ->getBuffer()->getBufferStart() + LocInfo.second;
}


/// getColumnNumber - Return the column # for the specified file position.
/// this is significantly cheaper to compute than the line number.  This returns
/// zero if the column number isn't known.
unsigned SourceManager::getColumnNumber(SourceLocation Loc) const {
  if (Loc.isInvalid()) return 0;
  assert(Loc.isFileID() && "Don't know what part of instantiation loc to get");
  
  std::pair<FileID, unsigned> LocInfo = getDecomposedLoc(Loc);
  unsigned FilePos = LocInfo.second;
  
  const char *Buf = getBuffer(LocInfo.first)->getBufferStart();

  unsigned LineStart = FilePos;
  while (LineStart && Buf[LineStart-1] != '\n' && Buf[LineStart-1] != '\r')
    --LineStart;
  return FilePos-LineStart+1;
}

static void ComputeLineNumbers(ContentCache* FI) DISABLE_INLINE;
static void ComputeLineNumbers(ContentCache* FI) {  
  // Note that calling 'getBuffer()' may lazily page in the file.
  const MemoryBuffer *Buffer = FI->getBuffer();
  
  // Find the file offsets of all of the *physical* source lines.  This does
  // not look at trigraphs, escaped newlines, or anything else tricky.
  std::vector<unsigned> LineOffsets;
  
  // Line #1 starts at char 0.
  LineOffsets.push_back(0);
  
  const unsigned char *Buf = (const unsigned char *)Buffer->getBufferStart();
  const unsigned char *End = (const unsigned char *)Buffer->getBufferEnd();
  unsigned Offs = 0;
  while (1) {
    // Skip over the contents of the line.
    // TODO: Vectorize this?  This is very performance sensitive for programs
    // with lots of diagnostics and in -E mode.
    const unsigned char *NextBuf = (const unsigned char *)Buf;
    while (*NextBuf != '\n' && *NextBuf != '\r' && *NextBuf != '\0')
      ++NextBuf;
    Offs += NextBuf-Buf;
    Buf = NextBuf;
    
    if (Buf[0] == '\n' || Buf[0] == '\r') {
      // If this is \n\r or \r\n, skip both characters.
      if ((Buf[1] == '\n' || Buf[1] == '\r') && Buf[0] != Buf[1])
        ++Offs, ++Buf;
      ++Offs, ++Buf;
      LineOffsets.push_back(Offs);
    } else {
      // Otherwise, this is a null.  If end of file, exit.
      if (Buf == End) break;
      // Otherwise, skip the null.
      ++Offs, ++Buf;
    }
  }
  
  // Copy the offsets into the FileInfo structure.
  FI->NumLines = LineOffsets.size();
  FI->SourceLineCache = new unsigned[LineOffsets.size()];
  std::copy(LineOffsets.begin(), LineOffsets.end(), FI->SourceLineCache);
}

/// getLineNumber - Given a SourceLocation, return the spelling line number
/// for the position indicated.  This requires building and caching a table of
/// line offsets for the MemoryBuffer, so this is not cheap: use only when
/// about to emit a diagnostic.
unsigned SourceManager::getLineNumber(SourceLocation Loc) const {
  if (Loc.isInvalid()) return 0;
  assert(Loc.isFileID() && "Don't know what part of instantiation loc to get");

  std::pair<FileID, unsigned> LocInfo = getDecomposedLoc(Loc);
  
  ContentCache *Content;
  if (LastLineNoFileIDQuery == LocInfo.first)
    Content = LastLineNoContentCache;
  else
    Content = const_cast<ContentCache*>(getSLocEntry(LocInfo.first)
                                        .getFile().getContentCache());
  
  // If this is the first use of line information for this buffer, compute the
  /// SourceLineCache for it on demand.
  if (Content->SourceLineCache == 0)
    ComputeLineNumbers(Content);

  // Okay, we know we have a line number table.  Do a binary search to find the
  // line number that this character position lands on.
  unsigned *SourceLineCache = Content->SourceLineCache;
  unsigned *SourceLineCacheStart = SourceLineCache;
  unsigned *SourceLineCacheEnd = SourceLineCache + Content->NumLines;
  
  unsigned QueriedFilePos = LocInfo.second+1;

  // If the previous query was to the same file, we know both the file pos from
  // that query and the line number returned.  This allows us to narrow the
  // search space from the entire file to something near the match.
  if (LastLineNoFileIDQuery == LocInfo.first) {
    if (QueriedFilePos >= LastLineNoFilePos) {
      SourceLineCache = SourceLineCache+LastLineNoResult-1;
      
      // The query is likely to be nearby the previous one.  Here we check to
      // see if it is within 5, 10 or 20 lines.  It can be far away in cases
      // where big comment blocks and vertical whitespace eat up lines but
      // contribute no tokens.
      if (SourceLineCache+5 < SourceLineCacheEnd) {
        if (SourceLineCache[5] > QueriedFilePos)
          SourceLineCacheEnd = SourceLineCache+5;
        else if (SourceLineCache+10 < SourceLineCacheEnd) {
          if (SourceLineCache[10] > QueriedFilePos)
            SourceLineCacheEnd = SourceLineCache+10;
          else if (SourceLineCache+20 < SourceLineCacheEnd) {
            if (SourceLineCache[20] > QueriedFilePos)
              SourceLineCacheEnd = SourceLineCache+20;
          }
        }
      }
    } else {
      SourceLineCacheEnd = SourceLineCache+LastLineNoResult+1;
    }
  }
  
  // If the spread is large, do a "radix" test as our initial guess, based on
  // the assumption that lines average to approximately the same length.
  // NOTE: This is currently disabled, as it does not appear to be profitable in
  // initial measurements.
  if (0 && SourceLineCacheEnd-SourceLineCache > 20) {
    unsigned FileLen = Content->SourceLineCache[Content->NumLines-1];
    
    // Take a stab at guessing where it is.
    unsigned ApproxPos = Content->NumLines*QueriedFilePos / FileLen;
    
    // Check for -10 and +10 lines.
    unsigned LowerBound = std::max(int(ApproxPos-10), 0);
    unsigned UpperBound = std::min(ApproxPos+10, FileLen);

    // If the computed lower bound is less than the query location, move it in.
    if (SourceLineCache < SourceLineCacheStart+LowerBound &&
        SourceLineCacheStart[LowerBound] < QueriedFilePos)
      SourceLineCache = SourceLineCacheStart+LowerBound;
    
    // If the computed upper bound is greater than the query location, move it.
    if (SourceLineCacheEnd > SourceLineCacheStart+UpperBound &&
        SourceLineCacheStart[UpperBound] >= QueriedFilePos)
      SourceLineCacheEnd = SourceLineCacheStart+UpperBound;
  }
  
  unsigned *Pos
    = std::lower_bound(SourceLineCache, SourceLineCacheEnd, QueriedFilePos);
  unsigned LineNo = Pos-SourceLineCacheStart;
  
  LastLineNoFileIDQuery = LocInfo.first;
  LastLineNoContentCache = Content;
  LastLineNoFilePos = QueriedFilePos;
  LastLineNoResult = LineNo;
  return LineNo;
}

/// getSourceName - This method returns the name of the file or buffer that
/// the SourceLocation specifies.  This can be modified with #line directives,
/// etc.
const char *SourceManager::getSourceName(SourceLocation Loc) const {
  if (Loc.isInvalid()) return "";
  
  const SrcMgr::ContentCache *C =
  getSLocEntry(getFileID(getSpellingLoc(Loc))).getFile().getContentCache();
  
  // To get the source name, first consult the FileEntry (if one exists) before
  // the MemBuffer as this will avoid unnecessarily paging in the MemBuffer.
  return C->Entry ? C->Entry->getName() : C->getBuffer()->getBufferIdentifier();
}

//===----------------------------------------------------------------------===//
// Other miscellaneous methods.
//===----------------------------------------------------------------------===//


/// PrintStats - Print statistics to stderr.
///
void SourceManager::PrintStats() const {
  llvm::cerr << "\n*** Source Manager Stats:\n";
  llvm::cerr << FileInfos.size() << " files mapped, " << MemBufferInfos.size()
             << " mem buffers mapped, " << SLocEntryTable.size() 
             << " SLocEntry's allocated.\n";
    
  unsigned NumLineNumsComputed = 0;
  unsigned NumFileBytesMapped = 0;
  for (std::set<ContentCache>::const_iterator I = 
       FileInfos.begin(), E = FileInfos.end(); I != E; ++I) {
    NumLineNumsComputed += I->SourceLineCache != 0;
    NumFileBytesMapped  += I->getSizeBytesMapped();
  }
  
  llvm::cerr << NumFileBytesMapped << " bytes of files mapped, "
             << NumLineNumsComputed << " files with line #'s computed.\n";
  llvm::cerr << "FileID scans: " << NumLinearScans << " linear, "
             << NumBinaryProbes << " binary.\n";
}

//===----------------------------------------------------------------------===//
// Serialization.
//===----------------------------------------------------------------------===//
  
void ContentCache::Emit(llvm::Serializer& S) const {
  S.FlushRecord();
  S.EmitPtr(this);

  if (Entry) {
    llvm::sys::Path Fname(Buffer->getBufferIdentifier());

    if (Fname.isAbsolute())
      S.EmitCStr(Fname.c_str());
    else {
      // Create an absolute path.
      // FIXME: This will potentially contain ".." and "." in the path.
      llvm::sys::Path path = llvm::sys::Path::GetCurrentDirectory();
      path.appendComponent(Fname.c_str());      
      S.EmitCStr(path.c_str());
    }
  }
  else {
    const char* p = Buffer->getBufferStart();
    const char* e = Buffer->getBufferEnd();
    
    S.EmitInt(e-p);
    
    for ( ; p != e; ++p)
      S.EmitInt(*p);    
  }
  
  S.FlushRecord();  
}

void ContentCache::ReadToSourceManager(llvm::Deserializer& D,
                                       SourceManager& SMgr,
                                       FileManager* FMgr,
                                       std::vector<char>& Buf) {
  if (FMgr) {
    llvm::SerializedPtrID PtrID = D.ReadPtrID();    
    D.ReadCStr(Buf,false);
    
    // Create/fetch the FileEntry.
    const char* start = &Buf[0];
    const FileEntry* E = FMgr->getFile(start,start+Buf.size());
    
    // FIXME: Ideally we want a lazy materialization of the ContentCache
    //  anyway, because we don't want to read in source files unless this
    //  is absolutely needed.
    if (!E)
      D.RegisterPtr(PtrID,NULL);
    else
      // Get the ContextCache object and register it with the deserializer.
      D.RegisterPtr(PtrID, SMgr.getOrCreateContentCache(E));
    return;
  }
  
  // Register the ContextCache object with the deserializer.
  SMgr.MemBufferInfos.push_back(ContentCache());
  ContentCache& Entry = const_cast<ContentCache&>(SMgr.MemBufferInfos.back());
  D.RegisterPtr(&Entry);
  
  // Create the buffer.
  unsigned Size = D.ReadInt();
  Entry.Buffer = MemoryBuffer::getNewUninitMemBuffer(Size);
  
  // Read the contents of the buffer.
  char* p = const_cast<char*>(Entry.Buffer->getBufferStart());
  for (unsigned i = 0; i < Size ; ++i)
    p[i] = D.ReadInt();    
}

void SourceManager::Emit(llvm::Serializer& S) const {
  S.EnterBlock();
  S.EmitPtr(this);
  S.EmitInt(MainFileID.getOpaqueValue());
  
  // Emit: FileInfos.  Just emit the file name.
  S.EnterBlock();    

  std::for_each(FileInfos.begin(),FileInfos.end(),
                S.MakeEmitter<ContentCache>());
  
  S.ExitBlock();
  
  // Emit: MemBufferInfos
  S.EnterBlock();

  std::for_each(MemBufferInfos.begin(), MemBufferInfos.end(),
                S.MakeEmitter<ContentCache>());
  
  S.ExitBlock();
  
  // FIXME: Emit SLocEntryTable.
  
  S.ExitBlock();
}

SourceManager*
SourceManager::CreateAndRegister(llvm::Deserializer& D, FileManager& FMgr){
  SourceManager *M = new SourceManager();
  D.RegisterPtr(M);
  
  // Read: the FileID of the main source file of the translation unit.
  M->MainFileID = FileID::get(D.ReadInt());
  
  std::vector<char> Buf;
    
  { // Read: FileInfos.
    llvm::Deserializer::Location BLoc = D.getCurrentBlockLocation();
    while (!D.FinishedBlock(BLoc))
    ContentCache::ReadToSourceManager(D,*M,&FMgr,Buf);
  }
    
  { // Read: MemBufferInfos.
    llvm::Deserializer::Location BLoc = D.getCurrentBlockLocation();
    while (!D.FinishedBlock(BLoc))
    ContentCache::ReadToSourceManager(D,*M,NULL,Buf);
  }
  
  // FIXME: Read SLocEntryTable.
  
  return M;
}
