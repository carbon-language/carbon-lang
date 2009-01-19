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

// This (temporary) directive toggles between lazy and eager creation of
// MemBuffers.  This directive is not permanent, and is here to test a few
// potential optimizations in PTH.  Once it is clear whether eager or lazy
// creation of MemBuffers is better this directive will get removed.
#define LAZY

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

const llvm::MemoryBuffer* ContentCache::getBuffer() const {  
#ifdef LAZY
  // Lazily create the Buffer for ContentCaches that wrap files.
  if (!Buffer && Entry) {
    // FIXME: Should we support a way to not have to do this check over
    //   and over if we cannot open the file?
    Buffer = MemoryBuffer::getFile(Entry->getName(), 0, Entry->getSize());
  }
#endif
  return Buffer;
}


/// getFileInfo - Create or return a cached FileInfo for the specified file.
///
const ContentCache* SourceManager::getContentCache(const FileEntry *FileEnt) {

  assert(FileEnt && "Didn't specify a file entry to use?");
  // Do we already have information about this file?
  std::set<ContentCache>::iterator I = 
    FileInfos.lower_bound(ContentCache(FileEnt));
  
  if (I != FileInfos.end() && I->Entry == FileEnt)
    return &*I;
  
  // Nope, get information.
#ifndef LAZY
  const MemoryBuffer *File =
    MemoryBuffer::getFile(FileEnt->getName(), 0, FileEnt->getSize());
  if (File == 0)
    return 0;
#endif
  
  ContentCache& Entry = const_cast<ContentCache&>(*FileInfos.insert(I,FileEnt));
#ifndef LAZY
  Entry.setBuffer(File);
#endif
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


/// createFileID - Create a new fileID for the specified ContentCache and
/// include position.  This works regardless of whether the ContentCache
/// corresponds to a file or some other input source.
FileID SourceManager::createFileID(const ContentCache *File,
                                     SourceLocation IncludePos,
                                     SrcMgr::CharacteristicKind FileCharacter) {
  // If FileEnt is really large (e.g. it's a large .i file), we may not be able
  // to fit an arbitrary position in the file in the FilePos field.  To handle
  // this, we create one FileID for each chunk of the file that fits in a
  // FilePos field.
  unsigned FileSize = File->getSize();
  if (FileSize+1 < (1 << SourceLocation::FilePosBits)) {
    FileIDs.push_back(FileIDInfo::get(IncludePos, 0, File, FileCharacter));
    assert(FileIDs.size() < (1 << SourceLocation::ChunkIDBits) &&
           "Ran out of file ID's!");
    return FileID::Create(FileIDs.size());
  }
  
  // Create one FileID for each chunk of the file.
  unsigned Result = FileIDs.size()+1;

  unsigned ChunkNo = 0;
  while (1) {
    FileIDs.push_back(FileIDInfo::get(IncludePos, ChunkNo++, File,
                                      FileCharacter));

    if (FileSize+1 < (1 << SourceLocation::FilePosBits)) break;
    FileSize -= (1 << SourceLocation::FilePosBits);
  }

  assert(FileIDs.size() < (1 << SourceLocation::ChunkIDBits) &&
         "Ran out of file ID's!");
  return FileID::Create(Result);
}

/// getInstantiationLoc - Return a new SourceLocation that encodes the fact
/// that a token from SpellingLoc should actually be referenced from
/// InstantiationLoc.
SourceLocation SourceManager::getInstantiationLoc(SourceLocation SpellingLoc,
                                                  SourceLocation InstantLoc) {
  // The specified source location may be a mapped location, due to a macro
  // instantiation or #line directive.  Strip off this information to find out
  // where the characters are actually located.
  SpellingLoc = getSpellingLoc(SpellingLoc);
  
  // Resolve InstantLoc down to a real instantiation location.
  InstantLoc = getInstantiationLoc(InstantLoc);
  
  
  // If the last macro id is close to the currently requested location, try to
  // reuse it.  This implements a small cache.
  for (int i = MacroIDs.size()-1, e = MacroIDs.size()-6; i >= 0 && i != e; --i){
    MacroIDInfo &LastOne = MacroIDs[i];
    
    // The instanitation point and source SpellingLoc have to exactly match to
    // reuse (for now).  We could allow "nearby" instantiations in the future.
    if (LastOne.getInstantiationLoc() != InstantLoc ||
        LastOne.getSpellingLoc().getChunkID() != SpellingLoc.getChunkID())
      continue;
  
    // Check to see if the spellloc of the token came from near enough to reuse.
    int SpellDelta = SpellingLoc.getRawFilePos() -
                     LastOne.getSpellingLoc().getRawFilePos();
    if (SourceLocation::isValidMacroSpellingOffs(SpellDelta))
      return SourceLocation::getMacroLoc(i, SpellDelta);
  }
  
 
  MacroIDs.push_back(MacroIDInfo::get(InstantLoc, SpellingLoc));
  return SourceLocation::getMacroLoc(MacroIDs.size()-1, 0);
}

/// getBufferData - Return a pointer to the start and end of the character
/// data for the specified location.
std::pair<const char*, const char*> 
SourceManager::getBufferData(SourceLocation Loc) const {
  const llvm::MemoryBuffer *Buf = getBuffer(getCanonicalFileID(Loc));
  return std::make_pair(Buf->getBufferStart(), Buf->getBufferEnd());
}

std::pair<const char*, const char*>
SourceManager::getBufferData(FileID FID) const {
  const llvm::MemoryBuffer *Buf = getBuffer(FID);
  return std::make_pair(Buf->getBufferStart(), Buf->getBufferEnd());
}



/// getCharacterData - Return a pointer to the start of the specified location
/// in the appropriate MemoryBuffer.
const char *SourceManager::getCharacterData(SourceLocation SL) const {
  // Note that this is a hot function in the getSpelling() path, which is
  // heavily used by -E mode.
  SL = getSpellingLoc(SL);
  
  std::pair<FileID, unsigned> LocInfo = getDecomposedFileLoc(SL);
  
  // Note that calling 'getBuffer()' may lazily page in a source file.
  return getContentCache(LocInfo.first)->getBuffer()->getBufferStart() + 
         LocInfo.second;
}


/// getColumnNumber - Return the column # for the specified file position.
/// this is significantly cheaper to compute than the line number.  This returns
/// zero if the column number isn't known.
unsigned SourceManager::getColumnNumber(SourceLocation Loc) const {
  if (Loc.getChunkID() == 0) return 0;
  
  std::pair<FileID, unsigned> LocInfo = getDecomposedFileLoc(Loc);
  unsigned FilePos = LocInfo.second;
  
  const char *Buf = getBuffer(LocInfo.first)->getBufferStart();

  unsigned LineStart = FilePos;
  while (LineStart && Buf[LineStart-1] != '\n' && Buf[LineStart-1] != '\r')
    --LineStart;
  return FilePos-LineStart+1;
}

/// getSourceName - This method returns the name of the file or buffer that
/// the SourceLocation specifies.  This can be modified with #line directives,
/// etc.
const char *SourceManager::getSourceName(SourceLocation Loc) const {
  if (Loc.getChunkID() == 0) return "";
  
  // To get the source name, first consult the FileEntry (if one exists) before
  // the MemBuffer as this will avoid unnecessarily paging in the MemBuffer.
  const SrcMgr::ContentCache *C = getContentCacheForLoc(Loc);
  return C->Entry ? C->Entry->getName() : C->getBuffer()->getBufferIdentifier();
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
  if (Loc.getChunkID() == 0) return 0;

  ContentCache *Content;
  
  std::pair<FileID, unsigned> LocInfo = getDecomposedFileLoc(Loc);
  
  if (LastLineNoFileIDQuery == LocInfo.first)
    Content = LastLineNoContentCache;
  else
    Content = const_cast<ContentCache*>(getContentCache(LocInfo.first));
  
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

/// PrintStats - Print statistics to stderr.
///
void SourceManager::PrintStats() const {
  llvm::cerr << "\n*** Source Manager Stats:\n";
  llvm::cerr << FileInfos.size() << " files mapped, " << MemBufferInfos.size()
             << " mem buffers mapped, " << FileIDs.size() 
             << " file ID's allocated.\n";
  llvm::cerr << "  " << FileIDs.size() << " normal buffer FileID's, "
             << MacroIDs.size() << " macro expansion FileID's.\n";
    
  unsigned NumLineNumsComputed = 0;
  unsigned NumFileBytesMapped = 0;
  for (std::set<ContentCache>::const_iterator I = 
       FileInfos.begin(), E = FileInfos.end(); I != E; ++I) {
    NumLineNumsComputed += I->SourceLineCache != 0;
    NumFileBytesMapped  += I->getSizeBytesMapped();
  }
  
  llvm::cerr << NumFileBytesMapped << " bytes of files mapped, "
             << NumLineNumsComputed << " files with line #'s computed.\n";
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
      D.RegisterPtr(PtrID,SMgr.getContentCache(E));
  }
  else {
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
}

void FileIDInfo::Emit(llvm::Serializer& S) const {
  S.Emit(IncludeLoc);
  S.EmitInt(ChunkNo);
  S.EmitPtr(Content);  
}

FileIDInfo FileIDInfo::ReadVal(llvm::Deserializer& D) {
  FileIDInfo I;
  I.IncludeLoc = SourceLocation::ReadVal(D);
  I.ChunkNo = D.ReadInt();
  D.ReadPtr(I.Content,false);
  return I;
}

void MacroIDInfo::Emit(llvm::Serializer& S) const {
  S.Emit(InstantiationLoc);
  S.Emit(SpellingLoc);
}

MacroIDInfo MacroIDInfo::ReadVal(llvm::Deserializer& D) {
  MacroIDInfo I;
  I.InstantiationLoc = SourceLocation::ReadVal(D);
  I.SpellingLoc = SourceLocation::ReadVal(D);
  return I;
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
  
  // Emit: FileIDs
  S.EmitInt(FileIDs.size());  
  std::for_each(FileIDs.begin(), FileIDs.end(), S.MakeEmitter<FileIDInfo>());
  
  // Emit: MacroIDs
  S.EmitInt(MacroIDs.size());  
  std::for_each(MacroIDs.begin(), MacroIDs.end(), S.MakeEmitter<MacroIDInfo>());
  
  S.ExitBlock();
}

SourceManager*
SourceManager::CreateAndRegister(llvm::Deserializer& D, FileManager& FMgr){
  SourceManager *M = new SourceManager();
  D.RegisterPtr(M);
  
  // Read: the FileID of the main source file of the translation unit.
  M->MainFileID = FileID::Create(D.ReadInt());
  
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
  
  // Read: FileIDs.
  unsigned Size = D.ReadInt();
  M->FileIDs.reserve(Size);
  for (; Size > 0 ; --Size)
    M->FileIDs.push_back(FileIDInfo::ReadVal(D));
  
  // Read: MacroIDs.
  Size = D.ReadInt();
  M->MacroIDs.reserve(Size);
  for (; Size > 0 ; --Size)
    M->MacroIDs.push_back(MacroIDInfo::ReadVal(D));
  
  return M;
}
