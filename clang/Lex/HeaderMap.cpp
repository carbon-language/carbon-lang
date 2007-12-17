//===--- HeaderMap.cpp - A file that acts like dir of symlinks ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HeaderMap interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/HeaderMap.h"
#include "clang/Basic/FileManager.h"
#include "llvm/ADT/scoped_ptr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
using namespace clang;

enum {
  HeaderMagicNumber = ('h' << 24) | ('m' << 16) | ('a' << 8) | 'p'
};

struct HMapHeader {
  uint32_t Magic;           // Magic word, also indicates byte order.
  uint16_t Version;         // Version number -- currently 1.
  uint16_t Reserved;        // Reserved for future use - zero for now.
  uint32_t StringsOffset;   // Offset to start of string pool.
  uint32_t Count;           // Number of entries in the string table.
  uint32_t Capacity;        // Number of buckets (always a power of 2).
  uint32_t MaxValueLength;  // Length of longest result path (excluding nul).
  // Strings follow the buckets, at StringsOffset.
};


/// HeaderMap::Create - This attempts to load the specified file as a header
/// map.  If it doesn't look like a HeaderMap, it gives up and returns null.
/// If it looks like a HeaderMap but is obviously corrupted, it puts a reason
/// into the string error argument and returns null.
const HeaderMap *HeaderMap::Create(const FileEntry *FE) {
  // If the file is too small to be a header map, ignore it.
  unsigned FileSize = FE->getSize();
  if (FileSize <= sizeof(HMapHeader)) return 0;
  
  llvm::scoped_ptr<const llvm::MemoryBuffer> FileBuffer( 
    llvm::MemoryBuffer::getFile(FE->getName(), strlen(FE->getName()), 0,
                                FE->getSize()));
  if (FileBuffer == 0) return 0;  // Unreadable file?
  const char *FileStart = FileBuffer->getBufferStart();

  // We know the file is at least as big as the header, check it now.
  const HMapHeader *Header = reinterpret_cast<const HMapHeader*>(FileStart);
  
  // Sniff it to see if it's a headermap.
  if (Header->Version != 1 || Header->Reserved != 0)
    return 0;
  
  // Check the magic number.
  bool NeedsByteSwap;
  if (Header->Magic == HeaderMagicNumber)
    NeedsByteSwap = false;
  else if (Header->Magic == llvm::ByteSwap_32(HeaderMagicNumber))
    NeedsByteSwap = true;  // Mixed endianness headermap.
  else 
    return 0;  // Not a header map.

  // Okay, everything looks good, create the header map.
  HeaderMap *NewHM = new HeaderMap(FileBuffer.get(), NeedsByteSwap);
  FileBuffer.reset();  // Don't deallocate the buffer on return.
  return NewHM; 
}

HeaderMap::~HeaderMap() {
  delete FileBuffer;
}


/// getFileName - Return the filename of the headermap.
const char *HeaderMap::getFileName() const {
  return FileBuffer->getBufferIdentifier();
}

/// LookupFile - Check to see if the specified relative filename is located in
/// this HeaderMap.  If so, open it and return its FileEntry.
const FileEntry *HeaderMap::LookupFile(const char *FilenameStart,
                                       const char *FilenameEnd,
                                       FileManager &FM) const {
  // FIXME: this needs work.
  return 0;
}
