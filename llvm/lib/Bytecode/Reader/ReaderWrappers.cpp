//===- ReaderWrappers.cpp - Parse bytecode from file or buffer  -----------===//
//
// This file implements loading and parsing a bytecode file and parsing a
// bytecode module from a given buffer.
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "Support/StringExtras.h"
#include "Config/fcntl.h"
#include "Config/unistd.h"
#include "Config/sys/mman.h"

namespace {
  /// FDHandle - Simple handle class to make sure a file descriptor gets closed
  /// when the object is destroyed.
  ///
  class FDHandle {
    int FD;
  public:
    FDHandle(int fd) : FD(fd) {}
    operator int() const { return FD; }
    ~FDHandle() {
      if (FD != -1) close(FD);
    }
  };

  /// BytecodeFileReader - parses a bytecode file from a file
  ///
  class BytecodeFileReader : public BytecodeParser {
  private:
    unsigned char *Buffer;
    int Length;

    BytecodeFileReader(const BytecodeFileReader&); // Do not implement
    void operator=(const BytecodeFileReader &BFR); // Do not implement

  public:
    BytecodeFileReader(const std::string &Filename);
    ~BytecodeFileReader();
  };
}

BytecodeFileReader::BytecodeFileReader(const std::string &Filename) {
  FDHandle FD = open(Filename.c_str(), O_RDONLY);
  if (FD == -1)
    throw std::string("Error opening file!");

  // Stat the file to get its length...
  struct stat StatBuf;
  if (fstat(FD, &StatBuf) == -1 || StatBuf.st_size == 0)
    throw std::string("Error stat'ing file!");

  // mmap in the file all at once...
  Length = StatBuf.st_size;
  Buffer = (unsigned char*)mmap(0, Length, PROT_READ, MAP_PRIVATE, FD, 0);

  if (Buffer == (unsigned char*)MAP_FAILED)
    throw std::string("Error mmapping file!");

  // Parse the bytecode we mmapped in
  ParseBytecode(Buffer, Length, Filename);
}

BytecodeFileReader::~BytecodeFileReader() {
  // Unmmap the bytecode...
  munmap((char*)Buffer, Length);
}

////////////////////////////////////////////////////////////////////////////

namespace {
  /// BytecodeBufferReader - parses a bytecode file from a buffer
  ///
  class BytecodeBufferReader : public BytecodeParser {
  private:
    const unsigned char *Buffer;
    bool MustDelete;

    BytecodeBufferReader(const BytecodeBufferReader&); // Do not implement
    void operator=(const BytecodeBufferReader &BFR);   // Do not implement

  public:
    BytecodeBufferReader(const unsigned char *Buf, unsigned Length,
                         const std::string &ModuleID);
    ~BytecodeBufferReader();

  };
}

BytecodeBufferReader::BytecodeBufferReader(const unsigned char *Buf,
                                           unsigned Length,
                                           const std::string &ModuleID)
{
  // If not aligned, allocate a new buffer to hold the bytecode...
  const unsigned char *ParseBegin = 0;
  if ((intptr_t)Buf & 3) {
    Buffer = new unsigned char[Length+4];
    unsigned Offset = 4 - ((intptr_t)Buffer & 3);   // Make sure it's aligned
    ParseBegin = Buffer + Offset;
    memcpy((unsigned char*)ParseBegin, Buf, Length);    // Copy it over
    MustDelete = true;
  } else {
    // If we don't need to copy it over, just use the caller's copy
    ParseBegin = Buffer = Buf;
    MustDelete = false;
  }
  ParseBytecode(ParseBegin, Length, ModuleID);
}

BytecodeBufferReader::~BytecodeBufferReader() {
  if (MustDelete) delete [] Buffer;
}


namespace {
  /// BytecodeStdinReader - parses a bytecode file from stdin
  /// 
  class BytecodeStdinReader : public BytecodeParser {
  private:
    std::vector<unsigned char> FileData;
    unsigned char *FileBuf;

    BytecodeStdinReader(const BytecodeStdinReader&); // Do not implement
    void operator=(const BytecodeStdinReader &BFR);  // Do not implement

  public:
    BytecodeStdinReader();
  };
}

BytecodeStdinReader::BytecodeStdinReader() {
  int BlockSize;
  unsigned char Buffer[4096*4];

  // Read in all of the data from stdin, we cannot mmap stdin...
  while ((BlockSize = read(0 /*stdin*/, Buffer, 4096*4))) {
    if (BlockSize == -1)
      throw std::string("Error reading from stdin!");
    
    FileData.insert(FileData.end(), Buffer, Buffer+BlockSize);
  }

  if (FileData.empty())
    throw std::string("Standard Input empty!");

  FileBuf = &FileData[0];
  ParseBytecode(FileBuf, FileData.size(), "<stdin>");
}

/////////////////////////////////////////////////////////////////////////////
//
// Wrapper functions
//
/////////////////////////////////////////////////////////////////////////////

/// getBytecodeBufferModuleProvider - lazy function-at-a-time loading from a
/// buffer
AbstractModuleProvider* 
getBytecodeBufferModuleProvider(const unsigned char *Buffer, unsigned Length,
                                const std::string &ModuleID) {
  return new BytecodeBufferReader(Buffer, Length, ModuleID);
}

/// ParseBytecodeBuffer - Parse a given bytecode buffer
///
Module *ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                            const std::string &ModuleID, std::string *ErrorStr){
  Module *M = 0;
  try {
    AbstractModuleProvider *AMP = 
      getBytecodeBufferModuleProvider(Buffer, Length, ModuleID);
    M = AMP->releaseModule();
    delete AMP;
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
  return M;
}

/// getBytecodeModuleProvider - lazy function-at-a-time loading from a file
///
AbstractModuleProvider*
getBytecodeModuleProvider(const std::string &Filename) {
  if (Filename != std::string("-"))        // Read from a file...
    return new BytecodeFileReader(Filename);
  else                                     // Read from stdin
    return new BytecodeStdinReader();
}

/// ParseBytecodeFile - Parse the given bytecode file
///
Module *ParseBytecodeFile(const std::string &Filename, std::string *ErrorStr) {
  Module *M = 0;
  try {
    AbstractModuleProvider *AMP = getBytecodeModuleProvider(Filename);
    M = AMP->releaseModule();
    delete AMP;
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
  return M;
}
