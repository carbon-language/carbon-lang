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

#define CHECK_ALIGN32(begin,end) \
  if (align32(begin,end)) \
    throw std::string("Alignment error: ReaderWrappers.cpp:" + \
                      utostr((unsigned)__LINE__));

namespace {

  /// BytecodeFileReader - parses a bytecode file from a file
  ///
  class BytecodeFileReader : public BytecodeParser {
  private:
    unsigned char *Buffer;
    int Length;

    BytecodeFileReader(const BytecodeFileReader&); // Do not implement
    void operator=(BytecodeFileReader &BFR);       // Do not implement

  public:
    BytecodeFileReader(const std::string &Filename);
    ~BytecodeFileReader();

  };

  /// BytecodeStdinReader - parses a bytecode file from stdin
  /// 
  class BytecodeStdinReader : public BytecodeParser {
  private:
    std::vector<unsigned char> FileData;
    unsigned char *FileBuf;

    BytecodeStdinReader(const BytecodeStdinReader&); // Do not implement
    void operator=(BytecodeStdinReader &BFR);        // Do not implement

  public:
    BytecodeStdinReader();
    ~BytecodeStdinReader();
  };

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
  unsigned char *Buffer = (unsigned char*)mmap(0, Length, PROT_READ, 
                                               MAP_PRIVATE, FD, 0);
  if (Buffer == (unsigned char*)MAP_FAILED)
    throw std::string("Error mmapping file!");

  // Parse the bytecode we mmapped in
  ParseBytecode(Buffer, Length, Filename);
}

BytecodeFileReader::~BytecodeFileReader() {
  // Unmmap the bytecode...
  munmap((char*)Buffer, Length);
}


#define ALIGN_PTRS 0

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

#if ALIGN_PTRS
  FileBuf = (unsigned char*)mmap(0, FileData.size(), PROT_READ|PROT_WRITE, 
                                 MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  assert((Buf != (unsigned char*)-1) && "mmap returned error!");
  memcpy(Buf, &FileData[0], FileData.size());
#else
  FileBuf = &FileData[0];
#endif

#if 0
  // Allocate a new buffer to hold the bytecode...
  unsigned char *ParseBegin=0;
  unsigned Offset=0;
  if ((intptr_t)Buffer & 3) {
    delete [] Buffer;
    Buffer = new unsigned char[Length+4];
    Offset = 4-((intptr_t)Buffer & 3);  // Make sure it's aligned
  }
  memcpy(Buffer+Offset, Buf, Length);          // Copy it over
  ParseBegin = Buffer+Offset;
#endif

  ParseBytecode(FileBuf, FileData.size(), "<stdin>");
}

BytecodeStdinReader::~BytecodeStdinReader() {
#if ALIGN_PTRS
  munmap((char*)FileBuf, FileData.size());   // Free mmap'd data area
#endif
}

///
///
AbstractModuleProvider* 
getBytecodeBufferModuleProvider(const unsigned char *Buffer, unsigned Length,
                                const std::string &ModuleID) {
  CHECK_ALIGN32(Buffer, Buffer+Length);
  BytecodeParser *Parser = new BytecodeParser();
  Parser->ParseBytecode(Buffer, Length, ModuleID);
  return Parser;
}

Module *ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                            const std::string &ModuleID, std::string *ErrorStr){
  AbstractModuleProvider *AMP = 
    getBytecodeBufferModuleProvider(Buffer, Length, ModuleID);
  Module *M = AMP->releaseModule();
  delete AMP;
  return M;
}


/// Parse and return a class file...
///
AbstractModuleProvider*
getBytecodeModuleProvider(const std::string &Filename) {
  if (Filename != std::string("-"))        // Read from a file...
    return new BytecodeFileReader(Filename);
  else                                     // Read from stdin
    return new BytecodeStdinReader();
}

Module *ParseBytecodeFile(const std::string &Filename, std::string *ErrorStr) {
  AbstractModuleProvider *AMP = getBytecodeModuleProvider(Filename);
  Module *M = AMP->releaseModule();
  delete AMP;
  return M;
}
