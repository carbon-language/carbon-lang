//===- ReaderWrappers.cpp - Parse bytecode from file or buffer  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements loading and parsing a bytecode file and parsing a
// bytecode module from a given buffer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Bytecode/Reader.h"
#include "Reader.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/System/MappedFile.h"
#include "llvm/System/Program.h"
#include <cerrno>
#include <memory>
using namespace llvm;

//===----------------------------------------------------------------------===//
// BytecodeFileReader - Read from an mmap'able file descriptor.
//

namespace {
  /// BytecodeFileReader - parses a bytecode file from a file
  ///
  class BytecodeFileReader : public BytecodeReader {
  private:
    std::string fileName;
    BCDecompressor_t *Decompressor;
    sys::MappedFile mapFile;

    BytecodeFileReader(const BytecodeFileReader&); // Do not implement
    void operator=(const BytecodeFileReader &BFR); // Do not implement

  public:
    BytecodeFileReader(const std::string &Filename, BCDecompressor_t *BCDC,
                       llvm::BytecodeHandler* H=0);
    bool read(std::string* ErrMsg);
    
    void freeState() {
      BytecodeReader::freeState();
      mapFile.close();
    }
  };
}

BytecodeFileReader::BytecodeFileReader(const std::string &Filename,
                                       BCDecompressor_t *BCDC,
                                       llvm::BytecodeHandler* H)
  : BytecodeReader(H), fileName(Filename), Decompressor(BCDC) {
}

bool BytecodeFileReader::read(std::string* ErrMsg) {
  if (mapFile.open(sys::Path(fileName), sys::MappedFile::READ_ACCESS, ErrMsg))
    return true;
  if (!mapFile.map(ErrMsg)) {
    mapFile.close();
    return true;
  }
  unsigned char* buffer = reinterpret_cast<unsigned char*>(mapFile.base());
  return ParseBytecode(buffer, mapFile.size(), fileName,
                       Decompressor, ErrMsg);
}

//===----------------------------------------------------------------------===//
// BytecodeBufferReader - Read from a memory buffer
//

namespace {
  /// BytecodeBufferReader - parses a bytecode file from a buffer
  ///
  class BytecodeBufferReader : public BytecodeReader {
  private:
    const unsigned char *Buffer;
    const unsigned char *Buf;
    unsigned Length;
    std::string ModuleID;
    BCDecompressor_t *Decompressor;
    bool MustDelete;

    BytecodeBufferReader(const BytecodeBufferReader&); // Do not implement
    void operator=(const BytecodeBufferReader &BFR);   // Do not implement

  public:
    BytecodeBufferReader(const unsigned char *Buf, unsigned Length,
                         const std::string &ModuleID, BCDecompressor_t *BCDC,
                         llvm::BytecodeHandler* Handler = 0);
    ~BytecodeBufferReader();

    bool read(std::string* ErrMsg);

  };
}

BytecodeBufferReader::BytecodeBufferReader(const unsigned char *buf,
                                           unsigned len,
                                           const std::string &modID,
                                           BCDecompressor_t *BCDC,
                                           llvm::BytecodeHandler *H)
  : BytecodeReader(H), Buffer(0), Buf(buf), Length(len), ModuleID(modID)
  , Decompressor(BCDC), MustDelete(false) {
}

BytecodeBufferReader::~BytecodeBufferReader() {
  if (MustDelete) delete [] Buffer;
}

bool
BytecodeBufferReader::read(std::string* ErrMsg) {
  // If not aligned, allocate a new buffer to hold the bytecode...
  const unsigned char *ParseBegin = 0;
  if (reinterpret_cast<uint64_t>(Buf) & 3) {
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
  if (ParseBytecode(ParseBegin, Length, ModuleID, Decompressor, ErrMsg)) {
    if (MustDelete) delete [] Buffer;
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
//  BytecodeStdinReader - Read bytecode from Standard Input
//

namespace {
  /// BytecodeStdinReader - parses a bytecode file from stdin
  ///
  class BytecodeStdinReader : public BytecodeReader {
  private:
    std::vector<unsigned char> FileData;
    BCDecompressor_t *Decompressor;
    unsigned char *FileBuf;

    BytecodeStdinReader(const BytecodeStdinReader&); // Do not implement
    void operator=(const BytecodeStdinReader &BFR);  // Do not implement

  public:
    BytecodeStdinReader(BCDecompressor_t *BCDC, llvm::BytecodeHandler* H = 0);
    bool read(std::string* ErrMsg);
  };
}

BytecodeStdinReader::BytecodeStdinReader(BCDecompressor_t *BCDC,
                                         BytecodeHandler* H)
  : BytecodeReader(H), Decompressor(BCDC) {
}

bool BytecodeStdinReader::read(std::string* ErrMsg) {
  sys::Program::ChangeStdinToBinary();
  char Buffer[4096*4];

  // Read in all of the data from stdin, we cannot mmap stdin...
  while (cin.stream()->good()) {
    cin.stream()->read(Buffer, 4096*4);
    int BlockSize = cin.stream()->gcount();
    if (0 >= BlockSize)
      break;
    FileData.insert(FileData.end(), Buffer, Buffer+BlockSize);
  }

  if (FileData.empty()) {
    if (ErrMsg)
      *ErrMsg = "Standard Input is empty!";
    return true;
  }

  FileBuf = &FileData[0];
  if (ParseBytecode(FileBuf, FileData.size(), "<stdin>", Decompressor, ErrMsg))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Wrapper functions
//===----------------------------------------------------------------------===//

/// getBytecodeBufferModuleProvider - lazy function-at-a-time loading from a
/// buffer
ModuleProvider*
llvm::getBytecodeBufferModuleProvider(const unsigned char *Buffer,
                                      unsigned Length,
                                      const std::string &ModuleID,
                                      BCDecompressor_t *BCDC,
                                      std::string *ErrMsg, 
                                      BytecodeHandler *H) {
  BytecodeBufferReader *rdr = 
    new BytecodeBufferReader(Buffer, Length, ModuleID, BCDC, H);
  if (rdr->read(ErrMsg))
    return 0;
  return rdr;
}

/// ParseBytecodeBuffer - Parse a given bytecode buffer
///
Module *llvm::ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                                  const std::string &ModuleID,
                                  BCDecompressor_t *BCDC,
                                  std::string *ErrMsg) {
  ModuleProvider *MP = 
    getBytecodeBufferModuleProvider(Buffer, Length, ModuleID, BCDC, ErrMsg, 0);
  if (!MP) return 0;
  Module *M = MP->releaseModule(ErrMsg);
  delete MP;
  return M;
}

/// getBytecodeModuleProvider - lazy function-at-a-time loading from a file
///
ModuleProvider *
llvm::getBytecodeModuleProvider(const std::string &Filename,
                                BCDecompressor_t *BCDC,
                                std::string* ErrMsg,
                                BytecodeHandler* H) {
  // Read from a file
  if (Filename != std::string("-")) {
    BytecodeFileReader *rdr = new BytecodeFileReader(Filename, BCDC, H);
    if (rdr->read(ErrMsg))
      return 0;
    return rdr;
  }

  // Read from stdin
  BytecodeStdinReader *rdr = new BytecodeStdinReader(BCDC, H);
  if (rdr->read(ErrMsg))
    return 0;
  return rdr;
}

/// ParseBytecodeFile - Parse the given bytecode file
///
Module *llvm::ParseBytecodeFile(const std::string &Filename,
                                BCDecompressor_t *BCDC,
                                std::string *ErrMsg) {
  ModuleProvider* MP = getBytecodeModuleProvider(Filename, BCDC, ErrMsg);
  if (!MP) return 0;
  Module *M = MP->releaseModule(ErrMsg);
  delete MP;
  return M;
}

// AnalyzeBytecodeFile - analyze one file
Module* llvm::AnalyzeBytecodeFile(
  const std::string &Filename,  ///< File to analyze
  BytecodeAnalysis& bca,        ///< Statistical output
  BCDecompressor_t *BCDC,
  std::string *ErrMsg,          ///< Error output
  std::ostream* output          ///< Dump output
) {
  BytecodeHandler* AH = createBytecodeAnalyzerHandler(bca,output);
  ModuleProvider* MP = getBytecodeModuleProvider(Filename, BCDC, ErrMsg, AH);
  if (!MP) return 0;
  Module *M = MP->releaseModule(ErrMsg);
  delete MP;
  return M;
}

bool llvm::GetBytecodeDependentLibraries(const std::string &fname,
                                         Module::LibraryListType& deplibs,
                                         BCDecompressor_t *BCDC,
                                         std::string* ErrMsg) {
  ModuleProvider* MP = getBytecodeModuleProvider(fname, BCDC, ErrMsg);
  if (!MP) {
    deplibs.clear();
    return true;
  }
  Module* M = MP->releaseModule(ErrMsg);
  deplibs = M->getLibraries();
  delete M;
  delete MP;
  return false;
}

static void getSymbols(Module*M, std::vector<std::string>& symbols) {
  // Loop over global variables
  for (Module::global_iterator GI = M->global_begin(), GE=M->global_end(); GI != GE; ++GI)
    if (!GI->isDeclaration() && !GI->hasInternalLinkage())
      if (!GI->getName().empty())
        symbols.push_back(GI->getName());

  // Loop over functions.
  for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI)
    if (!FI->isDeclaration() && !FI->hasInternalLinkage())
      if (!FI->getName().empty())
        symbols.push_back(FI->getName());
}

// Get just the externally visible defined symbols from the bytecode
bool llvm::GetBytecodeSymbols(const sys::Path& fName,
                              std::vector<std::string>& symbols,
                               BCDecompressor_t *BCDC,
                              std::string* ErrMsg) {
  ModuleProvider *MP = getBytecodeModuleProvider(fName.toString(), BCDC,ErrMsg);
  if (!MP)
    return true;

  // Get the module from the provider
  Module* M = MP->materializeModule();
  if (M == 0) {
    delete MP;
    return true;
  }

  // Get the symbols
  getSymbols(M, symbols);

  // Done with the module.
  delete MP;
  return true;
}

ModuleProvider*
llvm::GetBytecodeSymbols(const unsigned char*Buffer, unsigned Length,
                         const std::string& ModuleID,
                         std::vector<std::string>& symbols,
                          BCDecompressor_t *BCDC,
                         std::string* ErrMsg) {
  // Get the module provider
  ModuleProvider* MP = 
    getBytecodeBufferModuleProvider(Buffer, Length, ModuleID, BCDC, ErrMsg, 0);
  if (!MP)
    return 0;

  // Get the module from the provider
  Module* M = MP->materializeModule();
  if (M == 0) {
    delete MP;
    return 0;
  }

  // Get the symbols
  getSymbols(M, symbols);

  // Done with the module. Note that ModuleProvider will delete the
  // Module when it is deleted. Also note that its the caller's responsibility
  // to delete the ModuleProvider.
  return MP;
}
