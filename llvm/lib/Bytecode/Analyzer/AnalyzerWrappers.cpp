//===- AnalyzerWrappers.cpp - Analyze bytecode from file or buffer  -------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements loading and analysis of a bytecode file and analyzing a
// bytecode buffer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Analyzer.h"
#include "AnalyzerInternals.h"
#include "Support/FileUtilities.h"
#include "Support/StringExtras.h"
#include "Config/unistd.h"
#include <cerrno>

using namespace llvm;

//===----------------------------------------------------------------------===//
// BytecodeFileAnalyzer - Analyze from an mmap'able file descriptor.
//

namespace {
  /// BytecodeFileAnalyzer - parses a bytecode file from a file
  class BytecodeFileAnalyzer : public BytecodeAnalyzer {
  private:
    unsigned char *Buffer;
    unsigned Length;

    BytecodeFileAnalyzer(const BytecodeFileAnalyzer&); // Do not implement
    void operator=(const BytecodeFileAnalyzer &BFR); // Do not implement

  public:
    BytecodeFileAnalyzer(const std::string &Filename, BytecodeAnalysis& bca);
    ~BytecodeFileAnalyzer();
  };
}

static std::string ErrnoMessage (int savedErrNum, std::string descr) {
   return ::strerror(savedErrNum) + std::string(", while trying to ") + descr;
}

BytecodeFileAnalyzer::BytecodeFileAnalyzer(const std::string &Filename, 
	                                   BytecodeAnalysis& bca) {
  Buffer = (unsigned char*)ReadFileIntoAddressSpace(Filename, Length);
  if (Buffer == 0)
    throw "Error reading file '" + Filename + "'.";

  try {
    // Parse the bytecode we mmapped in
    if ( bca.dumpBytecode ) 
      DumpBytecode(Buffer, Length, bca, Filename);
    AnalyzeBytecode(Buffer, Length, bca, Filename);
  } catch (...) {
    UnmapFileFromAddressSpace(Buffer, Length);
    throw;
  }
}

BytecodeFileAnalyzer::~BytecodeFileAnalyzer() {
  // Unmmap the bytecode...
  UnmapFileFromAddressSpace(Buffer, Length);
}

//===----------------------------------------------------------------------===//
// BytecodeBufferAnalyzer - Read from a memory buffer
//

namespace {
  /// BytecodeBufferAnalyzer - parses a bytecode file from a buffer
  ///
  class BytecodeBufferAnalyzer : public BytecodeAnalyzer {
  private:
    const unsigned char *Buffer;
    bool MustDelete;

    BytecodeBufferAnalyzer(const BytecodeBufferAnalyzer&); // Do not implement
    void operator=(const BytecodeBufferAnalyzer &BFR);   // Do not implement

  public:
    BytecodeBufferAnalyzer(const unsigned char *Buf, unsigned Length,
	                   BytecodeAnalysis& bca, const std::string &ModuleID);
    ~BytecodeBufferAnalyzer();

  };
}

BytecodeBufferAnalyzer::BytecodeBufferAnalyzer(const unsigned char *Buf,
					       unsigned Length,
					       BytecodeAnalysis& bca,
					       const std::string &ModuleID) {
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
  try {
    if ( bca.dumpBytecode ) 
      DumpBytecode(ParseBegin, Length, bca, ModuleID);
    AnalyzeBytecode(ParseBegin, Length, bca, ModuleID);
  } catch (...) {
    if (MustDelete) delete [] Buffer;
    throw;
  }
}

BytecodeBufferAnalyzer::~BytecodeBufferAnalyzer() {
  if (MustDelete) delete [] Buffer;
}

//===----------------------------------------------------------------------===//
//  BytecodeStdinAnalyzer - Read bytecode from Standard Input
//

namespace {
  /// BytecodeStdinAnalyzer - parses a bytecode file from stdin
  /// 
  class BytecodeStdinAnalyzer : public BytecodeAnalyzer {
  private:
    std::vector<unsigned char> FileData;
    unsigned char *FileBuf;

    BytecodeStdinAnalyzer(const BytecodeStdinAnalyzer&); // Do not implement
    void operator=(const BytecodeStdinAnalyzer &BFR);  // Do not implement

  public:
    BytecodeStdinAnalyzer(BytecodeAnalysis& bca);
  };
}

BytecodeStdinAnalyzer::BytecodeStdinAnalyzer(BytecodeAnalysis& bca ) {
  int BlockSize;
  unsigned char Buffer[4096*4];

  // Read in all of the data from stdin, we cannot mmap stdin...
  while ((BlockSize = ::read(0 /*stdin*/, Buffer, 4096*4))) {
    if (BlockSize == -1)
      throw ErrnoMessage(errno, "read from standard input");
    
    FileData.insert(FileData.end(), Buffer, Buffer+BlockSize);
  }

  if (FileData.empty())
    throw std::string("Standard Input empty!");

  FileBuf = &FileData[0];
  if (bca.dumpBytecode)
    DumpBytecode(&FileData[0], FileData.size(), bca, "<stdin>");
  AnalyzeBytecode(FileBuf, FileData.size(), bca, "<stdin>");
}

//===----------------------------------------------------------------------===//
// Wrapper functions
//===----------------------------------------------------------------------===//

// AnalyzeBytecodeFile - analyze one file
void llvm::AnalyzeBytecodeFile(const std::string &Filename, 
                               BytecodeAnalysis& bca,
                               std::string *ErrorStr) 
{
  try {
    if ( Filename != "-" )
      BytecodeFileAnalyzer bfa(Filename,bca);
    else
      BytecodeStdinAnalyzer bsa(bca);
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
  }
}

// AnalyzeBytecodeBuffer - analyze a buffer
void llvm::AnalyzeBytecodeBuffer(
       const unsigned char* Buffer, ///< Pointer to start of bytecode buffer
       unsigned BufferSize,         ///< Size of the bytecode buffer
       BytecodeAnalysis& Results,   ///< The results of the analysis
       std::string* ErrorStr        ///< Errors, if any.
     ) 
{
  try {
    BytecodeBufferAnalyzer(Buffer, BufferSize, Results, "<buffer>" );
  } catch (std::string& err ) {
    if ( ErrorStr) *ErrorStr = err;
  }
}


/// This function prints the contents of rhe BytecodeAnalysis structure in
/// a human legible form.
/// @brief Print BytecodeAnalysis structure to an ostream
void llvm::PrintBytecodeAnalysis(BytecodeAnalysis& bca, std::ostream& Out )
{
  Out << "Not Implemented Yet.\n";
}

// vim: sw=2
