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
#include <iomanip>

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
namespace {
inline static void print(std::ostream& Out, const char*title, 
  unsigned val, bool nl = true ) {
  Out << std::setw(30) << std::right << title 
      << std::setw(0) << ": "
      << std::setw(9) << val << "\n";
}

inline static void print(std::ostream&Out, const char*title, 
  double val ) {
  Out << std::setw(30) << std::right << title 
      << std::setw(0) << ": "
      << std::setw(9) << std::setprecision(6) << val << "\n" ;
}

inline static void print(std::ostream&Out, const char*title, 
  double top, double bot ) {
  Out << std::setw(30) << std::right << title 
      << std::setw(0) << ": "
      << std::setw(9) << std::setprecision(6) << top 
      << " (" << std::left << std::setw(0) << std::setprecision(4) 
      << (top/bot)*100.0 << "%)\n";
}
inline static void print(std::ostream&Out, const char*title, 
  std::string val, bool nl = true) {
  Out << std::setw(30) << std::right << title 
      << std::setw(0) << ": "
      << std::left << val << (nl ? "\n" : "");
}

}

void llvm::PrintBytecodeAnalysis(BytecodeAnalysis& bca, std::ostream& Out )
{
  print(Out, "Bytecode Analysis Of Module",     bca.ModuleId);
  print(Out, "File Size",                       bca.byteSize);
  print(Out, "Bytecode Compression Index",std::string("TBD"));
  print(Out, "Number Of Bytecode Blocks",       bca.numBlocks);
  print(Out, "Number Of Types",                 bca.numTypes);
  print(Out, "Number Of Values",                bca.numValues);
  print(Out, "Number Of Constants",             bca.numConstants);
  print(Out, "Number Of Global Variables",      bca.numGlobalVars);
  print(Out, "Number Of Functions",             bca.numFunctions);
  print(Out, "Number Of Basic Blocks",          bca.numBasicBlocks);
  print(Out, "Number Of Instructions",          bca.numInstructions);
  print(Out, "Number Of Operands",              bca.numOperands);
  print(Out, "Number Of Compaction Tables",     bca.numCmpctnTables);
  print(Out, "Number Of Symbol Tables",         bca.numSymTab);
  print(Out, "Maximum Type Slot Number",        bca.maxTypeSlot);
  print(Out, "Maximum Value Slot Number",       bca.maxValueSlot);
  print(Out, "Bytes Thrown To Alignment",       double(bca.numAlignment), 
    double(bca.byteSize));
  print(Out, "File Density (bytes/def)",        bca.fileDensity);
  print(Out, "Globals Density (bytes/def)",     bca.globalsDensity);
  print(Out, "Function Density (bytes/func)",   bca.functionDensity);
  print(Out, "Number of VBR 32-bit Integers",   bca.vbrCount32);
  print(Out, "Number of VBR 64-bit Integers",   bca.vbrCount64);
  print(Out, "Number of VBR Compressed Bytes",  bca.vbrCompBytes);
  print(Out, "Number of VBR Expanded Bytes",    bca.vbrExpdBytes);
  print(Out, "VBR Savings", 
    double(bca.vbrExpdBytes)-double(bca.vbrCompBytes),
    double(bca.byteSize));

  if ( bca.detailedResults ) {
    print(Out, "Module Bytes",
        double(bca.BlockSizes[BytecodeFormat::Module]),
        double(bca.byteSize));
    print(Out, "Function Bytes", 
        double(bca.BlockSizes[BytecodeFormat::Function]),
        double(bca.byteSize));
    print(Out, "Constant Pool Bytes", 
        double(bca.BlockSizes[BytecodeFormat::ConstantPool]),
        double(bca.byteSize));
    print(Out, "Symbol Table Bytes", 
        double(bca.BlockSizes[BytecodeFormat::SymbolTable]),
        double(bca.byteSize));
    print(Out, "Module Global Info Bytes", 
        double(bca.BlockSizes[BytecodeFormat::ModuleGlobalInfo]),
        double(bca.byteSize));
    print(Out, "Global Type Plane Bytes", 
        double(bca.BlockSizes[BytecodeFormat::GlobalTypePlane]),
        double(bca.byteSize));
    print(Out, "Basic Block Bytes", 
        double(bca.BlockSizes[BytecodeFormat::BasicBlock]),
        double(bca.byteSize));
    print(Out, "Instruction List Bytes", 
        double(bca.BlockSizes[BytecodeFormat::InstructionList]),
        double(bca.byteSize));
    print(Out, "Compaction Table Bytes", 
        double(bca.BlockSizes[BytecodeFormat::CompactionTable]),
        double(bca.byteSize));

    std::map<unsigned,BytecodeAnalysis::BytecodeFunctionInfo>::iterator I = 
      bca.FunctionInfo.begin();
    std::map<unsigned,BytecodeAnalysis::BytecodeFunctionInfo>::iterator E = 
      bca.FunctionInfo.end();

    while ( I != E ) {
      Out << std::left << std::setw(0);
      Out << "Function: " << I->second.name << " Slot=" << I->first << "\n";
      print(Out,"Type:", I->second.description);
      print(Out,"Byte Size", I->second.byteSize);
      print(Out,"Instructions", I->second.numInstructions);
      print(Out,"Basic Blocks", I->second.numBasicBlocks);
      print(Out,"Operand", I->second.numOperands);
      print(Out,"Function Density", I->second.density);
      print(Out,"VBR Effectiveness", I->second.vbrEffectiveness);
      ++I;
    }
  }

  if ( bca.dumpBytecode )
    Out << bca.BytecodeDump;
}
// vim: sw=2
