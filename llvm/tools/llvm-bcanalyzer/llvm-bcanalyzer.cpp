//===-- llvm-bcanalyzer.cpp - Byte Code Analyzer --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool may be invoked in the following manner:
//  llvm-bcanalyzer [options]      - Read LLVM bytecode from stdin
//  llvm-bcanalyzer [options] x.bc - Read LLVM bytecode from the x.bc file
//
//  Options:
//      --help      - Output information about command line switches
//      --nodetails - Don't print out detailed informaton about individual
//                    blocks and functions
//      --dump      - Dump low-level bytecode structure in readable format
//
// This tool provides analytical information about a bytecode file. It is
// intended as an aid to developers of bytecode reading and writing software. It
// produces on std::out a summary of the bytecode file that shows various
// statistics about the contents of the file. By default this information is
// detailed and contains information about individual bytecode blocks and the
// functions in the module. To avoid this more detailed output, use the
// -nodetails option to limit the output to just module level information.
// The tool is also able to print a bytecode file in a straight forward text
// format that shows the containment and relationships of the information in
// the bytecode file (-dump option).
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compressor.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Signals.h"
#include <fstream>
#include <iostream>
using namespace llvm;

static cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string>
  OutputFilename("-o", cl::init("-"), cl::desc("<output file>"));

static cl::opt<bool> NoDetails("nodetails", cl::desc("Skip detailed output"));
static cl::opt<bool> Dump("dump", cl::desc("Dump low level bytecode trace"));
static cl::opt<bool> Verify("verify", cl::desc("Progressively verify module"));

//===----------------------------------------------------------------------===//
// Bitcode specific analysis.
//===----------------------------------------------------------------------===//

static cl::opt<bool> Bitcode("bitcode", cl::desc("Read a bitcode file"));

/// CurStreamType - If we can sniff the flavor of this stream, we can produce 
/// better dump info.
static enum {
  UnknownBitstream,
  LLVMIRBitstream
} CurStreamType;

/// Error - All bitcode analysis errors go through this function, making this a
/// good place to breakpoint if debugging.
static bool Error(const std::string &Err) {
  std::cerr << Err << "\n";
  return true;
}

/// ParseBlock - Read a block, updating statistics, etc.
static bool ParseBlock(BitstreamReader &Stream) {
  unsigned BlockID = Stream.ReadSubBlockID();
  
  // TODO: Compute per-block-id stats.
  BlockID = BlockID;
  
  if (Stream.EnterSubBlock())
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;

  // Read all the records for this block.
  while (1) {
    if (Stream.AtEndOfStream())
      return Error("Premature end of bitstream");

    // Read the code for this record.
    unsigned AbbrevID = Stream.ReadCode();
    switch (AbbrevID) {
    case bitc::END_BLOCK:
      if (Stream.ReadBlockEnd())
        return Error("Error at end of block");
      return false;
    case bitc::ENTER_SUBBLOCK:
      if (ParseBlock(Stream))
        return true;
      break;
    case bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      break;
    default:
      Record.clear();
      unsigned Code = Stream.ReadRecord(AbbrevID, Record);
      // TODO: Compute per-blockid/code stats.
      Code = Code;
      break;
    }
  }
}

/// AnalyzeBitcode - Analyze the bitcode file specified by InputFilename.
static int AnalyzeBitcode() {
  // Read the input file.
  MemoryBuffer *Buffer;
  if (InputFilename == "-")
    Buffer = MemoryBuffer::getSTDIN();
  else
    Buffer = MemoryBuffer::getFile(&InputFilename[0], InputFilename.size());

  if (Buffer == 0)
    return Error("Error reading '" + InputFilename + "'.");
  
  if (Buffer->getBufferSize() & 3)
    return Error("Bitcode stream should be a multiple of 4 bytes in length");
  
  unsigned char *BufPtr = (unsigned char *)Buffer->getBufferStart();
  BitstreamReader Stream(BufPtr, BufPtr+Buffer->getBufferSize());

  
  // Read the stream signature.
  char Signature[6];
  Signature[0] = Stream.Read(8);
  Signature[1] = Stream.Read(8);
  Signature[2] = Stream.Read(4);
  Signature[3] = Stream.Read(4);
  Signature[4] = Stream.Read(4);
  Signature[5] = Stream.Read(4);
  
  // Autodetect the file contents, if it is one we know.
  CurStreamType = UnknownBitstream;
  if (Signature[0] == 'B' && Signature[1] == 'C' &&
      Signature[2] == 0x0 && Signature[3] == 0xC &&
      Signature[4] == 0xE && Signature[5] == 0xD)
    CurStreamType = LLVMIRBitstream;

  // Parse the top-level structure.  We only allow blocks at the top-level.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code != bitc::ENTER_SUBBLOCK)
      return Error("Invalid record at top-level");
    
    if (ParseBlock(Stream))
      return true;
  }
  
  // Print a summary of the read file.
  
  std::cerr << "Summary of " << InputFilename << ":\n";
  std::cerr << "  Stream type: ";
  switch (CurStreamType) {
  default: assert(0 && "Unknown bitstream type");
  case UnknownBitstream: std::cerr << "unknown\n"; break;
  case LLVMIRBitstream:  std::cerr << "LLVM IR\n"; break;
  }
  
  // TODO: Stats!
  
  return 0;
}


//===----------------------------------------------------------------------===//
// Bytecode specific analysis.
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, " llvm-bcanalyzer file analyzer\n");
  
  sys::PrintStackTraceOnErrorSignal();
  
  if (Bitcode)
    return AnalyzeBitcode();
    
  try {
    std::ostream *Out = &std::cout;  // Default to printing to stdout...
    std::string ErrorMessage;
    BytecodeAnalysis bca;

    /// Determine what to generate
    bca.detailedResults = !NoDetails;
    bca.progressiveVerify = Verify;

    /// Analyze the bytecode file
    Module* M = AnalyzeBytecodeFile(InputFilename, bca, 
                                    Compressor::decompressToNewBuffer,
                                    &ErrorMessage, (Dump?Out:0));

    // All that bcanalyzer does is write the gathered statistics to the output
    PrintBytecodeAnalysis(bca,*Out);

    if (M && Verify) {
      std::string verificationMsg;
      if (verifyModule(*M, ReturnStatusAction, &verificationMsg))
        std::cerr << "Final Verification Message: " << verificationMsg << "\n";
    }

    if (Out != &std::cout) {
      ((std::ofstream*)Out)->close();
      delete Out;
    }
    return 0;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}
