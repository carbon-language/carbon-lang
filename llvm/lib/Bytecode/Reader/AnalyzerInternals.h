//===-- ReaderInternals.h - Definitions internal to the reader --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines various stuff that is used by the bytecode reader.
//
//===----------------------------------------------------------------------===//

#ifndef ANALYZER_INTERNALS_H
#define ANALYZER_INTERNALS_H

#include "ReaderPrimitives.h"
#include "Parser.h"
#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"

// Enable to trace to figure out what the heck is going on when parsing fails
//#define TRACE_LEVEL 10
//#define DEBUG_OUTPUT

#if TRACE_LEVEL    // ByteCodeReading_TRACEr
#define BCR_TRACE(n, X) \
    if (n < TRACE_LEVEL) std::cerr << std::string(n*2, ' ') << X
#else
#define BCR_TRACE(n, X)
#endif

namespace llvm {

inline void AbstractBytecodeParser::readBlock(const unsigned char *&Buf,
			       const unsigned char *EndBuf, 
			       unsigned &Type, unsigned &Size)
{
  Type = read(Buf, EndBuf);
  Size = read(Buf, EndBuf);
}

class BytecodeAnalyzer {
  BytecodeAnalyzer(const BytecodeAnalyzer &);  // DO NOT IMPLEMENT
  void operator=(const BytecodeAnalyzer &);  // DO NOT IMPLEMENT
public:
  BytecodeAnalyzer() { }
  ~BytecodeAnalyzer() { }

  void AnalyzeBytecode(
    const unsigned char *Buf, 
    unsigned Length,
    BytecodeAnalysis& bca,
    const std::string &ModuleID
  );

  void DumpBytecode(
    const unsigned char *Buf, 
    unsigned Length,
    BytecodeAnalysis& bca,
    const std::string &ModuleID
  );

  void dump() const {
    std::cerr << "BytecodeParser instance!\n";
  }
private:
  BytecodeAnalysis TheAnalysis;
};

} // End llvm namespace

#endif

// vim: sw=2
