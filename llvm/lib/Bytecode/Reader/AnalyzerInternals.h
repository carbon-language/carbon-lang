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

#include "Parser.h"
#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"


namespace llvm {

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
};

} // End llvm namespace

#endif

// vim: sw=2
