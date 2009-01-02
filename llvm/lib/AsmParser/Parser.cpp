//===- Parser.cpp - Main dispatch module for the Parser library -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/assembly/parser.h
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Parser.h"
#include "LLParser.h"
#include "llvm/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
using namespace llvm;

Module *llvm::ParseAssemblyFile(const std::string &Filename, ParseError &Err) {
  Err.setFilename(Filename);

  std::string ErrorStr;
  MemoryBuffer *F = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), &ErrorStr);
  if (F == 0) {
    Err.setError("Could not open input file '" + Filename + "'");
    return 0;
  }

  Module *Result = LLParser(F, Err).Run();
  delete F;
  return Result;
}

// FIXME: M is ignored??
Module *llvm::ParseAssemblyString(const char *AsmString, Module *M,
                                  ParseError &Err) {
  Err.setFilename("<string>");

  MemoryBuffer *F = MemoryBuffer::getMemBuffer(AsmString,
                                               AsmString+strlen(AsmString),
                                               "<string>");
  Module *Result = LLParser(F, Err).Run();
  delete F;
  return Result;
}


//===------------------------------------------------------------------------===
//                              ParseError Class
//===------------------------------------------------------------------------===

void ParseError::PrintError(const char *ProgName, raw_ostream &S) {
  errs() << ProgName << ": ";
  if (Filename == "-")
    errs() << "<stdin>";
  else
    errs() << Filename;

  if (LineNo != -1) {
    errs() << ':' << LineNo;
    if (ColumnNo != -1)
      errs() << ':' << (ColumnNo+1);
  }

  errs() << ": " << Message << '\n';

  if (LineNo != -1 && ColumnNo != -1) {
    errs() << LineContents << '\n';

    // Print out spaces/tabs before the caret.
    for (unsigned i = 0; i != unsigned(ColumnNo); ++i)
      errs() << (LineContents[i] == '\t' ? '\t' : ' ');
    errs() << "^\n";
  }
}
