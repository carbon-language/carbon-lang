//===- Parser.cpp - Main dispatch module for the Parser library -------------===
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/assembly/parser.h
//
//===------------------------------------------------------------------------===

#include "ParserInternals.h"
#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"

// The useful interface defined by this file... Parse an ASCII file, and return
// the internal representation in a nice slice'n'dice'able representation.
//
Module *ParseAssemblyFile(const std::string &Filename) {
  FILE *F = stdin;

  if (Filename != "-") {
    F = fopen(Filename.c_str(), "r");

    if (F == 0)
      throw ParseException(Filename, "Could not open file '" + Filename + "'");
  }

  Module *Result;
  try {
    Result = RunVMAsmParser(Filename, F);
  } catch (...) {
    if (F != stdin) fclose(F);      // Make sure to close file descriptor if an
    throw;                          // exception is thrown
  }

  if (F != stdin)
    fclose(F);

  return Result;
}


//===------------------------------------------------------------------------===
//                              ParseException Class
//===------------------------------------------------------------------------===


ParseException::ParseException(const std::string &filename,
                               const std::string &message, 
			       int lineNo, int colNo) 
  : Filename(filename), Message(message) {
  LineNo = lineNo; ColumnNo = colNo;
}

ParseException::ParseException(const ParseException &E) 
  : Filename(E.Filename), Message(E.Message) {
  LineNo = E.LineNo;
  ColumnNo = E.ColumnNo;
}

// Includes info from options
const std::string ParseException::getMessage() const { 
  std::string Result;
  char Buffer[10];

  if (Filename == "-") 
    Result += "<stdin>";
  else
    Result += Filename;

  if (LineNo != -1) {
    sprintf(Buffer, "%d", LineNo);
    Result += std::string(":") + Buffer;
    if (ColumnNo != -1) {
      sprintf(Buffer, "%d", ColumnNo);
      Result += std::string(",") + Buffer;
    }
  }
  
  return Result + ": " + Message;
}
