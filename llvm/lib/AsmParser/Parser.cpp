//===- Parser.cpp - Main dispatch module for the Parser library -------------===
//
// This library implements the functionality defined in llvm/assembly/parser.h
//
//===------------------------------------------------------------------------===

#include "llvm/Analysis/Verifier.h"
#include "llvm/Module.h"
#include "ParserInternals.h"
using std::string;

// The useful interface defined by this file... Parse an ASCII file, and return
// the internal representation in a nice slice'n'dice'able representation.
//
Module *ParseAssemblyFile(const string &Filename) { // throw (ParseException)
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


ParseException::ParseException(const string &filename, const string &message, 
			       int lineNo, int colNo) 
  : Filename(filename), Message(message) {
  LineNo = lineNo; ColumnNo = colNo;
}

ParseException::ParseException(const ParseException &E) 
  : Filename(E.Filename), Message(E.Message) {
  LineNo = E.LineNo;
  ColumnNo = E.ColumnNo;
}

const string ParseException::getMessage() const { // Includes info from options
  string Result;
  char Buffer[10];

  if (Filename == "-") 
    Result += "<stdin>";
  else
    Result += Filename;

  if (LineNo != -1) {
    sprintf(Buffer, "%d", LineNo);
    Result += string(":") + Buffer;
    if (ColumnNo != -1) {
      sprintf(Buffer, "%d", ColumnNo);
      Result += string(",") + Buffer;
    }
  }
  
  return Result + ": " + Message;
}
