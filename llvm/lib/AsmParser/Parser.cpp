//===- Parser.cpp - Main dispatch module for the Parser library -------------===
//
// This library implements the functionality defined in llvm/assembly/parser.h
//
//===------------------------------------------------------------------------===

#include "llvm/Analysis/Verifier.h"
#include "llvm/Module.h"
#include "ParserInternals.h"
#include <stdio.h>  // for sprintf

// The useful interface defined by this file... Parse an ascii file, and return
// the internal representation in a nice slice'n'dice'able representation.
//
Module *ParseAssemblyFile(const string &Filename) { // throw (ParseException)
  FILE *F = stdin;

  if (Filename != "-") 
    F = fopen(Filename.c_str(), "r");

  if (F == 0) {
    throw ParseException(Filename, string("Could not open file '") + 
			 Filename + "'");
  }

  // TODO: If this throws an exception, F is not closed.
  Module *Result = RunVMAsmParser(Filename, F);

  if (F != stdin)
    fclose(F);

  if (Result) {  // Check to see that it is valid...
    vector<string> Errors;
    if (verify(Result, Errors)) {
      delete Result; Result = 0;
      string Message;

      for (unsigned i = 0; i < Errors.size(); i++)
	Message += Errors[i] + "\n";

      throw ParseException(Filename, Message);
    }
  }
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
