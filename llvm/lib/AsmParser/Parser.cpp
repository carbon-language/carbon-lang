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
Module *ParseAssemblyFile(const ToolCommandLine &Opts) throw (ParseException) {
  FILE *F = stdin;

  if (Opts.getInputFilename() != "-") 
    F = fopen(Opts.getInputFilename().c_str(), "r");

  if (F == 0) {
    throw ParseException(Opts, string("Could not open file '") + 
			 Opts.getInputFilename() + "'");
  }

  // TODO: If this throws an exception, F is not closed.
  Module *Result = RunVMAsmParser(Opts, F);

  if (F != stdin)
    fclose(F);

  if (Result) {  // Check to see that it is valid...
    vector<string> Errors;
    if (verify(Result, Errors)) {
      delete Result; Result = 0;
      string Message;

      for (unsigned i = 0; i < Errors.size(); i++)
	Message += Errors[i] + "\n";

      throw ParseException(Opts, Message);
    }
  }
  return Result;
}


//===------------------------------------------------------------------------===
//                              ParseException Class
//===------------------------------------------------------------------------===


ParseException::ParseException(const ToolCommandLine &opts, 
			       const string &message, int lineNo, int colNo) 
  : Opts(opts), Message(message) {
  LineNo = lineNo; ColumnNo = colNo;
}

ParseException::ParseException(const ParseException &E) 
  : Opts(E.Opts), Message(E.Message) {
  LineNo = E.LineNo;
  ColumnNo = E.ColumnNo;
}

const string ParseException::getMessage() const { // Includes info from options
  string Result;
  char Buffer[10];

  if (Opts.getInputFilename() == "-") 
    Result += "<stdin>";
  else
    Result += Opts.getInputFilename();

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
