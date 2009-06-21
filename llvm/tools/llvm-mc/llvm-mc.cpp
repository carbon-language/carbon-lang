//===-- llvm-mc.cpp - Machine Code Hacking Driver -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility is a simple driver that allows command line hacking on machine
// code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include "AsmLexer.h"
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

static cl::list<std::string>
IncludeDirs("I", cl::desc("Directory of include files"),
            cl::value_desc("directory"), cl::Prefix);

enum ActionType {
  AC_Assemble
};

static cl::opt<ActionType>
Action(cl::desc("Action to perform:"),
       cl::values(clEnumValN(AC_Assemble, "assemble",
                             "Assemble a .s file (default)"),
                  clEnumValEnd));

static int AssembleInput(const char *ProgName) {
  std::string ErrorMessage;
  MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(InputFilename,
                                                      &ErrorMessage);
  if (Buffer == 0) {
    errs() << ProgName << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "input file didn't read correctly.\n";
    return 1;
  }

  SourceMgr SrcMgr;
  
  // Tell SrcMgr about this buffer, which is what TGParser will pick up.
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());
  
  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(IncludeDirs);

  
  
  AsmLexer Lexer(SrcMgr);
  
  asmtok::TokKind Tok = Lexer.Lex();
  while (Tok != asmtok::Eof) {
    switch (Tok) {
    default: outs() << "<<unknown token>>\n"; break;
    case asmtok::Error: outs() << "<<error>>\n"; break;
    case asmtok::Identifier:
      outs() << "identifier: " << Lexer.getCurStrVal() << '\n';
      break;
    case asmtok::IntVal:
      outs() << "int: " << Lexer.getCurIntVal() << '\n';
      break;
    case asmtok::Colon:  outs() << "Colon\n"; break;
    case asmtok::Plus:   outs() << "Plus\n"; break;
    case asmtok::Minus:  outs() << "Minus\n"; break;
    }
    
    Tok = Lexer.Lex();
  }
  
  return 1;
}


int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm machine code playground\n");

  switch (Action) {
  default:
  case AC_Assemble:
    return AssembleInput(argv[0]);
  }
  
  return 0;
}

