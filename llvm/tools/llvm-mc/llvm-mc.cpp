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

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"
#include "AsmParser.h"
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

static cl::list<std::string>
IncludeDirs("I", cl::desc("Directory of include files"),
            cl::value_desc("directory"), cl::Prefix);

static cl::opt<std::string>
Triple("triple", cl::desc("Target triple to assemble for,"
                          "see -version for available targets"),
       cl::init(LLVM_HOSTTRIPLE));

enum ActionType {
  AC_AsLex,
  AC_Assemble
};

static cl::opt<ActionType>
Action(cl::desc("Action to perform:"),
       cl::init(AC_Assemble),
       cl::values(clEnumValN(AC_AsLex, "as-lex",
                             "Lex tokens from a .s file"),
                  clEnumValN(AC_Assemble, "assemble",
                             "Assemble a .s file (default)"),
                  clEnumValEnd));

static int AsLexInput(const char *ProgName) {
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
  
  bool Error = false;
  
  asmtok::TokKind Tok = Lexer.Lex();
  while (Tok != asmtok::Eof) {
    switch (Tok) {
    default:
      Lexer.PrintMessage(Lexer.getLoc(), "unknown token", "warning");
      Error = true;
      break;
    case asmtok::Error:
      Error = true; // error already printed.
      break;
    case asmtok::Identifier:
      outs() << "identifier: " << Lexer.getCurStrVal() << '\n';
      break;
    case asmtok::Register:
      outs() << "register: " << Lexer.getCurStrVal() << '\n';
      break;
    case asmtok::String:
      outs() << "string: " << Lexer.getCurStrVal() << '\n';
      break;
    case asmtok::IntVal:
      outs() << "int: " << Lexer.getCurIntVal() << '\n';
      break;

    case asmtok::Amp:            outs() << "Amp\n"; break;
    case asmtok::AmpAmp:         outs() << "AmpAmp\n"; break;
    case asmtok::Caret:          outs() << "Caret\n"; break;
    case asmtok::Colon:          outs() << "Colon\n"; break;
    case asmtok::Comma:          outs() << "Comma\n"; break;
    case asmtok::Dollar:         outs() << "Dollar\n"; break;
    case asmtok::EndOfStatement: outs() << "EndOfStatement\n"; break;
    case asmtok::Eof:            outs() << "Eof\n"; break;
    case asmtok::Equal:          outs() << "Equal\n"; break;
    case asmtok::EqualEqual:     outs() << "EqualEqual\n"; break;
    case asmtok::Exclaim:        outs() << "Exclaim\n"; break;
    case asmtok::ExclaimEqual:   outs() << "ExclaimEqual\n"; break;
    case asmtok::Greater:        outs() << "Greater\n"; break;
    case asmtok::GreaterEqual:   outs() << "GreaterEqual\n"; break;
    case asmtok::GreaterGreater: outs() << "GreaterGreater\n"; break;
    case asmtok::LParen:         outs() << "LParen\n"; break;
    case asmtok::Less:           outs() << "Less\n"; break;
    case asmtok::LessEqual:      outs() << "LessEqual\n"; break;
    case asmtok::LessGreater:    outs() << "LessGreater\n"; break;
    case asmtok::LessLess:       outs() << "LessLess\n"; break;
    case asmtok::Minus:          outs() << "Minus\n"; break;
    case asmtok::Percent:        outs() << "Percent\n"; break;
    case asmtok::Pipe:           outs() << "Pipe\n"; break;
    case asmtok::PipePipe:       outs() << "PipePipe\n"; break;
    case asmtok::Plus:           outs() << "Plus\n"; break;
    case asmtok::RParen:         outs() << "RParen\n"; break;
    case asmtok::Slash:          outs() << "Slash\n"; break;
    case asmtok::Star:           outs() << "Star\n"; break;
    case asmtok::Tilde:          outs() << "Tilde\n"; break;
    }
    
    Tok = Lexer.Lex();
  }
  
  return Error;
}

static int AssembleInput(const char *ProgName) {
  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget =
    TargetRegistry::getClosestStaticTargetForTriple(Triple, Error);
  if (TheTarget == 0) {
    errs() << ProgName << ": error: unable to get target for '" << Triple
           << "', see --version and --triple.\n";
    return 1;
  }

  TargetAsmParser *TAP = TheTarget->createAsmParser();
  if (!TAP) {
    errs() << ProgName 
           << ": error: this target does not support assembly parsing.\n";
    return 1;    
  }

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
  
  MCContext Ctx;
  OwningPtr<MCStreamer> Str(createAsmStreamer(Ctx, outs()));

  // FIXME: Target hook & command line option for initial section.
  Str.get()->SwitchSection(Ctx.GetSection("__TEXT,__text,regular,pure_instructions"));

  AsmParser Parser(SrcMgr, Ctx, *Str.get());
  return Parser.Run();
}  


int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllAsmParsers();
  
  cl::ParseCommandLineOptions(argc, argv, "llvm machine code playground\n");

  switch (Action) {
  default:
  case AC_AsLex:
    return AsLexInput(argv[0]);
  case AC_Assemble:
    return AssembleInput(argv[0]);
  }
  
  return 0;
}

