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
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetAsmParser.h"
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
TripleName("triple", cl::desc("Target triple to assemble for,"
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
  
  while (Lexer.Lex().isNot(AsmToken::Eof)) {
    switch (Lexer.getKind()) {
    default:
      Lexer.PrintMessage(Lexer.getLoc(), "unknown token", "warning");
      Error = true;
      break;
    case AsmToken::Error:
      Error = true; // error already printed.
      break;
    case AsmToken::Identifier:
      outs() << "identifier: " << Lexer.getTok().getString() << '\n';
      break;
    case AsmToken::Register:
      outs() << "register: " << Lexer.getTok().getString() << '\n';
      break;
    case AsmToken::String:
      outs() << "string: " << Lexer.getTok().getString() << '\n';
      break;
    case AsmToken::Integer:
      outs() << "int: " << Lexer.getTok().getString() << '\n';
      break;

    case AsmToken::Amp:            outs() << "Amp\n"; break;
    case AsmToken::AmpAmp:         outs() << "AmpAmp\n"; break;
    case AsmToken::Caret:          outs() << "Caret\n"; break;
    case AsmToken::Colon:          outs() << "Colon\n"; break;
    case AsmToken::Comma:          outs() << "Comma\n"; break;
    case AsmToken::Dollar:         outs() << "Dollar\n"; break;
    case AsmToken::EndOfStatement: outs() << "EndOfStatement\n"; break;
    case AsmToken::Eof:            outs() << "Eof\n"; break;
    case AsmToken::Equal:          outs() << "Equal\n"; break;
    case AsmToken::EqualEqual:     outs() << "EqualEqual\n"; break;
    case AsmToken::Exclaim:        outs() << "Exclaim\n"; break;
    case AsmToken::ExclaimEqual:   outs() << "ExclaimEqual\n"; break;
    case AsmToken::Greater:        outs() << "Greater\n"; break;
    case AsmToken::GreaterEqual:   outs() << "GreaterEqual\n"; break;
    case AsmToken::GreaterGreater: outs() << "GreaterGreater\n"; break;
    case AsmToken::LParen:         outs() << "LParen\n"; break;
    case AsmToken::Less:           outs() << "Less\n"; break;
    case AsmToken::LessEqual:      outs() << "LessEqual\n"; break;
    case AsmToken::LessGreater:    outs() << "LessGreater\n"; break;
    case AsmToken::LessLess:       outs() << "LessLess\n"; break;
    case AsmToken::Minus:          outs() << "Minus\n"; break;
    case AsmToken::Percent:        outs() << "Percent\n"; break;
    case AsmToken::Pipe:           outs() << "Pipe\n"; break;
    case AsmToken::PipePipe:       outs() << "PipePipe\n"; break;
    case AsmToken::Plus:           outs() << "Plus\n"; break;
    case AsmToken::RParen:         outs() << "RParen\n"; break;
    case AsmToken::Slash:          outs() << "Slash\n"; break;
    case AsmToken::Star:           outs() << "Star\n"; break;
    case AsmToken::Tilde:          outs() << "Tilde\n"; break;
    }
  }
  
  return Error;
}

static TargetAsmParser *GetTargetAsmParser(const char *ProgName,
                                           MCAsmParser &Parser) {
  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget == 0) {
    errs() << ProgName << ": error: unable to get target for '" << TripleName
           << "', see --version and --triple.\n";
    return 0;
  }

  if (TargetAsmParser *TAP = TheTarget->createAsmParser(Parser))
    return TAP;
    
  errs() << ProgName 
         << ": error: this target does not support assembly parsing.\n";
  return 0;
}

static int AssembleInput(const char *ProgName) {
  std::string Error;
  MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(InputFilename, &Error);
  if (Buffer == 0) {
    errs() << ProgName << ": ";
    if (Error.size())
      errs() << Error << "\n";
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
  Str.get()->SwitchSection(MCSectionCOFF::Create("__TEXT,__text,"
                                             "regular,pure_instructions",
                                             false,
                                            SectionKind::getText(),
                                             Ctx));

  AsmParser Parser(SrcMgr, Ctx, *Str.get());
  OwningPtr<TargetAsmParser> TAP(GetTargetAsmParser(ProgName, Parser));
  if (!TAP)
    return 1;
  Parser.setTargetParser(*TAP.get());
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

