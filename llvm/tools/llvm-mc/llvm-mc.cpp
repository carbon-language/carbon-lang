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

#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"  // FIXME.
#include "llvm/Target/TargetSelect.h"
#include "llvm/MC/MCParser/AsmParser.h"
#include "Disassembler.h"
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
ShowEncoding("show-encoding", cl::desc("Show instruction encodings"));

static cl::opt<bool>
ShowFixups("show-fixups", cl::desc("Show fixups inside encodings"));

static cl::opt<bool>
ShowInst("show-inst", cl::desc("Show internal instruction representation"));

static cl::opt<unsigned>
OutputAsmVariant("output-asm-variant",
                 cl::desc("Syntax variant to use for output printing"));

enum OutputFileType {
  OFT_AssemblyFile,
  OFT_ObjectFile
};
static cl::opt<OutputFileType>
FileType("filetype", cl::init(OFT_AssemblyFile),
  cl::desc("Choose an output file type:"),
  cl::values(
       clEnumValN(OFT_AssemblyFile, "asm",
                  "Emit an assembly ('.s') file"),
       clEnumValN(OFT_ObjectFile, "obj",
                  "Emit a native object ('.o') file"),
       clEnumValEnd));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::list<std::string>
IncludeDirs("I", cl::desc("Directory of include files"),
            cl::value_desc("directory"), cl::Prefix);

static cl::opt<std::string>
TripleName("triple", cl::desc("Target triple to assemble for, "
                              "see -version for available targets"),
           cl::init(LLVM_HOSTTRIPLE));

enum ActionType {
  AC_AsLex,
  AC_Assemble,
  AC_Disassemble
};

static cl::opt<ActionType>
Action(cl::desc("Action to perform:"),
       cl::init(AC_Assemble),
       cl::values(clEnumValN(AC_AsLex, "as-lex",
                             "Lex tokens from a .s file"),
                  clEnumValN(AC_Assemble, "assemble",
                             "Assemble a .s file (default)"),
                  clEnumValN(AC_Disassemble, "disassemble",
                             "Disassemble strings of hex bytes"),
                  clEnumValEnd));

static const Target *GetTarget(const char *ProgName) {
  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget)
    return TheTarget;

  errs() << ProgName << ": error: unable to get target for '" << TripleName
         << "', see --version and --triple.\n";
  return 0;
}

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

  const Target *TheTarget = GetTarget(ProgName);
  if (!TheTarget)
    return 1;

  const MCAsmInfo *MAI = TheTarget->createAsmInfo(TripleName);
  assert(MAI && "Unable to create target asm info!");

  AsmLexer Lexer(*MAI);
  
  bool Error = false;
  
  while (Lexer.Lex().isNot(AsmToken::Eof)) {
    switch (Lexer.getKind()) {
    default:
      SrcMgr.PrintMessage(Lexer.getLoc(), "unknown token", "warning");
      Error = true;
      break;
    case AsmToken::Error:
      Error = true; // error already printed.
      break;
    case AsmToken::Identifier:
      outs() << "identifier: " << Lexer.getTok().getString() << '\n';
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

static formatted_raw_ostream *GetOutputStream() {
  if (OutputFilename == "")
    OutputFilename = "-";

  // Make sure that the Out file gets unlinked from the disk if we get a
  // SIGINT.
  if (OutputFilename != "-")
    sys::RemoveFileOnSignal(sys::Path(OutputFilename));

  std::string Err;
  raw_fd_ostream *Out = new raw_fd_ostream(OutputFilename.c_str(), Err,
                                           raw_fd_ostream::F_Binary);
  if (!Err.empty()) {
    errs() << Err << '\n';
    delete Out;
    return 0;
  }
  
  return new formatted_raw_ostream(*Out, formatted_raw_ostream::DELETE_STREAM);
}

static int AssembleInput(const char *ProgName) {
  const Target *TheTarget = GetTarget(ProgName);
  if (!TheTarget)
    return 1;

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
  
  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());
  
  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(IncludeDirs);
  
  MCContext Ctx;
  formatted_raw_ostream *Out = GetOutputStream();
  if (!Out)
    return 1;


  // FIXME: We shouldn't need to do this (and link in codegen).
  OwningPtr<TargetMachine> TM(TheTarget->createTargetMachine(TripleName, ""));

  if (!TM) {
    errs() << ProgName << ": error: could not create target for triple '"
           << TripleName << "'.\n";
    return 1;
  }

  OwningPtr<MCInstPrinter> IP;
  OwningPtr<MCCodeEmitter> CE;
  OwningPtr<MCStreamer> Str;

  const MCAsmInfo *MAI = TheTarget->createAsmInfo(TripleName);
  assert(MAI && "Unable to create target asm info!");

  if (FileType == OFT_AssemblyFile) {
    IP.reset(TheTarget->createMCInstPrinter(OutputAsmVariant, *MAI, *Out));
    if (ShowEncoding)
      CE.reset(TheTarget->createCodeEmitter(*TM));
    Str.reset(createAsmStreamer(Ctx, *Out, *MAI,
                                TM->getTargetData()->isLittleEndian(),
                                /*asmverbose*/true, IP.get(), CE.get(),
                                ShowInst, ShowFixups));
  } else {
    assert(FileType == OFT_ObjectFile && "Invalid file type!");
    CE.reset(TheTarget->createCodeEmitter(*TM));
    Str.reset(createMachOStreamer(Ctx, *Out, CE.get()));
  }

  AsmParser Parser(SrcMgr, Ctx, *Str.get(), *MAI);
  OwningPtr<TargetAsmParser> TAP(TheTarget->createAsmParser(Parser));
  if (!TAP) {
    errs() << ProgName 
           << ": error: this target does not support assembly parsing.\n";
    return 1;
  }

  Parser.setTargetParser(*TAP.get());

  int Res = Parser.Run();
  if (Out != &fouts())
    delete Out;

  return Res;
}

static int DisassembleInput(const char *ProgName) {
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget == 0) {
    errs() << ProgName << ": error: unable to get target for '" << TripleName
    << "', see --version and --triple.\n";
    return 0;
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
  
  return Disassembler::disassemble(*TheTarget, TripleName, *Buffer);
}


int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  // FIXME: We shouldn't need to initialize the Target(Machine)s.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();
  
  cl::ParseCommandLineOptions(argc, argv, "llvm machine code playground\n");

  switch (Action) {
  default:
  case AC_AsLex:
    return AsLexInput(argv[0]);
  case AC_Assemble:
    return AssembleInput(argv[0]);
  case AC_Disassemble:
    return DisassembleInput(argv[0]);
  }
  
  return 0;
}

