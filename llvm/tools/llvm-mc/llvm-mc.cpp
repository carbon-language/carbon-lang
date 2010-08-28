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

#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetAsmBackend.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"  // FIXME.
#include "llvm/Target/TargetSelect.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Host.h"
#include "llvm/System/Signals.h"
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
ShowInst("show-inst", cl::desc("Show internal instruction representation"));

static cl::opt<bool>
ShowInstOperands("show-inst-operands",
                 cl::desc("Show instructions operands as parsed"));

static cl::opt<unsigned>
OutputAsmVariant("output-asm-variant",
                 cl::desc("Syntax variant to use for output printing"));

static cl::opt<bool>
RelaxAll("mc-relax-all", cl::desc("Relax all fixups"));

static cl::opt<bool>
EnableLogging("enable-api-logging", cl::desc("Enable MC API logging"));

enum OutputFileType {
  OFT_Null,
  OFT_AssemblyFile,
  OFT_ObjectFile
};
static cl::opt<OutputFileType>
FileType("filetype", cl::init(OFT_AssemblyFile),
  cl::desc("Choose an output file type:"),
  cl::values(
       clEnumValN(OFT_AssemblyFile, "asm",
                  "Emit an assembly ('.s') file"),
       clEnumValN(OFT_Null, "null",
                  "Don't emit anything (for timing purposes)"),
       clEnumValN(OFT_ObjectFile, "obj",
                  "Emit a native object ('.o') file"),
       clEnumValEnd));

static cl::list<std::string>
IncludeDirs("I", cl::desc("Directory of include files"),
            cl::value_desc("directory"), cl::Prefix);

static cl::opt<std::string>
ArchName("arch", cl::desc("Target arch to assemble for, "
                            "see -version for available targets"));

static cl::opt<std::string>
TripleName("triple", cl::desc("Target triple to assemble for, "
                              "see -version for available targets"));

static cl::opt<bool>
NoInitialTextSection("n", cl::desc(
                   "Don't assume assembly file starts in the text section"));

enum ActionType {
  AC_AsLex,
  AC_Assemble,
  AC_Disassemble,
  AC_EDisassemble
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
                  clEnumValN(AC_EDisassemble, "edis",
                             "Enhanced disassembly of strings of hex bytes"),
                  clEnumValEnd));

static const Target *GetTarget(const char *ProgName) {
  // Figure out the target triple.
  if (TripleName.empty())
    TripleName = sys::getHostTriple();
  if (!ArchName.empty()) {
    llvm::Triple TT(TripleName);
    TT.setArchName(ArchName);
    TripleName = TT.str();
  }

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget)
    return TheTarget;

  errs() << ProgName << ": error: unable to get target for '" << TripleName
         << "', see --version and --triple.\n";
  return 0;
}

static formatted_tool_output_file *GetOutputStream() {
  if (OutputFilename == "")
    OutputFilename = "-";

  std::string Err;
  tool_output_file *Out = new tool_output_file(OutputFilename.c_str(), Err,
                                               raw_fd_ostream::F_Binary);
  if (!Err.empty()) {
    errs() << Err << '\n';
    delete Out;
    return 0;
  }
  
  return new formatted_tool_output_file(*Out,
                                        formatted_raw_ostream::DELETE_STREAM);
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

  llvm::OwningPtr<MCAsmInfo> MAI(TheTarget->createAsmInfo(TripleName));
  assert(MAI && "Unable to create target asm info!");

  AsmLexer Lexer(*MAI);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(0));

  OwningPtr<formatted_tool_output_file> Out(GetOutputStream());
  if (!Out)
    return 1;

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
      *Out << "identifier: " << Lexer.getTok().getString() << '\n';
      break;
    case AsmToken::String:
      *Out << "string: " << Lexer.getTok().getString() << '\n';
      break;
    case AsmToken::Integer:
      *Out << "int: " << Lexer.getTok().getString() << '\n';
      break;

    case AsmToken::Amp:            *Out << "Amp\n"; break;
    case AsmToken::AmpAmp:         *Out << "AmpAmp\n"; break;
    case AsmToken::Caret:          *Out << "Caret\n"; break;
    case AsmToken::Colon:          *Out << "Colon\n"; break;
    case AsmToken::Comma:          *Out << "Comma\n"; break;
    case AsmToken::Dollar:         *Out << "Dollar\n"; break;
    case AsmToken::EndOfStatement: *Out << "EndOfStatement\n"; break;
    case AsmToken::Eof:            *Out << "Eof\n"; break;
    case AsmToken::Equal:          *Out << "Equal\n"; break;
    case AsmToken::EqualEqual:     *Out << "EqualEqual\n"; break;
    case AsmToken::Exclaim:        *Out << "Exclaim\n"; break;
    case AsmToken::ExclaimEqual:   *Out << "ExclaimEqual\n"; break;
    case AsmToken::Greater:        *Out << "Greater\n"; break;
    case AsmToken::GreaterEqual:   *Out << "GreaterEqual\n"; break;
    case AsmToken::GreaterGreater: *Out << "GreaterGreater\n"; break;
    case AsmToken::LParen:         *Out << "LParen\n"; break;
    case AsmToken::Less:           *Out << "Less\n"; break;
    case AsmToken::LessEqual:      *Out << "LessEqual\n"; break;
    case AsmToken::LessGreater:    *Out << "LessGreater\n"; break;
    case AsmToken::LessLess:       *Out << "LessLess\n"; break;
    case AsmToken::Minus:          *Out << "Minus\n"; break;
    case AsmToken::Percent:        *Out << "Percent\n"; break;
    case AsmToken::Pipe:           *Out << "Pipe\n"; break;
    case AsmToken::PipePipe:       *Out << "PipePipe\n"; break;
    case AsmToken::Plus:           *Out << "Plus\n"; break;
    case AsmToken::RParen:         *Out << "RParen\n"; break;
    case AsmToken::Slash:          *Out << "Slash\n"; break;
    case AsmToken::Star:           *Out << "Star\n"; break;
    case AsmToken::Tilde:          *Out << "Tilde\n"; break;
    }
  }

  // Keep output if no errors.
  if (Error == 0) Out->keep();
 
  return Error;
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
  
  
  llvm::OwningPtr<MCAsmInfo> MAI(TheTarget->createAsmInfo(TripleName));
  assert(MAI && "Unable to create target asm info!");
  
  MCContext Ctx(*MAI);

  // FIXME: We shouldn't need to do this (and link in codegen).
  OwningPtr<TargetMachine> TM(TheTarget->createTargetMachine(TripleName, ""));

  if (!TM) {
    errs() << ProgName << ": error: could not create target for triple '"
           << TripleName << "'.\n";
    return 1;
  }

  OwningPtr<formatted_tool_output_file> Out(GetOutputStream());
  if (!Out)
    return 1;

  OwningPtr<MCStreamer> Str;

  if (FileType == OFT_AssemblyFile) {
    MCInstPrinter *IP =
      TheTarget->createMCInstPrinter(OutputAsmVariant, *MAI);
    MCCodeEmitter *CE = 0;
    if (ShowEncoding)
      CE = TheTarget->createCodeEmitter(*TM, Ctx);
    Str.reset(createAsmStreamer(Ctx, *Out,TM->getTargetData()->isLittleEndian(),
                                /*asmverbose*/true, IP, CE, ShowInst));
  } else if (FileType == OFT_Null) {
    Str.reset(createNullStreamer(Ctx));
  } else {
    assert(FileType == OFT_ObjectFile && "Invalid file type!");
    MCCodeEmitter *CE = TheTarget->createCodeEmitter(*TM, Ctx);
    TargetAsmBackend *TAB = TheTarget->createAsmBackend(TripleName);
    Str.reset(TheTarget->createObjectStreamer(TripleName, Ctx, *TAB,
                                              *Out, CE, RelaxAll));
  }

  if (EnableLogging) {
    Str.reset(createLoggingStreamer(Str.take(), errs()));
  }

  OwningPtr<MCAsmParser> Parser(createMCAsmParser(*TheTarget, SrcMgr, Ctx,
                                                   *Str.get(), *MAI));
  OwningPtr<TargetAsmParser> TAP(TheTarget->createAsmParser(*Parser, *TM));
  if (!TAP) {
    errs() << ProgName 
           << ": error: this target does not support assembly parsing.\n";
    return 1;
  }

  Parser->setShowParsedOperands(ShowInstOperands);
  Parser->setTargetParser(*TAP.get());

  int Res = Parser->Run(NoInitialTextSection);

  // Keep output if no errors.
  if (Res == 0) Out->keep();

  return Res;
}

static int DisassembleInput(const char *ProgName, bool Enhanced) {
  const Target *TheTarget = GetTarget(ProgName);
  if (!TheTarget)
    return 0;
  
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
  
  OwningPtr<formatted_tool_output_file> Out(GetOutputStream());
  if (!Out)
    return 1;

  int Res;
  if (Enhanced)
    Res = Disassembler::disassembleEnhanced(TripleName, *Buffer, *Out);
  else
    Res = Disassembler::disassemble(*TheTarget, TripleName, *Buffer, *Out);

  // Keep output if no errors.
  if (Res == 0) Out->keep();

  return Res;
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
  TripleName = Triple::normalize(TripleName);

  switch (Action) {
  default:
  case AC_AsLex:
    return AsLexInput(argv[0]);
  case AC_Assemble:
    return AssembleInput(argv[0]);
  case AC_Disassemble:
    return DisassembleInput(argv[0], false);
  case AC_EDisassemble:
    return DisassembleInput(argv[0], true);
  }
  
  return 0;
}

