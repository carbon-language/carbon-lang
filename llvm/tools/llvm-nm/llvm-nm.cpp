//===-- llvm-nm.cpp - Symbol table dumping utility for llvm ---------------===//
//
// This program is a utility that works like traditional Unix "nm",
// that is, it prints out the names of symbols in a bytecode file,
// along with some information about each symbol.
// 
// This "nm" does not print symbols' addresses. It supports many of
// the features of GNU "nm", including its different output formats.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/GlobalValue.h"
#include "llvm/Module.h"
#include <cctype>

namespace {
  enum OutputFormatTy { bsd, sysv, posix };
  cl::opt<OutputFormatTy>
  OutputFormat("format",
       cl::desc("Specify output format"),
         cl::values(clEnumVal(bsd,   "BSD format"),
                    clEnumVal(sysv,  "System V format"),
                    clEnumVal(posix, "POSIX.2 format"), 0), cl::init(bsd));
  cl::alias OutputFormat2("f", cl::desc("Alias for --format"),
                          cl::aliasopt(OutputFormat));

  cl::list<std::string> 
  InputFilenames(cl::Positional, cl::desc("<input bytecode files>"),
                 cl::OneOrMore);

  cl::opt<bool> UndefinedOnly("undefined-only",
                              cl::desc("Show only undefined symbols"));
  cl::alias UndefinedOnly2("u", cl::desc("Alias for --undefined-only"),
                           cl::aliasopt(UndefinedOnly));

  cl::opt<bool> DefinedOnly("defined-only",
                            cl::desc("Show only defined symbols"));

  cl::opt<bool> ExternalOnly("extern-only",
                             cl::desc("Show only external symbols"));
  cl::alias ExternalOnly2("g", cl::desc("Alias for --extern-only"),
                          cl::aliasopt(ExternalOnly));

  cl::opt<bool> BSDFormat("B", cl::desc("Alias for --format=bsd"));
  cl::opt<bool> POSIXFormat("P", cl::desc("Alias for --format=posix"));

  bool MultipleFiles = false;

  std::string ToolName;
};

char TypeCharForSymbol (GlobalValue &GV) {
  if (GV.isExternal ())                                     return 'U';
  if (GV.hasLinkOnceLinkage ())                             return 'C';
#ifdef WEAK_LINKAGE_EVENTUALLY_IMPLEMENTED
  if (GV.hasWeakLinkage ())                                 return 'W';
#endif
  if (isa<Function> (GV) && GV.hasInternalLinkage ())       return 't';
  if (isa<Function> (GV))                                   return 'T';
  if (isa<GlobalVariable> (GV) && GV.hasInternalLinkage ()) return 'd';
  if (isa<GlobalVariable> (GV))                             return 'D';
                                                            return '?';
}

void DumpSymbolNameForGlobalValue (GlobalValue &GV) {
  const std::string SymbolAddrStr = "        "; // Not used yet...
  char TypeChar = TypeCharForSymbol (GV);
  if ((TypeChar != 'U') && UndefinedOnly)
    return;
  if ((TypeChar == 'U') && DefinedOnly)
    return;
  if (GV.hasInternalLinkage () && ExternalOnly)
    return;
  if (OutputFormat == posix) {
    std::cout << GV.getName () << " " << TypeCharForSymbol (GV) << " "
              << SymbolAddrStr << "\n";
  } else if (OutputFormat == bsd) {
    std::cout << SymbolAddrStr << " " << TypeCharForSymbol (GV) << " "
              << GV.getName () << "\n";
  } else if (OutputFormat == sysv) {
    std::string PaddedName (GV.getName ());
    while (PaddedName.length () < 20)
      PaddedName += " ";
    std::cout << PaddedName << "|" << SymbolAddrStr << "|   "
              << TypeCharForSymbol (GV)
              << "  |                  |      |     |\n";
  }
}

void DumpSymbolNamesFromModule (Module *M) {
  std::for_each (M->begin (), M->end (), DumpSymbolNameForGlobalValue);
  std::for_each (M->gbegin (), M->gend (), DumpSymbolNameForGlobalValue);
}

void DumpSymbolNamesFromFile (std::string &Filename) {
  std::string ErrorMessage;
  Module *Result = ParseBytecodeFile(Filename, &ErrorMessage);
  if (Result) {
    if (OutputFormat == posix && MultipleFiles) {
      std::cout << Filename << ":\n";
    } else if (OutputFormat == bsd && MultipleFiles) {
      std::cout << "\n" << Filename << ":\n";
    } else if (OutputFormat == sysv) {
      std::cout << "\n\nSymbols from " << Filename << ":\n\n"
                << "Name                  Value   Class        Type"
                << "         Size   Line  Section\n";
    }
    DumpSymbolNamesFromModule (Result);
  } else {
    std::cerr << ToolName << ": " << Filename << ": " << ErrorMessage << "\n";
  }
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm symbol table dumper\n");
  ToolName = argv[0];
  if (BSDFormat) OutputFormat = bsd;
  if (POSIXFormat) OutputFormat = posix;
  if (InputFilenames.size () != 1)
    MultipleFiles = true;

  std::for_each (InputFilenames.begin (), InputFilenames.end (),
                 DumpSymbolNamesFromFile);
  return 0;
}
