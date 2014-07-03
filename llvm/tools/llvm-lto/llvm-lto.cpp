//===-- llvm-lto: a simple command-line program to link modules with LTO --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes in a list of bitcode files, links them, performs link-time
// optimization, and outputs an object file.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
DisableOpt("disable-opt", cl::init(false),
  cl::desc("Do not run any optimization passes"));

static cl::opt<bool>
DisableInline("disable-inlining", cl::init(false),
  cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
DisableGVNLoadPRE("disable-gvn-loadpre", cl::init(false),
  cl::desc("Do not run the GVN load PRE pass"));

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
  cl::desc("<input bitcode files>"));

static cl::opt<std::string>
OutputFilename("o", cl::init(""),
  cl::desc("Override output filename"),
  cl::value_desc("filename"));

static cl::list<std::string>
ExportedSymbols("exported-symbol",
  cl::desc("Symbol to export from the resulting object file"),
  cl::ZeroOrMore);

static cl::list<std::string>
DSOSymbols("dso-symbol",
  cl::desc("Symbol to put in the symtab in the resulting dso"),
  cl::ZeroOrMore);

namespace {
struct ModuleInfo {
  std::vector<bool> CanBeHidden;
};
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm LTO linker\n");

  // Initialize the configured targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // set up the TargetOptions for the machine
  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();

  unsigned BaseArg = 0;

  LTOCodeGenerator CodeGen;

  switch (RelocModel) {
  case Reloc::Static:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_STATIC);
    break;
  case Reloc::PIC_:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_DYNAMIC);
    break;
  case Reloc::DynamicNoPIC:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC);
    break;
  default:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_DEFAULT);
  }

  CodeGen.setDebugInfo(LTO_DEBUG_MODEL_DWARF);
  CodeGen.setTargetOptions(Options);

  llvm::StringSet<llvm::MallocAllocator> DSOSymbolsSet;
  for (unsigned i = 0; i < DSOSymbols.size(); ++i)
    DSOSymbolsSet.insert(DSOSymbols[i]);

  std::vector<std::string> KeptDSOSyms;

  for (unsigned i = BaseArg; i < InputFilenames.size(); ++i) {
    std::string error;
    std::unique_ptr<LTOModule> Module(
        LTOModule::createFromFile(InputFilenames[i].c_str(), Options, error));
    if (!error.empty()) {
      errs() << argv[0] << ": error loading file '" << InputFilenames[i]
             << "': " << error << "\n";
      return 1;
    }


    if (!CodeGen.addModule(Module.get(), error)) {
      errs() << argv[0] << ": error adding file '" << InputFilenames[i]
             << "': " << error << "\n";
      return 1;
    }

    unsigned NumSyms = Module->getSymbolCount();
    for (unsigned I = 0; I < NumSyms; ++I) {
      StringRef Name = Module->getSymbolName(I);
      if (!DSOSymbolsSet.count(Name))
        continue;
      lto_symbol_attributes Attrs = Module->getSymbolAttributes(I);
      unsigned Scope = Attrs & LTO_SYMBOL_SCOPE_MASK;
      if (Scope != LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN)
        KeptDSOSyms.push_back(Name);
    }
  }

  // Add all the exported symbols to the table of symbols to preserve.
  for (unsigned i = 0; i < ExportedSymbols.size(); ++i)
    CodeGen.addMustPreserveSymbol(ExportedSymbols[i].c_str());

  // Add all the dso symbols to the table of symbols to expose.
  for (unsigned i = 0; i < KeptDSOSyms.size(); ++i)
    CodeGen.addMustPreserveSymbol(KeptDSOSyms[i].c_str());

  std::string attrs;
  for (unsigned i = 0; i < MAttrs.size(); ++i) {
    if (i > 0)
      attrs.append(",");
    attrs.append(MAttrs[i]);
  }

  if (!attrs.empty())
    CodeGen.setAttr(attrs.c_str());

  if (!OutputFilename.empty()) {
    size_t len = 0;
    std::string ErrorInfo;
    const void *Code = CodeGen.compile(&len, DisableOpt, DisableInline,
                                       DisableGVNLoadPRE, ErrorInfo);
    if (!Code) {
      errs() << argv[0]
             << ": error compiling the code: " << ErrorInfo << "\n";
      return 1;
    }

    raw_fd_ostream FileStream(OutputFilename.c_str(), ErrorInfo,
                              sys::fs::F_None);
    if (!ErrorInfo.empty()) {
      errs() << argv[0] << ": error opening the file '" << OutputFilename
             << "': " << ErrorInfo << "\n";
      return 1;
    }

    FileStream.write(reinterpret_cast<const char *>(Code), len);
  } else {
    std::string ErrorInfo;
    const char *OutputName = nullptr;
    if (!CodeGen.compile_to_file(&OutputName, DisableOpt, DisableInline,
                                 DisableGVNLoadPRE, ErrorInfo)) {
      errs() << argv[0]
             << ": error compiling the code: " << ErrorInfo
             << "\n";
      return 1;
    }

    outs() << "Wrote native object file '" << OutputName << "'\n";
  }

  return 0;
}
