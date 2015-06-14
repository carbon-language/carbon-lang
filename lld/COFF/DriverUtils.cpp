//===- DriverUtils.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the driver. Because there
// are so many small functions, we created this separate file to make
// Driver.cpp less cluttered.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/COFF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm::COFF;
using namespace llvm;
using llvm::cl::ExpandResponseFiles;
using llvm::cl::TokenizeWindowsCommandLine;
using llvm::sys::Process;

namespace lld {
namespace coff {

// Returns /machine's value.
ErrorOr<MachineTypes> getMachineType(llvm::opt::InputArgList *Args) {
  if (auto *Arg = Args->getLastArg(OPT_machine)) {
    StringRef S(Arg->getValue());
    MachineTypes MT = StringSwitch<MachineTypes>(S.lower())
                          .Case("arm", IMAGE_FILE_MACHINE_ARMNT)
                          .Case("x64", IMAGE_FILE_MACHINE_AMD64)
                          .Case("x86", IMAGE_FILE_MACHINE_I386)
                          .Default(IMAGE_FILE_MACHINE_UNKNOWN);
    if (MT == IMAGE_FILE_MACHINE_UNKNOWN) {
      llvm::errs() << "unknown /machine argument" << S << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
    return MT;
  }
  return IMAGE_FILE_MACHINE_UNKNOWN;
}

// Parses a string in the form of "<integer>[,<integer>]".
std::error_code parseNumbers(StringRef Arg, uint64_t *Addr, uint64_t *Size) {
  StringRef S1, S2;
  std::tie(S1, S2) = Arg.split(',');
  if (S1.getAsInteger(0, *Addr)) {
    llvm::errs() << "invalid number: " << S1 << "\n";
    return make_error_code(LLDError::InvalidOption);
  }
  if (Size && !S2.empty() && S2.getAsInteger(0, *Size)) {
    llvm::errs() << "invalid number: " << S2 << "\n";
    return make_error_code(LLDError::InvalidOption);
  }
  return std::error_code();
}

// Parses a string in the form of "<integer>[.<integer>]".
// If second number is not present, Minor is set to 0.
std::error_code parseVersion(StringRef Arg, uint32_t *Major, uint32_t *Minor) {
  StringRef S1, S2;
  std::tie(S1, S2) = Arg.split('.');
  if (S1.getAsInteger(0, *Major)) {
    llvm::errs() << "invalid number: " << S1 << "\n";
    return make_error_code(LLDError::InvalidOption);
  }
  *Minor = 0;
  if (!S2.empty() && S2.getAsInteger(0, *Minor)) {
    llvm::errs() << "invalid number: " << S2 << "\n";
    return make_error_code(LLDError::InvalidOption);
  }
  return std::error_code();
}

// Parses a string in the form of "<subsystem>[,<integer>[.<integer>]]".
std::error_code parseSubsystem(StringRef Arg, WindowsSubsystem *Sys,
                               uint32_t *Major, uint32_t *Minor) {
  StringRef SysStr, Ver;
  std::tie(SysStr, Ver) = Arg.split(',');
  *Sys = StringSwitch<WindowsSubsystem>(SysStr.lower())
    .Case("boot_application", IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION)
    .Case("console", IMAGE_SUBSYSTEM_WINDOWS_CUI)
    .Case("efi_application", IMAGE_SUBSYSTEM_EFI_APPLICATION)
    .Case("efi_boot_service_driver", IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER)
    .Case("efi_rom", IMAGE_SUBSYSTEM_EFI_ROM)
    .Case("efi_runtime_driver", IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER)
    .Case("native", IMAGE_SUBSYSTEM_NATIVE)
    .Case("posix", IMAGE_SUBSYSTEM_POSIX_CUI)
    .Case("windows", IMAGE_SUBSYSTEM_WINDOWS_GUI)
    .Default(IMAGE_SUBSYSTEM_UNKNOWN);
  if (*Sys == IMAGE_SUBSYSTEM_UNKNOWN) {
    llvm::errs() << "unknown subsystem: " << SysStr << "\n";
    return make_error_code(LLDError::InvalidOption);
  }
  if (!Ver.empty())
    if (auto EC = parseVersion(Ver, Major, Minor))
      return EC;
  return std::error_code();
}

// Parses a string in the form of "key=value" and check
// if value matches previous values for the same key.
std::error_code checkFailIfMismatch(llvm::opt::InputArgList *Args) {
  for (auto *Arg : Args->filtered(OPT_failifmismatch)) {
    StringRef K, V;
    std::tie(K, V) = StringRef(Arg->getValue()).split('=');
    if (K.empty() || V.empty()) {
      llvm::errs() << "/failifmismatch: invalid argument: "
                   << Arg->getValue() << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
    StringRef Existing = Config->MustMatch[K];
    if (!Existing.empty() && V != Existing) {
      llvm::errs() << "/failifmismatch: mismatch detected: "
                   << Existing << " and " << V
                   << " for key " << K << "\n";
      return make_error_code(LLDError::InvalidOption);
    }
    Config->MustMatch[K] = V;
  }
  return std::error_code();
}

// Convert Windows resource files (.res files) to a .obj file
// using cvtres.exe.
ErrorOr<std::unique_ptr<MemoryBuffer>>
convertResToCOFF(const std::vector<MemoryBufferRef> &MBs) {
  // Find cvtres.exe.
  std::string Prog = "cvtres.exe";
  ErrorOr<std::string> ExeOrErr = llvm::sys::findProgramByName(Prog);
  if (auto EC = ExeOrErr.getError()) {
    llvm::errs() << "unable to find " << Prog << " in PATH: "
                 << EC.message() << "\n";
    return make_error_code(LLDError::InvalidOption);
  }
  llvm::BumpPtrAllocator Alloc;
  llvm::BumpPtrStringSaver S(Alloc);
  const char *Exe = S.save(ExeOrErr.get());

  // Create an output file path.
  SmallString<128> Path;
  if (llvm::sys::fs::createTemporaryFile("resource", "obj", Path))
    return make_error_code(LLDError::InvalidOption);

  // Execute cvtres.exe.
  std::vector<const char *> Args;
  Args.push_back(Exe);
  Args.push_back("/machine:x64");
  Args.push_back("/readonly");
  Args.push_back("/nologo");
  Args.push_back(S.save("/out:" + Path));
  for (MemoryBufferRef MB : MBs)
    Args.push_back(S.save(MB.getBufferIdentifier()));
  Args.push_back(nullptr);
  llvm::errs() << "\n";
  if (llvm::sys::ExecuteAndWait(Args[0], Args.data()) != 0) {
    llvm::errs() << Exe << " failed\n";
    return make_error_code(LLDError::InvalidOption);
  }
  return MemoryBuffer::getFile(Path);
}

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)    \
  {                                                                    \
    X1, X2, X9, X10, OPT_##ID, llvm::opt::Option::KIND##Class, X8, X7, \
    OPT_##GROUP, OPT_##ALIAS, X6                                       \
  },
#include "Options.inc"
#undef OPTION
};

class COFFOptTable : public llvm::opt::OptTable {
public:
  COFFOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable), true) {}
};

// Parses a given list of options.
ErrorOr<std::unique_ptr<llvm::opt::InputArgList>>
ArgParser::parse(std::vector<const char *> Argv) {
  // First, replace respnose files (@<file>-style options).
  auto ArgvOrErr = replaceResponseFiles(Argv);
  if (auto EC = ArgvOrErr.getError()) {
    llvm::errs() << "error while reading response file: " << EC.message()
                 << "\n";
    return EC;
  }
  Argv = std::move(ArgvOrErr.get());

  // Make InputArgList from string vectors.
  COFFOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;
  std::unique_ptr<llvm::opt::InputArgList> Args(Table.ParseArgs(
      &Argv[0], &Argv[0] + Argv.size(), MissingIndex, MissingCount));
  if (MissingCount) {
    llvm::errs() << "missing arg value for \""
                 << Args->getArgString(MissingIndex)
                 << "\", expected " << MissingCount
                 << (MissingCount == 1 ? " argument.\n" : " arguments.\n");
    return make_error_code(LLDError::InvalidOption);
  }
  for (auto *Arg : Args->filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getSpelling() << "\n";
  return std::move(Args);
}

ErrorOr<std::unique_ptr<llvm::opt::InputArgList>>
ArgParser::parse(int Argc, const char *Argv[]) {
  std::vector<const char *> V(Argv + 1, Argv + Argc);
  return parse(V);
}

std::vector<const char *> ArgParser::tokenize(StringRef S) {
  SmallVector<const char *, 16> Tokens;
  BumpPtrStringSaver Saver(AllocAux);
  llvm::cl::TokenizeWindowsCommandLine(S, Saver, Tokens);
  return std::vector<const char *>(Tokens.begin(), Tokens.end());
}

// Creates a new command line by replacing options starting with '@'
// character. '@<filename>' is replaced by the file's contents.
ErrorOr<std::vector<const char *>>
ArgParser::replaceResponseFiles(std::vector<const char *> Argv) {
  SmallVector<const char *, 256> Tokens(&Argv[0], &Argv[0] + Argv.size());
  BumpPtrStringSaver Saver(AllocAux);
  ExpandResponseFiles(Saver, TokenizeWindowsCommandLine, Tokens);
  return std::vector<const char *>(Tokens.begin(), Tokens.end());
}

void printHelp(const char *Argv0) {
  COFFOptTable Table;
  Table.PrintHelp(llvm::outs(), Argv0, "LLVM Linker", false);
}

} // namespace coff
} // namespace lld
