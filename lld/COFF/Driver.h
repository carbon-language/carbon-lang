//===- Driver.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DRIVER_H
#define LLD_COFF_DRIVER_H

#include "Memory.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include <memory>
#include <system_error>
#include <vector>

namespace lld {
namespace coff {

using llvm::COFF::MachineTypes;
class InputFile;

ErrorOr<std::unique_ptr<llvm::opt::InputArgList>>
parseArgs(int Argc, const char *Argv[]);

std::error_code parseDirectives(StringRef S,
                                std::vector<std::unique_ptr<InputFile>> *Res,
                                StringAllocator *Alloc);

// Functions below this line are defined in DriverUtils.cpp.

void printHelp(const char *Argv0);

// "ENV" environment variable-aware file finders.
std::string findLib(StringRef Filename);
std::string findFile(StringRef Filename);

// For /machine option.
ErrorOr<MachineTypes> getMachineType(llvm::opt::InputArgList *Args);

// Parses a string in the form of "<integer>[,<integer>]".
std::error_code parseNumbers(StringRef Arg, uint64_t *Addr,
                             uint64_t *Size = nullptr);

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

} // namespace coff
} // namespace lld

#endif
