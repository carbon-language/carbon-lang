//===-- sanitizer_symbolize.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of weak hooks from sanitizer_symbolizer_posix_libcdep.cpp.
//
//===----------------------------------------------------------------------===//

#include <inttypes.h>
#include <stdio.h>

#include <string>

#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"

static llvm::symbolize::LLVMSymbolizer *Symbolizer = nullptr;
static bool Demangle = true;
static bool InlineFrames = true;

static llvm::symbolize::LLVMSymbolizer *getDefaultSymbolizer() {
  if (Symbolizer)
    return Symbolizer;
  llvm::symbolize::LLVMSymbolizer::Options Opts;
  Opts.Demangle = Demangle;
  Symbolizer = new llvm::symbolize::LLVMSymbolizer(Opts);
  return Symbolizer;
}

static llvm::symbolize::PrinterConfig getDefaultPrinterConfig() {
  llvm::symbolize::PrinterConfig Config;
  Config.Pretty = false;
  Config.Verbose = false;
  Config.PrintFunctions = true;
  Config.PrintAddress = false;
  Config.SourceContextLines = 0;
  return Config;
}

namespace __sanitizer {
int internal_snprintf(char *buffer, uintptr_t length, const char *format,
                      ...);
}  // namespace __sanitizer

extern "C" {

typedef uint64_t u64;

bool __sanitizer_symbolize_code(const char *ModuleName, uint64_t ModuleOffset,
                                char *Buffer, int MaxLength) {
  std::string Result;
  {
    llvm::raw_string_ostream OS(Result);
    llvm::symbolize::PrinterConfig Config = getDefaultPrinterConfig();
    llvm::symbolize::Request Request{ModuleName, ModuleOffset};
    auto Printer =
        std::make_unique<llvm::symbolize::LLVMPrinter>(OS, OS, Config);

    // TODO: it is neccessary to set proper SectionIndex here.
    // object::SectionedAddress::UndefSection works for only absolute addresses.
    if (InlineFrames) {
      auto ResOrErr = getDefaultSymbolizer()->symbolizeInlinedCode(
          ModuleName,
          {ModuleOffset, llvm::object::SectionedAddress::UndefSection});
      Printer->print(Request,
                     ResOrErr ? ResOrErr.get() : llvm::DIInliningInfo());
    } else {
      auto ResOrErr = getDefaultSymbolizer()->symbolizeCode(
          ModuleName,
          {ModuleOffset, llvm::object::SectionedAddress::UndefSection});
      Printer->print(Request, ResOrErr ? ResOrErr.get() : llvm::DILineInfo());
    }
  }
  return __sanitizer::internal_snprintf(Buffer, MaxLength, "%s",
                                        Result.c_str()) < MaxLength;
}

bool __sanitizer_symbolize_data(const char *ModuleName, uint64_t ModuleOffset,
                                char *Buffer, int MaxLength) {
  std::string Result;
  {
    llvm::symbolize::PrinterConfig Config = getDefaultPrinterConfig();
    llvm::raw_string_ostream OS(Result);
    llvm::symbolize::Request Request{ModuleName, ModuleOffset};
    auto Printer =
        std::make_unique<llvm::symbolize::LLVMPrinter>(OS, OS, Config);

    // TODO: it is neccessary to set proper SectionIndex here.
    // object::SectionedAddress::UndefSection works for only absolute addresses.
    auto ResOrErr = getDefaultSymbolizer()->symbolizeData(
        ModuleName,
        {ModuleOffset, llvm::object::SectionedAddress::UndefSection});
    Printer->print(Request, ResOrErr ? ResOrErr.get() : llvm::DIGlobal());
  }
  return __sanitizer::internal_snprintf(Buffer, MaxLength, "%s",
                                        Result.c_str()) < MaxLength;
}

void __sanitizer_symbolize_flush() {
  if (Symbolizer)
    Symbolizer->flush();
}

int __sanitizer_symbolize_demangle(const char *Name, char *Buffer,
                                   int MaxLength) {
  std::string Result =
      llvm::symbolize::LLVMSymbolizer::DemangleName(Name, nullptr);
  return __sanitizer::internal_snprintf(Buffer, MaxLength, "%s",
                                        Result.c_str()) < MaxLength
             ? static_cast<int>(Result.size() + 1)
             : 0;
}

bool __sanitizer_symbolize_set_demangle(bool Value) {
  // Must be called before LLVMSymbolizer created.
  if (Symbolizer)
    return false;
  Demangle = Value;
  return true;
}

bool __sanitizer_symbolize_set_inline_frames(bool Value) {
  InlineFrames = Value;
  return true;
}

// Override __cxa_atexit and ignore callbacks.
// This prevents crashes in a configuration when the symbolizer
// is built into sanitizer runtime and consequently into the test process.
// LLVM libraries have some global objects destroyed during exit,
// so if the test process triggers any bugs after that, the symbolizer crashes.
// An example stack trace of such crash:
//
// #1  __cxa_throw
// #2  std::__u::__throw_system_error
// #3  std::__u::recursive_mutex::lock
// #4  __sanitizer_llvm::ManagedStaticBase::RegisterManagedStatic
// #5  __sanitizer_llvm::errorToErrorCode
// #6  __sanitizer_llvm::getFileAux
// #7  __sanitizer_llvm::MemoryBuffer::getFileOrSTDIN
// #10 __sanitizer_llvm::symbolize::LLVMSymbolizer::getOrCreateModuleInfo
// #13 __sanitizer::Symbolizer::SymbolizeData
// #14 __tsan::SymbolizeData
// #16 __tsan::ReportRace
// #18 __tsan_write4
// #19 race() () at test/tsan/atexit4.cpp
// #20 cxa_at_exit_wrapper
// #21 __cxa_finalize
// #22 __do_fini
//
// For the standalone llvm-symbolizer this does not hurt,
// we just don't destroy few global objects on exit.
int __cxa_atexit(void (*f)(void *a), void *arg, void *dso) { return 0; }

}  // extern "C"
