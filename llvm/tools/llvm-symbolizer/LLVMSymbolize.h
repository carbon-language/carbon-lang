//===-- LLVMSymbolize.h ----------------------------------------- C++ -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Header for LLVM symbolization library.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SYMBOLIZE_H
#define LLVM_SYMBOLIZE_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include <map>
#include <string>

namespace llvm {

using namespace object;

namespace symbolize {

class ModuleInfo;

class LLVMSymbolizer {
public:
  struct Options {
    bool UseSymbolTable : 1;
    bool PrintFunctions : 1;
    bool PrintInlining : 1;
    bool Demangle : 1;
    Options(bool UseSymbolTable = true, bool PrintFunctions = true,
            bool PrintInlining = true, bool Demangle = true)
        : UseSymbolTable(UseSymbolTable), PrintFunctions(PrintFunctions),
          PrintInlining(PrintInlining), Demangle(Demangle) {
    }
  };

  LLVMSymbolizer(const Options &Opts = Options()) : Opts(Opts) {}

  // Returns the result of symbolization for module name/offset as
  // a string (possibly containing newlines).
  std::string
  symbolizeCode(const std::string &ModuleName, uint64_t ModuleOffset);
  std::string
  symbolizeData(const std::string &ModuleName, uint64_t ModuleOffset);
  void flush();
private:
  ModuleInfo *getOrCreateModuleInfo(const std::string &ModuleName);
  std::string printDILineInfo(DILineInfo LineInfo) const;
  void DemangleName(std::string &Name) const;

  typedef std::map<std::string, ModuleInfo *> ModuleMapTy;
  ModuleMapTy Modules;
  Options Opts;
  static const char kBadString[];
};

class ModuleInfo {
public:
  ModuleInfo(ObjectFile *Obj, DIContext *DICtx);

  DILineInfo symbolizeCode(uint64_t ModuleOffset,
                           const LLVMSymbolizer::Options &Opts) const;
  DIInliningInfo symbolizeInlinedCode(
      uint64_t ModuleOffset, const LLVMSymbolizer::Options &Opts) const;
  bool symbolizeData(uint64_t ModuleOffset, std::string &Name, uint64_t &Start,
                     uint64_t &Size) const;

private:
  bool getNameFromSymbolTable(SymbolRef::Type Type, uint64_t Address,
                              std::string &Name, uint64_t &Addr,
                              uint64_t &Size) const;
  OwningPtr<ObjectFile> Module;
  OwningPtr<DIContext> DebugInfoContext;

  struct SymbolDesc {
    uint64_t Addr;
    uint64_t AddrEnd;
    friend bool operator<(const SymbolDesc &s1, const SymbolDesc &s2) {
      return s1.AddrEnd <= s2.Addr;
    }
  };
  typedef std::map<SymbolDesc, StringRef> SymbolMapTy;
  SymbolMapTy Functions;
  SymbolMapTy Objects;
};

} // namespace symbolize
} // namespace llvm

#endif // LLVM_SYMBOLIZE_H
