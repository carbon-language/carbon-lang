//===-- Symbolize.h --------------------------------------------- C++ -----===//
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
#ifndef LLVM_DEBUGINFO_SYMBOLIZE_SYMBOLIZE_H
#define LLVM_DEBUGINFO_SYMBOLIZE_SYMBOLIZE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include <map>
#include <memory>
#include <string>

namespace llvm {
namespace symbolize {

using namespace object;
using FunctionNameKind = DILineInfoSpecifier::FunctionNameKind;

class LLVMSymbolizer {
public:
  struct Options {
    FunctionNameKind PrintFunctions;
    bool UseSymbolTable : 1;
    bool Demangle : 1;
    bool RelativeAddresses : 1;
    std::string DefaultArch;
    std::vector<std::string> DsymHints;
    Options(FunctionNameKind PrintFunctions = FunctionNameKind::LinkageName,
            bool UseSymbolTable = true, bool Demangle = true,
            bool RelativeAddresses = false, std::string DefaultArch = "")
        : PrintFunctions(PrintFunctions), UseSymbolTable(UseSymbolTable),
          Demangle(Demangle), RelativeAddresses(RelativeAddresses),
          DefaultArch(DefaultArch) {}
  };

  LLVMSymbolizer(const Options &Opts = Options()) : Opts(Opts) {}
  ~LLVMSymbolizer() {
    flush();
  }

  // Returns the result of symbolization for module name/offset as
  // a string (possibly containing newlines).
  std::string
  symbolizeCode(const std::string &ModuleName, uint64_t ModuleOffset);
  std::string symbolizeInlinedCode(const std::string &ModuleName,
                                   uint64_t ModuleOffset);
  std::string
  symbolizeData(const std::string &ModuleName, uint64_t ModuleOffset);
  void flush();
  static std::string DemangleName(const std::string &Name,
                                  const SymbolizableModule *ModInfo);

private:
  typedef std::pair<ObjectFile*, ObjectFile*> ObjectPair;

  SymbolizableModule *getOrCreateModuleInfo(const std::string &ModuleName);
  ObjectFile *lookUpDsymFile(const std::string &Path,
                             const MachOObjectFile *ExeObj,
                             const std::string &ArchName);

  /// \brief Returns pair of pointers to object and debug object.
  ObjectPair getOrCreateObjects(const std::string &Path,
                                const std::string &ArchName);
  /// \brief Returns a parsed object file for a given architecture in a
  /// universal binary (or the binary itself if it is an object file).
  ObjectFile *getObjectFileFromBinary(Binary *Bin, const std::string &ArchName);

  std::string printDILineInfo(DILineInfo LineInfo) const;
  std::string printDIInliningInfo(DIInliningInfo InlinedContext) const;
  std::string printDIGlobal(DIGlobal Global) const;

  // Owns all the parsed binaries and object files.
  SmallVector<std::unique_ptr<Binary>, 4> ParsedBinariesAndObjects;
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> MemoryBuffers;
  void addOwningBinary(OwningBinary<Binary> OwningBin) {
    std::unique_ptr<Binary> Bin;
    std::unique_ptr<MemoryBuffer> MemBuf;
    std::tie(Bin, MemBuf) = OwningBin.takeBinary();
    ParsedBinariesAndObjects.push_back(std::move(Bin));
    MemoryBuffers.push_back(std::move(MemBuf));
  }

  std::map<std::string, std::unique_ptr<SymbolizableModule>> Modules;
  std::map<std::pair<MachOUniversalBinary *, std::string>, ObjectFile *>
      ObjectFileForArch;
  std::map<std::pair<std::string, std::string>, ObjectPair>
      ObjectPairForPathArch;

  Options Opts;
  static const char kBadString[];
};

} // namespace symbolize
} // namespace llvm

#endif
