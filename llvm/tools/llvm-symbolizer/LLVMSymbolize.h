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
#ifndef LLVM_TOOLS_LLVM_SYMBOLIZER_LLVMSYMBOLIZE_H
#define LLVM_TOOLS_LLVM_SYMBOLIZER_LLVMSYMBOLIZE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"
#include <map>
#include <memory>
#include <string>

namespace llvm {

typedef DILineInfoSpecifier::FunctionNameKind FunctionNameKind;
using namespace object;

namespace symbolize {

class ModuleInfo;

class LLVMSymbolizer {
public:
  struct Options {
    FunctionNameKind PrintFunctions;
    bool UseSymbolTable : 1;
    bool PrintInlining : 1;
    bool Demangle : 1;
    bool RelativeAddresses : 1;
    std::string DefaultArch;
    std::vector<std::string> DsymHints;
    Options(FunctionNameKind PrintFunctions = FunctionNameKind::LinkageName,
            bool UseSymbolTable = true, bool PrintInlining = true,
            bool Demangle = true, bool RelativeAddresses = false,
            std::string DefaultArch = "")
        : PrintFunctions(PrintFunctions), UseSymbolTable(UseSymbolTable),
          PrintInlining(PrintInlining), Demangle(Demangle),
          RelativeAddresses(RelativeAddresses), DefaultArch(DefaultArch) {}
  };

  LLVMSymbolizer(const Options &Opts = Options()) : Opts(Opts) {}
  ~LLVMSymbolizer() {
    flush();
  }

  // Returns the result of symbolization for module name/offset as
  // a string (possibly containing newlines).
  std::string
  symbolizeCode(const std::string &ModuleName, uint64_t ModuleOffset);
  std::string
  symbolizeData(const std::string &ModuleName, uint64_t ModuleOffset);
  void flush();
  static std::string DemangleName(const std::string &Name, ModuleInfo *ModInfo);

private:
  typedef std::pair<ObjectFile*, ObjectFile*> ObjectPair;

  ModuleInfo *getOrCreateModuleInfo(const std::string &ModuleName);
  ObjectFile *lookUpDsymFile(const std::string &Path, const MachOObjectFile *ExeObj,
                             const std::string &ArchName);

  /// \brief Returns pair of pointers to object and debug object.
  ObjectPair getOrCreateObjects(const std::string &Path,
                                const std::string &ArchName);
  /// \brief Returns a parsed object file for a given architecture in a
  /// universal binary (or the binary itself if it is an object file).
  ObjectFile *getObjectFileFromBinary(Binary *Bin, const std::string &ArchName);

  std::string printDILineInfo(DILineInfo LineInfo, ModuleInfo *ModInfo) const;

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

  // Owns module info objects.
  std::map<std::string, ModuleInfo *> Modules;
  std::map<std::pair<MachOUniversalBinary *, std::string>, ObjectFile *>
      ObjectFileForArch;
  std::map<std::pair<std::string, std::string>, ObjectPair>
      ObjectPairForPathArch;

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

  // Return true if this is a 32-bit x86 PE COFF module.
  bool isWin32Module() const;

  // Returns the preferred base of the module, i.e. where the loader would place
  // it in memory assuming there were no conflicts.
  uint64_t getModulePreferredBase() const;

private:
  bool getNameFromSymbolTable(SymbolRef::Type Type, uint64_t Address,
                              std::string &Name, uint64_t &Addr,
                              uint64_t &Size) const;
  // For big-endian PowerPC64 ELF, OpdAddress is the address of the .opd
  // (function descriptor) section and OpdExtractor refers to its contents.
  void addSymbol(const SymbolRef &Symbol, uint64_t SymbolSize,
                 DataExtractor *OpdExtractor = nullptr,
                 uint64_t OpdAddress = 0);
  ObjectFile *Module;
  std::unique_ptr<DIContext> DebugInfoContext;

  struct SymbolDesc {
    uint64_t Addr;
    // If size is 0, assume that symbol occupies the whole memory range up to
    // the following symbol.
    uint64_t Size;
    friend bool operator<(const SymbolDesc &s1, const SymbolDesc &s2) {
      return s1.Addr < s2.Addr;
    }
  };
  std::map<SymbolDesc, StringRef> Functions;
  std::map<SymbolDesc, StringRef> Objects;
};

} // namespace symbolize
} // namespace llvm

#endif
