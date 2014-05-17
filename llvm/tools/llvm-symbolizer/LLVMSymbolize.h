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

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
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
    bool UseSymbolTable : 1;
    FunctionNameKind PrintFunctions;
    bool PrintInlining : 1;
    bool Demangle : 1;
    std::string DefaultArch;
    Options(bool UseSymbolTable = true,
            FunctionNameKind PrintFunctions = FunctionNameKind::LinkageName,
            bool PrintInlining = true, bool Demangle = true,
            std::string DefaultArch = "")
        : UseSymbolTable(UseSymbolTable), PrintFunctions(PrintFunctions),
          PrintInlining(PrintInlining), Demangle(Demangle),
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
  std::string
  symbolizeData(const std::string &ModuleName, uint64_t ModuleOffset);
  void flush();
  static std::string DemangleName(const std::string &Name);
private:
  typedef std::pair<Binary*, Binary*> BinaryPair;

  ModuleInfo *getOrCreateModuleInfo(const std::string &ModuleName);
  /// \brief Returns pair of pointers to binary and debug binary.
  BinaryPair getOrCreateBinary(const std::string &Path);
  /// \brief Returns a parsed object file for a given architecture in a
  /// universal binary (or the binary itself if it is an object file).
  ObjectFile *getObjectFileFromBinary(Binary *Bin, const std::string &ArchName);

  std::string printDILineInfo(DILineInfo LineInfo) const;

  // Owns all the parsed binaries and object files.
  SmallVector<std::unique_ptr<Binary>, 4> ParsedBinariesAndObjects;
  // Owns module info objects.
  typedef std::map<std::string, ModuleInfo *> ModuleMapTy;
  ModuleMapTy Modules;
  typedef std::map<std::string, BinaryPair> BinaryMapTy;
  BinaryMapTy BinaryForPath;
  typedef std::map<std::pair<MachOUniversalBinary *, std::string>, ObjectFile *>
      ObjectFileForArchMapTy;
  ObjectFileForArchMapTy ObjectFileForArch;

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
  void addSymbol(const SymbolRef &Symbol);
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
  typedef std::map<SymbolDesc, StringRef> SymbolMapTy;
  SymbolMapTy Functions;
  SymbolMapTy Objects;
};

} // namespace symbolize
} // namespace llvm

#endif // LLVM_SYMBOLIZE_H
