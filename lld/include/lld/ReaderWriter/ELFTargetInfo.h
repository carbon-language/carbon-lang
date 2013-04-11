//===- lld/ReaderWriter/ELFTargetInfo.h -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_INFO_H
#define LLD_READER_WRITER_ELF_TARGET_INFO_H

#include "lld/Core/PassManager.h"
#include "lld/Core/Pass.h"
#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

#include <memory>

namespace lld {
class DefinedAtom;
class Reference;

namespace elf { template <typename ELFT> class TargetHandler; }

class TargetHandlerBase {
public:
  virtual ~TargetHandlerBase() {}
};

class ELFTargetInfo : public TargetInfo {
public:
  llvm::Triple getTriple() const { return _triple; }
  virtual bool is64Bits() const;
  virtual bool isLittleEndian() const;
  virtual uint64_t getPageSize() const { return 0x1000; }
  uint16_t getOutputType() const { return _outputFileType; }
  uint16_t getOutputMachine() const;
  bool outputYAML() const { return _outputYAML; }
  bool mergeCommonStrings() const { return _mergeCommonStrings; }
  virtual uint64_t getBaseAddress() const { return _baseAddress; }

  /// This controls if undefined atoms need to be created for undefines that are
  /// present in a SharedLibrary. If this option is set, undefined atoms are
  /// created for every undefined symbol that are present in the dynamic table
  /// in the shared library
  bool useShlibUndefines() const { return _useShlibUndefines; }
  /// @}

  /// \brief Does this relocation belong in the dynamic relocation table?
  ///
  /// This table is evaluated at loadtime by the dynamic loader and is
  /// referenced by the DT_RELA{,ENT,SZ} entries in the dynamic table.
  /// Relocations that return true will be added to the dynamic relocation
  /// table.
  virtual bool isDynamicRelocation(const DefinedAtom &,
                                   const Reference &) const {
    return false;
  }
  virtual bool validate(raw_ostream &diagnostics);


  virtual error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                        std::vector<std::unique_ptr<File>> &result) const;

  static std::unique_ptr<ELFTargetInfo> create(llvm::Triple);

  /// \brief Does this relocation belong in the dynamic plt relocation table?
  ///
  /// This table holds all of the relocations used for delayed symbol binding.
  /// It will be evaluated at load time if LD_BIND_NOW is set. It is referenced
  /// by the DT_{JMPREL,PLTRELSZ} entries in the dynamic table.
  /// Relocations that return true will be added to the dynamic plt relocation
  /// table.
  virtual bool isPLTRelocation(const DefinedAtom &, const Reference &) const {
    return false;
  }

  virtual StringRef getInterpreter() const {
    return "/lib64/ld-linux-x86-64.so.2";
  }

  /// \brief Does the output have dynamic sections.
  bool isDynamic() const;

  template <typename ELFT>
  lld::elf::TargetHandler<ELFT> &getTargetHandler() const {
    assert(_targetHandler && "Got null TargetHandler!");
    return static_cast<lld::elf::TargetHandler<ELFT> &>(*_targetHandler.get());
  }

  virtual void addPasses(PassManager &pm) const;

  void setTriple(llvm::Triple trip) { _triple = trip; }
  void setOutputFileType(uint32_t type) { _outputFileType = type; }
  void setOutputYAML(bool v) { _outputYAML = v; }
  void setNoInhibitExec(bool v) { _noInhibitExec = v; }
  void setIsStaticExecutable(bool v) { _isStaticExecutable = v; }
  void setMergeCommonStrings(bool v) { _mergeCommonStrings = v; }
  void setUseShlibUndefines(bool use) { _useShlibUndefines = use; }
  void appendSearchPath(StringRef dirPath) {
    _inputSearchPaths.push_back(dirPath);
  }
  /// Searches directories then calls appendInputFile()
  bool appendLibrary(StringRef libName);

private:
  ELFTargetInfo() LLVM_DELETED_FUNCTION;
protected:
  ELFTargetInfo(llvm::Triple);

  virtual Writer &writer() const;

  uint16_t                           _outputFileType; // e.g ET_EXEC
  llvm::Triple                       _triple;
  std::unique_ptr<TargetHandlerBase> _targetHandler;
  uint64_t                           _baseAddress;
  bool                               _isStaticExecutable;
  bool                               _outputYAML;
  bool                               _noInhibitExec;
  bool                               _mergeCommonStrings;
  bool                               _runLayoutPass;
  bool                               _useShlibUndefines;
  std::vector<StringRef>             _inputSearchPaths;
  llvm::BumpPtrAllocator             _extraStrings;
  mutable std::unique_ptr<Reader>    _elfReader;
  mutable std::unique_ptr<Reader>    _yamlReader;
  mutable std::unique_ptr<Writer>    _writer;
  mutable std::unique_ptr<Reader>    _linkerScriptReader;
};
} // end namespace lld

#endif
