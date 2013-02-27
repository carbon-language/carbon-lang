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

#include "lld/Core/LinkerOptions.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Pass.h"
#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

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
protected:
  ELFTargetInfo(const LinkerOptions &lo);

public:
  uint16_t getOutputType() const;
  uint16_t getOutputMachine() const;

  virtual StringRef getEntry() const;
  virtual uint64_t getBaseAddress() const { return _options._baseAddress; }

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
  bool isDynamic() const {
    return _options._outputKind == OutputKind::DynamicExecutable ||
           _options._outputKind == OutputKind::Shared;
  }

  virtual ErrorOr<Reader &> getReader(const LinkerInput &input) const;

  virtual ErrorOr<Writer &> getWriter() const;

  static std::unique_ptr<ELFTargetInfo> create(const LinkerOptions &lo);

  template <typename ELFT>
  lld::elf::TargetHandler<ELFT> &getTargetHandler() const {
    assert(_targetHandler && "Got null TargetHandler!");
    return static_cast<lld::elf::TargetHandler<ELFT> &>(*_targetHandler.get());
  }

  virtual void addPasses(PassManager &pm) const;

protected:
  std::unique_ptr<TargetHandlerBase> _targetHandler;
  mutable std::unique_ptr<Reader> _reader;
  mutable std::unique_ptr<Writer> _writer;
};
} // end namespace lld

#endif
