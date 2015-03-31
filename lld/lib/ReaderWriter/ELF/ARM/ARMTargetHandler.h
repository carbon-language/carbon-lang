//===--------- lib/ReaderWriter/ELF/ARM/ARMTargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARM_ARM_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_ARM_ARM_TARGET_HANDLER_H

#include "ARMELFFile.h"
#include "ARMELFReader.h"
#include "ARMRelocationHandler.h"
#include "DefaultTargetHandler.h"
#include "TargetLayout.h"
#include "llvm/ADT/Optional.h"

namespace lld {
namespace elf {
class ARMLinkingContext;

template <class ELFT> class ARMTargetLayout : public TargetLayout<ELFT> {
public:
  ARMTargetLayout(ARMLinkingContext &ctx) : TargetLayout<ELFT>(ctx) {}

  uint64_t getGOTSymAddr() {
    if (!_gotSymAddr.hasValue()) {
      auto gotAtomIter = this->findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
      _gotSymAddr = (gotAtomIter != this->absoluteAtoms().end())
                        ? (*gotAtomIter)->_virtualAddr
                        : 0;
    }
    return *_gotSymAddr;
  }

  uint64_t getTPOffset() {
    if (_tpOff.hasValue())
      return *_tpOff;

    for (const auto &phdr : *this->_programHeader) {
      if (phdr->p_type == llvm::ELF::PT_TLS) {
        _tpOff = llvm::RoundUpToAlignment(TCB_SIZE, phdr->p_align);
        return *_tpOff;
      }
    }
    llvm_unreachable("TLS segment not found");
  }

private:
  // TCB block size of the TLS.
  enum { TCB_SIZE = 0x8 };

private:
  // Cached value of the GOT origin symbol address.
  llvm::Optional<uint64_t> _gotSymAddr;

  // Cached value of the TLS offset from the $tp pointer.
  llvm::Optional<uint64_t> _tpOff;
};

class ARMTargetHandler final : public DefaultTargetHandler<ARMELFType> {
public:
  ARMTargetHandler(ARMLinkingContext &ctx);

  ARMTargetLayout<ARMELFType> &getTargetLayout() override {
    return *_armTargetLayout;
  }

  void registerRelocationNames(Registry &registry) override;

  const ARMTargetRelocationHandler &getRelocationHandler() const override {
    return *_armRelocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return std::unique_ptr<Reader>(new ARMELFObjectReader(_ctx));
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return std::unique_ptr<Reader>(new ARMELFDSOReader(_ctx));
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  static const Registry::KindStrings kindStrings[];
  ARMLinkingContext &_ctx;
  std::unique_ptr<ARMTargetLayout<ARMELFType>> _armTargetLayout;
  std::unique_ptr<ARMTargetRelocationHandler> _armRelocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_TARGET_HANDLER_H
