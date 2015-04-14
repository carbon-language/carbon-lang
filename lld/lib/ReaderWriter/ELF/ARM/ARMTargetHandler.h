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
#include "ARMRelocationHandler.h"
#include "ELFReader.h"
#include "TargetLayout.h"

namespace lld {
class ELFLinkingContext;

namespace elf {

class ARMTargetLayout : public TargetLayout<ELF32LE> {
public:
  ARMTargetLayout(ELFLinkingContext &ctx) : TargetLayout<ELF32LE>(ctx) {}

  uint64_t getGOTSymAddr() {
    std::call_once(_gotSymOnce, [this]() {
      if (AtomLayout *gotAtom = this->findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_"))
        _gotSymAddr = gotAtom->_virtualAddr;
    });
    return _gotSymAddr;
  }

  uint64_t getTPOffset() {
    std::call_once(_tpOffOnce, [this]() {
      for (const auto &phdr : *this->_programHeader) {
        if (phdr->p_type == llvm::ELF::PT_TLS) {
          _tpOff = llvm::RoundUpToAlignment(TCB_SIZE, phdr->p_align);
          break;
        }
      }
      assert(_tpOff != 0 && "TLS segment not found");
    });
    return _tpOff;
  }

  bool target1Rel() const { return this->_ctx.armTarget1Rel(); }

private:
  // TCB block size of the TLS.
  enum { TCB_SIZE = 0x8 };

private:
  uint64_t _gotSymAddr = 0;
  uint64_t _tpOff = 0;
  std::once_flag _gotSymOnce;
  std::once_flag _tpOffOnce;
};

class ARMTargetHandler final : public TargetHandler {
  typedef ELFReader<ARMELFFile> ObjReader;
  typedef ELFReader<DynamicFile<ELF32LE>> DSOReader;

public:
  ARMTargetHandler(ARMLinkingContext &ctx);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ObjReader>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<DSOReader>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  ARMLinkingContext &_ctx;
  std::unique_ptr<ARMTargetLayout> _targetLayout;
  std::unique_ptr<ARMTargetRelocationHandler> _relocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_TARGET_HANDLER_H
