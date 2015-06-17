//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_X86_64_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_X86_64_X86_64_TARGET_HANDLER_H

#include "ELFReader.h"
#include "TargetLayout.h"
#include "X86_64LinkingContext.h"
#include "X86_64RelocationHandler.h"
#include "X86_64SectionChunks.h"
#include "lld/Core/Simple.h"

namespace lld {
namespace elf {


class X86_64TargetLayout : public TargetLayout<ELF64LE> {
public:
  X86_64TargetLayout(X86_64LinkingContext &ctx) : TargetLayout(ctx),
    _gotSection(new (this->_allocator) X86_64GOTSection(ctx)) {}

  AtomSection<ELF64LE> *
  createSection(StringRef name, int32_t type,
                DefinedAtom::ContentPermissions permissions,
                TargetLayout<ELF64LE>::SectionOrder order) override {
    if (type == DefinedAtom::typeGOT && name == ".got")
      return _gotSection;
    return TargetLayout<ELF64LE>::createSection(name, type, permissions, order);
  }

  void finalizeOutputSectionLayout() override {
    sortOutputSectionByPriority<ELF64LE>(".init_array");
    sortOutputSectionByPriority<ELF64LE>(".fini_array");
  }

 const X86_64GOTSection &getGOTSection() const { return *_gotSection; }

private:
  uint32_t getPriority(StringRef sectionName) const {
    StringRef priority = sectionName.drop_front().rsplit('.').second;
    uint32_t prio;
    if (priority.getAsInteger(10, prio))
      return std::numeric_limits<uint32_t>::max();
    return prio;
  }

  template <typename T> void sortOutputSectionByPriority(StringRef prefix) {
    OutputSection<T> *section = findOutputSection(prefix);
    if (!section)
      return;
    auto sections = section->sections();
    std::sort(sections.begin(), sections.end(),
              [&](Chunk<T> *lhs, Chunk<T> *rhs) {
                Section<T> *lhsSection = dyn_cast<Section<T>>(lhs);
                Section<T> *rhsSection = dyn_cast<Section<T>>(rhs);
                if (!lhsSection || !rhsSection)
                  return false;
                StringRef lhsName = lhsSection->inputSectionName();
                StringRef rhsName = rhsSection->inputSectionName();
                if (!lhsName.startswith(prefix) || !rhsName.startswith(prefix))
                  return false;
                return getPriority(lhsName) < getPriority(rhsName);
              });
  }

private:
  X86_64GOTSection *_gotSection;
};

class X86_64TargetHandler : public TargetHandler {
public:
  X86_64TargetHandler(X86_64LinkingContext &ctx);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ELFReader<ELFFile<ELF64LE>>>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<ELFReader<DynamicFile<ELF64LE>>>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

protected:
  X86_64LinkingContext &_ctx;
  std::unique_ptr<X86_64TargetLayout> _targetLayout;
  std::unique_ptr<X86_64TargetRelocationHandler> _relocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif
