//===- lib/ReaderWriter/ELF/AArch64/AArch64LinkingContext.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64LinkingContext.h"
#include "AArch64RelocationPass.h"
#include "Atoms.h"
#include "lld/Core/File.h"
#include "lld/Core/Instrumentation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace lld;
using namespace lld::elf;

using llvm::makeArrayRef;

namespace {
using namespace llvm::ELF;

const uint8_t AArch64InitFiniAtomContent[8] = {0};

// AArch64InitFini Atom
class AArch64InitAtom : public InitFiniAtom {
public:
  AArch64InitAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".init_array") {
#ifndef NDEBUG
    _name = "__init_fn_";
    _name += function;
#endif
  }
  ArrayRef<uint8_t> rawContent() const override {
    return makeArrayRef(AArch64InitFiniAtomContent);
  }
  Alignment alignment() const override { return Alignment(3); }
};

class AArch64FiniAtom : public InitFiniAtom {
public:
  AArch64FiniAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".fini_array") {
#ifndef NDEBUG
    _name = "__fini_fn_";
    _name += function;
#endif
  }
  ArrayRef<uint8_t> rawContent() const override {
    return makeArrayRef(AArch64InitFiniAtomContent);
  }

  Alignment alignment() const override { return Alignment(3); }
};

class AArch64InitFiniFile : public SimpleFile {
public:
  AArch64InitFiniFile(const ELFLinkingContext &context)
      : SimpleFile("command line option -init/-fini"), _ordinal(0) {}

  void addInitFunction(StringRef name) {
    Atom *initFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    AArch64InitAtom *initAtom = (new (_allocator) AArch64InitAtom(*this, name));
    initAtom->addReferenceELF_AArch64(llvm::ELF::R_AARCH64_ABS64, 0,
                                      initFunctionAtom, 0);
    initAtom->setOrdinal(_ordinal++);
    addAtom(*initFunctionAtom);
    addAtom(*initAtom);
  }

  void addFiniFunction(StringRef name) {
    Atom *finiFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    AArch64FiniAtom *finiAtom = (new (_allocator) AArch64FiniAtom(*this, name));
    finiAtom->addReferenceELF_AArch64(llvm::ELF::R_AARCH64_ABS64, 0,
                                      finiFunctionAtom, 0);
    finiAtom->setOrdinal(_ordinal++);
    addAtom(*finiFunctionAtom);
    addAtom(*finiAtom);
  }

private:
  llvm::BumpPtrAllocator _allocator;
  uint64_t _ordinal;
};

} // end anon namespace

void elf::AArch64LinkingContext::addPasses(PassManager &pm) {
  auto pass = createAArch64RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

void elf::AArch64LinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File>> &result) const {
  ELFLinkingContext::createInternalFiles(result);
  std::unique_ptr<AArch64InitFiniFile> initFiniFile(
      new AArch64InitFiniFile(*this));
  for (auto ai : initFunctions())
    initFiniFile->addInitFunction(ai);
  for (auto ai : finiFunctions())
    initFiniFile->addFiniFunction(ai);
  result.push_back(std::move(initFiniFile));
}
