//===- lib/ReaderWriter/ELF/Hexagon/HexagonLinkingContext.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "HexagonLinkingContext.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/Core/PassManager.h"
#include "lld/ReaderWriter/Simple.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace lld;
using namespace lld::elf;

namespace {

const uint8_t hexagonInitFiniAtomContent[4] = { 0 };

// HexagonInitFini Atom
class HexagonInitAtom : public InitFiniAtom {
public:
  HexagonInitAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".init_array") {
#ifndef NDEBUG
    _name = "__init_fn_";
    _name += function;
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(hexagonInitFiniAtomContent, 4);
  }

  virtual Alignment alignment() const { return Alignment(2); }
};

class HexagonFiniAtom : public InitFiniAtom {
public:
  HexagonFiniAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".fini_array") {
#ifndef NDEBUG
    _name = "__fini_fn_";
    _name += function;
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(hexagonInitFiniAtomContent, 4);
  }
  virtual Alignment alignment() const { return Alignment(2); }
};

class HexagonInitFiniFile : public SimpleFile {
public:
  HexagonInitFiniFile(const ELFLinkingContext &context)
      : SimpleFile("command line option -init/-fini"), _ordinal(0) {}

  void addInitFunction(StringRef name) {
    Atom *initFuncAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    HexagonInitAtom *initAtom =
           (new (_allocator) HexagonInitAtom(*this, name));
    initAtom->addReferenceELF_Hexagon(llvm::ELF::R_HEX_32, 0, initFuncAtom, 0);
    initAtom->setOrdinal(_ordinal++);
    addAtom(*initFuncAtom);
    addAtom(*initAtom);
  }

  void addFiniFunction(StringRef name) {
    Atom *finiFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    HexagonFiniAtom *finiAtom =
           (new (_allocator) HexagonFiniAtom(*this, name));
    finiAtom->addReferenceELF_Hexagon(llvm::ELF::R_HEX_32, 0, finiFunctionAtom,
                                      0);
    finiAtom->setOrdinal(_ordinal++);
    addAtom(*finiFunctionAtom);
    addAtom(*finiAtom);
  }

private:
  llvm::BumpPtrAllocator _allocator;
  uint64_t _ordinal;
};
}

void elf::HexagonLinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File> > &result) const {
  ELFLinkingContext::createInternalFiles(result);
  std::unique_ptr<HexagonInitFiniFile> initFiniFile(
      new HexagonInitFiniFile(*this));
  for (auto ai : initFunctions())
    initFiniFile->addInitFunction(ai);
  for (auto ai:finiFunctions())
    initFiniFile->addFiniFunction(ai);
  result.push_back(std::move(initFiniFile));
}
