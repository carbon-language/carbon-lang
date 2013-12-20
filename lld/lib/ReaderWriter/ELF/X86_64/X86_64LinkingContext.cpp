//===- lib/ReaderWriter/ELF/X86_64/X86_64LinkingContext.cpp ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64LinkingContext.h"

#include "lld/Core/File.h"
#include "lld/Core/Instrumentation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "Atoms.h"
#include "X86_64RelocationPass.h"

using namespace lld;
using namespace lld::elf;

using llvm::makeArrayRef;

namespace {
using namespace llvm::ELF;

const uint8_t x86_64InitFiniAtomContent[8] = { 0 };

// X86_64InitFini Atom
class X86_64InitAtom : public InitFiniAtom {
public:
  X86_64InitAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".init_array") {
#ifndef NDEBUG
    _name = "__init_fn_";
    _name += function;
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return makeArrayRef(x86_64InitFiniAtomContent);
  }
  virtual Alignment alignment() const { return Alignment(3); }
};

class X86_64FiniAtom : public InitFiniAtom {
public:
  X86_64FiniAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".fini_array") {
#ifndef NDEBUG
    _name = "__fini_fn_";
    _name += function;
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return makeArrayRef(x86_64InitFiniAtomContent);
  }

  virtual Alignment alignment() const { return Alignment(3); }
};

class X86_64InitFiniFile : public SimpleFile {
public:
  X86_64InitFiniFile(const ELFLinkingContext &context)
      : SimpleFile("command line option -init/-fini"), _ordinal(0) {}

  void addInitFunction(StringRef name) {
    Atom *initFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    X86_64InitAtom *initAtom =
           (new (_allocator) X86_64InitAtom(*this, name));
    initAtom->addReferenceELF_x86_64(llvm::ELF::R_X86_64_64, 0,
                                     initFunctionAtom, 0);
    initAtom->setOrdinal(_ordinal++);
    addAtom(*initFunctionAtom);
    addAtom(*initAtom);
  }

  void addFiniFunction(StringRef name) {
    Atom *finiFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    X86_64FiniAtom *finiAtom =
           (new (_allocator) X86_64FiniAtom(*this, name));
    finiAtom->addReferenceELF_x86_64(llvm::ELF::R_X86_64_64, 0,
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

void elf::X86_64LinkingContext::addPasses(PassManager &pm) {
  auto pass = createX86_64RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

bool elf::X86_64LinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File> > &result) const {
  ELFLinkingContext::createInternalFiles(result);
  std::unique_ptr<X86_64InitFiniFile> initFiniFile(
      new X86_64InitFiniFile(*this));
  for (auto ai : initFunctions())
    initFiniFile->addInitFunction(ai);
  for (auto ai:finiFunctions())
    initFiniFile->addFiniFunction(ai);
  result.push_back(std::move(initFiniFile));
  return true;
}

