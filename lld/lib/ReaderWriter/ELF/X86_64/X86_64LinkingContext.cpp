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
    return ArrayRef<uint8_t>(x86_64InitFiniAtomContent, 8);
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
    return ArrayRef<uint8_t>(x86_64InitFiniAtomContent, 8);
  }

  virtual Alignment alignment() const { return Alignment(3); }
};

class X86_64InitFiniFile : public SimpleFile {
public:
  X86_64InitFiniFile(const ELFLinkingContext &context)
      : SimpleFile(context, "command line option -init/-fini"), _ordinal(0) {}

  void addInitFunction(StringRef name) {
    Atom *initFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    X86_64InitAtom *initAtom =
           (new (_allocator) X86_64InitAtom(*this, name));
    initAtom->addReference(llvm::ELF::R_X86_64_64, 0, initFunctionAtom, 0);
    initAtom->setOrdinal(_ordinal++);
    addAtom(*initFunctionAtom);
    addAtom(*initAtom);
  }

  void addFiniFunction(StringRef name) {
    Atom *finiFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    X86_64FiniAtom *finiAtom =
           (new (_allocator) X86_64FiniAtom(*this, name));
    finiAtom->addReference(llvm::ELF::R_X86_64_64, 0, finiFunctionAtom, 0);
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

#define LLD_CASE(name) .Case(#name, llvm::ELF::name)

ErrorOr<Reference::Kind>
elf::X86_64LinkingContext::relocKindFromString(StringRef str) const {
  int32_t ret = llvm::StringSwitch<int32_t>(str) LLD_CASE(R_X86_64_NONE)
      LLD_CASE(R_X86_64_64) LLD_CASE(R_X86_64_PC32) LLD_CASE(R_X86_64_GOT32)
      LLD_CASE(R_X86_64_PLT32) LLD_CASE(R_X86_64_COPY)
      LLD_CASE(R_X86_64_GLOB_DAT) LLD_CASE(R_X86_64_JUMP_SLOT)
      LLD_CASE(R_X86_64_RELATIVE) LLD_CASE(R_X86_64_GOTPCREL)
      LLD_CASE(R_X86_64_32) LLD_CASE(R_X86_64_32S) LLD_CASE(R_X86_64_16)
      LLD_CASE(R_X86_64_PC16) LLD_CASE(R_X86_64_8) LLD_CASE(R_X86_64_PC8)
      LLD_CASE(R_X86_64_DTPMOD64) LLD_CASE(R_X86_64_DTPOFF64)
      LLD_CASE(R_X86_64_TPOFF64) LLD_CASE(R_X86_64_TLSGD)
      LLD_CASE(R_X86_64_TLSLD) LLD_CASE(R_X86_64_DTPOFF32)
      LLD_CASE(R_X86_64_GOTTPOFF) LLD_CASE(R_X86_64_TPOFF32)
      LLD_CASE(R_X86_64_PC64) LLD_CASE(R_X86_64_GOTOFF64)
      LLD_CASE(R_X86_64_GOTPC32) LLD_CASE(R_X86_64_GOT64)
      LLD_CASE(R_X86_64_GOTPCREL64) LLD_CASE(R_X86_64_GOTPC64)
      LLD_CASE(R_X86_64_GOTPLT64) LLD_CASE(R_X86_64_PLTOFF64)
      LLD_CASE(R_X86_64_SIZE32) LLD_CASE(R_X86_64_SIZE64)
      LLD_CASE(R_X86_64_GOTPC32_TLSDESC) LLD_CASE(R_X86_64_TLSDESC_CALL)
      LLD_CASE(R_X86_64_TLSDESC) LLD_CASE(R_X86_64_IRELATIVE)
          .Case("LLD_R_X86_64_GOTRELINDEX", LLD_R_X86_64_GOTRELINDEX)
          .Default(-1);

  if (ret == -1)
    return make_error_code(YamlReaderError::illegal_value);
  return ret;
}

#undef LLD_CASE

#define LLD_CASE(name)                                                         \
  case llvm::ELF::name:                                                        \
  return std::string(#name);

ErrorOr<std::string>
elf::X86_64LinkingContext::stringFromRelocKind(Reference::Kind kind) const {
  switch (kind) {
    LLD_CASE(R_X86_64_NONE)
    LLD_CASE(R_X86_64_64)
    LLD_CASE(R_X86_64_PC32)
    LLD_CASE(R_X86_64_GOT32)
    LLD_CASE(R_X86_64_PLT32)
    LLD_CASE(R_X86_64_COPY)
    LLD_CASE(R_X86_64_GLOB_DAT)
    LLD_CASE(R_X86_64_JUMP_SLOT)
    LLD_CASE(R_X86_64_RELATIVE)
    LLD_CASE(R_X86_64_GOTPCREL)
    LLD_CASE(R_X86_64_32)
    LLD_CASE(R_X86_64_32S)
    LLD_CASE(R_X86_64_16)
    LLD_CASE(R_X86_64_PC16)
    LLD_CASE(R_X86_64_8)
    LLD_CASE(R_X86_64_PC8)
    LLD_CASE(R_X86_64_DTPMOD64)
    LLD_CASE(R_X86_64_DTPOFF64)
    LLD_CASE(R_X86_64_TPOFF64)
    LLD_CASE(R_X86_64_TLSGD)
    LLD_CASE(R_X86_64_TLSLD)
    LLD_CASE(R_X86_64_DTPOFF32)
    LLD_CASE(R_X86_64_GOTTPOFF)
    LLD_CASE(R_X86_64_TPOFF32)
    LLD_CASE(R_X86_64_PC64)
    LLD_CASE(R_X86_64_GOTOFF64)
    LLD_CASE(R_X86_64_GOTPC32)
    LLD_CASE(R_X86_64_GOT64)
    LLD_CASE(R_X86_64_GOTPCREL64)
    LLD_CASE(R_X86_64_GOTPC64)
    LLD_CASE(R_X86_64_GOTPLT64)
    LLD_CASE(R_X86_64_PLTOFF64)
    LLD_CASE(R_X86_64_SIZE32)
    LLD_CASE(R_X86_64_SIZE64)
    LLD_CASE(R_X86_64_GOTPC32_TLSDESC)
    LLD_CASE(R_X86_64_TLSDESC_CALL)
    LLD_CASE(R_X86_64_TLSDESC)
    LLD_CASE(R_X86_64_IRELATIVE)
  case LLD_R_X86_64_GOTRELINDEX:
    return std::string("LLD_R_X86_64_GOTRELINDEX");
  }

  return make_error_code(YamlReaderError::illegal_value);
}

