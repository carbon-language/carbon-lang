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

#define LLD_CASE(name) .Case(#name, llvm::ELF::name)

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
  HexagonInitFiniFile(const ELFLinkingContext &context):
    SimpleFile(context, "command line option -init/-fini")
  {}

  void addInitFunction(StringRef name) {
    Atom *initFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    HexagonInitAtom *initAtom =
           (new (_allocator) HexagonInitAtom(*this, name));
    initAtom->addReference(llvm::ELF::R_HEX_32, 0, initFunctionAtom, 0);
    initAtom->setOrdinal(_ordinal++);
    addAtom(*initFunctionAtom);
    addAtom(*initAtom);
  }

  void addFiniFunction(StringRef name) {
    Atom *finiFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    HexagonFiniAtom *finiAtom =
           (new (_allocator) HexagonFiniAtom(*this, name));
    finiAtom->addReference(llvm::ELF::R_HEX_32, 0, finiFunctionAtom, 0);
    finiAtom->setOrdinal(_ordinal++);
    addAtom(*finiFunctionAtom);
    addAtom(*finiAtom);
  }

private:
  llvm::BumpPtrAllocator _allocator;
};
}

bool elf::HexagonLinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File> > &result) const {
  ELFLinkingContext::createInternalFiles(result);
  std::unique_ptr<HexagonInitFiniFile> initFiniFile(
      new HexagonInitFiniFile(*this));
  for (auto ai : initFunctions())
    initFiniFile->addInitFunction(ai);
  for (auto ai:finiFunctions())
    initFiniFile->addFiniFunction(ai);
  result.push_back(std::move(initFiniFile));
  return true;
}

ErrorOr<Reference::Kind>
elf::HexagonLinkingContext::relocKindFromString(StringRef str) const {
  int32_t ret = llvm::StringSwitch<int32_t>(str) LLD_CASE(R_HEX_NONE)
      LLD_CASE(R_HEX_B22_PCREL) LLD_CASE(R_HEX_B15_PCREL)
      LLD_CASE(R_HEX_B7_PCREL) LLD_CASE(R_HEX_LO16) LLD_CASE(R_HEX_HI16)
      LLD_CASE(R_HEX_32) LLD_CASE(R_HEX_16) LLD_CASE(R_HEX_8)
      LLD_CASE(R_HEX_GPREL16_0) LLD_CASE(R_HEX_GPREL16_1)
      LLD_CASE(R_HEX_GPREL16_2) LLD_CASE(R_HEX_GPREL16_3) LLD_CASE(R_HEX_HL16)
      LLD_CASE(R_HEX_B13_PCREL) LLD_CASE(R_HEX_B9_PCREL)
      LLD_CASE(R_HEX_B32_PCREL_X) LLD_CASE(R_HEX_32_6_X)
      LLD_CASE(R_HEX_B22_PCREL_X) LLD_CASE(R_HEX_B15_PCREL_X)
      LLD_CASE(R_HEX_B13_PCREL_X) LLD_CASE(R_HEX_B9_PCREL_X)
      LLD_CASE(R_HEX_B7_PCREL_X) LLD_CASE(R_HEX_16_X) LLD_CASE(R_HEX_12_X)
      LLD_CASE(R_HEX_11_X) LLD_CASE(R_HEX_10_X) LLD_CASE(R_HEX_9_X)
      LLD_CASE(R_HEX_8_X) LLD_CASE(R_HEX_7_X) LLD_CASE(R_HEX_6_X)
      LLD_CASE(R_HEX_32_PCREL) LLD_CASE(R_HEX_COPY) LLD_CASE(R_HEX_GLOB_DAT)
      LLD_CASE(R_HEX_JMP_SLOT) LLD_CASE(R_HEX_RELATIVE)
      LLD_CASE(R_HEX_PLT_B22_PCREL) LLD_CASE(R_HEX_GOTREL_LO16)
      LLD_CASE(R_HEX_GOTREL_HI16) LLD_CASE(R_HEX_GOTREL_32)
      LLD_CASE(R_HEX_GOT_LO16) LLD_CASE(R_HEX_GOT_HI16) LLD_CASE(R_HEX_GOT_32)
      LLD_CASE(R_HEX_GOT_16) LLD_CASE(R_HEX_DTPMOD_32)
      LLD_CASE(R_HEX_DTPREL_LO16) LLD_CASE(R_HEX_DTPREL_HI16)
      LLD_CASE(R_HEX_DTPREL_32) LLD_CASE(R_HEX_DTPREL_16)
      LLD_CASE(R_HEX_GD_PLT_B22_PCREL) LLD_CASE(R_HEX_GD_GOT_LO16)
      LLD_CASE(R_HEX_GD_GOT_HI16) LLD_CASE(R_HEX_GD_GOT_32)
      LLD_CASE(R_HEX_GD_GOT_16) LLD_CASE(R_HEX_IE_LO16) LLD_CASE(R_HEX_IE_HI16)
      LLD_CASE(R_HEX_IE_32) LLD_CASE(R_HEX_IE_GOT_LO16)
      LLD_CASE(R_HEX_IE_GOT_HI16) LLD_CASE(R_HEX_IE_GOT_32)
      LLD_CASE(R_HEX_IE_GOT_16) LLD_CASE(R_HEX_TPREL_LO16)
      LLD_CASE(R_HEX_TPREL_HI16) LLD_CASE(R_HEX_TPREL_32)
      LLD_CASE(R_HEX_TPREL_16) LLD_CASE(R_HEX_6_PCREL_X)
      LLD_CASE(R_HEX_GOTREL_32_6_X) LLD_CASE(R_HEX_GOTREL_16_X)
      LLD_CASE(R_HEX_GOTREL_11_X) LLD_CASE(R_HEX_GOT_32_6_X)
      LLD_CASE(R_HEX_GOT_16_X) LLD_CASE(R_HEX_GOT_11_X)
      LLD_CASE(R_HEX_DTPREL_32_6_X) LLD_CASE(R_HEX_DTPREL_16_X)
      LLD_CASE(R_HEX_DTPREL_11_X) LLD_CASE(R_HEX_GD_GOT_32_6_X)
      LLD_CASE(R_HEX_GD_GOT_16_X) LLD_CASE(R_HEX_GD_GOT_11_X)
      LLD_CASE(R_HEX_IE_32_6_X) LLD_CASE(R_HEX_IE_16_X)
      LLD_CASE(R_HEX_IE_GOT_32_6_X) LLD_CASE(R_HEX_IE_GOT_16_X)
      LLD_CASE(R_HEX_IE_GOT_11_X) LLD_CASE(R_HEX_TPREL_32_6_X)
      LLD_CASE(R_HEX_TPREL_16_X) LLD_CASE(R_HEX_TPREL_11_X).Default(-1);

  if (ret == -1)
    return make_error_code(YamlReaderError::illegal_value);
  return ret;
}

#undef LLD_CASE

#define LLD_CASE(name)                                                         \
  case llvm::ELF::name:                                                        \
  return std::string(#name);

ErrorOr<std::string>
elf::HexagonLinkingContext::stringFromRelocKind(int32_t kind) const {
  switch (kind) {
    LLD_CASE(R_HEX_NONE)
    LLD_CASE(R_HEX_B22_PCREL)
    LLD_CASE(R_HEX_B15_PCREL)
    LLD_CASE(R_HEX_B7_PCREL)
    LLD_CASE(R_HEX_LO16)
    LLD_CASE(R_HEX_HI16)
    LLD_CASE(R_HEX_32)
    LLD_CASE(R_HEX_16)
    LLD_CASE(R_HEX_8)
    LLD_CASE(R_HEX_GPREL16_0)
    LLD_CASE(R_HEX_GPREL16_1)
    LLD_CASE(R_HEX_GPREL16_2)
    LLD_CASE(R_HEX_GPREL16_3)
    LLD_CASE(R_HEX_HL16)
    LLD_CASE(R_HEX_B13_PCREL)
    LLD_CASE(R_HEX_B9_PCREL)
    LLD_CASE(R_HEX_B32_PCREL_X)
    LLD_CASE(R_HEX_32_6_X)
    LLD_CASE(R_HEX_B22_PCREL_X)
    LLD_CASE(R_HEX_B15_PCREL_X)
    LLD_CASE(R_HEX_B13_PCREL_X)
    LLD_CASE(R_HEX_B9_PCREL_X)
    LLD_CASE(R_HEX_B7_PCREL_X)
    LLD_CASE(R_HEX_16_X)
    LLD_CASE(R_HEX_12_X)
    LLD_CASE(R_HEX_11_X)
    LLD_CASE(R_HEX_10_X)
    LLD_CASE(R_HEX_9_X)
    LLD_CASE(R_HEX_8_X)
    LLD_CASE(R_HEX_7_X)
    LLD_CASE(R_HEX_6_X)
    LLD_CASE(R_HEX_32_PCREL)
    LLD_CASE(R_HEX_COPY)
    LLD_CASE(R_HEX_GLOB_DAT)
    LLD_CASE(R_HEX_JMP_SLOT)
    LLD_CASE(R_HEX_RELATIVE)
    LLD_CASE(R_HEX_PLT_B22_PCREL)
    LLD_CASE(R_HEX_GOTREL_LO16)
    LLD_CASE(R_HEX_GOTREL_HI16)
    LLD_CASE(R_HEX_GOTREL_32)
    LLD_CASE(R_HEX_GOT_LO16)
    LLD_CASE(R_HEX_GOT_HI16)
    LLD_CASE(R_HEX_GOT_32)
    LLD_CASE(R_HEX_GOT_16)
    LLD_CASE(R_HEX_DTPMOD_32)
    LLD_CASE(R_HEX_DTPREL_LO16)
    LLD_CASE(R_HEX_DTPREL_HI16)
    LLD_CASE(R_HEX_DTPREL_32)
    LLD_CASE(R_HEX_DTPREL_16)
    LLD_CASE(R_HEX_GD_PLT_B22_PCREL)
    LLD_CASE(R_HEX_GD_GOT_LO16)
    LLD_CASE(R_HEX_GD_GOT_HI16)
    LLD_CASE(R_HEX_GD_GOT_32)
    LLD_CASE(R_HEX_GD_GOT_16)
    LLD_CASE(R_HEX_IE_LO16)
    LLD_CASE(R_HEX_IE_HI16)
    LLD_CASE(R_HEX_IE_32)
    LLD_CASE(R_HEX_IE_GOT_LO16)
    LLD_CASE(R_HEX_IE_GOT_HI16)
    LLD_CASE(R_HEX_IE_GOT_32)
    LLD_CASE(R_HEX_IE_GOT_16)
    LLD_CASE(R_HEX_TPREL_LO16)
    LLD_CASE(R_HEX_TPREL_HI16)
    LLD_CASE(R_HEX_TPREL_32)
    LLD_CASE(R_HEX_TPREL_16)
    LLD_CASE(R_HEX_6_PCREL_X)
    LLD_CASE(R_HEX_GOTREL_32_6_X)
    LLD_CASE(R_HEX_GOTREL_16_X)
    LLD_CASE(R_HEX_GOTREL_11_X)
    LLD_CASE(R_HEX_GOT_32_6_X)
    LLD_CASE(R_HEX_GOT_16_X)
    LLD_CASE(R_HEX_GOT_11_X)
    LLD_CASE(R_HEX_DTPREL_32_6_X)
    LLD_CASE(R_HEX_DTPREL_16_X)
    LLD_CASE(R_HEX_DTPREL_11_X)
    LLD_CASE(R_HEX_GD_GOT_32_6_X)
    LLD_CASE(R_HEX_GD_GOT_16_X)
    LLD_CASE(R_HEX_GD_GOT_11_X)
    LLD_CASE(R_HEX_IE_32_6_X)
    LLD_CASE(R_HEX_IE_16_X)
    LLD_CASE(R_HEX_IE_GOT_32_6_X)
    LLD_CASE(R_HEX_IE_GOT_16_X)
    LLD_CASE(R_HEX_IE_GOT_11_X)
    LLD_CASE(R_HEX_TPREL_32_6_X)
    LLD_CASE(R_HEX_TPREL_16_X)
    LLD_CASE(R_HEX_TPREL_11_X)
  }

  return make_error_code(YamlReaderError::illegal_value);
}
