//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetInfo.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64TargetInfo.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/Core/PassManager.h"
#include "lld/ReaderWriter/Simple.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace lld;

namespace {
class GOTAtom : public SimpleDefinedAtom {
  static const uint8_t _defaultContent[8];

public:
  GOTAtom(const File &f, const DefinedAtom *target) : SimpleDefinedAtom(f) {
    if (target->contentType() == typeResolver) {
      DEBUG_WITH_TYPE("GOTAtom", llvm::dbgs() << "IRELATIVE relocation to "
                                              << target->name());
      addReference(llvm::ELF::R_X86_64_IRELATIVE, 0, target, 0);
    }
  }

  virtual StringRef name() const { return "ELF-GOTAtom"; }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return ".got.plt"; }

  virtual ContentType contentType() const { return typeGOT; }

  virtual uint64_t size() const { return rawContent().size(); }

  virtual ContentPermissions permissions() const { return permRW_; }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_defaultContent, 8);
  }
};

const uint8_t GOTAtom::_defaultContent[8] = { 0 };

class PLTAtom : public SimpleDefinedAtom {
  static const uint8_t _defaultContent[16];

public:
  PLTAtom(const File &f, GOTAtom *ga) : SimpleDefinedAtom(f) {
    addReference(llvm::ELF::R_X86_64_PC32, 2, ga, -4);
  }

  virtual StringRef name() const { return "ELF-PLTAtom"; }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return ".plt"; }

  virtual ContentType contentType() const { return typeStub; }

  virtual uint64_t size() const { return rawContent().size(); }

  virtual ContentPermissions permissions() const { return permR_X; }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_defaultContent, 16);
  }
};

const uint8_t PLTAtom::_defaultContent[16] = {
  0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmpq *gotatom(%rip)
  0x68, 0x00, 0x00, 0x00, 0x00,       // pushq pltentry
  0xe9, 0x00, 0x00, 0x00, 0x00        // jmpq plt[-1]
};

class ELFPassFile : public SimpleFile {
public:
  ELFPassFile(const ELFTargetInfo &eti) : SimpleFile(eti, "ELFPassFile") {}

  llvm::BumpPtrAllocator _alloc;
};

class PLTPass : public Pass {
public:
  PLTPass(const ELFTargetInfo &ti) : _file(ti) {}

  virtual void perform(MutableFile &mf) {
    for (const auto &atom : mf.defined())
      for (const auto &ref : *atom) {
        if (ref->kind() != llvm::ELF::R_X86_64_PC32)
          continue;
        if (const DefinedAtom *da =
                dyn_cast<const DefinedAtom>(ref->target())) {
          if (da->contentType() != DefinedAtom::typeResolver)
            continue;
          // We have a PC32 call to a IFUNC. Create a plt and got entry.
          // Look it up first.
          const PLTAtom *pa;
          auto plt = _pltMap.find(da);
          if (plt == _pltMap.end()) {
            // Add an entry.
            auto ga = new (_file._alloc) GOTAtom(_file, da);
            mf.addAtom(*ga);
            pa = new (_file._alloc) PLTAtom(_file, ga);
            mf.addAtom(*pa);
            _pltMap[da] = pa;
          } else
            pa = plt->second;
          // This is dirty.
          const_cast<Reference *>(ref)->setTarget(pa);
        }
      }
  }

private:
  llvm::DenseMap<const DefinedAtom *, const PLTAtom *> _pltMap;
  ELFPassFile _file;
};
} // end anon namespace

void elf::X86_64TargetInfo::addPasses(PassManager &pm) const {
  pm.add(std::unique_ptr<Pass>(new PLTPass(*this)));
}

#define LLD_CASE(name) .Case(#name, llvm::ELF::name)

ErrorOr<int32_t> elf::X86_64TargetInfo::relocKindFromString(
    StringRef str) const {
  int32_t ret = llvm::StringSwitch<int32_t>(str)
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
    .Default(-1);

  if (ret == -1)
    return make_error_code(yaml_reader_error::illegal_value);
  return ret;
}

#undef LLD_CASE

#define LLD_CASE(name) case llvm::ELF::name: return std::string(#name);

ErrorOr<std::string> elf::X86_64TargetInfo::stringFromRelocKind(
    int32_t kind) const {
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
  }

  return make_error_code(yaml_reader_error::illegal_value);
}
