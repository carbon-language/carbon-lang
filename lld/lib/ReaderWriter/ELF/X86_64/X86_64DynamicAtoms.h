//===- lib/ReaderWriter/ELF/X86_64/X86_64DynamicAtoms.h -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_64_DYNAMIC_ATOMS_H
#define X86_64_DYNAMIC_ATOMS_H

#include "Atoms.h"
#include "X86_64TargetInfo.h"

/// \brief Specify various atom contents that are used by X86_64 dynamic
/// linking
namespace {
// .got values
const uint8_t x86_64GotAtomContent[8] = { 0 };

// .plt value (entry 0)
const uint8_t x86_64Plt0AtomContent[16] = {
  0xff, 0x35, 0x00, 0x00, 0x00, 0x00, // pushq GOT+8(%rip)
  0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp *GOT+16(%rip)
  0x90, 0x90, 0x90, 0x90              // nopnopnop
};

// .plt values (other entries)
const uint8_t x86_64PltAtomContent[16] = {
  0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmpq *gotatom(%rip)
  0x68, 0x00, 0x00, 0x00, 0x00,       // pushq reloc-index
  0xe9, 0x00, 0x00, 0x00, 0x00        // jmpq plt[-1]
};
}

namespace lld {
namespace elf {

class X86_64GOTAtom : public GOTAtom {
public:
  X86_64GOTAtom(const File &f, StringRef secName) : GOTAtom(f, secName) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(x86_64GotAtomContent, 8);
  }
};

class X86_64PLT0Atom : public PLT0Atom {
public:
  X86_64PLT0Atom(const File &f) : PLT0Atom(f) {
#ifndef NDEBUG
    _name = ".PLT0";
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(x86_64Plt0AtomContent, 16);
  }
};

class X86_64PLTAtom : public PLTAtom {
public:
  X86_64PLTAtom(const File &f, StringRef secName) : PLTAtom(f, secName) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(x86_64PltAtomContent, 16);
  }
};

} // elf
} // lld

#endif
