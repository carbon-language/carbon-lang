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

namespace lld {
namespace elf {

class X86_64GOTAtom : public GOTAtom {
  static const uint8_t _defaultContent[8];

public:
  X86_64GOTAtom(const File &f, StringRef secName) : GOTAtom(f, secName) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_defaultContent, 8);
  }
};

const uint8_t X86_64GOTAtom::_defaultContent[8] = { 0 };

class X86_64PLTAtom : public PLTAtom {
  static const uint8_t _defaultContent[16];

public:
  X86_64PLTAtom(const File &f, StringRef secName) : PLTAtom(f, secName) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_defaultContent, 16);
  }
};

const uint8_t X86_64PLTAtom::_defaultContent[16] = {
  0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmpq *gotatom(%rip)
  0x68, 0x00, 0x00, 0x00, 0x00,       // pushq reloc-index
  0xe9, 0x00, 0x00, 0x00, 0x00        // jmpq plt[-1]
};

class X86_64PLT0Atom : public PLT0Atom {
  static const uint8_t _plt0Content[16];

public:
  X86_64PLT0Atom(const File &f) : PLT0Atom(f) {
#ifndef NDEBUG
    _name = ".PLT0";
#endif
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_plt0Content, 16);
  }
};

const uint8_t X86_64PLT0Atom::_plt0Content[16] = {
  0xff, 0x35, 0x00, 0x00, 0x00, 0x00, // pushq GOT+8(%rip)
  0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp *GOT+16(%rip)
  0x90, 0x90, 0x90, 0x90              // nopnopnop
};

} // elf
} // lld

#endif
