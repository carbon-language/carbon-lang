//===- lib/ReaderWriter/ELF/Hexagon/HexagonDynamicAtoms.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_DYNAMIC_ATOMS_H
#define LLD_READER_WRITER_ELF_HEXAGON_DYNAMIC_ATOMS_H

#include "Atoms.h"
#include "HexagonTargetInfo.h"

namespace lld {
namespace elf {

class HexagonGOTAtom : public GOTAtom {
  static const uint8_t _defaultContent[8];

public:
  HexagonGOTAtom(const File &f, StringRef secName)
      : GOTAtom(f, secName) {
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_defaultContent, 8);
  }
};

const uint8_t HexagonGOTAtom::_defaultContent[8] = { 0 };

class HexagonPLTAtom : public PLTAtom {
  static const uint8_t _defaultContent[16];

public:
  HexagonPLTAtom(const File &f, StringRef secName)
      : PLTAtom(f, secName) {
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_defaultContent, 16);
  }
};

const uint8_t HexagonPLTAtom::_defaultContent[16] = {
  0x00, 0x40, 0x00, 0x00, // { immext (#0)                                     
  0x0e, 0xc0, 0x49, 0x6a, //   r14 = add (pc, ##GOTn@PCREL) } # address of GOTn
  0x1c, 0xc0, 0x8e, 0x91, // r28 = memw (r14)                 # contents of GOTn
  0x00, 0xc0, 0x9c, 0x52, // jumpr r28                        # call it        
};

class HexagonPLT0Atom : public PLT0Atom {
  static const uint8_t _plt0Content[28];

public:
  HexagonPLT0Atom(const File &f) : PLT0Atom(f) {
#ifndef NDEBUG
    _name = ".PLT0";
#endif
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(_plt0Content, 28);
  }
};

const uint8_t HexagonPLT0Atom::_plt0Content[28] = {
 0x00, 0x40, 0x00, 0x00,  // { immext (#0)                                      
 0x1c, 0xc0, 0x49, 0x6a,  //   r28 = add (pc, ##GOT0@PCREL) } # address of GOT0
 0x0e, 0x42, 0x9c, 0xe2,  // { r14 -= add (r28, #16)  # offset of GOTn from GOTa
 0x4f, 0x40, 0x9c, 0x91,  //   r15 = memw (r28 + #8)  # object ID at GOT2      
 0x3c, 0xc0, 0x9c, 0x91,  //   r28 = memw (r28 + #4) }# dynamic link at GOT1  
 0x0e, 0x42, 0x0e, 0x8c,  // { r14 = asr (r14, #2)    # index of PLTn         
 0x00, 0xc0, 0x9c, 0x52,  //   jumpr r28 }            # call dynamic linker  
};

} // elf 
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_DYNAMIC_ATOMS_H
