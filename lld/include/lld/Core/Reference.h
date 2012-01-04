//===- Core/References.h - A Reference to Another Atom --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_REFERENCES_H_
#define LLD_CORE_REFERENCES_H_

#include <stdint.h>


namespace lld {

class Atom;

class Reference {
public:
  typedef Reference *iterator;

  const Atom *target;
  uint64_t    addend;
  uint32_t    offsetInAtom;
  uint16_t    kind;
  uint16_t    flags;
};

} // namespace lld

#endif // LLD_CORE_REFERENCES_H_
