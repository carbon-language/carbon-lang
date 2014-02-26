//===- lib/ReaderWriter/PECOFF/Pass.h -------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_PASS_H
#define LLD_READER_WRITER_PE_COFF_PASS_H

#include "Atoms.h"

namespace lld {
namespace pecoff {

void addDir32Reloc(COFFBaseDefinedAtom *atom, const Atom *target,
                   size_t offsetInAtom = 0);

void addDir32NBReloc(COFFBaseDefinedAtom *atom, const Atom *target,
                     size_t offsetInAtom = 0);

} // namespace pecoff
} // namespace lld

#endif
