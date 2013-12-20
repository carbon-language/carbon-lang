//===- lib/ReaderWriter/MachO/GOTPass.hpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_GOT_PASS_H
#define LLD_READER_WRITER_MACHO_GOT_PASS_H

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
#include "lld/Core/Pass.h"

#include "ReferenceKinds.h"
#include "StubAtoms.hpp"

namespace lld {
namespace mach_o {


class GOTPass : public lld::GOTPass {
public:
  virtual bool noTextRelocs() {
    return true;
  }

  virtual bool isGOTAccess(const Reference &, bool &canBypassGOT) {
    return false;
  }

  virtual void updateReferenceToGOT(const Reference*, bool targetIsNowGOT) {
  
  }

  virtual const DefinedAtom* makeGOTEntry(const Atom&) {
    return nullptr;
  }
  
};


} // namespace mach_o 
} // namespace lld 


#endif // LLD_READER_WRITER_MACHO_GOT_PASS_H
