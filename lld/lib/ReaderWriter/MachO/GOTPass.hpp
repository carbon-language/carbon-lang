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
  bool noTextRelocs() override {
    return true;
  }

  bool isGOTAccess(const Reference &, bool &canBypassGOT) override {
    return false;
  }

  void updateReferenceToGOT(const Reference*, bool targetIsNowGOT) override {

  }

  const DefinedAtom* makeGOTEntry(const Atom&) override {
    return nullptr;
  }

};


} // namespace mach_o
} // namespace lld


#endif // LLD_READER_WRITER_MACHO_GOT_PASS_H
