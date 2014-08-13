//===- lib/ReaderWriter/MachO/ExecutableAtoms.hpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_EXECUTABLE_ATOMS_H
#define LLD_READER_WRITER_MACHO_EXECUTABLE_ATOMS_H

#include "llvm/Support/MachO.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Reference.h"
#include "lld/Core/Simple.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"

namespace lld {
namespace mach_o {


//
// CEntryFile adds an UndefinedAtom for "_main" so that the Resolving
// phase will fail if "_main" is undefined.
//
class CEntryFile : public SimpleFile {
public:
  CEntryFile(const MachOLinkingContext &context)
      : SimpleFile("C entry"),
       _undefMain(*this, context.entrySymbolName()) {
    this->addAtom(_undefMain);
  }

private:
  SimpleUndefinedAtom   _undefMain;
};


//
// StubHelperFile adds an UndefinedAtom for "dyld_stub_binder" so that
// the Resolveing phase will fail if "dyld_stub_binder" is undefined.
//
class StubHelperFile : public SimpleFile {
public:
  StubHelperFile(const MachOLinkingContext &context)
      : SimpleFile("stub runtime"),
        _undefBinder(*this, context.binderSymbolName()) {
    this->addAtom(_undefBinder);
  }

private:
  SimpleUndefinedAtom   _undefBinder;
};





} // namespace mach_o
} // namespace lld

#endif // LLD_READER_WRITER_MACHO_EXECUTABLE_ATOMS_H
