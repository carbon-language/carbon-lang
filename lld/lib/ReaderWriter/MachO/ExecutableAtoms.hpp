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
#include "lld/Core/UndefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
#include "lld/Core/LinkingContext.h"
#include "lld/ReaderWriter/Simple.h"

namespace lld {
namespace mach_o {


//
// CRuntimeFile adds an UndefinedAtom for "_main" so that the Resolving
// phase will fail if "_main" is undefined.
//
class CRuntimeFile : public SimpleFile {
public:
    CRuntimeFile(const MachOLinkingContext &context)
      : SimpleFile(context, "C runtime"),
        _undefMain(*this, context.entrySymbolName()) {
      // only main executables need _main
      if (context.outputFileType() == llvm::MachO::MH_EXECUTE) {
        this->addAtom(_undefMain);
      }
   }

private:
  SimpleUndefinedAtom   _undefMain;
};

} // namespace mach_o
} // namespace lld

#endif // LLD_READER_WRITER_MACHO_EXECUTABLE_ATOMS_H
