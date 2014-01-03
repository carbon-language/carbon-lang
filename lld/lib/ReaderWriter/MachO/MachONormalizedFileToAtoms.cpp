//===- lib/ReaderWriter/MachO/MachONormalizedFileToAtoms.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

///
/// \file Converts from in-memory normalized mach-o to in-memory Atoms.
///
///                  +------------+
///                  | normalized |
///                  +------------+
///                        |
///                        |
///                        v
///                    +-------+
///                    | Atoms |
///                    +-------+

#include "MachONormalizedFile.h"
#include "File.h"
#include "Atoms.h"

#include "lld/Core/LLVM.h"

#include "llvm/Support/MachO.h"

using namespace llvm::MachO;

namespace lld {
namespace mach_o {
namespace normalized {

static ErrorOr<std::unique_ptr<lld::File>>
normalizedObjectToAtoms(const NormalizedFile &normalizedFile, StringRef path) {
  std::unique_ptr<MachOFile> file(new MachOFile(path));

  for (auto &sym : normalizedFile.globalSymbols) {
    file->addDefinedAtom(sym.name,
                         normalizedFile.sections[sym.sect - 1].content);
  }

  assert(normalizedFile.localSymbols.empty() &&
         "local symbols not supported yet!");
  assert(normalizedFile.undefinedSymbols.empty() &&
         "undefined symbols not supported yet!");

  return std::unique_ptr<File>(std::move(file));
}

ErrorOr<std::unique_ptr<lld::File>>
normalizedToAtoms(const NormalizedFile &normalizedFile, StringRef path) {
  switch (normalizedFile.fileType) {
  case MH_OBJECT:
    return normalizedObjectToAtoms(normalizedFile, path);
  default:
    llvm_unreachable("unhandled MachO file type!");
  }
}

} // namespace normalized
} // namespace mach_o
} // namespace lld
