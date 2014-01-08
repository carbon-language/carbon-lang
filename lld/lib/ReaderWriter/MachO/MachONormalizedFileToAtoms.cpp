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

static uint64_t nextSymbolAddress(const NormalizedFile &normalizedFile, 
                                const Symbol &symbol) {
  uint64_t symbolAddr = symbol.value;
  uint8_t symbolSectionIndex = symbol.sect;
  const Section &section = normalizedFile.sections[symbolSectionIndex - 1];
  // If no symbol after this address, use end of section address.
  uint64_t closestAddr = section.address + section.content.size();
  for (const Symbol &s : normalizedFile.globalSymbols) {
    if (s.sect != symbolSectionIndex)
      continue;
    uint64_t sValue = s.value;
    if (sValue <= symbolAddr)
      continue;
    if (sValue < closestAddr)
      closestAddr = s.value;
  }
  return closestAddr;
}

static ErrorOr<std::unique_ptr<lld::File>>
normalizedObjectToAtoms(const NormalizedFile &normalizedFile, StringRef path) {
  std::unique_ptr<MachOFile> file(new MachOFile(path));

 for (const Symbol &sym : normalizedFile.globalSymbols) {
    // Mach-O symbol table does have size in it, so need to scan ahead
    // to find symbol with next highest address.
    const Section &section = normalizedFile.sections[sym.sect - 1];
    uint64_t offset = sym.value - section.address;
    uint64_t size = nextSymbolAddress(normalizedFile, sym) - sym.value;
    ArrayRef<uint8_t> atomContent = llvm::makeArrayRef(&section.content[offset],
                                                      size);
    file->addDefinedAtom(sym.name, atomContent);
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
