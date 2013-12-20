//===- lib/ReaderWriter/Reader.cpp ----------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Reader.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/system_error.h"

namespace lld {

Reader::~Reader() {
}

void Registry::add(std::unique_ptr<Reader> reader) {
  _readers.push_back(std::move(reader));
}

error_code
Registry::parseFile(std::unique_ptr<MemoryBuffer> &mb,
                    std::vector<std::unique_ptr<File>> &result) const {
  // Get file type.
  StringRef content(mb->getBufferStart(), mb->getBufferSize());
  llvm::sys::fs::file_magic fileType = llvm::sys::fs::identify_magic(content);
  // Get file extension.
  StringRef extension = llvm::sys::path::extension(mb->getBufferIdentifier());

  // Ask each registered reader if it can handle this file type or extension.
  for (const std::unique_ptr<Reader> &reader : _readers) {
    if (reader->canParse(fileType, extension, *mb))
      return reader->parseFile(mb, *this, result);
  }

  // No Reader could parse this file.
  return llvm::make_error_code(llvm::errc::executable_format_error);
}

static const Registry::KindStrings kindStrings[] = {
  { Reference::kindInGroup,      "in-group" },
  { Reference::kindLayoutAfter,  "layout-after" },
  { Reference::kindLayoutBefore, "layout-before" },
  LLD_KIND_STRING_END
};

Registry::Registry() {
  addKindTable(Reference::KindNamespace::all, Reference::KindArch::all,
               kindStrings);
}

void Registry::addKindTable(Reference::KindNamespace ns,
                            Reference::KindArch arch,
                            const KindStrings array[]) {
  KindEntry entry = { ns, arch, array };
  _kindEntries.push_back(entry);
}

bool Registry::referenceKindFromString(StringRef inputStr,
                                       Reference::KindNamespace &ns,
                                       Reference::KindArch &arch,
                                       Reference::KindValue &value) const {
  for (const KindEntry &entry : _kindEntries) {
    for (const KindStrings *pair = entry.array; !pair->name.empty(); ++pair) {
      if (!inputStr.equals(pair->name))
        continue;
      ns = entry.ns;
      arch = entry.arch;
      value = pair->value;
      return true;
    }
  }
  return false;
}

bool Registry::referenceKindToString(Reference::KindNamespace ns,
                                     Reference::KindArch arch,
                                     Reference::KindValue value,
                                     StringRef &str) const {
  for (const KindEntry &entry : _kindEntries) {
    if (entry.ns != ns)
      continue;
    if (entry.arch != arch)
      continue;
    for (const KindStrings *pair = entry.array; !pair->name.empty(); ++pair) {
      if (pair->value != value)
        continue;
      str = pair->name;
      return true;
    }
  }
  return false;
}

} // end namespace lld
