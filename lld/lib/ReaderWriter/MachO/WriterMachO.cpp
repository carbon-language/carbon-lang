//===- lib/ReaderWriter/MachO/WriterMachO.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Writer.h"
#include "ExecutableAtoms.hpp"
#include "MachONormalizedFile.h"
#include "lld/Core/File.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

using lld::mach_o::normalized::NormalizedFile;

namespace lld {
namespace mach_o {

class MachOWriter : public Writer {
public:
  MachOWriter(const MachOLinkingContext &ctxt) : _context(ctxt) { }

  std::error_code writeFile(const lld::File &file, StringRef path) override {
    // Construct empty normalized file from atoms.
    ErrorOr<std::unique_ptr<NormalizedFile>> nFile =
                                normalized::normalizedFromAtoms(file, _context);
    if (std::error_code ec = nFile.getError())
      return ec;

    // For testing, write out yaml form of normalized file.
    if (_context.printAtoms()) {
      std::unique_ptr<Writer> yamlWriter = createWriterYAML(_context);
      yamlWriter->writeFile(file, "-");
    }

    // Write normalized file as mach-o binary.
    return writeBinary(*nFile->get(), path);
  }

  bool createImplicitFiles(std::vector<std::unique_ptr<File> > &r) override {
    // When building main executables, add _main as required entry point.
    if (_context.outputTypeHasEntry())
      r.emplace_back(new CEntryFile(_context));
    // If this can link with dylibs, need helper function (dyld_stub_binder).
    if (_context.needsStubsPass())
      r.emplace_back(new StubHelperFile(_context));

    return true;
  }
private:
   const MachOLinkingContext  &_context;
 };


} // namespace mach_o

std::unique_ptr<Writer> createWriterMachO(const MachOLinkingContext &context) {
  return std::unique_ptr<Writer>(new lld::mach_o::MachOWriter(context));
}

} // namespace lld
