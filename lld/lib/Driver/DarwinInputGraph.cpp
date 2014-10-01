//===- lib/ReaderWriter/MachO/DarwinInputGraph.cpp ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/DarwinInputGraph.h"

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/Core/SharedLibraryFile.h"

#include "lld/ReaderWriter/MachOLinkingContext.h"

namespace lld {

/// \brief Parse the input file to lld::File.
std::error_code MachOFileNode::parse(const LinkingContext &ctx,
                                     raw_ostream &diagnostics)  {
  ErrorOr<StringRef> filePath = getPath(ctx);
  if (std::error_code ec = filePath.getError())
    return ec;

  if (std::error_code ec = getBuffer(*filePath))
    return ec;

  if (ctx.logInputFiles())
    diagnostics << *filePath << "\n";

  std::vector<std::unique_ptr<File>> parsedFiles;
  if (_isWholeArchive) {
    std::error_code ec = ctx.registry().parseFile(_buffer, parsedFiles);
    if (ec)
      return ec;
    assert(parsedFiles.size() == 1);
    std::unique_ptr<File> f(parsedFiles[0].release());
    if (auto archive =
            reinterpret_cast<const ArchiveLibraryFile *>(f.get())) {
      // FIXME: something needs to own archive File
      //_files.push_back(std::move(archive));
      return archive->parseAllMembers(_files);
    } else {
      // if --whole-archive is around non-archive, just use it as normal.
      _files.push_back(std::move(f));
      return std::error_code();
    }
  }
  if (std::error_code ec = ctx.registry().parseFile(_buffer, parsedFiles))
    return ec;
  for (std::unique_ptr<File> &pf : parsedFiles) {
    // If a dylib was parsed, inform LinkingContext about it.
    if (SharedLibraryFile *shl = dyn_cast<SharedLibraryFile>(pf.get())) {
      _context.registerDylib(reinterpret_cast<mach_o::MachODylibFile*>(shl));
    }
    _files.push_back(std::move(pf));
  }
  return std::error_code();
}



} // end namesapce lld
