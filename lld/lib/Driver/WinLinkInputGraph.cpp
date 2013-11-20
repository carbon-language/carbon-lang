//===- lib/Driver/WinLinkDriver.cpp ---------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/WinLinkInputGraph.h"

namespace lld {

/// \brief Parse the input file to lld::File.
error_code PECOFFFileNode::parse(const LinkingContext &ctx,
                                 raw_ostream &diagnostics) {
  ErrorOr<StringRef> filePath = getPath(ctx);
  if (!filePath)
    return error_code(filePath);

  if (error_code ec = getBuffer(*filePath))
    return ec;

  if (ctx.logInputFiles())
    diagnostics << *filePath << "\n";

  if (filePath->endswith(".objtxt"))
    return ctx.getYAMLReader().parseFile(_buffer, _files);

  llvm::sys::fs::file_magic FileType =
      llvm::sys::fs::identify_magic(_buffer->getBuffer());
  std::unique_ptr<File> f;

  switch (FileType) {
  case llvm::sys::fs::file_magic::archive: {
    // Archive File
    error_code ec;
    f.reset(new FileArchive(ctx, std::move(_buffer), ec, false));
    _files.push_back(std::move(f));
    return ec;
  }
  case llvm::sys::fs::file_magic::coff_object:
  default:
    return _ctx.getDefaultReader().parseFile(_buffer, _files);
  }
}

ErrorOr<File &> PECOFFFileNode::getNextFile() {
  if (_nextFileIndex == _files.size())
    return make_error_code(InputGraphError::no_more_files);
  return *_files[_nextFileIndex++];
}

ErrorOr<StringRef> PECOFFFileNode::getPath(const LinkingContext &) const {
  if (_path.endswith_lower(".lib"))
    return _ctx.searchLibraryFile(_path);
  if (llvm::sys::path::extension(_path).empty())
    return _ctx.allocateString(_path.str() + ".obj");
  return _path;
}

ErrorOr<StringRef> PECOFFLibraryNode::getPath(const LinkingContext &) const {
  if (_path.endswith_lower(".lib"))
    return _ctx.searchLibraryFile(_path);
  return _ctx.searchLibraryFile(_ctx.allocateString(_path.str() + ".lib"));
}

} // end anonymous namespace
