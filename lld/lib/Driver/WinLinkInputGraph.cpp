//===- lib/Driver/WinLinkInputGraph.cpp -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/WinLinkInputGraph.h"

using llvm::sys::fs::file_magic;
using llvm::sys::fs::identify_magic;

namespace lld {

/// \brief Parse the input file to lld::File.
error_code PECOFFFileNode::parse(const LinkingContext &ctx,
                                 raw_ostream &diagnostics) {
  ErrorOr<StringRef> filePath = getPath(ctx);
  if (!filePath) {
    diagnostics << "File not found: " << _path << "\n";
    return error_code(filePath);
  }

  if (error_code ec = getBuffer(*filePath)) {
    diagnostics << "Cannot open file: " << *filePath << "\n";
    return ec;
  }

  if (ctx.logInputFiles())
    diagnostics << *filePath << "\n";

  if (filePath->endswith(".objtxt"))
    return ctx.getYAMLReader().parseFile(_buffer, _files);
  if (identify_magic(_buffer->getBuffer()) == file_magic::archive) {
    // Archive File
    error_code ec;
    _files.push_back(std::unique_ptr<File>(
                       new FileArchive(ctx, std::move(_buffer), ec, false)));
    return ec;
  }
  return _ctx.getDefaultReader().parseFile(_buffer, _files);
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
    return _ctx.allocate(_path.str() + ".obj");
  return _path;
}

ErrorOr<StringRef> PECOFFLibraryNode::getPath(const LinkingContext &) const {
  if (_path.endswith_lower(".lib"))
    return _ctx.searchLibraryFile(_path);
  return _ctx.searchLibraryFile(_ctx.allocate(_path.str() + ".lib"));
}

} // end anonymous namespace
