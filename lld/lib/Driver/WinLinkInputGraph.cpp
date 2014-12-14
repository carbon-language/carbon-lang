//===- lib/Driver/WinLinkInputGraph.cpp -----------------------------------===//
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
std::error_code PECOFFFileNode::parse(const LinkingContext &ctx,
                                      raw_ostream &diagnostics) {
  if (_parsed)
    return std::error_code();
  _parsed = true;
  ErrorOr<StringRef> filePath = getPath(ctx);
  if (std::error_code ec = filePath.getError()) {
    diagnostics << "File not found: " << _path << "\n";
    return ec;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> mb =
      MemoryBuffer::getFileOrSTDIN(*filePath);
  if (std::error_code ec = mb.getError()) {
    diagnostics << "Cannot open file: " << *filePath << "\n";
    return ec;
  }

  if (ctx.logInputFiles())
    diagnostics << *filePath << "\n";

  return ctx.registry().parseFile(std::move(mb.get()), _files);
}

} // end anonymous namespace
