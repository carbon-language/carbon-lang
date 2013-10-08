//===- lld/Driver/CoreInputGraph.h - Input Graph Node for Core linker -----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Handles Options for CORE linking and provides InputElements
/// for the CORE linker
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_CORE_INPUT_GRAPH_H
#define LLD_DRIVER_CORE_INPUT_GRAPH_H

#include "lld/Driver/InputGraph.h"
#include "lld/ReaderWriter/CoreLinkingContext.h"

#include <map>

namespace lld {

/// \brief Represents a CORE File
class COREFileNode : public FileNode {
public:
  COREFileNode(CoreLinkingContext &ctx, StringRef path)
      : FileNode(path), _ctx(ctx) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  /// \brief validates the Input Element
  virtual bool validate() {
    (void)_ctx;
    return true;
  }

  /// \brief Parse the input file to lld::File.
  llvm::error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) {
    ErrorOr<StringRef> filePath = path(ctx);
    if (!filePath &&
        error_code(filePath) == llvm::errc::no_such_file_or_directory)
      return make_error_code(llvm::errc::no_such_file_or_directory);

    // Create a memory buffer
    OwningPtr<llvm::MemoryBuffer> opmb;
    llvm::error_code ec;

    if ((ec = llvm::MemoryBuffer::getFileOrSTDIN(*filePath, opmb)))
      return ec;

    std::unique_ptr<MemoryBuffer> mb(opmb.take());
    _buffer = std::move(mb);

    ec = _ctx.getDefaultReader().parseFile(_buffer, _files);
    return ec;
  }

  /// \brief Return the file that has to be processed by the resolver
  /// to resolve atoms. This iterates over all the files thats part
  /// of this node. Returns no_more_files when there are no files to be
  /// processed
  virtual ErrorOr<File &> getNextFile() {
    if (_files.size() == _nextFileIndex)
      return make_error_code(input_graph_error::no_more_files);
    return *_files[_nextFileIndex++];
  }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

private:
  CoreLinkingContext &_ctx;
};

} // namespace lld

#endif
