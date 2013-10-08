//===- lld/Driver/WinLinkInputGraph.h - Input Graph Node for COFF linker --===//
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
/// Handles Options for PECOFF linking and provides InputElements
/// for PECOFF linker
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_WIN_LINK_INPUT_GRAPH_H
#define LLD_DRIVER_WIN_LINK_INPUT_GRAPH_H

#include "lld/Driver/InputGraph.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/FileArchive.h"

#include <map>

namespace lld {

/// \brief Represents a PECOFF File
class PECOFFFileNode : public FileNode {
public:
  PECOFFFileNode(PECOFFLinkingContext &ctx, StringRef path)
      : FileNode(path), _ctx(ctx) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  virtual llvm::ErrorOr<StringRef> path(const LinkingContext &ctx) const;

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

    if (ctx.logInputFiles())
      diagnostics << _buffer->getBufferIdentifier() << "\n";

    // YAML file is identified by a .objtxt extension
    // FIXME : Identify YAML files by using a magic
    if (filePath->endswith(".objtxt")) {
      ec = _ctx.getYAMLReader().parseFile(_buffer, _files);
      if (!ec)
        return ec;
    }

    llvm::sys::fs::file_magic FileType =
        llvm::sys::fs::identify_magic(_buffer->getBuffer());

    std::unique_ptr<File> f;

    switch (FileType) {
    case llvm::sys::fs::file_magic::archive:
      // Archive File
      f.reset(new FileArchive(ctx, std::move(_buffer), ec, false));
      _files.push_back(std::move(f));
      break;

    case llvm::sys::fs::file_magic::coff_object:
    default:
      ec = _ctx.getDefaultReader().parseFile(_buffer, _files);
      break;
    }
    return ec;
  }

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

  virtual ErrorOr<File &> getNextFile() {
    if (_nextFileIndex == _files.size())
      return make_error_code(input_graph_error::no_more_files);
    return *_files[_nextFileIndex++];
  }

protected:
  const PECOFFLinkingContext &_ctx;
};

/// \brief Represents a PECOFF Library File
class PECOFFLibraryNode : public PECOFFFileNode {
public:
  PECOFFLibraryNode(PECOFFLinkingContext &ctx, StringRef path)
      : PECOFFFileNode(ctx, path) {}

  virtual llvm::ErrorOr<StringRef> path(const LinkingContext &ctx) const;
};

} // namespace lld

#endif
