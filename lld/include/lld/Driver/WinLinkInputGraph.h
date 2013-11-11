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

  virtual ErrorOr<StringRef> getPath(const LinkingContext &ctx) const;

  /// \brief Parse the input file to lld::File.
  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) {
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

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

  virtual ErrorOr<File &> getNextFile() {
    if (_nextFileIndex == _files.size())
      return make_error_code(InputGraphError::no_more_files);
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

  virtual ErrorOr<StringRef> getPath(const LinkingContext &ctx) const;
};

} // namespace lld

#endif
