//===- lld/Driver/DarwinInputGraph.h - Input Graph Node for Mach-O linker -===//
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
/// Handles Options for MachO linking and provides InputElements
/// for MachO linker
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_DARWIN_INPUT_GRAPH_H
#define LLD_DRIVER_DARWIN_INPUT_GRAPH_H

#include "lld/Core/InputGraph.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"

#include <map>

namespace lld {

/// \brief Represents a MachO File
class MachOFileNode : public FileNode {
public:
  MachOFileNode(StringRef path, bool isWholeArchive)
      : FileNode(path), _isWholeArchive(isWholeArchive) {}

  /// \brief Parse the input file to lld::File.
  std::error_code parse(const LinkingContext &ctx,
                        raw_ostream &diagnostics) override {
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
        MachOLinkingContext *mctx = (MachOLinkingContext*)(&ctx);
        mctx->registerDylib(reinterpret_cast<mach_o::MachODylibFile*>(shl));
      }
      _files.push_back(std::move(pf));
    }
    return std::error_code();
  }

  /// \brief Return the file that has to be processed by the resolver
  /// to resolve atoms. This iterates over all the files thats part
  /// of this node. Returns no_more_files when there are no files to be
  /// processed
  ErrorOr<File &> getNextFile() override {
    if (_files.size() == _nextFileIndex)
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }

private:
  bool _isWholeArchive;
};

} // namespace lld

#endif
