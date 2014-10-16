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
#include "lld/ReaderWriter/MachOLinkingContext.h"

namespace lld {

/// \brief Represents a MachO File
class MachOFileNode : public FileNode {
public:
  MachOFileNode(StringRef path, MachOLinkingContext &ctx)
      : FileNode(path), _context(ctx), _isWholeArchive(false),
        _upwardDylib(false) {}

  /// \brief Parse the input file to lld::File.
  std::error_code parse(const LinkingContext &ctx,
                        raw_ostream &diagnostics) override;

  /// \brief Return the file that has to be processed by the resolver
  /// to resolve atoms. This iterates over all the files thats part
  /// of this node. Returns no_more_files when there are no files to be
  /// processed
  ErrorOr<File &> getNextFile() override {
    if (_files.size() == _nextFileIndex)
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }

  void setLoadWholeArchive(bool value=true) {
    _isWholeArchive = value;
  }

  void setUpwardDylib(bool value=true) {
    _upwardDylib = value;
  }

private:
 void narrowFatBuffer(StringRef filePath);

  MachOLinkingContext &_context;
  bool _isWholeArchive;
  bool _upwardDylib;
};

} // namespace lld

#endif
