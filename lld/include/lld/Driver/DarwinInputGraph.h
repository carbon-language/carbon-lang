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

#include "lld/Core/ArchiveLibraryFile.h"
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

  void setLoadWholeArchive(bool value=true) {
    _isWholeArchive = value;
  }

  void setUpwardDylib(bool value=true) {
    _upwardDylib = value;
  }

private:
  void narrowFatBuffer(std::unique_ptr<MemoryBuffer> &mb, StringRef filePath);

  MachOLinkingContext &_context;
  std::unique_ptr<const ArchiveLibraryFile> _archiveFile;
  bool _isWholeArchive;
  bool _upwardDylib;
};

} // namespace lld

#endif
