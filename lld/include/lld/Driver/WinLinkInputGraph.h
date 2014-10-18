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

#include "lld/Core/InputGraph.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include <map>

namespace lld {

extern bool isCOFFLibraryFileExtension(StringRef path);

/// \brief Represents a PECOFF File
class PECOFFFileNode : public FileNode {
public:
  PECOFFFileNode(PECOFFLinkingContext &ctx, StringRef path)
      : FileNode(path), _ctx(ctx), _parsed(false) {}

  ErrorOr<StringRef> getPath(const LinkingContext &ctx) const override;

  /// \brief Parse the input file to lld::File.
  std::error_code parse(const LinkingContext &ctx,
                        raw_ostream &diagnostics) override;

  ErrorOr<File &> getNextFile() override;

protected:
  const PECOFFLinkingContext &_ctx;

private:
  bool _parsed;
};

/// \brief Represents a PECOFF Library File
class PECOFFLibraryNode : public PECOFFFileNode {
public:
  PECOFFLibraryNode(PECOFFLinkingContext &ctx, StringRef path)
      : PECOFFFileNode(ctx, path) {}

  ErrorOr<StringRef> getPath(const LinkingContext &ctx) const override;
};

/// \brief Represents a ELF control node
class PECOFFGroup : public Group {
public:
  PECOFFGroup(PECOFFLinkingContext &ctx) : Group(), _ctx(ctx) {}

  /// \brief Parse the group members.
  std::error_code parse(const LinkingContext &ctx, raw_ostream &diag) override {
    std::lock_guard<std::recursive_mutex> lock(_ctx.getMutex());
    return Group::parse(ctx, diag);
  }

private:
  PECOFFLinkingContext &_ctx;
};

} // namespace lld

#endif
