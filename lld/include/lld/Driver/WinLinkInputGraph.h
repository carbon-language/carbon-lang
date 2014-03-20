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

/// \brief Represents a PECOFF File
class PECOFFFileNode : public FileNode {
public:
  PECOFFFileNode(PECOFFLinkingContext &ctx, StringRef path)
      : FileNode(path), _ctx(ctx) {}

  ErrorOr<StringRef> getPath(const LinkingContext &ctx) const override;

  /// \brief Parse the input file to lld::File.
  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) override;

  /// \brief validates the Input Element
  bool validate() override { return true; }

  /// \brief Dump the Input Element
  bool dump(raw_ostream &) override { return true; }

  ErrorOr<File &> getNextFile() override;

protected:
  const PECOFFLinkingContext &_ctx;
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
  PECOFFGroup() : Group(0) {}

  bool validate() override { return true; }
  bool dump(raw_ostream &) override { return true; }

  /// \brief Parse the group members.
  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) override {
    auto *pctx = (PECOFFLinkingContext *)(&ctx);
    error_code ec = error_code::success();
    pctx->lock();
    for (auto &elem : _elements)
      if ((ec = elem->parse(ctx, diagnostics)))
        break;
    pctx->unlock();
    return ec;
  }
};

} // namespace lld

#endif
