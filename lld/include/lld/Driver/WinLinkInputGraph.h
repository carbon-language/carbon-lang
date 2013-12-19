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

  virtual ErrorOr<StringRef> getPath(const LinkingContext &ctx) const;

  /// \brief Parse the input file to lld::File.
  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics);

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

  virtual ErrorOr<File &> getNextFile();

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

/// \brief Represents a ELF control node
class PECOFFGroup : public Group {
public:
  PECOFFGroup() : Group(0) {}

  virtual bool validate() { return true; }
  virtual bool dump(raw_ostream &) { return true; }

  /// \brief Parse the group members.
  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) {
    for (auto &elem : _elements)
      if (error_code ec = elem->parse(ctx, diagnostics))
        return ec;
    return error_code::success();
  }
};

} // namespace lld

#endif
