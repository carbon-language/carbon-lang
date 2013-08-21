//===- lld/Driver/WinLinkInputGraph.h - Files to be linked for PECOFF
// linking---===//
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

#ifndef LLD_PECOFF_INPUT_GRAPH_H
#define LLD_PECOFF_INPUT_GRAPH_H

#include "lld/Driver/InputGraph.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"

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

  virtual StringRef path(const LinkingContext &ctx) const;

  virtual std::unique_ptr<lld::LinkerInput>
  createLinkerInput(const lld::LinkingContext &);

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

private:
  const PECOFFLinkingContext &_ctx;
};

/// \brief Represents a PECOFF Library File
class PECOFFLibraryNode : public FileNode {
public:
  PECOFFLibraryNode(PECOFFLinkingContext &ctx, StringRef path)
      : FileNode(path), _ctx(ctx) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  virtual StringRef path(const LinkingContext &ctx) const;

  virtual std::unique_ptr<lld::LinkerInput>
  createLinkerInput(const lld::LinkingContext &);

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

private:
  const PECOFFLinkingContext &_ctx;
};

} // namespace lld

#endif
