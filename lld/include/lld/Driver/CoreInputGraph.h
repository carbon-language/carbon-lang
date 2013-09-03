//===- lld/Driver/CoreInputGraph.h - Files to be linked for CORE linking---==//
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

#ifndef LLD_CORE_INPUT_GRAPH_H
#define LLD_CORE_INPUT_GRAPH_H

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

  virtual llvm::ErrorOr<std::unique_ptr<lld::LinkerInput> >
  createLinkerInput(const lld::LinkingContext &);

  /// \brief validates the Input Element
  virtual bool validate() {
    (void)_ctx;
    return true;
  }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

private:
  const CoreLinkingContext &_ctx;
};

} // namespace lld

#endif
