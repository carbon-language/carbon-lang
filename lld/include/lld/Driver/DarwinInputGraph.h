//===- lld/Driver/DarwinInputGraph.h - Files to be linked for MachO
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
/// Handles Options for MachO linking and provides InputElements
/// for MachO linker
///
//===----------------------------------------------------------------------===//

#ifndef LLD_MachO_INPUT_GRAPH_H
#define LLD_MachO_INPUT_GRAPH_H

#include "lld/Driver/InputGraph.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"

#include <map>

namespace lld {

/// \brief Represents a MachO File
class MachOFileNode : public FileNode {
public:
  MachOFileNode(MachOLinkingContext &ctx, StringRef path)
      : FileNode(path), _ctx(ctx) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  virtual std::unique_ptr<lld::LinkerInput>
  createLinkerInput(const lld::LinkingContext &);

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &) { return true; }

private:
  const MachOLinkingContext &_ctx;
};

} // namespace lld

#endif
