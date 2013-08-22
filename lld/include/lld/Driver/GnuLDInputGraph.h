//===- lld/Driver/GnuLDInputGraph.h - Files to be linked for ELF linking---===//
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
/// Handles Options for the GNU style linker for ELF and provides InputElements
/// for the GNU style linker for ELF
///
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_INPUT_GRAPH_H
#define LLD_ELF_INPUT_GRAPH_H

#include "lld/Driver/InputGraph.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"

namespace lld {

/// \brief Represents a ELF File
class ELFFileNode : public FileNode {
public:
  ELFFileNode(ELFLinkingContext &ctx, StringRef path,
              std::vector<StringRef> searchPath,
              bool isWholeArchive = false, bool asNeeded = false)
      : FileNode(path), _elfLinkingContext(ctx),
        _isWholeArchive(isWholeArchive), _asNeeded(asNeeded) {
    std::copy(searchPath.begin(), searchPath.end(),
              std::back_inserter(_libraryPaths));
  }

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  virtual StringRef path(const LinkingContext &ctx) const;

  virtual std::unique_ptr<lld::LinkerInput>
  createLinkerInput(const lld::LinkingContext &);

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &diagnostics) {
    diagnostics << "Name    : " << path(_elfLinkingContext) << "\n";
    diagnostics << "Type    : "
                << "ELF File"
                << "\n";
    diagnostics << "Ordinal : " << getOrdinal() << "\n";
    diagnostics << "Attributes : "
                << "\n";
    diagnostics << "  - wholeArchive : "
                << ((_isWholeArchive) ? "true" : "false") << "\n";
    diagnostics << "  - asNeeded : " << ((_asNeeded) ? "true" : "false")
                << "\n";
    diagnostics << "  contextPath : " << ((_libraryPaths.size()) ? "" : "None")
                << "\n";
    for (auto path : _libraryPaths)
      diagnostics << "    - " << path << "\n";
    return true;
  }

private:
  ELFLinkingContext &_elfLinkingContext;
  bool _isWholeArchive : 1;
  bool _asNeeded : 1;
  std::vector<StringRef> _libraryPaths;
};

/// \brief Represents a ELF control node
class ELFGroup : public Group {
public:
  ELFGroup(ELFLinkingContext &ctx) : Group(), _elfLinkingContext(ctx) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::Control;
  }

  virtual std::unique_ptr<lld::LinkerInput>
  createLinkerInput(const lld::LinkingContext &) {
    // FIXME : create a linker input to handle groups
    return nullptr;
  }

  /// \brief Validate the options
  virtual bool validate() { return true; }

  /// \brief Dump the ELFGroup
  virtual bool dump(llvm::raw_ostream &) { return true; }

protected:
  ELFLinkingContext &_elfLinkingContext;
};

} // namespace lld

#endif
