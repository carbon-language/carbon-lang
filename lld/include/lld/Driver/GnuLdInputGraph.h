//===- lld/Driver/GnuLdInputGraph.h - Input Graph Node for ELF linker------===//
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

#ifndef LLD_DRIVER_GNU_LD_INPUT_GRAPH_H
#define LLD_DRIVER_GNU_LD_INPUT_GRAPH_H

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/InputGraph.h"
#include "lld/Core/Resolver.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "lld/ReaderWriter/LinkerScript.h"

namespace lld {

/// \brief Represents a ELF File
class ELFFileNode : public FileNode {
public:
  ELFFileNode(ELFLinkingContext &ctx, StringRef path, int64_t ordinal = -1,
              bool isWholeArchive = false, bool asNeeded = false,
              bool dashlPrefix = false)
      : FileNode(path, ordinal), _elfLinkingContext(ctx),
        _isWholeArchive(isWholeArchive), _asNeeded(asNeeded),
        _isDashlPrefix(dashlPrefix) {}

  ErrorOr<StringRef> getPath(const LinkingContext &ctx) const override;

  /// \brief create an error string for printing purposes
  std::string errStr(error_code) override;

  /// \brief Dump the Input Element
  bool dump(raw_ostream &diagnostics) override {
    diagnostics << "Name    : " << *getPath(_elfLinkingContext) << "\n"
                << "Type    : "
                << "ELF File"
                << "\n"
                << "Ordinal : " << getOrdinal() << "\n"
                << "Attributes : "
                << "\n"
                << "  - wholeArchive : "
                << ((_isWholeArchive) ? "true" : "false") << "\n"
                << "  - asNeeded : " << ((_asNeeded) ? "true" : "false")
                << "\n";
    return true;
  }

  /// \brief Parse the input file to lld::File.
  error_code parse(const LinkingContext &, raw_ostream &) override;

  /// \brief This is used by Group Nodes, when there is a need to reset the
  /// the file to be processed next. When handling a group node that contains
  /// Input elements, if the group node has to be reprocessed, the linker needs
  /// to start processing files as part of the inputelement from beginning.
  /// reset the next file index to 0 only if the node is an archive library or
  /// a shared library
  void resetNextIndex() override {
    if ((!_isWholeArchive && (_files[0]->kind() == File::kindArchiveLibrary)) ||
        (_files[0]->kind() == File::kindSharedLibrary)) {
      _nextFileIndex = 0;
    }
    setResolveState(Resolver::StateNoChange);
  }

  /// \brief Return the file that has to be processed by the resolver
  /// to resolve atoms. This iterates over all the files thats part
  /// of this node. Returns no_more_files when there are no files to be
  /// processed
  ErrorOr<File &> getNextFile() override {
    if (_nextFileIndex == _files.size())
      return make_error_code(InputGraphError::no_more_files);
    return *_files[_nextFileIndex++];
  }

private:
  llvm::BumpPtrAllocator _alloc;
  const ELFLinkingContext &_elfLinkingContext;
  bool _isWholeArchive;
  bool _asNeeded;
  bool _isDashlPrefix;
  std::unique_ptr<const ArchiveLibraryFile> _archiveFile;
};

/// \brief Represents a ELF control node
class ELFGroup : public Group {
public:
  ELFGroup(const ELFLinkingContext &, int64_t ordinal)
      : Group(ordinal) {}

  /// \brief Parse the group members.
  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) override {
    for (auto &ei : _elements)
      if (error_code ec = ei->parse(ctx, diagnostics))
        return ec;
    return error_code::success();
  }
};

/// \brief Parse GNU Linker scripts.
class GNULdScript : public FileNode {
public:
  GNULdScript(ELFLinkingContext &ctx, StringRef userPath, int64_t ordinal)
      : FileNode(userPath, ordinal), _elfLinkingContext(ctx),
        _linkerScript(nullptr) {}

  /// \brief Parse the linker script.
  error_code parse(const LinkingContext &, raw_ostream &) override;

protected:
  ELFLinkingContext &_elfLinkingContext;
  std::unique_ptr<script::Parser> _parser;
  std::unique_ptr<script::Lexer> _lexer;
  script::LinkerScript *_linkerScript;
};

/// \brief Handle ELF style with GNU Linker scripts.
class ELFGNULdScript : public GNULdScript {
public:
  ELFGNULdScript(ELFLinkingContext &ctx, StringRef userPath, int64_t ordinal)
      : GNULdScript(ctx, userPath, ordinal) {}

  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) override;

  bool shouldExpand() const override { return true; }

  /// Unused functions for ELFGNULdScript Nodes.
  ErrorOr<File &> getNextFile() override {
    return make_error_code(InputGraphError::no_more_files);
  }

  /// Return the elements that we would want to expand with.
  range<InputGraph::InputElementIterT> expandElements() override {
    return make_range(_expandElements.begin(), _expandElements.end());
  }

  void setResolveState(uint32_t) override {
    llvm_unreachable("cannot use this function: setResolveState");
  }

  uint32_t getResolveState() const override {
    llvm_unreachable("cannot use this function: getResolveState");
  }

  // Do nothing here.
  void resetNextIndex() override {}

private:
  InputGraph::InputElementVectorT _expandElements;
};

} // namespace lld

#endif
