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

  virtual ErrorOr<StringRef> getPath(const LinkingContext &ctx) const;

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief create an error string for printing purposes
  virtual std::string errStr(error_code);

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &diagnostics) {
    diagnostics << "Name    : " << *getPath(_elfLinkingContext) << "\n";
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
    return true;
  }

  /// \brief Parse the input file to lld::File.
  error_code parse(const LinkingContext &, raw_ostream &);

  /// \brief This is used by Group Nodes, when there is a need to reset the
  /// the file to be processed next. When handling a group node that contains
  /// Input elements, if the group node has to be reprocessed, the linker needs
  /// to start processing files as part of the inputelement from beginning.
  /// reset the next file index to 0 only if the node is an archive library or
  /// a shared library
  virtual void resetNextIndex() {
    if ((!_isWholeArchive && (_files[0]->kind() == File::kindArchiveLibrary)) ||
        (_files[0]->kind() == File::kindSharedLibrary))
      _nextFileIndex = 0;
    setResolveState(Resolver::StateNoChange);
  }

  /// \brief Return the file that has to be processed by the resolver
  /// to resolve atoms. This iterates over all the files thats part
  /// of this node. Returns no_more_files when there are no files to be
  /// processed
  virtual ErrorOr<File &> getNextFile() {
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
  ELFGroup(const ELFLinkingContext &ctx, int64_t ordinal)
      : Group(ordinal), _elfLinkingContext(ctx) {}

  /// \brief Validate the options
  virtual bool validate() {
    (void)_elfLinkingContext;
    return true;
  }

  /// \brief Dump the ELFGroup
  virtual bool dump(raw_ostream &) { return true; }

  /// \brief Parse the group members.
  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) {
    for (auto &ei : _elements)
      if (error_code ec = ei->parse(ctx, diagnostics))
        return ec;
    return error_code::success();
  }

private:
  const ELFLinkingContext &_elfLinkingContext;
};

/// \brief Parse GNU Linker scripts.
class GNULdScript : public FileNode {
public:
  GNULdScript(ELFLinkingContext &ctx, StringRef userPath, int64_t ordinal)
      : FileNode(userPath, ordinal), _elfLinkingContext(ctx),
        _linkerScript(nullptr)
  {}

  /// \brief Is this node part of resolution ?
  virtual bool isHidden() const { return true; }

  /// \brief Validate the options
  virtual bool validate() {
    (void)_elfLinkingContext;
    return true;
  }

  /// \brief Dump the Linker script.
  virtual bool dump(raw_ostream &) { return true; }

  /// \brief Parse the linker script.
  virtual error_code parse(const LinkingContext &, raw_ostream &);

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

  virtual error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics);

  virtual ExpandType expandType() const {
    return InputElement::ExpandType::ExpandOnly;
  }

  /// Unused functions for ELFGNULdScript Nodes.
  virtual ErrorOr<File &> getNextFile() {
    return make_error_code(InputGraphError::no_more_files);
  }

  /// Return the elements that we would want to expand with.
  range<InputGraph::InputElementIterT> expandElements() {
    return make_range(_expandElements.begin(), _expandElements.end());
  }

  virtual void setResolveState(uint32_t) {
    llvm_unreachable("cannot use this function: setResolveState");
  }

  virtual uint32_t getResolveState() const {
    llvm_unreachable("cannot use this function: getResolveState");
  }

  // Do nothing here.
  virtual void resetNextIndex() {}

private:
  InputGraph::InputElementVectorT _expandElements;
};

} // namespace lld

#endif
