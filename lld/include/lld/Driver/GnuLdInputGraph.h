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
  /// \brief The attributes class provides a way for a input file to look into
  /// all the positional attributes that were specified in the command line.
  /// There are few positional operators and the number of arguments to the
  /// ELFFileNode class keeps growing. This achieves code to be clean as well.
  class Attributes {
  public:
    Attributes()
        : _isWholeArchive(false), _asNeeded(false), _isDashlPrefix(false),
          _isSysRooted(false) {}
    void setWholeArchive(bool isWholeArchive) {
      _isWholeArchive = isWholeArchive;
    }
    void setAsNeeded(bool asNeeded) { _asNeeded = asNeeded; }
    void setDashlPrefix(bool isDashlPrefix) { _isDashlPrefix = isDashlPrefix; }
    void setSysRooted(bool isSysRooted) { _isSysRooted = isSysRooted; }

  public:
    bool _isWholeArchive;
    bool _asNeeded;
    bool _isDashlPrefix;
    bool _isSysRooted;
  };

  ELFFileNode(ELFLinkingContext &ctx, StringRef path, Attributes &attributes)
      : FileNode(path), _elfLinkingContext(ctx), _attributes(attributes) {}

  ErrorOr<StringRef> getPath(const LinkingContext &ctx) const override;

  /// \brief create an error string for printing purposes
  std::string errStr(error_code) override;

  /// \brief Dump the Input Element
  bool dump(raw_ostream &diagnostics) override {
    diagnostics << "Name    : " << *getPath(_elfLinkingContext) << "\n"
                << "Type    : ELF File\n"
                << "Attributes :\n"
                << "  - wholeArchive : "
                << ((_attributes._isWholeArchive) ? "true" : "false") << "\n"
                << "  - asNeeded : "
                << ((_attributes._asNeeded) ? "true" : "false") << "\n";
    return true;
  }

  /// \brief Parse the input file to lld::File.
  error_code parse(const LinkingContext &, raw_ostream &) override;

  /// \brief This is used by Group Nodes, when there is a need to reset the
  /// the file to be processed next. When handling a group node that contains
  /// Input elements, if the group node has to be reprocessed, the linker needs
  /// to start processing files as part of the input element from beginning.
  /// Reset the next file index to 0 only if the node is an archive library.
  void resetNextIndex() override {
    auto kind = _files[0]->kind();
    if (kind == File::kindSharedLibrary ||
        (kind == File::kindArchiveLibrary && !_attributes._isWholeArchive)) {
      _nextFileIndex = 0;
    }
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
  std::unique_ptr<const ArchiveLibraryFile> _archiveFile;
  const Attributes _attributes;
};

/// \brief Parse GNU Linker scripts.
class GNULdScript : public FileNode {
public:
  GNULdScript(ELFLinkingContext &ctx, StringRef userPath)
      : FileNode(userPath), _elfLinkingContext(ctx), _linkerScript(nullptr) {}

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
  ELFGNULdScript(ELFLinkingContext &ctx, StringRef userPath)
      : GNULdScript(ctx, userPath) {}

  error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) override;

  bool getReplacements(InputGraph::InputElementVectorT &result) override {
    for (std::unique_ptr<InputElement> &elt : _expandElements)
      result.push_back(std::move(elt));
    return true;
  }

  /// Unused functions for ELFGNULdScript Nodes.
  ErrorOr<File &> getNextFile() override {
    return make_error_code(InputGraphError::no_more_files);
  }

  // Linker Script will be expanded and replaced with other elements
  // by InputGraph::normalize(), so at link time it does not exist in
  // the tree. No need to handle this message.
  void resetNextIndex() override {}

private:
  InputGraph::InputElementVectorT _expandElements;
};

} // namespace lld

#endif
