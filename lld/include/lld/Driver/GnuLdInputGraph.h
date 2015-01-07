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
  std::string errStr(std::error_code) override;

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
  std::error_code parse(const LinkingContext &, raw_ostream &) override;

private:
  llvm::BumpPtrAllocator _alloc;
  const ELFLinkingContext &_elfLinkingContext;
  std::unique_ptr<const ArchiveLibraryFile> _archiveFile;
  const Attributes _attributes;
};

} // namespace lld

#endif
