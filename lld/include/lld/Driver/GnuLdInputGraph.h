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

#include "lld/Core/Resolver.h"
#include "lld/Driver/InputGraph.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "lld/ReaderWriter/FileArchive.h"

namespace lld {

/// \brief Represents a ELF File
class ELFFileNode : public FileNode {
public:
  ELFFileNode(ELFLinkingContext &ctx, StringRef path,
              std::vector<StringRef> searchPath, int64_t ordinal = -1,
              bool isWholeArchive = false, bool asNeeded = false,
              bool dashlPrefix = false)
      : FileNode(path, ordinal), _elfLinkingContext(ctx),
        _isWholeArchive(isWholeArchive), _asNeeded(asNeeded),
        _isDashlPrefix(dashlPrefix) {
    std::copy(searchPath.begin(), searchPath.end(),
              std::back_inserter(_libraryPaths));
  }

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  virtual ErrorOr<StringRef> getPath(const LinkingContext &ctx) const;

  /// \brief validates the Input Element
  virtual bool validate() { return true; }

  /// \brief create an error string for printing purposes
  virtual std::string errStr(llvm::error_code);

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
    diagnostics << "  contextPath : " << ((_libraryPaths.size()) ? "" : "None")
                << "\n";
    for (auto path : _libraryPaths)
      diagnostics << "    - " << path << "\n";
    return true;
  }

  /// \brief Parse the input file to lld::File.
  llvm::error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) {
    ErrorOr<StringRef> filePath = getPath(ctx);
    if (!filePath &&
        error_code(filePath) == llvm::errc::no_such_file_or_directory)
      return make_error_code(llvm::errc::no_such_file_or_directory);

    // Create a memory buffer
    OwningPtr<llvm::MemoryBuffer> opmb;
    llvm::error_code ec;

    if ((ec = llvm::MemoryBuffer::getFileOrSTDIN(*filePath, opmb)))
      return ec;

    std::unique_ptr<MemoryBuffer> mb(opmb.take());
    _buffer = std::move(mb);

    // If tracing is enabled, print the files being processed.
    if (ctx.logInputFiles())
      diagnostics << _buffer->getBufferIdentifier() << "\n";

    // YAML file is identified by a .objtxt extension
    // FIXME : Identify YAML files by using a magic
    if (filePath->endswith(".objtxt")) {
      ec = _elfLinkingContext.getYAMLReader().parseFile(_buffer, _files);
      if (!ec)
        return ec;
    }

    // Identify File type
    llvm::sys::fs::file_magic FileType =
        llvm::sys::fs::identify_magic(_buffer->getBuffer());

    switch (FileType) {
    case llvm::sys::fs::file_magic::elf_relocatable:
    case llvm::sys::fs::file_magic::elf_shared_object:
      // Call the default reader to read object files and shared objects
      ec = _elfLinkingContext.getDefaultReader().parseFile(_buffer, _files);
      break;

    case llvm::sys::fs::file_magic::archive: {
      // Process archive files. If Whole Archive option is set,
      // parse all members of the archive.
      std::unique_ptr<FileArchive> fileArchive(
          new FileArchive(ctx, std::move(_buffer), ec, _isWholeArchive));
      if (_isWholeArchive) {
        fileArchive->parseAllMembers(_files);
        _archiveFile = std::move(fileArchive);
      } else {
        _files.push_back(std::move(fileArchive));
      }
      break;
    }

    default:
      // Process Linker script
      _elfLinkingContext.getLinkerScriptReader().parseFile(_buffer, _files);
      break;
    }
    return ec;
  }

  /// \brief This is used by Group Nodes, when there is a need to reset the
  /// the file to be processed next. When handling a group node that contains
  /// Input elements, if the group node has to be reprocessed, the linker needs
  /// to start processing files as part of the inputelement from beginning.
  /// reset the next file index to 0 only if the node is an archive library or
  /// a shared library
  virtual void resetNextFileIndex() {
    if ((!_isWholeArchive && (_files[0]->kind() == File::kindArchiveLibrary)) ||
        (_files[0]->kind() == File::kindSharedLibrary))
      _nextFileIndex = 0;
    return;
  }

  /// \brief Return the file that has to be processed by the resolver
  /// to resolve atoms. This iterates over all the files thats part
  /// of this node. Returns no_more_files when there are no files to be
  /// processed
  virtual ErrorOr<File &> getNextFile() {
    if (_nextFileIndex == _files.size())
      return make_error_code(input_graph_error::no_more_files);
    return *_files[_nextFileIndex++];
  }

private:
  llvm::BumpPtrAllocator _alloc;
  const ELFLinkingContext &_elfLinkingContext;
  bool _isWholeArchive;
  bool _asNeeded;
  bool _isDashlPrefix;
  std::vector<StringRef> _libraryPaths;
  std::unique_ptr<FileArchive> _archiveFile;
};

/// \brief Represents a ELF control node
class ELFGroup : public Group {
public:
  ELFGroup(ELFLinkingContext &ctx) : Group(), _elfLinkingContext(ctx) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::Control;
  }

  /// \brief Validate the options
  virtual bool validate() {
    (void)_elfLinkingContext;
    return true;
  }

  /// \brief Dump the ELFGroup
  virtual bool dump(llvm::raw_ostream &) { return true; }

  /// \brief Parse the group members.
  llvm::error_code parse(const LinkingContext &ctx, raw_ostream &diagnostics) {
    for (auto &ei : _elements)
      if (error_code ec = ei->parse(ctx, diagnostics))
        return ec;
    return error_code::success();
  }

  /// \brief Return the file that has to be processed by the resolver
  /// to resolve atoms. This iterates over all the elements thats part
  /// of this node.
  virtual ErrorOr<File &> getNextFile() {
    // Does the linker need to process the elements again ?
    bool again = false;
    // If there are no elements, move on to the next input element
    if (_elements.size() == 0)
      return make_error_code(input_graph_error::no_more_files);
    // If we have processed all the elements as part of this node
    // check the resolver status for each input element and if the status
    // has not changed, move onto the next file.
    if (_nextElementIndex == _elements.size()) {
      for (auto &elem : _elements) {
        if (elem->getResolverState() == Resolver::StateNoChange) {
          again = true;
          break;
        }
      }
      if (!again)
        return make_error_code(input_graph_error::no_more_files);
      _nextElementIndex = 0;
      // Reset the next file to be processed as part of each element
      for (auto &elem : _elements)
        elem->resetNextFileIndex();
    }
    auto file = _elements[_nextElementIndex]->getNextFile();
    // Move on to the next element if we have finished processing all
    // the files in the input element
    if (error_code(file) == input_graph_error::no_more_files)
      _nextElementIndex++;
    else
      return *file;
    return getNextFile();
  }

private:
  const ELFLinkingContext &_elfLinkingContext;
};

} // namespace lld

#endif
