//===- lib/Driver/GnuLdInputGraph.cpp -------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/GnuLdInputGraph.h"
#include "lld/ReaderWriter/LinkerScript.h"

using namespace lld;

/// \brief Parse the input file to lld::File.
error_code ELFFileNode::parse(const LinkingContext &ctx,
                              raw_ostream &diagnostics) {
  ErrorOr<StringRef> filePath = getPath(ctx);
  if (!filePath)
    return error_code(filePath);

  if (error_code ec = getBuffer(*filePath))
    return ec;

  if (ctx.logInputFiles())
    diagnostics << *filePath << "\n";

  if (filePath->endswith(".objtxt"))
    return ctx.getYAMLReader().parseFile(_buffer, _files);

  // Identify File type
  llvm::sys::fs::file_magic FileType =
      llvm::sys::fs::identify_magic(_buffer->getBuffer());

  switch (FileType) {
  case llvm::sys::fs::file_magic::elf_relocatable:
  case llvm::sys::fs::file_magic::elf_shared_object:
    // Call the default reader to read object files and shared objects
    return _elfLinkingContext.getDefaultReader().parseFile(_buffer, _files);

  case llvm::sys::fs::file_magic::archive: {
    // Process archive files. If Whole Archive option is set,
    // parse all members of the archive.
    error_code ec;
    std::unique_ptr<FileArchive> fileArchive(
        new FileArchive(ctx, std::move(_buffer), ec, _isWholeArchive));
    if (_isWholeArchive) {
      fileArchive->parseAllMembers(_files);
      _archiveFile = std::move(fileArchive);
    } else {
      _files.push_back(std::move(fileArchive));
    }
    return ec;
  }
  default:
    // Process Linker script
    return _elfLinkingContext.getLinkerScriptReader().parseFile(_buffer,
                                                                _files);
    break;
  }
}
