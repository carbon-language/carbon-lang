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
    break;
  }
  return error_code::success();
}

/// \brief Parse the GnuLD Script
error_code GNULdScript::parse(const LinkingContext &ctx,
                              raw_ostream &diagnostics) {
  ErrorOr<StringRef> filePath = getPath(ctx);
  if (!filePath)
    return error_code(filePath);

  if (error_code ec = getBuffer(*filePath))
    return ec;

  if (ctx.logInputFiles())
    diagnostics << *filePath << "\n";

  _lexer.reset(new script::Lexer(std::move(_buffer)));
  _parser.reset(new script::Parser(*_lexer.get()));

  _linkerScript = _parser->parse();

  if (!_linkerScript)
    return LinkerScriptReaderError::parse_error;

  return error_code::success();
}

/// \brief Handle GnuLD script for ELF.
error_code ELFGNULdScript::parse(const LinkingContext &ctx,
                                 raw_ostream &diagnostics) {
  int64_t index = 0;
  if (error_code ec = GNULdScript::parse(ctx, diagnostics))
    return ec;
  for (const auto &c : _linkerScript->_commands) {
    if (auto group = dyn_cast<script::Group>(c)) {
      std::unique_ptr<InputElement> controlStart(
          new ELFGroup(_elfLinkingContext, index++));
      for (auto &path : group->getPaths()) {
        // TODO : Propagate Set WholeArchive/dashlPrefix
        auto inputNode = new ELFFileNode(
            _elfLinkingContext, _elfLinkingContext.allocateString(path._path),
            index++, false, path._asNeeded, false);
        std::unique_ptr<InputElement> inputFile(inputNode);
        dyn_cast<ControlNode>(controlStart.get())
            ->processInputElement(std::move(inputFile));
      }
      _expandElements.push_back(std::move(controlStart));
    }
  }
  return error_code::success();
}
