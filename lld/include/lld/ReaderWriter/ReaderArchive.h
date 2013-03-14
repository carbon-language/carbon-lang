//===- lld/ReaderWriter/ReaderArchive.h - Archive Library Reader ----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_ARCHIVE_H
#define LLD_READER_ARCHIVE_H

#include "lld/Core/LLVM.h"
#include "lld/ReaderWriter/Reader.h"

#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"

#include <functional>
#include <memory>
#include <vector>

namespace lld {
class File;
class LinkerInput;
class TargetInfo;

/// \brief ReaderArchive is a class for reading archive libraries
class ReaderArchive : public Reader {
public:
  ReaderArchive(const TargetInfo &ti,
                std::function<ErrorOr<Reader&> (const LinkerInput &)> getReader)
      : Reader(ti),
        _getReader(getReader) {}

  /// \brief Returns a vector of Files that are contained in the archive file
  ///        pointed to by the Memorybuffer
  error_code parseFile(std::unique_ptr<llvm::MemoryBuffer> mb,
                       std::vector<std::unique_ptr<File>> &result);

private:
  std::function<ErrorOr<Reader&> (const LinkerInput &)> _getReader;
  std::unique_ptr<llvm::object::Archive> _archive;
};
} // end namespace lld

#endif
