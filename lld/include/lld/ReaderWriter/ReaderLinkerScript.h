//===- lld/ReaderWriter/ReaderLinkerScript.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_READER_LINKER_SCRIPT_H
#define LLD_READER_WRITER_READER_LINKER_SCRIPT_H

#include "lld/Core/LLVM.h"
#include "lld/ReaderWriter/Reader.h"

namespace lld {
class File;
class LinkerInput;
class TargetInfo;

/// \brief ReaderLinkerScript is a class for reading linker scripts
class ReaderLinkerScript : public Reader {
public:
  ReaderLinkerScript(const TargetInfo &ti)
      : Reader(ti) {}

  /// \brief Returns a vector of Files that are contained in the archive file
  ///        pointed to by the Memorybuffer
  error_code parseFile(std::unique_ptr<llvm::MemoryBuffer> mb,
                       std::vector<std::unique_ptr<File>> &result) const;
};

} // end namespace lld

#endif
