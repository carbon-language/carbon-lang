//===- lld/ReaderWriter/Reader.h - Abstract File Format Reading Interface -===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_READER_H
#define LLD_READERWRITER_READER_H

#include "lld/Core/LLVM.h"

#include <functional>
#include <memory>
#include <vector>

namespace lld {
class ELFTargetInfo;
class File;
class LinkerInput;
class TargetInfo;

/// \brief An abstract class for reading object files, library files, and
/// executable files.
///
/// Each file format (e.g. ELF, mach-o, PECOFF, native, etc) have a concrete
/// subclass of Reader.
class Reader {
public:
  virtual ~Reader();

  /// \brief Parse a file given its file system path and create a File object.
  virtual error_code readFile(StringRef path,
                              std::vector<std::unique_ptr<File>> &result) const;

  /// \brief Parse a supplied buffer (already filled with the contents of a
  /// file) and create a File object.
  ///
  /// On success, the resulting File object takes ownership of the MemoryBuffer.
  virtual error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                          std::vector<std::unique_ptr<File>> &result) const = 0;

protected:
  // only concrete subclasses can be instantiated
  Reader(const TargetInfo &ti)
      : _targetInfo(ti) {}

  const TargetInfo &_targetInfo;
};

typedef ErrorOr<Reader &> ReaderFunc(const LinkerInput &);

std::unique_ptr<Reader> createReaderELF(const ELFTargetInfo &);
std::unique_ptr<Reader> createReaderMachO(const TargetInfo &);
std::unique_ptr<Reader> createReaderNative(const TargetInfo &);
std::unique_ptr<Reader> createReaderPECOFF(const TargetInfo &,
                                           std::function<ReaderFunc>);
std::unique_ptr<Reader> createReaderYAML(const TargetInfo &);
} // end namespace lld

#endif
