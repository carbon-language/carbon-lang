//===- FileOutputBuffer.cpp - File Output Buffer ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility for creating a in-memory buffer that will be written to a file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

using llvm::sys::fs::mapped_file_region;

namespace llvm {
FileOutputBuffer::FileOutputBuffer(mapped_file_region * R,
                                   StringRef Path, StringRef TmpPath)
  : Region(R)
  , FinalPath(Path)
  , TempPath(TmpPath) {
}

FileOutputBuffer::~FileOutputBuffer() {
  sys::fs::remove(Twine(TempPath));
}

error_code FileOutputBuffer::create(StringRef FilePath,
                                    size_t Size,
                                    std::unique_ptr<FileOutputBuffer> &Result,
                                    unsigned Flags) {
  // If file already exists, it must be a regular file (to be mappable).
  sys::fs::file_status Stat;
  error_code EC = sys::fs::status(FilePath, Stat);
  switch (Stat.type()) {
    case sys::fs::file_type::file_not_found:
      // If file does not exist, we'll create one.
      break;
    case sys::fs::file_type::regular_file: {
        // If file is not currently writable, error out.
        // FIXME: There is no sys::fs:: api for checking this.
        // FIXME: In posix, you use the access() call to check this.
      }
      break;
    default:
      if (EC)
        return EC;
      else
        return make_error_code(errc::operation_not_permitted);
  }

  // Delete target file.
  EC = sys::fs::remove(FilePath);
  if (EC)
    return EC;

  unsigned Mode = sys::fs::all_read | sys::fs::all_write;
  // If requested, make the output file executable.
  if (Flags & F_executable)
    Mode |= sys::fs::all_exe;

  // Create new file in same directory but with random name.
  SmallString<128> TempFilePath;
  int FD;
  EC = sys::fs::createUniqueFile(Twine(FilePath) + ".tmp%%%%%%%", FD,
                                 TempFilePath, Mode);
  if (EC)
    return EC;

  std::unique_ptr<mapped_file_region> MappedFile(new mapped_file_region(
      FD, true, mapped_file_region::readwrite, Size, 0, EC));
  if (EC)
    return EC;

  Result.reset(new FileOutputBuffer(MappedFile.get(), FilePath, TempFilePath));
  if (Result)
    MappedFile.release();

  return error_code::success();
}

error_code FileOutputBuffer::commit(int64_t NewSmallerSize) {
  // Unmap buffer, letting OS flush dirty pages to file on disk.
  Region.reset(nullptr);

  // If requested, resize file as part of commit.
  if ( NewSmallerSize != -1 ) {
    error_code EC = sys::fs::resize_file(Twine(TempPath), NewSmallerSize);
    if (EC)
      return EC;
  }

  // Rename file to final name.
  return sys::fs::rename(Twine(TempPath), Twine(FinalPath));
}
} // namespace
