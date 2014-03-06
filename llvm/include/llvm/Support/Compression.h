//===-- llvm/Support/Compression.h ---Compression----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains basic functions for compression/uncompression.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMPRESSION_H
#define LLVM_SUPPORT_COMPRESSION_H

#include "llvm/Support/DataTypes.h"
#include <memory>

namespace llvm {

class MemoryBuffer;
class StringRef;

namespace zlib {

enum CompressionLevel {
  NoCompression,
  DefaultCompression,
  BestSpeedCompression,
  BestSizeCompression
};

enum Status {
  StatusOK,
  StatusUnsupported,    // zlib is unavailable
  StatusOutOfMemory,    // there was not enough memory
  StatusBufferTooShort, // there was not enough room in the output buffer
  StatusInvalidArg,     // invalid input parameter
  StatusInvalidData     // data was corrupted or incomplete
};

bool isAvailable();

Status compress(StringRef InputBuffer,
                std::unique_ptr<MemoryBuffer> &CompressedBuffer,
                CompressionLevel Level = DefaultCompression);

Status uncompress(StringRef InputBuffer,
                  std::unique_ptr<MemoryBuffer> &UncompressedBuffer,
                  size_t UncompressedSize);

uint32_t crc32(StringRef Buffer);

}  // End of namespace zlib

} // End of namespace llvm

#endif

