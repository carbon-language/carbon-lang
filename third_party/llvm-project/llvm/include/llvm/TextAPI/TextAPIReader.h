//===--- TextAPIReader.h - Text API Reader ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_TEXTAPIREADER_H
#define LLVM_TEXTAPI_TEXTAPIREADER_H

#include "llvm/Support/Error.h"

namespace llvm {

class MemoryBufferRef;

namespace MachO {

class InterfaceFile;

class TextAPIReader {
public:
  static Expected<std::unique_ptr<InterfaceFile>>
  get(MemoryBufferRef InputBuffer);

  TextAPIReader() = delete;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_TEXTAPIREADER_H
