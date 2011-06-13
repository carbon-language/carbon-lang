//===- Binary.cpp - A generic binary file -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Binary class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace object;

Binary::~Binary() {
  delete Data;
}

Binary::Binary(unsigned int Type, MemoryBuffer *Source)
  : TypeID(Type)
  , Data(Source) {}

StringRef Binary::getData() const {
  return Data->getBuffer();
}

StringRef Binary::getFileName() const {
  return Data->getBufferIdentifier();
}

error_code object::createBinary(MemoryBuffer *Source,
                                OwningPtr<Binary> &Result) {
  // We don't support any at the moment.
  delete Source;
  return object_error::invalid_file_type;
}

error_code object::createBinary(StringRef Path, OwningPtr<Binary> &Result) {
  OwningPtr<MemoryBuffer> File;
  if (error_code ec = MemoryBuffer::getFile(Path, File))
    return ec;
  return createBinary(File.take(), Result);
}
