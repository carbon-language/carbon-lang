//===- Binary.h - A generic binary file -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Binary class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_BINARY_H
#define LLVM_OBJECT_BINARY_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Object/Error.h"

namespace llvm {

class MemoryBuffer;
class StringRef;

namespace object {

class Binary {
private:
  Binary(); // = delete
  Binary(const Binary &other); // = delete

  unsigned int TypeID;

protected:
  MemoryBuffer *Data;

  Binary(unsigned int Type, MemoryBuffer *Source);

  enum {
    isArchive,
    isCOFF,
    isELF,
    isMachO,
    isObject
  };

public:
  virtual ~Binary();

  StringRef getData() const;
  StringRef getFileName() const;

  // Cast methods.
  unsigned int getType() const { return TypeID; }
  static inline bool classof(Binary const *v) { return true; }
};

error_code createBinary(MemoryBuffer *Source, OwningPtr<Binary> &Result);
error_code createBinary(StringRef Path, OwningPtr<Binary> &Result);

}
}

#endif
