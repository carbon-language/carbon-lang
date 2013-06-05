//===- YAML.h - YAMLIO utilities for object files ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares utility classes for handling the YAML representation of
// object files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_YAML_H
#define LLVM_OBJECT_YAML_H

#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace object {
namespace yaml {

/// In an object file this is just a binary blob. In an yaml file it is an hex
/// string. Using this avoid having to allocate temporary strings.
class BinaryRef {
  ArrayRef<uint8_t> Data;
  bool isBinary;

public:
  BinaryRef(ArrayRef<uint8_t> Data) : Data(Data), isBinary(true) {}
  BinaryRef(StringRef Data)
      : Data(reinterpret_cast<const uint8_t *>(Data.data()), Data.size()),
        isBinary(false) {}
  BinaryRef() : isBinary(false) {}
  StringRef getHex() const {
    assert(!isBinary);
    return StringRef(reinterpret_cast<const char *>(Data.data()), Data.size());
  }
  ArrayRef<uint8_t> getBinary() const {
    assert(isBinary);
    return Data;
  }
  bool operator==(const BinaryRef &Ref) {
    // Special case for default constructed BinaryRef.
    if (Ref.Data.empty() && Data.empty())
      return true;

    return Ref.isBinary == isBinary && Ref.Data == Data;
  }
};

}
}

namespace yaml {
template <> struct ScalarTraits<object::yaml::BinaryRef> {
  static void output(const object::yaml::BinaryRef &, void *,
                     llvm::raw_ostream &);
  static StringRef input(StringRef, void *, object::yaml::BinaryRef &);
};
}

}

#endif
