//===- MachOUniversal.h - Mach-O universal binaries -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares Mach-O fat/universal binaries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHOUNIVERSAL_H
#define LLVM_OBJECT_MACHOUNIVERSAL_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/MachOFormat.h"

namespace llvm {
namespace object {

class ObjectFile;

class MachOUniversalBinary : public Binary {
  virtual void anchor();

  uint32_t NumberOfObjects;
public:
  class ObjectForArch {
    const MachOUniversalBinary *Parent;
    /// \brief Index of object in the universal binary.
    uint32_t Index;
    /// \brief Descriptor of the object.
    macho::FatArchHeader Header;

  public:
    ObjectForArch(const MachOUniversalBinary *Parent, uint32_t Index);

    void clear() {
      Parent = 0;
      Index = 0;
    }

    bool operator==(const ObjectForArch &Other) const {
      return (Parent == Other.Parent) && (Index == Other.Index);
    }

    ObjectForArch getNext() const { return ObjectForArch(Parent, Index + 1); }
    uint32_t getCPUType() const { return Header.CPUType; }

    error_code getAsObjectFile(OwningPtr<ObjectFile> &Result) const;
  };

  class object_iterator {
    ObjectForArch Obj;
  public:
    object_iterator(const ObjectForArch &Obj) : Obj(Obj) {}
    const ObjectForArch* operator->() const {
      return &Obj;
    }

    bool operator==(const object_iterator &Other) const {
      return Obj == Other.Obj;
    }
    bool operator!=(const object_iterator &Other) const {
      return !(*this == Other);
    }

    object_iterator& operator++() {  // Preincrement
      Obj = Obj.getNext();
      return *this;
    }
  };

  MachOUniversalBinary(MemoryBuffer *Source, error_code &ec);

  object_iterator begin_objects() const {
    return ObjectForArch(this, 0);
  }
  object_iterator end_objects() const {
    return ObjectForArch(0, 0);
  }

  uint32_t getNumberOfObjects() const { return NumberOfObjects; }

  // Cast methods.
  static inline bool classof(Binary const *V) {
    return V->isMachOUniversalBinary();
  }

  error_code getObjectForArch(Triple::ArchType Arch,
                              OwningPtr<ObjectFile> &Result) const;
};

}
}

#endif
