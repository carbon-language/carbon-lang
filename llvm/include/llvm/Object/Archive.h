//===- Archive.h - ar archive file format -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ar archive file format class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ARCHIVE_H
#define LLVM_OBJECT_ARCHIVE_H

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
namespace object {

class Archive : public Binary {
public:
  class Child {
    const Archive *Parent;
    StringRef Data;

  public:
    Child(const Archive *p, StringRef d) : Parent(p), Data(d) {}

    bool operator ==(const Child &other) const {
      return (Parent == other.Parent) && (Data.begin() == other.Data.begin());
    }

    Child getNext() const;
    error_code getName(StringRef &Result) const;
    int getLastModified() const;
    int getUID() const;
    int getGID() const;
    int getAccessMode() const;
    ///! Return the size of the archive member without the header or padding.
    uint64_t getSize() const;

    MemoryBuffer *getBuffer() const;
    error_code getAsBinary(OwningPtr<Binary> &Result) const;
  };

  class child_iterator {
    Child child;
  public:
    child_iterator(const Child &c) : child(c) {}
    const Child* operator->() const {
      return &child;
    }

    bool operator==(const child_iterator &other) const {
      return child == other.child;
    }

    bool operator!=(const child_iterator &other) const {
      return !(*this == other);
    }

    child_iterator& operator++() {  // Preincrement
      child = child.getNext();
      return *this;
    }
  };

  Archive(MemoryBuffer *source, error_code &ec);

  child_iterator begin_children() const;
  child_iterator end_children() const;

  // Cast methods.
  static inline bool classof(Archive const *v) { return true; }
  static inline bool classof(Binary const *v) {
    return v->getType() == Binary::isArchive;
  }

private:
  child_iterator StringTable;
};

}
}

#endif
