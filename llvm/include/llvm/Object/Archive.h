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

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace object {
struct ArchiveMemberHeader {
  char Name[16];
  char LastModified[12];
  char UID[6];
  char GID[6];
  char AccessMode[8];
  char Size[10]; ///< Size of data, not including header or padding.
  char Terminator[2];

  /// Get the name without looking up long names.
  llvm::StringRef getName() const;

  /// Members are not larger than 4GB.
  uint32_t getSize() const;
};

class Archive : public Binary {
  virtual void anchor();
public:
  class Child {
    const Archive *Parent;
    /// \brief Includes header but not padding byte.
    StringRef Data;
    /// \brief Offset from Data to the start of the file.
    uint16_t StartOfFile;

    const ArchiveMemberHeader *getHeader() const {
      return reinterpret_cast<const ArchiveMemberHeader *>(Data.data());
    }

  public:
    Child(const Archive *Parent, const char *Start);

    bool operator ==(const Child &other) const {
      assert(Parent == other.Parent);
      return Data.begin() == other.Data.begin();
    }

    bool operator <(const Child &other) const {
      return Data.begin() < other.Data.begin();
    }

    Child getNext() const;

    error_code getName(StringRef &Result) const;
    StringRef getRawName() const { return getHeader()->getName(); }
    /// \return the size of the archive member without the header or padding.
    uint64_t getSize() const { return Data.size() - StartOfFile; }

    StringRef getBuffer() const {
      return StringRef(Data.data() + StartOfFile, getSize());
    }

    error_code getMemoryBuffer(OwningPtr<MemoryBuffer> &Result,
                               bool FullPath = false) const;

    error_code getAsBinary(OwningPtr<Binary> &Result) const;
  };

  class child_iterator {
    Child child;
  public:
    child_iterator() : child(Child(0, 0)) {}
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

    bool operator <(const child_iterator &other) const {
      return child < other.child;
    }

    child_iterator& operator++() {  // Preincrement
      child = child.getNext();
      return *this;
    }
  };

  class Symbol {
    const Archive *Parent;
    uint32_t SymbolIndex;
    uint32_t StringIndex; // Extra index to the string.

  public:
    bool operator ==(const Symbol &other) const {
      return (Parent == other.Parent) && (SymbolIndex == other.SymbolIndex);
    }

    Symbol(const Archive *p, uint32_t symi, uint32_t stri)
      : Parent(p)
      , SymbolIndex(symi)
      , StringIndex(stri) {}
    error_code getName(StringRef &Result) const;
    error_code getMember(child_iterator &Result) const;
    Symbol getNext() const;
  };

  class symbol_iterator {
    Symbol symbol;
  public:
    symbol_iterator(const Symbol &s) : symbol(s) {}
    const Symbol *operator->() const {
      return &symbol;
    }

    bool operator==(const symbol_iterator &other) const {
      return symbol == other.symbol;
    }

    bool operator!=(const symbol_iterator &other) const {
      return !(*this == other);
    }

    symbol_iterator& operator++() {  // Preincrement
      symbol = symbol.getNext();
      return *this;
    }
  };

  Archive(MemoryBuffer *source, error_code &ec);

  enum Kind {
    K_GNU,
    K_BSD,
    K_COFF
  };

  Kind kind() const { 
    return Format;
  }

  child_iterator begin_children(bool skip_internal = true) const;
  child_iterator end_children() const;

  symbol_iterator begin_symbols() const;
  symbol_iterator end_symbols() const;

  // Cast methods.
  static inline bool classof(Binary const *v) {
    return v->isArchive();
  }

  // check if a symbol is in the archive
  child_iterator findSym(StringRef name) const;

private:
  child_iterator SymbolTable;
  child_iterator StringTable;
  Kind Format;
};

}
}

#endif
