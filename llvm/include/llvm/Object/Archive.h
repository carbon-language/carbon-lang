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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/DataTypes.h"
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

  ///! Get the name without looking up long names.
  llvm::StringRef getName() const {
    char EndCond;
    if (Name[0] == '/' || Name[0] == '#')
      EndCond = ' ';
    else
      EndCond = '/';
    llvm::StringRef::size_type end =
        llvm::StringRef(Name, sizeof(Name)).find(EndCond);
    if (end == llvm::StringRef::npos)
      end = sizeof(Name);
    assert(end <= sizeof(Name) && end > 0);
    // Don't include the EndCond if there is one.
    return llvm::StringRef(Name, end);
  }

  uint64_t getSize() const {
    uint64_t ret;
    if (llvm::StringRef(Size, sizeof(Size)).rtrim(" ").getAsInteger(10, ret))
      llvm_unreachable("Size is not an integer.");
    return ret;
  }
};

static const ArchiveMemberHeader *ToHeader(const char *base) {
  return reinterpret_cast<const ArchiveMemberHeader *>(base);
}

class Archive : public Binary {
  virtual void anchor();
public:
  class Child {
    const Archive *Parent;
    /// \brief Includes header but not padding byte.
    StringRef Data;
    /// \brief Offset from Data to the start of the file.
    uint16_t StartOfFile;

  public:
    Child(const Archive *p, StringRef d) : Parent(p), Data(d) {
      if (!p || d.empty())
        return;
      // Setup StartOfFile and PaddingBytes.
      StartOfFile = sizeof(ArchiveMemberHeader);
      // Don't include attached name.
      StringRef Name = ToHeader(Data.data())->getName();
      if (Name.startswith("#1/")) {
        uint64_t NameSize;
        if (Name.substr(3).rtrim(" ").getAsInteger(10, NameSize))
          llvm_unreachable("Long name length is not an integer");
        StartOfFile += NameSize;
      }
    }

    bool operator ==(const Child &other) const {
      return (Parent == other.Parent) && (Data.begin() == other.Data.begin());
    }

    bool operator <(const Child &other) const {
      return Data.begin() < other.Data.begin();
    }

    Child getNext() const {
      size_t SpaceToSkip = Data.size();
      // If it's odd, add 1 to make it even.
      if (SpaceToSkip & 1)
        ++SpaceToSkip;

      const char *NextLoc = Data.data() + SpaceToSkip;

      // Check to see if this is past the end of the archive.
      if (NextLoc >= Parent->Data->getBufferEnd())
        return Child(Parent, StringRef(0, 0));

      size_t NextSize =
          sizeof(ArchiveMemberHeader) + ToHeader(NextLoc)->getSize();

      return Child(Parent, StringRef(NextLoc, NextSize));
    }

    error_code getName(StringRef &Result) const;
    int getLastModified() const;
    int getUID() const;
    int getGID() const;
    int getAccessMode() const;
    /// \return the size of the archive member without the header or padding.
    uint64_t getSize() const { return Data.size() - StartOfFile; }

    StringRef getBuffer() const {
      return StringRef(Data.data() + StartOfFile, getSize());
    }

    error_code getMemoryBuffer(OwningPtr<MemoryBuffer> &Result,
                               bool FullPath = false) const {
      StringRef Name;
      if (error_code ec = getName(Name))
        return ec;
      SmallString<128> Path;
      Result.reset(MemoryBuffer::getMemBuffer(
          getBuffer(), FullPath ? (Twine(Parent->getFileName()) + "(" + Name +
                                   ")").toStringRef(Path) : Name, false));
      return error_code::success();
    }

    error_code getAsBinary(OwningPtr<Binary> &Result) const;
  };

  class child_iterator {
    Child child;
  public:
    child_iterator() : child(Child(0, StringRef())) {}
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
