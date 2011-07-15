//===- ObjectFile.h - File format independent object file -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a file format independent ObjectFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_OBJECT_FILE_H
#define LLVM_OBJECT_OBJECT_FILE_H

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>

namespace llvm {
namespace object {

class ObjectFile;

union DataRefImpl {
  struct {
    uint32_t a, b;
  } d;
  uintptr_t p;
};

static bool operator ==(const DataRefImpl &a, const DataRefImpl &b) {
  // Check bitwise identical. This is the only legal way to compare a union w/o
  // knowing which member is in use.
  return std::memcmp(&a, &b, sizeof(DataRefImpl)) == 0;
}

class RelocationRef {
  DataRefImpl RelocationPimpl;
  const ObjectFile *OwningObject;

public:
  RelocationRef() : OwningObject(NULL) {
    std::memset(&RelocationPimpl, 0, sizeof(RelocationPimpl));
  }

  RelocationRef(DataRefImpl RelocationP, const ObjectFile *Owner);

  bool operator==(const RelocationRef &Other) const;

  error_code getNext(RelocationRef &Result);
};

/// SymbolRef - This is a value type class that represents a single symbol in
/// the list of symbols in the object file.
class SymbolRef {
  friend class SectionRef;
  DataRefImpl SymbolPimpl;
  const ObjectFile *OwningObject;

public:
  SymbolRef() : OwningObject(NULL) {
    std::memset(&SymbolPimpl, 0, sizeof(SymbolPimpl));
  }

  SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner);

  bool operator==(const SymbolRef &Other) const;

  error_code getNext(SymbolRef &Result) const;

  error_code getName(StringRef &Result) const;
  error_code getAddress(uint64_t &Result) const;
  error_code getSize(uint64_t &Result) const;

  /// Returns the ascii char that should be displayed in a symbol table dump via
  /// nm for this symbol.
  error_code getNMTypeChar(char &Result) const;

  /// Returns true for symbols that are internal to the object file format such
  /// as section symbols.
  error_code isInternal(bool &Result) const;
};

/// SectionRef - This is a value type class that represents a single section in
/// the list of sections in the object file.
class SectionRef {
  friend class SymbolRef;
  DataRefImpl SectionPimpl;
  const ObjectFile *OwningObject;

public:
  SectionRef() : OwningObject(NULL) {
    std::memset(&SectionPimpl, 0, sizeof(SectionPimpl));
  }

  SectionRef(DataRefImpl SectionP, const ObjectFile *Owner);

  bool operator==(const SectionRef &Other) const;

  error_code getNext(SectionRef &Result) const;

  error_code getName(StringRef &Result) const;
  error_code getAddress(uint64_t &Result) const;
  error_code getSize(uint64_t &Result) const;
  error_code getContents(StringRef &Result) const;

  // FIXME: Move to the normalization layer when it's created.
  error_code isText(bool &Result) const;

  error_code containsSymbol(SymbolRef S, bool &Result) const;
};

const uint64_t UnknownAddressOrSize = ~0ULL;

/// ObjectFile - This class is the base class for all object file types.
/// Concrete instances of this object are created by createObjectFile, which
/// figure out which type to create.
class ObjectFile : public Binary {
private:
  ObjectFile(); // = delete
  ObjectFile(const ObjectFile &other); // = delete

protected:
  ObjectFile(unsigned int Type, MemoryBuffer *source, error_code &ec);

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Data->getBufferStart());
  }

  // These functions are for SymbolRef to call internally. The main goal of
  // this is to allow SymbolRef::SymbolPimpl to point directly to the symbol
  // entry in the memory mapped object file. SymbolPimpl cannot contain any
  // virtual functions because then it could not point into the memory mapped
  // file.
  //
  // Implementations assume that the DataRefImpl is valid and has not been
  // modified externally. It's UB otherwise.
  friend class SymbolRef;
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const = 0;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const = 0;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const =0;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const = 0;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const = 0;
  virtual error_code isSymbolInternal(DataRefImpl Symb, bool &Res) const = 0;

  // Same as above for SectionRef.
  friend class SectionRef;
  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const = 0;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const = 0;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const =0;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const = 0;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res)const=0;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const = 0;


public:
  template<class content_type>
  class content_iterator {
    content_type Current;
  public:
    content_iterator(content_type symb)
      : Current(symb) {}

    const content_type* operator->() const {
      return &Current;
    }

    const content_type &operator*() const {
      return Current;
    }

    bool operator==(const content_iterator &other) const {
      return Current == other.Current;
    }

    bool operator!=(const content_iterator &other) const {
      return !(*this == other);
    }

    content_iterator& increment(error_code &err) {
      content_type next;
      if (error_code ec = Current.getNext(next))
        err = ec;
      else
        Current = next;
      return *this;
    }
  };

  typedef content_iterator<SymbolRef> symbol_iterator;
  typedef content_iterator<SectionRef> section_iterator;

  virtual symbol_iterator begin_symbols() const = 0;
  virtual symbol_iterator end_symbols() const = 0;

  virtual section_iterator begin_sections() const = 0;
  virtual section_iterator end_sections() const = 0;

  /// @brief The number of bytes used to represent an address in this object
  ///        file format.
  virtual uint8_t getBytesInAddress() const = 0;

  virtual StringRef getFileFormatName() const = 0;
  virtual /* Triple::ArchType */ unsigned getArch() const = 0;

  /// @returns Pointer to ObjectFile subclass to handle this type of object.
  /// @param ObjectPath The path to the object file. ObjectPath.isObject must
  ///        return true.
  /// @brief Create ObjectFile from path.
  static ObjectFile *createObjectFile(StringRef ObjectPath);
  static ObjectFile *createObjectFile(MemoryBuffer *Object);

  static inline bool classof(const Binary *v) {
    return v->getType() >= isObject &&
           v->getType() < lastObject;
  }
  static inline bool classof(const ObjectFile *v) { return true; }

public:
  static ObjectFile *createCOFFObjectFile(MemoryBuffer *Object);
  static ObjectFile *createELFObjectFile(MemoryBuffer *Object);
  static ObjectFile *createMachOObjectFile(MemoryBuffer *Object);
};

// Inline function definitions.
inline SymbolRef::SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner)
  : SymbolPimpl(SymbolP)
  , OwningObject(Owner) {}

inline bool SymbolRef::operator==(const SymbolRef &Other) const {
  return SymbolPimpl == Other.SymbolPimpl;
}

inline error_code SymbolRef::getNext(SymbolRef &Result) const {
  return OwningObject->getSymbolNext(SymbolPimpl, Result);
}

inline error_code SymbolRef::getName(StringRef &Result) const {
  return OwningObject->getSymbolName(SymbolPimpl, Result);
}

inline error_code SymbolRef::getAddress(uint64_t &Result) const {
  return OwningObject->getSymbolAddress(SymbolPimpl, Result);
}

inline error_code SymbolRef::getSize(uint64_t &Result) const {
  return OwningObject->getSymbolSize(SymbolPimpl, Result);
}

inline error_code SymbolRef::getNMTypeChar(char &Result) const {
  return OwningObject->getSymbolNMTypeChar(SymbolPimpl, Result);
}

inline error_code SymbolRef::isInternal(bool &Result) const {
  return OwningObject->isSymbolInternal(SymbolPimpl, Result);
}


/// SectionRef
inline SectionRef::SectionRef(DataRefImpl SectionP,
                              const ObjectFile *Owner)
  : SectionPimpl(SectionP)
  , OwningObject(Owner) {}

inline bool SectionRef::operator==(const SectionRef &Other) const {
  return SectionPimpl == Other.SectionPimpl;
}

inline error_code SectionRef::getNext(SectionRef &Result) const {
  return OwningObject->getSectionNext(SectionPimpl, Result);
}

inline error_code SectionRef::getName(StringRef &Result) const {
  return OwningObject->getSectionName(SectionPimpl, Result);
}

inline error_code SectionRef::getAddress(uint64_t &Result) const {
  return OwningObject->getSectionAddress(SectionPimpl, Result);
}

inline error_code SectionRef::getSize(uint64_t &Result) const {
  return OwningObject->getSectionSize(SectionPimpl, Result);
}

inline error_code SectionRef::getContents(StringRef &Result) const {
  return OwningObject->getSectionContents(SectionPimpl, Result);
}

inline error_code SectionRef::isText(bool &Result) const {
  return OwningObject->isSectionText(SectionPimpl, Result);
}

inline error_code SectionRef::containsSymbol(SymbolRef S, bool &Result) const {
  return OwningObject->sectionContainsSymbol(SectionPimpl, S.SymbolPimpl,
                                             Result);
}

} // end namespace object
} // end namespace llvm

#endif
