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
    // ELF needs this for relocations. This entire union should probably be a
    // char[max(8, sizeof(uintptr_t))] and require the impl to cast.
    uint16_t a, b;
    uint32_t c;
  } w;
  struct {
    uint32_t a, b;
  } d;
  uintptr_t p;
};

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

static bool operator ==(const DataRefImpl &a, const DataRefImpl &b) {
  // Check bitwise identical. This is the only legal way to compare a union w/o
  // knowing which member is in use.
  return std::memcmp(&a, &b, sizeof(DataRefImpl)) == 0;
}

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

  enum SymbolType {
    ST_Function,
    ST_Data,
    ST_External,    // Defined in another object file
    ST_Other
  };

  SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner);

  bool operator==(const SymbolRef &Other) const;

  error_code getNext(SymbolRef &Result) const;

  error_code getName(StringRef &Result) const;
  error_code getAddress(uint64_t &Result) const;
  error_code getOffset(uint64_t &Result) const;
  error_code getSize(uint64_t &Result) const;
  error_code getSymbolType(SymbolRef::SymbolType &Result) const;

  /// Returns the ascii char that should be displayed in a symbol table dump via
  /// nm for this symbol.
  error_code getNMTypeChar(char &Result) const;

  /// Returns true for symbols that are internal to the object file format such
  /// as section symbols.
  error_code isInternal(bool &Result) const;

  /// Returns true for symbols that can be used in another objects,
  /// such as library functions
  error_code isGlobal(bool &Result) const;
};
typedef content_iterator<SymbolRef> symbol_iterator;

/// RelocationRef - This is a value type class that represents a single
/// relocation in the list of relocations in the object file.
class RelocationRef {
  DataRefImpl RelocationPimpl;
  const ObjectFile *OwningObject;

public:
  RelocationRef() : OwningObject(NULL) {
    std::memset(&RelocationPimpl, 0, sizeof(RelocationPimpl));
  }

  RelocationRef(DataRefImpl RelocationP, const ObjectFile *Owner);

  bool operator==(const RelocationRef &Other) const;

  error_code getNext(RelocationRef &Result) const;

  error_code getAddress(uint64_t &Result) const;
  error_code getSymbol(SymbolRef &Result) const;
  error_code getType(uint32_t &Result) const;

  /// @brief Get a string that represents the type of this relocation.
  ///
  /// This is for display purposes only.
  error_code getTypeName(SmallVectorImpl<char> &Result) const;
  error_code getAdditionalInfo(int64_t &Result) const;

  /// @brief Get a string that represents the calculation of the value of this
  ///        relocation.
  ///
  /// This is for display purposes only.
  error_code getValueString(SmallVectorImpl<char> &Result) const;
};
typedef content_iterator<RelocationRef> relocation_iterator;

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

  /// @brief Get the alignment of this section as the actual value (not log 2).
  error_code getAlignment(uint64_t &Result) const;

  // FIXME: Move to the normalization layer when it's created.
  error_code isText(bool &Result) const;
  error_code isData(bool &Result) const;
  error_code isBSS(bool &Result) const;

  error_code containsSymbol(SymbolRef S, bool &Result) const;

  relocation_iterator begin_relocations() const;
  relocation_iterator end_relocations() const;
};
typedef content_iterator<SectionRef> section_iterator;

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
  virtual error_code getSymbolOffset(DataRefImpl Symb, uint64_t &Res) const =0;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const = 0;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const = 0;
  virtual error_code isSymbolInternal(DataRefImpl Symb, bool &Res) const = 0;
  virtual error_code isSymbolGlobal(DataRefImpl Symb, bool &Res) const = 0;
  virtual error_code getSymbolType(DataRefImpl Symb, SymbolRef::SymbolType &Res) const = 0;

  // Same as above for SectionRef.
  friend class SectionRef;
  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const = 0;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const = 0;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const =0;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const = 0;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res)const=0;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res)const=0;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const = 0;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const = 0;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const = 0;


  // Same as above for RelocationRef.
  friend class RelocationRef;
  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const = 0;
  virtual error_code getRelocationAddress(DataRefImpl Rel,
                                          uint64_t &Res) const =0;
  virtual error_code getRelocationSymbol(DataRefImpl Rel,
                                         SymbolRef &Res) const = 0;
  virtual error_code getRelocationType(DataRefImpl Rel,
                                       uint32_t &Res) const = 0;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                       SmallVectorImpl<char> &Result) const = 0;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const = 0;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                       SmallVectorImpl<char> &Result) const = 0;

public:

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

inline error_code SymbolRef::getOffset(uint64_t &Result) const {
  return OwningObject->getSymbolOffset(SymbolPimpl, Result);
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

inline error_code SymbolRef::isGlobal(bool &Result) const {
  return OwningObject->isSymbolGlobal(SymbolPimpl, Result);
}

inline error_code SymbolRef::getSymbolType(SymbolRef::SymbolType &Result) const {
  return OwningObject->getSymbolType(SymbolPimpl, Result);
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

inline error_code SectionRef::getAlignment(uint64_t &Result) const {
  return OwningObject->getSectionAlignment(SectionPimpl, Result);
}

inline error_code SectionRef::isText(bool &Result) const {
  return OwningObject->isSectionText(SectionPimpl, Result);
}

inline error_code SectionRef::isData(bool &Result) const {
  return OwningObject->isSectionData(SectionPimpl, Result);
}

inline error_code SectionRef::isBSS(bool &Result) const {
  return OwningObject->isSectionBSS(SectionPimpl, Result);
}

inline error_code SectionRef::containsSymbol(SymbolRef S, bool &Result) const {
  return OwningObject->sectionContainsSymbol(SectionPimpl, S.SymbolPimpl,
                                             Result);
}

inline relocation_iterator SectionRef::begin_relocations() const {
  return OwningObject->getSectionRelBegin(SectionPimpl);
}

inline relocation_iterator SectionRef::end_relocations() const {
  return OwningObject->getSectionRelEnd(SectionPimpl);
}


/// RelocationRef
inline RelocationRef::RelocationRef(DataRefImpl RelocationP,
                              const ObjectFile *Owner)
  : RelocationPimpl(RelocationP)
  , OwningObject(Owner) {}

inline bool RelocationRef::operator==(const RelocationRef &Other) const {
  return RelocationPimpl == Other.RelocationPimpl;
}

inline error_code RelocationRef::getNext(RelocationRef &Result) const {
  return OwningObject->getRelocationNext(RelocationPimpl, Result);
}

inline error_code RelocationRef::getAddress(uint64_t &Result) const {
  return OwningObject->getRelocationAddress(RelocationPimpl, Result);
}

inline error_code RelocationRef::getSymbol(SymbolRef &Result) const {
  return OwningObject->getRelocationSymbol(RelocationPimpl, Result);
}

inline error_code RelocationRef::getType(uint32_t &Result) const {
  return OwningObject->getRelocationType(RelocationPimpl, Result);
}

inline error_code RelocationRef::getTypeName(SmallVectorImpl<char> &Result)
  const {
  return OwningObject->getRelocationTypeName(RelocationPimpl, Result);
}

inline error_code RelocationRef::getAdditionalInfo(int64_t &Result) const {
  return OwningObject->getRelocationAdditionalInfo(RelocationPimpl, Result);
}

inline error_code RelocationRef::getValueString(SmallVectorImpl<char> &Result)
  const {
  return OwningObject->getRelocationValueString(RelocationPimpl, Result);
}

} // end namespace object
} // end namespace llvm

#endif
