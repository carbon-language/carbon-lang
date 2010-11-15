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

#include "llvm/ADT/Triple.h"
#include "llvm/System/Path.h"

namespace llvm {

class MemoryBuffer;

namespace object {

class ObjectFile;
typedef uint64_t DataRefImpl;

/// SymbolRef - This is a value type class that represents a single symbol in
/// the list of symbols in the object file.
class SymbolRef {
  DataRefImpl SymbolPimpl;
  const ObjectFile *OwningObject;

public:
  SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner);

  bool operator==(const SymbolRef &Other) const;

  SymbolRef getNext() const;

  StringRef getName() const;
  uint64_t  getAddress() const;
  uint64_t  getSize() const;

  /// Returns the ascii char that should be displayed in a symbol table dump via
  /// nm for this symbol.
  char      getNMTypeChar() const;

  /// Returns true for symbols that are internal to the object file format such
  /// as section symbols.
  bool      isInternal() const;
};

/// SectionRef - This is a value type class that represents a single section in
/// the list of sections in the object file.
class SectionRef {
  DataRefImpl SectionPimpl;
  const ObjectFile *OwningObject;

public:
  SectionRef(DataRefImpl SectionP, const ObjectFile *Owner);

  bool operator==(const SectionRef &Other) const;

  SectionRef getNext() const;

  StringRef getName() const;
  uint64_t  getAddress() const;
  uint64_t  getSize() const;
  StringRef getContents() const;

  // FIXME: Move to the normalization layer when it's created.
  bool      isText() const;
};

const uint64_t UnknownAddressOrSize = ~0ULL;

/// ObjectFile - This class is the base class for all object file types.
/// Concrete instances of this object are created by createObjectFile, which
/// figure out which type to create.
class ObjectFile {
private:
  ObjectFile(); // = delete
  ObjectFile(const ObjectFile &other); // = delete

protected:
  MemoryBuffer *MapFile;
  const uint8_t *base;

  ObjectFile(MemoryBuffer *Object);

  // These functions are for SymbolRef to call internally. The main goal of
  // this is to allow SymbolRef::SymbolPimpl to point directly to the symbol
  // entry in the memory mapped object file. SymbolPimpl cannot contain any
  // virtual functions because then it could not point into the memory mapped
  // file.
  friend class SymbolRef;
  virtual SymbolRef getSymbolNext(DataRefImpl Symb) const = 0;
  virtual StringRef getSymbolName(DataRefImpl Symb) const = 0;
  virtual uint64_t  getSymbolAddress(DataRefImpl Symb) const = 0;
  virtual uint64_t  getSymbolSize(DataRefImpl Symb) const = 0;
  virtual char      getSymbolNMTypeChar(DataRefImpl Symb) const = 0;
  virtual bool      isSymbolInternal(DataRefImpl Symb) const = 0;

  // Same as above for SectionRef.
  friend class SectionRef;
  virtual SectionRef getSectionNext(DataRefImpl Sec) const = 0;
  virtual StringRef  getSectionName(DataRefImpl Sec) const = 0;
  virtual uint64_t   getSectionAddress(DataRefImpl Sec) const = 0;
  virtual uint64_t   getSectionSize(DataRefImpl Sec) const = 0;
  virtual StringRef  getSectionContents(DataRefImpl Sec) const = 0;
  virtual bool       isSectionText(DataRefImpl Sec) const = 0;


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

    bool operator==(const content_iterator &other) const {
      return Current == other.Current;
    }

    bool operator!=(const content_iterator &other) const {
      return !(*this == other);
    }

    content_iterator& operator++() {  // Preincrement
      Current = Current.getNext();
      return *this;
    }
  };

  typedef content_iterator<SymbolRef> symbol_iterator;
  typedef content_iterator<SectionRef> section_iterator;

  virtual ~ObjectFile();

  virtual symbol_iterator begin_symbols() const = 0;
  virtual symbol_iterator end_symbols() const = 0;

  virtual section_iterator begin_sections() const = 0;
  virtual section_iterator end_sections() const = 0;

  /// @brief The number of bytes used to represent an address in this object
  ///        file format.
  virtual uint8_t getBytesInAddress() const = 0;

  virtual StringRef getFileFormatName() const = 0;
  virtual Triple::ArchType getArch() const = 0;

  StringRef getFilename() const;

  /// @returns Pointer to ObjectFile subclass to handle this type of object.
  /// @param ObjectPath The path to the object file. ObjectPath.isObject must
  ///        return true.
  /// @brief Create ObjectFile from path.
  static ObjectFile *createObjectFile(const sys::Path &ObjectPath);
  static ObjectFile *createObjectFile(MemoryBuffer *Object);

private:
  static ObjectFile *createCOFFObjectFile(MemoryBuffer *Object);
  static ObjectFile *createELFObjectFile(MemoryBuffer *Object);
  static ObjectFile *createMachOObjectFile(MemoryBuffer *Object);
  static ObjectFile *createArchiveObjectFile(MemoryBuffer *Object);
  static ObjectFile *createLibObjectFile(MemoryBuffer *Object);
};

// Inline function definitions.
inline SymbolRef::SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner)
  : SymbolPimpl(SymbolP)
  , OwningObject(Owner) {}

inline bool SymbolRef::operator==(const SymbolRef &Other) const {
  return SymbolPimpl == Other.SymbolPimpl;
}

inline SymbolRef SymbolRef::getNext() const {
  return OwningObject->getSymbolNext(SymbolPimpl);
}

inline StringRef SymbolRef::getName() const {
  return OwningObject->getSymbolName(SymbolPimpl);
}

inline uint64_t SymbolRef::getAddress() const {
  return OwningObject->getSymbolAddress(SymbolPimpl);
}

inline uint64_t SymbolRef::getSize() const {
  return OwningObject->getSymbolSize(SymbolPimpl);
}

inline char SymbolRef::getNMTypeChar() const {
  return OwningObject->getSymbolNMTypeChar(SymbolPimpl);
}

inline bool SymbolRef::isInternal() const {
  return OwningObject->isSymbolInternal(SymbolPimpl);
}


/// SectionRef
inline SectionRef::SectionRef(DataRefImpl SectionP,
                              const ObjectFile *Owner)
  : SectionPimpl(SectionP)
  , OwningObject(Owner) {}

inline bool SectionRef::operator==(const SectionRef &Other) const {
  return SectionPimpl == Other.SectionPimpl;
}

inline SectionRef SectionRef::getNext() const {
  return OwningObject->getSectionNext(SectionPimpl);
}

inline StringRef SectionRef::getName() const {
  return OwningObject->getSectionName(SectionPimpl);
}

inline uint64_t SectionRef::getAddress() const {
  return OwningObject->getSectionAddress(SectionPimpl);
}

inline uint64_t SectionRef::getSize() const {
  return OwningObject->getSectionSize(SectionPimpl);
}

inline StringRef SectionRef::getContents() const {
  return OwningObject->getSectionContents(SectionPimpl);
}

inline bool SectionRef::isText() const {
  return OwningObject->isSectionText(SectionPimpl);
}

} // end namespace object
} // end namespace llvm

#endif
