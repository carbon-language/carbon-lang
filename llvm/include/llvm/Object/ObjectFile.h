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

#ifndef LLVM_OBJECT_OBJECTFILE_H
#define LLVM_OBJECT_OBJECTFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>
#include <vector>

namespace llvm {
namespace object {

class ObjectFile;

class SymbolRef;
class symbol_iterator;

/// RelocationRef - This is a value type class that represents a single
/// relocation in the list of relocations in the object file.
class RelocationRef {
  DataRefImpl RelocationPimpl;
  const ObjectFile *OwningObject;

public:
  RelocationRef() : OwningObject(nullptr) { }

  RelocationRef(DataRefImpl RelocationP, const ObjectFile *Owner);

  bool operator==(const RelocationRef &Other) const;

  void moveNext();

  error_code getAddress(uint64_t &Result) const;
  error_code getOffset(uint64_t &Result) const;
  symbol_iterator getSymbol() const;
  error_code getType(uint64_t &Result) const;

  /// @brief Indicates whether this relocation should hidden when listing
  /// relocations, usually because it is the trailing part of a multipart
  /// relocation that will be printed as part of the leading relocation.
  error_code getHidden(bool &Result) const;

  /// @brief Get a string that represents the type of this relocation.
  ///
  /// This is for display purposes only.
  error_code getTypeName(SmallVectorImpl<char> &Result) const;

  /// @brief Get a string that represents the calculation of the value of this
  ///        relocation.
  ///
  /// This is for display purposes only.
  error_code getValueString(SmallVectorImpl<char> &Result) const;

  DataRefImpl getRawDataRefImpl() const;
  const ObjectFile *getObjectFile() const;
};
typedef content_iterator<RelocationRef> relocation_iterator;

/// SectionRef - This is a value type class that represents a single section in
/// the list of sections in the object file.
class SectionRef;
typedef content_iterator<SectionRef> section_iterator;
class SectionRef {
  friend class SymbolRef;
  DataRefImpl SectionPimpl;
  const ObjectFile *OwningObject;

public:
  SectionRef() : OwningObject(nullptr) { }

  SectionRef(DataRefImpl SectionP, const ObjectFile *Owner);

  bool operator==(const SectionRef &Other) const;
  bool operator!=(const SectionRef &Other) const;
  bool operator<(const SectionRef &Other) const;

  void moveNext();

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
  error_code isRequiredForExecution(bool &Result) const;
  error_code isVirtual(bool &Result) const;
  error_code isZeroInit(bool &Result) const;
  error_code isReadOnlyData(bool &Result) const;

  error_code containsSymbol(SymbolRef S, bool &Result) const;

  relocation_iterator relocation_begin() const;
  relocation_iterator relocation_end() const;
  typedef iterator_range<relocation_iterator> relocation_iterator_range;
  relocation_iterator_range relocations() const {
    return relocation_iterator_range(relocation_begin(), relocation_end());
  }
  section_iterator getRelocatedSection() const;

  DataRefImpl getRawDataRefImpl() const;
};

/// SymbolRef - This is a value type class that represents a single symbol in
/// the list of symbols in the object file.
class SymbolRef : public BasicSymbolRef {
  friend class SectionRef;

public:
  SymbolRef() : BasicSymbolRef() {}

  enum Type {
    ST_Unknown, // Type not specified
    ST_Data,
    ST_Debug,
    ST_File,
    ST_Function,
    ST_Other
  };

  SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner);

  error_code getName(StringRef &Result) const;
  /// Returns the symbol virtual address (i.e. address at which it will be
  /// mapped).
  error_code getAddress(uint64_t &Result) const;
  /// @brief Get the alignment of this symbol as the actual value (not log 2).
  error_code getAlignment(uint32_t &Result) const;
  error_code getSize(uint64_t &Result) const;
  error_code getType(SymbolRef::Type &Result) const;

  /// @brief Get section this symbol is defined in reference to. Result is
  /// end_sections() if it is undefined or is an absolute symbol.
  error_code getSection(section_iterator &Result) const;

  const ObjectFile *getObject() const;
};

class symbol_iterator : public basic_symbol_iterator {
public:
  symbol_iterator(SymbolRef Sym) : basic_symbol_iterator(Sym) {}
  symbol_iterator(const basic_symbol_iterator &B)
      : basic_symbol_iterator(SymbolRef(B->getRawDataRefImpl(),
                                        cast<ObjectFile>(B->getObject()))) {}

  const SymbolRef *operator->() const {
    const BasicSymbolRef &P = basic_symbol_iterator::operator *();
    return static_cast<const SymbolRef*>(&P);
  }

  const SymbolRef &operator*() const {
    const BasicSymbolRef &P = basic_symbol_iterator::operator *();
    return static_cast<const SymbolRef&>(P);
  }
};

/// LibraryRef - This is a value type class that represents a single library in
/// the list of libraries needed by a shared or dynamic object.
class LibraryRef {
  friend class SectionRef;
  DataRefImpl LibraryPimpl;
  const ObjectFile *OwningObject;

public:
  LibraryRef() : OwningObject(nullptr) { }

  LibraryRef(DataRefImpl LibraryP, const ObjectFile *Owner);

  bool operator==(const LibraryRef &Other) const;
  bool operator<(const LibraryRef &Other) const;

  error_code getNext(LibraryRef &Result) const;

  // Get the path to this library, as stored in the object file.
  error_code getPath(StringRef &Result) const;

  DataRefImpl getRawDataRefImpl() const;
};
typedef content_iterator<LibraryRef> library_iterator;

/// ObjectFile - This class is the base class for all object file types.
/// Concrete instances of this object are created by createObjectFile, which
/// figures out which type to create.
class ObjectFile : public SymbolicFile {
  virtual void anchor();
  ObjectFile() LLVM_DELETED_FUNCTION;
  ObjectFile(const ObjectFile &other) LLVM_DELETED_FUNCTION;

protected:
  ObjectFile(unsigned int Type, MemoryBuffer *Source, bool BufferOwned = true);

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
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const = 0;
  error_code printSymbolName(raw_ostream &OS, DataRefImpl Symb) const override;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const = 0;
  virtual error_code getSymbolAlignment(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const = 0;
  virtual error_code getSymbolType(DataRefImpl Symb,
                                   SymbolRef::Type &Res) const = 0;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const = 0;

  // Same as above for SectionRef.
  friend class SectionRef;
  virtual void moveSectionNext(DataRefImpl &Sec) const = 0;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const = 0;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const =0;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const = 0;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res)const=0;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res)const=0;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const = 0;
  // A section is 'virtual' if its contents aren't present in the object image.
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const = 0;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const =0;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const = 0;
  virtual relocation_iterator section_rel_begin(DataRefImpl Sec) const = 0;
  virtual relocation_iterator section_rel_end(DataRefImpl Sec) const = 0;
  virtual section_iterator getRelocatedSection(DataRefImpl Sec) const;

  // Same as above for RelocationRef.
  friend class RelocationRef;
  virtual void moveRelocationNext(DataRefImpl &Rel) const = 0;
  virtual error_code getRelocationAddress(DataRefImpl Rel,
                                          uint64_t &Res) const =0;
  virtual error_code getRelocationOffset(DataRefImpl Rel,
                                         uint64_t &Res) const =0;
  virtual symbol_iterator getRelocationSymbol(DataRefImpl Rel) const = 0;
  virtual error_code getRelocationType(DataRefImpl Rel,
                                       uint64_t &Res) const = 0;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                       SmallVectorImpl<char> &Result) const = 0;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                       SmallVectorImpl<char> &Result) const = 0;
  virtual error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const {
    Result = false;
    return object_error::success;
  }

  // Same for LibraryRef
  friend class LibraryRef;
  virtual error_code getLibraryNext(DataRefImpl Lib, LibraryRef &Res) const = 0;
  virtual error_code getLibraryPath(DataRefImpl Lib, StringRef &Res) const = 0;

public:
  typedef iterator_range<symbol_iterator> symbol_iterator_range;
  symbol_iterator_range symbols() const {
    return symbol_iterator_range(symbol_begin(), symbol_end());
  }

  virtual section_iterator section_begin() const = 0;
  virtual section_iterator section_end() const = 0;

  typedef iterator_range<section_iterator> section_iterator_range;
  section_iterator_range sections() const {
    return section_iterator_range(section_begin(), section_end());
  }

  virtual library_iterator needed_library_begin() const = 0;
  virtual library_iterator needed_library_end() const = 0;

  /// @brief The number of bytes used to represent an address in this object
  ///        file format.
  virtual uint8_t getBytesInAddress() const = 0;

  virtual StringRef getFileFormatName() const = 0;
  virtual /* Triple::ArchType */ unsigned getArch() const = 0;

  /// For shared objects, returns the name which this object should be
  /// loaded from at runtime. This corresponds to DT_SONAME on ELF and
  /// LC_ID_DYLIB (install name) on MachO.
  virtual StringRef getLoadName() const = 0;

  /// @returns Pointer to ObjectFile subclass to handle this type of object.
  /// @param ObjectPath The path to the object file. ObjectPath.isObject must
  ///        return true.
  /// @brief Create ObjectFile from path.
  static ErrorOr<ObjectFile *> createObjectFile(StringRef ObjectPath);
  static ErrorOr<ObjectFile *> createObjectFile(MemoryBuffer *Object,
                                                bool BufferOwned,
                                                sys::fs::file_magic Type);
  static ErrorOr<ObjectFile *> createObjectFile(MemoryBuffer *Object) {
    return createObjectFile(Object, true, sys::fs::file_magic::unknown);
  }


  static inline bool classof(const Binary *v) {
    return v->isObject();
  }

public:
  static ErrorOr<ObjectFile *> createCOFFObjectFile(MemoryBuffer *Object,
                                                    bool BufferOwned = true);
  static ErrorOr<ObjectFile *> createELFObjectFile(MemoryBuffer *Object,
                                                   bool BufferOwned = true);
  static ErrorOr<ObjectFile *> createMachOObjectFile(MemoryBuffer *Object,
                                                     bool BufferOwned = true);
};

// Inline function definitions.
inline SymbolRef::SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner)
    : BasicSymbolRef(SymbolP, Owner) {}

inline error_code SymbolRef::getName(StringRef &Result) const {
  return getObject()->getSymbolName(getRawDataRefImpl(), Result);
}

inline error_code SymbolRef::getAddress(uint64_t &Result) const {
  return getObject()->getSymbolAddress(getRawDataRefImpl(), Result);
}

inline error_code SymbolRef::getAlignment(uint32_t &Result) const {
  return getObject()->getSymbolAlignment(getRawDataRefImpl(), Result);
}

inline error_code SymbolRef::getSize(uint64_t &Result) const {
  return getObject()->getSymbolSize(getRawDataRefImpl(), Result);
}

inline error_code SymbolRef::getSection(section_iterator &Result) const {
  return getObject()->getSymbolSection(getRawDataRefImpl(), Result);
}

inline error_code SymbolRef::getType(SymbolRef::Type &Result) const {
  return getObject()->getSymbolType(getRawDataRefImpl(), Result);
}

inline const ObjectFile *SymbolRef::getObject() const {
  const SymbolicFile *O = BasicSymbolRef::getObject();
  return cast<ObjectFile>(O);
}


/// SectionRef
inline SectionRef::SectionRef(DataRefImpl SectionP,
                              const ObjectFile *Owner)
  : SectionPimpl(SectionP)
  , OwningObject(Owner) {}

inline bool SectionRef::operator==(const SectionRef &Other) const {
  return SectionPimpl == Other.SectionPimpl;
}

inline bool SectionRef::operator!=(const SectionRef &Other) const {
  return SectionPimpl != Other.SectionPimpl;
}

inline bool SectionRef::operator<(const SectionRef &Other) const {
  return SectionPimpl < Other.SectionPimpl;
}

inline void SectionRef::moveNext() {
  return OwningObject->moveSectionNext(SectionPimpl);
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

inline error_code SectionRef::isRequiredForExecution(bool &Result) const {
  return OwningObject->isSectionRequiredForExecution(SectionPimpl, Result);
}

inline error_code SectionRef::isVirtual(bool &Result) const {
  return OwningObject->isSectionVirtual(SectionPimpl, Result);
}

inline error_code SectionRef::isZeroInit(bool &Result) const {
  return OwningObject->isSectionZeroInit(SectionPimpl, Result);
}

inline error_code SectionRef::isReadOnlyData(bool &Result) const {
  return OwningObject->isSectionReadOnlyData(SectionPimpl, Result);
}

inline error_code SectionRef::containsSymbol(SymbolRef S, bool &Result) const {
  return OwningObject->sectionContainsSymbol(SectionPimpl,
                                             S.getRawDataRefImpl(), Result);
}

inline relocation_iterator SectionRef::relocation_begin() const {
  return OwningObject->section_rel_begin(SectionPimpl);
}

inline relocation_iterator SectionRef::relocation_end() const {
  return OwningObject->section_rel_end(SectionPimpl);
}

inline section_iterator SectionRef::getRelocatedSection() const {
  return OwningObject->getRelocatedSection(SectionPimpl);
}

inline DataRefImpl SectionRef::getRawDataRefImpl() const {
  return SectionPimpl;
}

/// RelocationRef
inline RelocationRef::RelocationRef(DataRefImpl RelocationP,
                              const ObjectFile *Owner)
  : RelocationPimpl(RelocationP)
  , OwningObject(Owner) {}

inline bool RelocationRef::operator==(const RelocationRef &Other) const {
  return RelocationPimpl == Other.RelocationPimpl;
}

inline void RelocationRef::moveNext() {
  return OwningObject->moveRelocationNext(RelocationPimpl);
}

inline error_code RelocationRef::getAddress(uint64_t &Result) const {
  return OwningObject->getRelocationAddress(RelocationPimpl, Result);
}

inline error_code RelocationRef::getOffset(uint64_t &Result) const {
  return OwningObject->getRelocationOffset(RelocationPimpl, Result);
}

inline symbol_iterator RelocationRef::getSymbol() const {
  return OwningObject->getRelocationSymbol(RelocationPimpl);
}

inline error_code RelocationRef::getType(uint64_t &Result) const {
  return OwningObject->getRelocationType(RelocationPimpl, Result);
}

inline error_code RelocationRef::getTypeName(SmallVectorImpl<char> &Result)
  const {
  return OwningObject->getRelocationTypeName(RelocationPimpl, Result);
}

inline error_code RelocationRef::getValueString(SmallVectorImpl<char> &Result)
  const {
  return OwningObject->getRelocationValueString(RelocationPimpl, Result);
}

inline error_code RelocationRef::getHidden(bool &Result) const {
  return OwningObject->getRelocationHidden(RelocationPimpl, Result);
}

inline DataRefImpl RelocationRef::getRawDataRefImpl() const {
  return RelocationPimpl;
}

inline const ObjectFile *RelocationRef::getObjectFile() const {
  return OwningObject;
}

// Inline function definitions.
inline LibraryRef::LibraryRef(DataRefImpl LibraryP, const ObjectFile *Owner)
  : LibraryPimpl(LibraryP)
  , OwningObject(Owner) {}

inline bool LibraryRef::operator==(const LibraryRef &Other) const {
  return LibraryPimpl == Other.LibraryPimpl;
}

inline bool LibraryRef::operator<(const LibraryRef &Other) const {
  return LibraryPimpl < Other.LibraryPimpl;
}

inline error_code LibraryRef::getNext(LibraryRef &Result) const {
  return OwningObject->getLibraryNext(LibraryPimpl, Result);
}

inline error_code LibraryRef::getPath(StringRef &Result) const {
  return OwningObject->getLibraryPath(LibraryPimpl, Result);
}

} // end namespace object
} // end namespace llvm

#endif
