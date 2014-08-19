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
class COFFObjectFile;
class MachOObjectFile;

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

  std::error_code getAddress(uint64_t &Result) const;
  std::error_code getOffset(uint64_t &Result) const;
  symbol_iterator getSymbol() const;
  std::error_code getType(uint64_t &Result) const;

  /// @brief Indicates whether this relocation should hidden when listing
  /// relocations, usually because it is the trailing part of a multipart
  /// relocation that will be printed as part of the leading relocation.
  std::error_code getHidden(bool &Result) const;

  /// @brief Get a string that represents the type of this relocation.
  ///
  /// This is for display purposes only.
  std::error_code getTypeName(SmallVectorImpl<char> &Result) const;

  /// @brief Get a string that represents the calculation of the value of this
  ///        relocation.
  ///
  /// This is for display purposes only.
  std::error_code getValueString(SmallVectorImpl<char> &Result) const;

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

  std::error_code getName(StringRef &Result) const;
  std::error_code getAddress(uint64_t &Result) const;
  std::error_code getSize(uint64_t &Result) const;
  std::error_code getContents(StringRef &Result) const;

  /// @brief Get the alignment of this section as the actual value (not log 2).
  std::error_code getAlignment(uint64_t &Result) const;

  // FIXME: Move to the normalization layer when it's created.
  std::error_code isText(bool &Result) const;
  std::error_code isData(bool &Result) const;
  std::error_code isBSS(bool &Result) const;
  std::error_code isRequiredForExecution(bool &Result) const;
  std::error_code isVirtual(bool &Result) const;
  std::error_code isZeroInit(bool &Result) const;
  std::error_code isReadOnlyData(bool &Result) const;

  std::error_code containsSymbol(SymbolRef S, bool &Result) const;

  relocation_iterator relocation_begin() const;
  relocation_iterator relocation_end() const;
  iterator_range<relocation_iterator> relocations() const {
    return iterator_range<relocation_iterator>(relocation_begin(),
                                               relocation_end());
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

  std::error_code getName(StringRef &Result) const;
  /// Returns the symbol virtual address (i.e. address at which it will be
  /// mapped).
  std::error_code getAddress(uint64_t &Result) const;
  /// @brief Get the alignment of this symbol as the actual value (not log 2).
  std::error_code getAlignment(uint32_t &Result) const;
  std::error_code getSize(uint64_t &Result) const;
  std::error_code getType(SymbolRef::Type &Result) const;
  std::error_code getOther(uint8_t &Result) const;

  /// @brief Get section this symbol is defined in reference to. Result is
  /// end_sections() if it is undefined or is an absolute symbol.
  std::error_code getSection(section_iterator &Result) const;

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

/// ObjectFile - This class is the base class for all object file types.
/// Concrete instances of this object are created by createObjectFile, which
/// figures out which type to create.
class ObjectFile : public SymbolicFile {
  virtual void anchor();
  ObjectFile() LLVM_DELETED_FUNCTION;
  ObjectFile(const ObjectFile &other) LLVM_DELETED_FUNCTION;

protected:
  ObjectFile(unsigned int Type, MemoryBufferRef Source);

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Data.getBufferStart());
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
  virtual std::error_code getSymbolName(DataRefImpl Symb,
                                        StringRef &Res) const = 0;
  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override;
  virtual std::error_code getSymbolAddress(DataRefImpl Symb,
                                           uint64_t &Res) const = 0;
  virtual std::error_code getSymbolAlignment(DataRefImpl Symb,
                                             uint32_t &Res) const;
  virtual std::error_code getSymbolSize(DataRefImpl Symb,
                                        uint64_t &Res) const = 0;
  virtual std::error_code getSymbolType(DataRefImpl Symb,
                                        SymbolRef::Type &Res) const = 0;
  virtual std::error_code getSymbolSection(DataRefImpl Symb,
                                           section_iterator &Res) const = 0;
  virtual std::error_code getSymbolOther(DataRefImpl Symb,
                                         uint8_t &Res) const {
    return object_error::invalid_file_type;
  }

  // Same as above for SectionRef.
  friend class SectionRef;
  virtual void moveSectionNext(DataRefImpl &Sec) const = 0;
  virtual std::error_code getSectionName(DataRefImpl Sec,
                                         StringRef &Res) const = 0;
  virtual std::error_code getSectionAddress(DataRefImpl Sec,
                                            uint64_t &Res) const = 0;
  virtual std::error_code getSectionSize(DataRefImpl Sec,
                                         uint64_t &Res) const = 0;
  virtual std::error_code getSectionContents(DataRefImpl Sec,
                                             StringRef &Res) const = 0;
  virtual std::error_code getSectionAlignment(DataRefImpl Sec,
                                              uint64_t &Res) const = 0;
  virtual std::error_code isSectionText(DataRefImpl Sec, bool &Res) const = 0;
  virtual std::error_code isSectionData(DataRefImpl Sec, bool &Res) const = 0;
  virtual std::error_code isSectionBSS(DataRefImpl Sec, bool &Res) const = 0;
  virtual std::error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                        bool &Res) const = 0;
  // A section is 'virtual' if its contents aren't present in the object image.
  virtual std::error_code isSectionVirtual(DataRefImpl Sec,
                                           bool &Res) const = 0;
  virtual std::error_code isSectionZeroInit(DataRefImpl Sec,
                                            bool &Res) const = 0;
  virtual std::error_code isSectionReadOnlyData(DataRefImpl Sec,
                                                bool &Res) const = 0;
  virtual std::error_code sectionContainsSymbol(DataRefImpl Sec,
                                                DataRefImpl Symb,
                                                bool &Result) const = 0;
  virtual relocation_iterator section_rel_begin(DataRefImpl Sec) const = 0;
  virtual relocation_iterator section_rel_end(DataRefImpl Sec) const = 0;
  virtual section_iterator getRelocatedSection(DataRefImpl Sec) const;

  // Same as above for RelocationRef.
  friend class RelocationRef;
  virtual void moveRelocationNext(DataRefImpl &Rel) const = 0;
  virtual std::error_code getRelocationAddress(DataRefImpl Rel,
                                               uint64_t &Res) const = 0;
  virtual std::error_code getRelocationOffset(DataRefImpl Rel,
                                              uint64_t &Res) const = 0;
  virtual symbol_iterator getRelocationSymbol(DataRefImpl Rel) const = 0;
  virtual std::error_code getRelocationType(DataRefImpl Rel,
                                            uint64_t &Res) const = 0;
  virtual std::error_code
  getRelocationTypeName(DataRefImpl Rel,
                        SmallVectorImpl<char> &Result) const = 0;
  virtual std::error_code
  getRelocationValueString(DataRefImpl Rel,
                           SmallVectorImpl<char> &Result) const = 0;
  virtual std::error_code getRelocationHidden(DataRefImpl Rel,
                                              bool &Result) const {
    Result = false;
    return object_error::success;
  }

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

  /// @brief The number of bytes used to represent an address in this object
  ///        file format.
  virtual uint8_t getBytesInAddress() const = 0;

  virtual StringRef getFileFormatName() const = 0;
  virtual /* Triple::ArchType */ unsigned getArch() const = 0;

  /// Returns platform-specific object flags, if any.
  virtual std::error_code getPlatformFlags(unsigned &Result) const {
    Result = 0;
    return object_error::invalid_file_type;
  }

  /// True if this is a relocatable object (.o/.obj).
  virtual bool isRelocatableObject() const = 0;

  /// @returns Pointer to ObjectFile subclass to handle this type of object.
  /// @param ObjectPath The path to the object file. ObjectPath.isObject must
  ///        return true.
  /// @brief Create ObjectFile from path.
  static ErrorOr<OwningBinary<ObjectFile>>
  createObjectFile(StringRef ObjectPath);

  static ErrorOr<std::unique_ptr<ObjectFile>>
  createObjectFile(MemoryBufferRef Object, sys::fs::file_magic Type);
  static ErrorOr<std::unique_ptr<ObjectFile>>
  createObjectFile(MemoryBufferRef Object) {
    return createObjectFile(Object, sys::fs::file_magic::unknown);
  }


  static inline bool classof(const Binary *v) {
    return v->isObject();
  }

  static ErrorOr<std::unique_ptr<COFFObjectFile>>
  createCOFFObjectFile(MemoryBufferRef Object);

  static ErrorOr<std::unique_ptr<ObjectFile>>
  createELFObjectFile(MemoryBufferRef Object);

  static ErrorOr<std::unique_ptr<MachOObjectFile>>
  createMachOObjectFile(MemoryBufferRef Object);
};

// Inline function definitions.
inline SymbolRef::SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner)
    : BasicSymbolRef(SymbolP, Owner) {}

inline std::error_code SymbolRef::getName(StringRef &Result) const {
  return getObject()->getSymbolName(getRawDataRefImpl(), Result);
}

inline std::error_code SymbolRef::getAddress(uint64_t &Result) const {
  return getObject()->getSymbolAddress(getRawDataRefImpl(), Result);
}

inline std::error_code SymbolRef::getAlignment(uint32_t &Result) const {
  return getObject()->getSymbolAlignment(getRawDataRefImpl(), Result);
}

inline std::error_code SymbolRef::getSize(uint64_t &Result) const {
  return getObject()->getSymbolSize(getRawDataRefImpl(), Result);
}

inline std::error_code SymbolRef::getSection(section_iterator &Result) const {
  return getObject()->getSymbolSection(getRawDataRefImpl(), Result);
}

inline std::error_code SymbolRef::getType(SymbolRef::Type &Result) const {
  return getObject()->getSymbolType(getRawDataRefImpl(), Result);
}

inline std::error_code SymbolRef::getOther(uint8_t &Result) const {
  return getObject()->getSymbolOther(getRawDataRefImpl(), Result);
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

inline std::error_code SectionRef::getName(StringRef &Result) const {
  return OwningObject->getSectionName(SectionPimpl, Result);
}

inline std::error_code SectionRef::getAddress(uint64_t &Result) const {
  return OwningObject->getSectionAddress(SectionPimpl, Result);
}

inline std::error_code SectionRef::getSize(uint64_t &Result) const {
  return OwningObject->getSectionSize(SectionPimpl, Result);
}

inline std::error_code SectionRef::getContents(StringRef &Result) const {
  return OwningObject->getSectionContents(SectionPimpl, Result);
}

inline std::error_code SectionRef::getAlignment(uint64_t &Result) const {
  return OwningObject->getSectionAlignment(SectionPimpl, Result);
}

inline std::error_code SectionRef::isText(bool &Result) const {
  return OwningObject->isSectionText(SectionPimpl, Result);
}

inline std::error_code SectionRef::isData(bool &Result) const {
  return OwningObject->isSectionData(SectionPimpl, Result);
}

inline std::error_code SectionRef::isBSS(bool &Result) const {
  return OwningObject->isSectionBSS(SectionPimpl, Result);
}

inline std::error_code SectionRef::isRequiredForExecution(bool &Result) const {
  return OwningObject->isSectionRequiredForExecution(SectionPimpl, Result);
}

inline std::error_code SectionRef::isVirtual(bool &Result) const {
  return OwningObject->isSectionVirtual(SectionPimpl, Result);
}

inline std::error_code SectionRef::isZeroInit(bool &Result) const {
  return OwningObject->isSectionZeroInit(SectionPimpl, Result);
}

inline std::error_code SectionRef::isReadOnlyData(bool &Result) const {
  return OwningObject->isSectionReadOnlyData(SectionPimpl, Result);
}

inline std::error_code SectionRef::containsSymbol(SymbolRef S,
                                                  bool &Result) const {
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

inline std::error_code RelocationRef::getAddress(uint64_t &Result) const {
  return OwningObject->getRelocationAddress(RelocationPimpl, Result);
}

inline std::error_code RelocationRef::getOffset(uint64_t &Result) const {
  return OwningObject->getRelocationOffset(RelocationPimpl, Result);
}

inline symbol_iterator RelocationRef::getSymbol() const {
  return OwningObject->getRelocationSymbol(RelocationPimpl);
}

inline std::error_code RelocationRef::getType(uint64_t &Result) const {
  return OwningObject->getRelocationType(RelocationPimpl, Result);
}

inline std::error_code
RelocationRef::getTypeName(SmallVectorImpl<char> &Result) const {
  return OwningObject->getRelocationTypeName(RelocationPimpl, Result);
}

inline std::error_code
RelocationRef::getValueString(SmallVectorImpl<char> &Result) const {
  return OwningObject->getRelocationValueString(RelocationPimpl, Result);
}

inline std::error_code RelocationRef::getHidden(bool &Result) const {
  return OwningObject->getRelocationHidden(RelocationPimpl, Result);
}

inline DataRefImpl RelocationRef::getRawDataRefImpl() const {
  return RelocationPimpl;
}

inline const ObjectFile *RelocationRef::getObjectFile() const {
  return OwningObject;
}


} // end namespace object
} // end namespace llvm

#endif
