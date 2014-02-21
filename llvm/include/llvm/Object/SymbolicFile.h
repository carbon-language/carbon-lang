//===- SymbolicFile.h - Interface that only provides symbols ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SymbolicFile interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_SYMBOLIC_FILE_H
#define LLVM_OBJECT_SYMBOLIC_FILE_H

#include "llvm/Object/Binary.h"

namespace llvm {
namespace object {

union DataRefImpl {
  // This entire union should probably be a
  // char[max(8, sizeof(uintptr_t))] and require the impl to cast.
  struct {
    uint32_t a, b;
  } d;
  uintptr_t p;
  DataRefImpl() { std::memset(this, 0, sizeof(DataRefImpl)); }
};

inline bool operator==(const DataRefImpl &a, const DataRefImpl &b) {
  // Check bitwise identical. This is the only legal way to compare a union w/o
  // knowing which member is in use.
  return std::memcmp(&a, &b, sizeof(DataRefImpl)) == 0;
}

inline bool operator<(const DataRefImpl &a, const DataRefImpl &b) {
  // Check bitwise identical. This is the only legal way to compare a union w/o
  // knowing which member is in use.
  return std::memcmp(&a, &b, sizeof(DataRefImpl)) < 0;
}

template <class content_type> class content_iterator {
  content_type Current;

public:
  content_iterator(content_type symb) : Current(symb) {}

  const content_type *operator->() const { return &Current; }

  const content_type &operator*() const { return Current; }

  bool operator==(const content_iterator &other) const {
    return Current == other.Current;
  }

  bool operator!=(const content_iterator &other) const {
    return !(*this == other);
  }

  content_iterator &operator++() { // preincrement
    Current.moveNext();
    return *this;
  }
};

class SymbolicFile;

/// This is a value type class that represents a single symbol in the list of
/// symbols in the object file.
class BasicSymbolRef {
  DataRefImpl SymbolPimpl;
  const SymbolicFile *OwningObject;

public:
  // FIXME: should we add a SF_Text?
  enum Flags LLVM_ENUM_INT_TYPE(unsigned) {
    SF_None = 0,
    SF_Undefined = 1U << 0,      // Symbol is defined in another object file
    SF_Global = 1U << 1,         // Global symbol
    SF_Weak = 1U << 2,           // Weak symbol
    SF_Absolute = 1U << 3,       // Absolute symbol
    SF_Common = 1U << 4,         // Symbol has common linkage
    SF_FormatSpecific = 1U << 5  // Specific to the object file format
                                 // (e.g. section symbols)
  };

  BasicSymbolRef() : OwningObject(NULL) { }
  BasicSymbolRef(DataRefImpl SymbolP, const SymbolicFile *Owner);

  bool operator==(const BasicSymbolRef &Other) const;
  bool operator<(const BasicSymbolRef &Other) const;

  void moveNext();

  error_code printName(raw_ostream &OS) const;

  /// Get symbol flags (bitwise OR of SymbolRef::Flags)
  uint32_t getFlags() const;

  DataRefImpl getRawDataRefImpl() const;
  const SymbolicFile *getObject() const;
};

typedef content_iterator<BasicSymbolRef> basic_symbol_iterator;

const uint64_t UnknownAddressOrSize = ~0ULL;

class SymbolicFile : public Binary {
public:
  virtual ~SymbolicFile();
  SymbolicFile(unsigned int Type, MemoryBuffer *Source, bool BufferOwned);

  // virtual interface.
  virtual void moveSymbolNext(DataRefImpl &Symb) const = 0;

  virtual error_code printSymbolName(raw_ostream &OS,
                                     DataRefImpl Symb) const = 0;

  virtual uint32_t getSymbolFlags(DataRefImpl Symb) const = 0;

  virtual basic_symbol_iterator symbol_begin_impl() const = 0;

  virtual basic_symbol_iterator symbol_end_impl() const = 0;

  // convenience wrappers.
  basic_symbol_iterator symbol_begin() const {
    return symbol_begin_impl();
  }
  basic_symbol_iterator symbol_end() const {
    return symbol_end_impl();
  }

  // construction aux.
  static ErrorOr<SymbolicFile *> createIRObjectFile(MemoryBuffer *Object,
                                                    LLVMContext &Context,
                                                    bool BufferOwned = true);

  static ErrorOr<SymbolicFile *> createSymbolicFile(MemoryBuffer *Object,
                                                    bool BufferOwned,
                                                    sys::fs::file_magic Type,
                                                    LLVMContext *Context);

  static ErrorOr<SymbolicFile *> createSymbolicFile(MemoryBuffer *Object) {
    return createSymbolicFile(Object, true, sys::fs::file_magic::unknown, 0);
  }
  static ErrorOr<SymbolicFile *> createSymbolicFile(StringRef ObjectPath);

  static inline bool classof(const Binary *v) {
    return v->isSymbolic();
  }
};

inline BasicSymbolRef::BasicSymbolRef(DataRefImpl SymbolP,
                                      const SymbolicFile *Owner)
    : SymbolPimpl(SymbolP), OwningObject(Owner) {}

inline bool BasicSymbolRef::operator==(const BasicSymbolRef &Other) const {
  return SymbolPimpl == Other.SymbolPimpl;
}

inline bool BasicSymbolRef::operator<(const BasicSymbolRef &Other) const {
  return SymbolPimpl < Other.SymbolPimpl;
}

inline void BasicSymbolRef::moveNext() {
  return OwningObject->moveSymbolNext(SymbolPimpl);
}

inline error_code BasicSymbolRef::printName(raw_ostream &OS) const {
  return OwningObject->printSymbolName(OS, SymbolPimpl);
}

inline uint32_t BasicSymbolRef::getFlags() const {
  return OwningObject->getSymbolFlags(SymbolPimpl);
}

inline DataRefImpl BasicSymbolRef::getRawDataRefImpl() const {
  return SymbolPimpl;
}

inline const SymbolicFile *BasicSymbolRef::getObject() const {
  return OwningObject;
}

}
}

#endif
