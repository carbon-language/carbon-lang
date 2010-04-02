//===--- PartialDiagnostic.h - Diagnostic "closures" ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a partial diagnostic that can be emitted anwyhere
//  in a DiagnosticBuilder stream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARTIALDIAGNOSTIC_H
#define LLVM_CLANG_PARTIALDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/System/DataTypes.h"
#include <cassert>

namespace clang {

class PartialDiagnostic {
public:
  struct Storage {
    Storage() : NumDiagArgs(0), NumDiagRanges(0), NumFixItHints(0) { }

    enum {
        /// MaxArguments - The maximum number of arguments we can hold. We 
        /// currently only support up to 10 arguments (%0-%9).
        /// A single diagnostic with more than that almost certainly has to
        /// be simplified anyway.
        MaxArguments = 10
    };
  
    /// NumDiagArgs - This contains the number of entries in Arguments.
    unsigned char NumDiagArgs;
  
    /// NumDiagRanges - This is the number of ranges in the DiagRanges array.
    unsigned char NumDiagRanges;

    /// \brief The number of code modifications hints in the
    /// FixItHints array.
    unsigned char NumFixItHints;
    
    /// DiagArgumentsKind - This is an array of ArgumentKind::ArgumentKind enum
    /// values, with one for each argument.  This specifies whether the argument
    /// is in DiagArgumentsStr or in DiagArguments.
    unsigned char DiagArgumentsKind[MaxArguments];
  
    /// DiagArgumentsVal - The values for the various substitution positions. 
    /// This is used when the argument is not an std::string. The specific value 
    /// is mangled into an intptr_t and the intepretation depends on exactly
    /// what sort of argument kind it is.
    intptr_t DiagArgumentsVal[MaxArguments];
  
    /// DiagRanges - The list of ranges added to this diagnostic.  It currently
    /// only support 10 ranges, could easily be extended if needed.
    SourceRange DiagRanges[10];
    
    enum { MaxFixItHints = 3 };
    
    /// FixItHints - If valid, provides a hint with some code
    /// to insert, remove, or modify at a particular position.
    FixItHint FixItHints[MaxFixItHints];    
  };

  /// \brief An allocator for Storage objects, which uses a small cache to 
  /// objects, used to reduce malloc()/free() traffic for partial diagnostics.
  class StorageAllocator {
    static const unsigned NumCached = 4;
    Storage Cached[NumCached];
    Storage *FreeList[NumCached];
    unsigned NumFreeListEntries;
    
  public:
    StorageAllocator();
    ~StorageAllocator();
    
    /// \brief Allocate new storage.
    Storage *Allocate() {
      if (NumFreeListEntries == 0)
        return new Storage;
      
      Storage *Result = FreeList[--NumFreeListEntries];
      Result->NumDiagArgs = 0;
      Result->NumDiagRanges = 0;
      Result->NumFixItHints = 0;
      return Result;
    }
    
    /// \brief Free the given storage object.
    void Deallocate(Storage *S) {
      if (S >= Cached && S <= Cached + NumCached) {
        FreeList[NumFreeListEntries++] = S;
        return;
      }
      
      delete S;
    }
  };
  
private:
  // NOTE: Sema assumes that PartialDiagnostic is location-invariant
  // in the sense that its bits can be safely memcpy'ed and destructed
  // in the new location.

  /// DiagID - The diagnostic ID.
  mutable unsigned DiagID;
  
  /// DiagStorage - Storage for args and ranges.
  mutable Storage *DiagStorage;

  /// \brief Allocator used to allocate storage for this diagnostic.
  StorageAllocator *Allocator;
  
  /// \brief Retrieve storage for this particular diagnostic.
  Storage *getStorage() const {
    if (DiagStorage)
      return DiagStorage;
    
    if (Allocator)
      DiagStorage = Allocator->Allocate();
    else {
      assert(Allocator != reinterpret_cast<StorageAllocator *>(~uintptr_t(0)));
      DiagStorage = new Storage;
    }
    return DiagStorage;
  }
  
  void freeStorage() { 
    if (!DiagStorage)
      return;
    
    if (Allocator)
      Allocator->Deallocate(DiagStorage);
    else if (Allocator != reinterpret_cast<StorageAllocator *>(~uintptr_t(0)))
      delete DiagStorage;
    DiagStorage = 0;
  }
  
  void AddSourceRange(const SourceRange &R) const {
    if (!DiagStorage)
      DiagStorage = getStorage();

    assert(DiagStorage->NumDiagRanges < 
           llvm::array_lengthof(DiagStorage->DiagRanges) &&
           "Too many arguments to diagnostic!");
    DiagStorage->DiagRanges[DiagStorage->NumDiagRanges++] = R;
  }  

  void AddFixItHint(const FixItHint &Hint) const {
    if (Hint.isNull())
      return;
    
    if (!DiagStorage)
      DiagStorage = getStorage();

    assert(DiagStorage->NumFixItHints < Storage::MaxFixItHints &&
           "Too many code modification hints!");
    DiagStorage->FixItHints[DiagStorage->NumFixItHints++]
      = Hint;
  }
  
public:
  PartialDiagnostic(unsigned DiagID, StorageAllocator &Allocator)
    : DiagID(DiagID), DiagStorage(0), Allocator(&Allocator) { }
  
  PartialDiagnostic(const PartialDiagnostic &Other) 
    : DiagID(Other.DiagID), DiagStorage(0), Allocator(Other.Allocator)
  {
    if (Other.DiagStorage) {
      DiagStorage = getStorage();
      *DiagStorage = *Other.DiagStorage;
    }
  }

  PartialDiagnostic(const PartialDiagnostic &Other, Storage *DiagStorage) 
    : DiagID(Other.DiagID), DiagStorage(DiagStorage), 
      Allocator(reinterpret_cast<StorageAllocator *>(~uintptr_t(0)))
  {
    if (Other.DiagStorage)
      *this->DiagStorage = *Other.DiagStorage;
  }
  
  PartialDiagnostic &operator=(const PartialDiagnostic &Other) {
    DiagID = Other.DiagID;
    if (Other.DiagStorage) {
      if (!DiagStorage)
        DiagStorage = getStorage();
      
      *DiagStorage = *Other.DiagStorage;
    } else {
      freeStorage();
    }

    return *this;
  }

  ~PartialDiagnostic() {
    freeStorage();
  }

  unsigned getDiagID() const { return DiagID; }

  void AddTaggedVal(intptr_t V, Diagnostic::ArgumentKind Kind) const {
    if (!DiagStorage)
      DiagStorage = getStorage();

    assert(DiagStorage->NumDiagArgs < Storage::MaxArguments &&
           "Too many arguments to diagnostic!");
    DiagStorage->DiagArgumentsKind[DiagStorage->NumDiagArgs] = Kind;
    DiagStorage->DiagArgumentsVal[DiagStorage->NumDiagArgs++] = V;
  }

  void Emit(const DiagnosticBuilder &DB) const {
    if (!DiagStorage)
      return;
    
    // Add all arguments.
    for (unsigned i = 0, e = DiagStorage->NumDiagArgs; i != e; ++i) {
      DB.AddTaggedVal(DiagStorage->DiagArgumentsVal[i],
                   (Diagnostic::ArgumentKind)DiagStorage->DiagArgumentsKind[i]);
    }
    
    // Add all ranges.
    for (unsigned i = 0, e = DiagStorage->NumDiagRanges; i != e; ++i)
      DB.AddSourceRange(DiagStorage->DiagRanges[i]);
    
    // Add all code modification hints
    for (unsigned i = 0, e = DiagStorage->NumFixItHints; i != e; ++i)
      DB.AddFixItHint(DiagStorage->FixItHints[i]);
  }
  
  /// \brief Clear out this partial diagnostic, giving it a new diagnostic ID
  /// and removing all of its arguments, ranges, and fix-it hints.
  void Reset(unsigned DiagID = 0) {
    this->DiagID = DiagID;
    freeStorage();
  }
  
  bool hasStorage() const { return DiagStorage != 0; }
  
  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                             unsigned I) {
    PD.AddTaggedVal(I, Diagnostic::ak_uint);
    return PD;
  }

  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                             int I) {
    PD.AddTaggedVal(I, Diagnostic::ak_sint);
    return PD;
  }

  friend inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                                    const char *S) {
    PD.AddTaggedVal(reinterpret_cast<intptr_t>(S), Diagnostic::ak_c_string);
    return PD;
  }

  friend inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                                    const SourceRange &R) {
    PD.AddSourceRange(R);
    return PD;
  }

  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                             const FixItHint &Hint) {
    PD.AddFixItHint(Hint);
    return PD;
  }
  
};

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const PartialDiagnostic &PD) {
  PD.Emit(DB);
  return DB;
}
  

}  // end namespace clang
#endif
