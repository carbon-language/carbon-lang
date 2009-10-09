//===--- PartialDiagnostic.h - Diagnostic "closures" ----------------------===//
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

#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {

class DeclarationName;
  
class PartialDiagnostic {
  struct Storage {
    Storage() : NumDiagArgs(0), NumDiagRanges(0) { }

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

    /// DiagArgumentsKind - This is an array of ArgumentKind::ArgumentKind enum
    /// values, with one for each argument.  This specifies whether the argument
    /// is in DiagArgumentsStr or in DiagArguments.
    unsigned char DiagArgumentsKind[MaxArguments];
  
    /// DiagArgumentsVal - The values for the various substitution positions. 
    /// This is used when the argument is not an std::string. The specific value 
    /// is mangled into an intptr_t and the intepretation depends on exactly
    /// what sort of argument kind it is.
    mutable intptr_t DiagArgumentsVal[MaxArguments];
  
    /// DiagRanges - The list of ranges added to this diagnostic.  It currently
    /// only support 10 ranges, could easily be extended if needed.
    mutable const SourceRange *DiagRanges[10];
  };

  /// DiagID - The diagnostic ID.
  mutable unsigned DiagID;
  
  /// DiagStorare - Storge for args and ranges.
  mutable Storage *DiagStorage;

  void AddTaggedVal(intptr_t V, Diagnostic::ArgumentKind Kind) const {
    if (!DiagStorage)
      DiagStorage = new Storage;
    
    assert(DiagStorage->NumDiagArgs < Storage::MaxArguments &&
           "Too many arguments to diagnostic!");
    DiagStorage->DiagArgumentsKind[DiagStorage->NumDiagArgs] = Kind;
    DiagStorage->DiagArgumentsVal[DiagStorage->NumDiagArgs++] = V;
  }

  void AddSourceRange(const SourceRange &R) const {
    if (!DiagStorage)
      DiagStorage = new Storage;

    assert(DiagStorage->NumDiagRanges < 
           llvm::array_lengthof(DiagStorage->DiagRanges) &&
           "Too many arguments to diagnostic!");
    DiagStorage->DiagRanges[DiagStorage->NumDiagRanges++] = &R;
  }  

  void operator=(const PartialDiagnostic &); // DO NOT IMPLEMENT

public:
  PartialDiagnostic(unsigned DiagID)
    : DiagID(DiagID), DiagStorage(0) { }

  PartialDiagnostic(const PartialDiagnostic &Other) 
    : DiagID(Other.DiagID), DiagStorage(Other.DiagStorage) {
    Other.DiagID = 0;
    Other.DiagStorage = 0;
  }

  ~PartialDiagnostic() {
    delete DiagStorage;
  }

  unsigned getDiagID() const { return DiagID; }

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
      DB.AddSourceRange(*DiagStorage->DiagRanges[i]);
  }
  
  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                             QualType T) {
    PD.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                    Diagnostic::ak_qualtype);
    return PD;
  }

  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                             unsigned I) {
    PD.AddTaggedVal(I, Diagnostic::ak_uint);
    return PD;
  }
  
  friend inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                                    const SourceRange &R) {
    PD.AddSourceRange(R);
    return PD;
  }
  
  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                             DeclarationName N);
};

inline PartialDiagnostic PDiag(unsigned DiagID = 0) {
  return PartialDiagnostic(DiagID);
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const PartialDiagnostic &PD) {
  PD.Emit(DB);
  return DB;
}
  

}  // end namespace clang
#endif 
