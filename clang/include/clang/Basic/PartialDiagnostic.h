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

#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {

class DeclarationName;
  
class PartialDiagnostic {
  struct Storage {
    Storage() : NumDiagArgs(0), NumDiagRanges(0), NumCodeModificationHints(0) {
    }

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
    /// CodeModificationHints array.
    unsigned char NumCodeModificationHints;
    
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
    
    enum { MaxCodeModificationHints = 3 };
    
    /// CodeModificationHints - If valid, provides a hint with some code
    /// to insert, remove, or modify at a particular position.
    CodeModificationHint CodeModificationHints[MaxCodeModificationHints];    
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
    DiagStorage->DiagRanges[DiagStorage->NumDiagRanges++] = R;
  }  

  void AddCodeModificationHint(const CodeModificationHint &Hint) const {
    if (Hint.isNull())
      return;
    
    if (!DiagStorage)
      DiagStorage = new Storage;

    assert(DiagStorage->NumCodeModificationHints < 
             Storage::MaxCodeModificationHints &&
           "Too many code modification hints!");
    DiagStorage->CodeModificationHints[DiagStorage->NumCodeModificationHints++]
      = Hint;
  }
  
public:
  PartialDiagnostic(unsigned DiagID)
    : DiagID(DiagID), DiagStorage(0) { }

  PartialDiagnostic(const PartialDiagnostic &Other) 
    : DiagID(Other.DiagID), DiagStorage(0) 
  {
    if (Other.DiagStorage)
      DiagStorage = new Storage(*Other.DiagStorage);
  }

  PartialDiagnostic &operator=(const PartialDiagnostic &Other) {
    DiagID = Other.DiagID;
    if (Other.DiagStorage) {
      if (DiagStorage)
        *DiagStorage = *Other.DiagStorage;
      else
        DiagStorage = new Storage(*Other.DiagStorage);
    } else {
      delete DiagStorage;
      DiagStorage = 0;
    }

    return *this;
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
      DB.AddSourceRange(DiagStorage->DiagRanges[i]);
    
    // Add all code modification hints
    for (unsigned i = 0, e = DiagStorage->NumCodeModificationHints; i != e; ++i)
      DB.AddCodeModificationHint(DiagStorage->CodeModificationHints[i]);
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
                                             DeclarationName N);
  
  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                             const CodeModificationHint &Hint) {
    PD.AddCodeModificationHint(Hint);
    return PD;
  }
  
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
