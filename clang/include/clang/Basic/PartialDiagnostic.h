#ifndef LLVM_CLANG_PARTIALDIAGNOSTIC_H
#define LLVM_CLANG_PARTIALDIAGNOSTIC_H

#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

class PartialDiagnostic {
  unsigned DiagID;
  
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

  mutable Storage *DiagStorage;

  void AddTaggedVal(intptr_t V, Diagnostic::ArgumentKind Kind) const {
    assert(DiagStorage->NumDiagArgs < Storage::MaxArguments &&
           "Too many arguments to diagnostic!");
    DiagStorage->DiagArgumentsKind[DiagStorage->NumDiagArgs] = Kind;
    DiagStorage->DiagArgumentsVal[DiagStorage->NumDiagArgs++] = V;
  }

  void AddSourceRange(const SourceRange &R) const {
    assert(DiagStorage->NumDiagRanges < 
           llvm::array_lengthof(DiagStorage->DiagRanges) &&
           "Too many arguments to diagnostic!");
    DiagStorage->DiagRanges[DiagStorage->NumDiagRanges++] = &R;
  }  

  void operator=(const PartialDiagnostic &); // DO NOT IMPLEMENT

public:
  explicit PartialDiagnostic(unsigned DiagID)
    : DiagID(DiagID), DiagStorage(new Storage) { }

  PartialDiagnostic(const PartialDiagnostic &Other) 
    : DiagStorage(Other.DiagStorage) {
    Other.DiagStorage = 0;
  }

  ~PartialDiagnostic() {
    delete DiagStorage;
  }

  friend const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                                    QualType T) {
    PD.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                    Diagnostic::ak_qualtype);
    return PD;
  }

  friend inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                                    const SourceRange &R) {
    PD.AddSourceRange(R);
    return PD;
  }
};

PartialDiagnostic PDiag(unsigned DiagID) {
  return PartialDiagnostic(DiagID);
}


}  // end namespace clang
#endif 
