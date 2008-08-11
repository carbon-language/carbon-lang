//===--- LangOptions.h - C Language Family Language Options -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LangOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LANGOPTIONS_H
#define LLVM_CLANG_LANGOPTIONS_H

#include "llvm/Bitcode/SerializationFwd.h"

namespace clang {

/// LangOptions - This class keeps track of the various options that can be
/// enabled, which controls the dialect of C that is accepted.
struct LangOptions {
  
  unsigned Trigraphs         : 1;  // Trigraphs in source files.
  unsigned BCPLComment       : 1;  // BCPL-style '//' comments.
  unsigned DollarIdents      : 1;  // '$' allowed in identifiers.
  unsigned ImplicitInt       : 1;  // C89 implicit 'int'.
  unsigned Digraphs          : 1;  // C94, C99 and C++
  unsigned HexFloats         : 1;  // C99 Hexadecimal float constants.
  unsigned C99               : 1;  // C99 Support
  unsigned Microsoft         : 1;  // Microsoft extensions.
  unsigned CPlusPlus         : 1;  // C++ Support
  unsigned CPlusPlus0x       : 1;  // C++0x Support
  unsigned NoExtensions      : 1;  // All extensions are disabled, strict mode.
  unsigned CXXOperatorNames  : 1;  // Treat C++ operator names as keywords.
    
  unsigned ObjC1             : 1;  // Objective-C 1 support enabled.
  unsigned ObjC2             : 1;  // Objective-C 2 support enabled.
    
  unsigned PascalStrings     : 1;  // Allow Pascal strings
  unsigned Boolean           : 1;  // Allow bool/true/false
  unsigned WritableStrings   : 1;  // Allow writable strings
  unsigned LaxVectorConversions : 1;
  unsigned Exceptions        : 1;  // Support exception handling.
  
private:
  unsigned GC : 2; // Objective-C Garbage Collection modes.  We declare
                   // this enum as unsigned because MSVC insists on making enums
                   // signed.  Set/Query this value using accessors.  
public:  

  enum GCMode { NonGC, GCOnly, HybridGC };
  
  LangOptions() {
    Trigraphs = BCPLComment = DollarIdents = ImplicitInt = Digraphs = 0;
    HexFloats = 0;
    GC = ObjC1 = ObjC2 = 0;
    C99 = Microsoft = CPlusPlus = CPlusPlus0x = NoExtensions = 0;
    CXXOperatorNames = PascalStrings = Boolean = WritableStrings = 0;
    LaxVectorConversions = Exceptions = 0;
  }
  
  GCMode getGCMode() const { return (GCMode) GC; }
  void setGCMode(GCMode m) { GC = (unsigned) m; }
  
  /// Emit - Emit this LangOptions object to bitcode.
  void Emit(llvm::Serializer& S) const;
  
  /// Read - Read new values for this LangOption object from bitcode.
  void Read(llvm::Deserializer& S);  
};

}  // end namespace clang

#endif
