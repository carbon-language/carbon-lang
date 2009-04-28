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

namespace clang {

/// LangOptions - This class keeps track of the various options that can be
/// enabled, which controls the dialect of C that is accepted.
class LangOptions {
public:
  unsigned Trigraphs         : 1;  // Trigraphs in source files.
  unsigned BCPLComment       : 1;  // BCPL-style '//' comments.
  unsigned DollarIdents      : 1;  // '$' allowed in identifiers.
  unsigned AsmPreprocessor   : 1;  // Preprocessor in asm mode.
  unsigned GNUMode           : 1;  // True in gnu99 mode false in c99 mode (etc)
  unsigned ImplicitInt       : 1;  // C89 implicit 'int'.
  unsigned Digraphs          : 1;  // C94, C99 and C++
  unsigned HexFloats         : 1;  // C99 Hexadecimal float constants.
  unsigned C99               : 1;  // C99 Support
  unsigned Microsoft         : 1;  // Microsoft extensions.
  unsigned CPlusPlus         : 1;  // C++ Support
  unsigned CPlusPlus0x       : 1;  // C++0x Support
  unsigned CXXOperatorNames  : 1;  // Treat C++ operator names as keywords.
    
  unsigned ObjC1             : 1;  // Objective-C 1 support enabled.
  unsigned ObjC2             : 1;  // Objective-C 2 support enabled.
  unsigned ObjCNonFragileABI : 1;  // Objective-C modern abi enabled
    
  unsigned PascalStrings     : 1;  // Allow Pascal strings
  unsigned WritableStrings   : 1;  // Allow writable strings
  unsigned LaxVectorConversions : 1;
  unsigned Exceptions        : 1;  // Support exception handling.

  unsigned NeXTRuntime       : 1; // Use NeXT runtime.
  unsigned Freestanding      : 1; // Freestanding implementation
  unsigned NoBuiltin         : 1; // Do not use builtin functions (-fno-builtin)

  unsigned ThreadsafeStatics : 1; // Whether static initializers are protected
                                  // by locks.
  unsigned Blocks            : 1; // block extension to C
  unsigned EmitAllDecls      : 1; // Emit all declarations, even if
                                  // they are unused.
  unsigned MathErrno         : 1; // Math functions must respect errno
                                  // (modulo the platform support).

  unsigned OverflowChecking  : 1; // Extension to call a handler function when
                                  // signed integer arithmetic overflows.

  unsigned HeinousExtensions : 1; // Extensions that we really don't like and
                                  // may be ripped out at any time.

  unsigned Optimize          : 1; // Whether __OPTIMIZE__ should be defined.
  unsigned OptimizeSize      : 1; // Whether __OPTIMIZE_SIZE__ should be 
                                  // defined.
  unsigned Static            : 1; // Should __STATIC__ be defined (as
                                  // opposed to __DYNAMIC__).
  unsigned PICLevel          : 2; // The value for __PIC__, if non-zero.

  unsigned GNUInline         : 1; // Should GNU inline semantics be
                                  // used (instead of C99 semantics).
  unsigned NoInline          : 1; // Should __NO_INLINE__ be defined.

  unsigned ObjCGCBitmapPrint : 1; // Enable printing of gc's bitmap layout
                                  // for __weak/__strong ivars.

private:
  unsigned GC : 2; // Objective-C Garbage Collection modes.  We declare
                   // this enum as unsigned because MSVC insists on making enums
                   // signed.  Set/Query this value using accessors.  
  unsigned SymbolVisibility  : 3; // Symbol's visibility.

  /// The user provided name for the "main file", if non-null. This is
  /// useful in situations where the input file name does not match
  /// the original input file, for example with -save-temps.
  const char *MainFileName;

public:  
  unsigned InstantiationDepth;    // Maximum template instantiation depth.

  enum GCMode { NonGC, GCOnly, HybridGC };
  enum VisibilityMode { 
    Default, 
    Protected, 
    Hidden
  };
  
  LangOptions() {
    Trigraphs = BCPLComment = DollarIdents = AsmPreprocessor = 0;
    GNUMode = ImplicitInt = Digraphs = 0;
    HexFloats = 0;
    GC = ObjC1 = ObjC2 = ObjCNonFragileABI = 0;
    C99 = Microsoft = CPlusPlus = CPlusPlus0x = 0;
    CXXOperatorNames = PascalStrings = WritableStrings = 0;
    Exceptions = NeXTRuntime = Freestanding = NoBuiltin = 0;
    LaxVectorConversions = 1;
    HeinousExtensions = 0;
    
    SymbolVisibility = (unsigned) Default;
    
    // FIXME: The default should be 1.
    ThreadsafeStatics = 0;
    Blocks = 0;
    EmitAllDecls = 0;
    MathErrno = 1;

    OverflowChecking = 0;
    ObjCGCBitmapPrint = 0;

    InstantiationDepth = 99;
    
    Optimize = 0;
    OptimizeSize = 0;

    Static = 0;
    PICLevel = 0;

    GNUInline = 0;
    NoInline = 0;

    MainFileName = 0;
  }
  
  GCMode getGCMode() const { return (GCMode) GC; }
  void setGCMode(GCMode m) { GC = (unsigned) m; }

  const char *getMainFileName() const { return MainFileName; }
  void setMainFileName(const char *Name) { MainFileName = Name; }

  VisibilityMode getVisibilityMode() const { 
    return (VisibilityMode) SymbolVisibility; 
  }
  void setVisibilityMode(VisibilityMode v) { SymbolVisibility = (unsigned) v; }
};

}  // end namespace clang

#endif
