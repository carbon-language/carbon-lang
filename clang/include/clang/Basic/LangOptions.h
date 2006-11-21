//===--- LangOptions.h - C Language Family Language Options -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LangOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LANGOPTIONS_H
#define LLVM_CLANG_LANGOPTIONS_H

namespace llvm {
namespace clang {

/// LangOptions - This class keeps track of the various options that can be
/// enabled, which controls the dialect of C that is accepted.
struct LangOptions {
  unsigned Trigraphs         : 1;  // Trigraphs in source files.
  unsigned BCPLComment       : 1;  // BCPL-style // comments.
  unsigned DollarIdents      : 1;  // '$' allowed in identifiers.
  unsigned Digraphs          : 1;  // When added to C?  C99?
  unsigned HexFloats         : 1;  // C99 Hexadecimal float constants.
  unsigned C99               : 1;  // C99 Support
  unsigned Microsoft         : 1;  // Microsoft extensions.
  unsigned CPlusPlus         : 1;  // C++ Support
  unsigned NoExtensions      : 1;  // All extensions are disabled, strict mode.
  
  unsigned ObjC1             : 1;  // Objective C 1 support enabled.
  unsigned ObjC2             : 1;  // Objective C 2 support enabled.
  
  LangOptions() {
    Trigraphs = BCPLComment = DollarIdents = Digraphs = ObjC1 = ObjC2 = 0;
    C99 = Microsoft = CPlusPlus = NoExtensions = 0;
  }
};

}  // end namespace clang
}  // end namespace llvm

#endif
