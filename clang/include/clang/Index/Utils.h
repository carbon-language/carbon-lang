//===--- Utils.h - Misc utilities for indexing-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for indexing related
//  functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_UTILS_H
#define LLVM_CLANG_INDEX_UTILS_H

namespace clang {
  class ASTContext;
  class SourceLocation;
  
namespace idx {
  class ASTLocation;

/// \brief Returns the ASTLocation that a source location points to.
///
/// \returns the resolved ASTLocation or an invalid ASTLocation if the source
/// location could not be resolved.
ASTLocation ResolveLocationInAST(ASTContext &Ctx, SourceLocation Loc, 
                                 ASTLocation *LastLoc = 0);

} // end namespace idx

}  // end namespace clang

#endif
