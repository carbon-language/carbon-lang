//===- CXSourceLocation.h - Routines for manipulating CXSourceLocations ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXSourceLocations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CXSOURCELOCATION_H
#define LLVM_CLANG_CXSOURCELOCATION_H

#include "clang-c/Index.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
      
class ASTContext;

namespace cxloc {
  
typedef llvm::PointerIntPair<ASTContext *, 1, bool> CXSourceLocationPtr;

/// \brief Translate a Clang source location into a CIndex source location.
static inline CXSourceLocation translateSourceLocation(ASTContext &Context,
                                                       SourceLocation Loc,
                                                       bool AtEnd = false) {
  CXSourceLocationPtr Ptr(&Context, AtEnd);
  CXSourceLocation Result = { Ptr.getOpaqueValue(), Loc.getRawEncoding() };
  return Result;
}

/// \brief Translate a Clang source range into a CIndex source range.
static inline CXSourceRange translateSourceRange(ASTContext &Context,
                                                 SourceRange R) {
  CXSourceRange Result = { &Context, 
    R.getBegin().getRawEncoding(),
    R.getEnd().getRawEncoding() };
  return Result;
}

static inline SourceLocation translateSourceLocation(CXSourceLocation L) {
  return SourceLocation::getFromRawEncoding(L.int_data);
}

static inline SourceRange translateSourceRange(CXSourceRange R) {
  return SourceRange(SourceLocation::getFromRawEncoding(R.begin_int_data),
                     SourceLocation::getFromRawEncoding(R.end_int_data));
}


}} // end namespace: clang::cxloc

#endif
