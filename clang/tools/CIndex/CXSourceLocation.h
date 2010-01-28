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
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/AST/ASTContext.h"

namespace clang {
      
class ASTContext;

namespace cxloc {
  
typedef llvm::PointerIntPair<const SourceManager *, 1, bool> 
  CXSourceLocationPtr;

/// \brief Translate a Clang source location into a CIndex source location.
static inline CXSourceLocation 
translateSourceLocation(const SourceManager &SM, const LangOptions &LangOpts,
                        SourceLocation Loc, bool AtEnd = false) {
  CXSourceLocationPtr Ptr(&SM, AtEnd);
  CXSourceLocation Result = { { Ptr.getOpaqueValue(), (void *)&LangOpts, },
                              Loc.getRawEncoding() };
  return Result;
}
  
/// \brief Translate a Clang source location into a CIndex source location.
static inline CXSourceLocation translateSourceLocation(ASTContext &Context,
                                                       SourceLocation Loc,
                                                       bool AtEnd = false) {
  return translateSourceLocation(Context.getSourceManager(), 
                                 Context.getLangOptions(),
                                 Loc, AtEnd);
}

/// \brief Translate a Clang source range into a CIndex source range.
static inline CXSourceRange translateSourceRange(const SourceManager &SM, 
                                                 const LangOptions &LangOpts,
                                                 SourceRange R) {
  CXSourceRange Result = { { (void *)&SM, (void *)&LangOpts },
                           R.getBegin().getRawEncoding(),
                           R.getEnd().getRawEncoding() };
  return Result;
}
  
/// \brief Translate a Clang source range into a CIndex source range.
static inline CXSourceRange translateSourceRange(ASTContext &Context,
                                                 SourceRange R) {
  return translateSourceRange(Context.getSourceManager(),
                              Context.getLangOptions(),
                              R);
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
