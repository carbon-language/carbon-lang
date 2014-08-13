//=== SelectorExtras.h - Helpers for checkers using selectors -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_SELECTOREXTRAS_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_SELECTOREXTRAS_H

#include "clang/AST/ASTContext.h"
#include <cstdarg>

namespace clang {
namespace ento {

static inline Selector getKeywordSelectorImpl(ASTContext &Ctx,
                                              const char *First,
                                              va_list argp) {
  SmallVector<IdentifierInfo*, 10> II;
  II.push_back(&Ctx.Idents.get(First));

  while (const char *s = va_arg(argp, const char *))
    II.push_back(&Ctx.Idents.get(s));

  return Ctx.Selectors.getSelector(II.size(), &II[0]);
}

static inline Selector getKeywordSelector(ASTContext &Ctx, va_list argp) {
  const char *First = va_arg(argp, const char *);
  assert(First && "keyword selectors must have at least one argument");
  return getKeywordSelectorImpl(Ctx, First, argp);
}

END_WITH_NULL
static inline Selector getKeywordSelector(ASTContext &Ctx,
                                          const char *First, ...) {
  va_list argp;
  va_start(argp, First);
  Selector result = getKeywordSelectorImpl(Ctx, First, argp);
  va_end(argp);
  return result;
}

END_WITH_NULL
static inline void lazyInitKeywordSelector(Selector &Sel, ASTContext &Ctx,
                                           const char *First, ...) {
  if (!Sel.isNull())
    return;
  va_list argp;
  va_start(argp, First);
  Sel = getKeywordSelectorImpl(Ctx, First, argp);
  va_end(argp);
}

static inline void lazyInitNullarySelector(Selector &Sel, ASTContext &Ctx,
                                           const char *Name) {
  if (!Sel.isNull())
    return;
  Sel = GetNullarySelector(Name, Ctx);
}

} // end namespace ento
} // end namespace clang

#endif
