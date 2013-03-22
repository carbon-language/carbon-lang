//===--- ParseOpenMP.cpp - OpenMP directives parsing ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements parsing of all OpenMP directives and clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "RAIIObjectsForParser.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// OpenMP declarative directives.
//===----------------------------------------------------------------------===//

/// \brief Parses OpenMP declarative directive
///       threadprivate-directive
///         annot_pragma_openmp threadprivate simple-variable-list
///
Parser::DeclGroupPtrTy Parser::ParseOpenMPDeclarativeDirective() {
  assert(Tok.is(tok::annot_pragma_openmp) && "Not an OpenMP directive!");

  SourceLocation Loc = ConsumeToken();
  SmallVector<DeclarationNameInfo, 5> Identifiers;
  OpenMPDirectiveKind Kind = Tok.isAnnotation() ?
                                 OMPD_unknown :
                                 getOpenMPDirectiveKind(PP.getSpelling(Tok));
  switch(Kind) {
  case OMPD_threadprivate:
    ConsumeToken();
    if (!ParseOpenMPSimpleVarList(OMPD_threadprivate, Identifiers)) {
      // The last seen token is annot_pragma_openmp_end - need to check for
      // extra tokens.
      if (Tok.isNot(tok::annot_pragma_openmp_end)) {
        Diag(Tok, diag::warn_omp_extra_tokens_at_eol)
          << getOpenMPDirectiveName(OMPD_threadprivate);
        SkipUntil(tok::annot_pragma_openmp_end, false, true);
      }
      ConsumeToken();
      return Actions.ActOnOpenMPThreadprivateDirective(Loc,
                                                       getCurScope(),
                                                       Identifiers);
    }
    break;
  case OMPD_unknown:
    Diag(Tok, diag::err_omp_unknown_directive);
    break;
  default:
    Diag(Tok, diag::err_omp_unexpected_directive)
      << getOpenMPDirectiveName(Kind);
    break;
  }
  SkipUntil(tok::annot_pragma_openmp_end, false);
  return DeclGroupPtrTy();
}

/// \brief Parses list of simple variables for '#pragma omp threadprivate'
/// directive
/// simple-variable-list:
///   ( unqualified-id {, unqualified-id} ) annot_pragma_openmp_end
///
bool Parser::ParseOpenMPSimpleVarList(
  OpenMPDirectiveKind Kind,
  SmallVectorImpl<DeclarationNameInfo> &IdList) {
  // Parse '('.
  bool IsCorrect = true;
  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPDirectiveName(Kind))) {
    SkipUntil(tok::annot_pragma_openmp_end, false, true);
    return false;
  }

  // Read tokens while ')' or annot_pragma_openmp_end is not found.
  do {
    CXXScopeSpec SS;
    SourceLocation TemplateKWLoc;
    UnqualifiedId Name;
    // Read var name.
    Token PrevTok = Tok;

    if (ParseUnqualifiedId(SS, false, false, false, ParsedType(),
                           TemplateKWLoc, Name)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                false, true);
    }
    else if (Tok.isNot(tok::comma) && Tok.isNot(tok::r_paren) &&
             Tok.isNot(tok::annot_pragma_openmp_end)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                false, true);
      Diag(PrevTok.getLocation(), diag::err_expected_unqualified_id)
        << getLangOpts().CPlusPlus
        << SourceRange(PrevTok.getLocation(), PrevTokLocation);
    } else {
      IdList.push_back(Actions.GetNameFromUnqualifiedId(Name));
    }
    // Consume ','.
    if (Tok.is(tok::comma)) {
      ConsumeToken();
    }
  } while (Tok.isNot(tok::r_paren) && Tok.isNot(tok::annot_pragma_openmp_end));

  if (IsCorrect || Tok.is(tok::r_paren)) {
    IsCorrect = !T.consumeClose() && IsCorrect;
  }

  return !IsCorrect && IdList.empty();
}
