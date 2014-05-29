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

#include "RAIIObjectsForParser.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/PointerIntPair.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// OpenMP declarative directives.
//===----------------------------------------------------------------------===//

/// \brief Parsing of declarative OpenMP directives.
///
///       threadprivate-directive:
///         annot_pragma_openmp 'threadprivate' simple-variable-list
///
Parser::DeclGroupPtrTy Parser::ParseOpenMPDeclarativeDirective() {
  assert(Tok.is(tok::annot_pragma_openmp) && "Not an OpenMP directive!");
  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  SourceLocation Loc = ConsumeToken();
  SmallVector<Expr *, 5> Identifiers;
  OpenMPDirectiveKind DKind = Tok.isAnnotation()
                                  ? OMPD_unknown
                                  : getOpenMPDirectiveKind(PP.getSpelling(Tok));

  switch (DKind) {
  case OMPD_threadprivate:
    ConsumeToken();
    if (!ParseOpenMPSimpleVarList(OMPD_threadprivate, Identifiers, true)) {
      // The last seen token is annot_pragma_openmp_end - need to check for
      // extra tokens.
      if (Tok.isNot(tok::annot_pragma_openmp_end)) {
        Diag(Tok, diag::warn_omp_extra_tokens_at_eol)
            << getOpenMPDirectiveName(OMPD_threadprivate);
        SkipUntil(tok::annot_pragma_openmp_end, StopBeforeMatch);
      }
      // Skip the last annot_pragma_openmp_end.
      ConsumeToken();
      return Actions.ActOnOpenMPThreadprivateDirective(Loc, Identifiers);
    }
    break;
  case OMPD_unknown:
    Diag(Tok, diag::err_omp_unknown_directive);
    break;
  case OMPD_parallel:
  case OMPD_simd:
  case OMPD_task:
    Diag(Tok, diag::err_omp_unexpected_directive)
        << getOpenMPDirectiveName(DKind);
    break;
  }
  SkipUntil(tok::annot_pragma_openmp_end);
  return DeclGroupPtrTy();
}

/// \brief Parsing of declarative or executable OpenMP directives.
///
///       threadprivate-directive:
///         annot_pragma_openmp 'threadprivate' simple-variable-list
///         annot_pragma_openmp_end
///
///       parallel-directive:
///         annot_pragma_openmp 'parallel' {clause} annot_pragma_openmp_end
///
StmtResult Parser::ParseOpenMPDeclarativeOrExecutableDirective() {
  assert(Tok.is(tok::annot_pragma_openmp) && "Not an OpenMP directive!");
  ParenBraceBracketBalancer BalancerRAIIObj(*this);
  SmallVector<Expr *, 5> Identifiers;
  SmallVector<OMPClause *, 5> Clauses;
  SmallVector<llvm::PointerIntPair<OMPClause *, 1, bool>, OMPC_unknown + 1>
  FirstClauses(OMPC_unknown + 1);
  const unsigned ScopeFlags =
      Scope::FnScope | Scope::DeclScope | Scope::OpenMPDirectiveScope;
  SourceLocation Loc = ConsumeToken(), EndLoc;
  OpenMPDirectiveKind DKind = Tok.isAnnotation()
                                  ? OMPD_unknown
                                  : getOpenMPDirectiveKind(PP.getSpelling(Tok));
  // Name of critical directive.
  DeclarationNameInfo DirName;
  StmtResult Directive = StmtError();

  switch (DKind) {
  case OMPD_threadprivate:
    ConsumeToken();
    if (!ParseOpenMPSimpleVarList(OMPD_threadprivate, Identifiers, false)) {
      // The last seen token is annot_pragma_openmp_end - need to check for
      // extra tokens.
      if (Tok.isNot(tok::annot_pragma_openmp_end)) {
        Diag(Tok, diag::warn_omp_extra_tokens_at_eol)
            << getOpenMPDirectiveName(OMPD_threadprivate);
        SkipUntil(tok::annot_pragma_openmp_end, StopBeforeMatch);
      }
      DeclGroupPtrTy Res =
          Actions.ActOnOpenMPThreadprivateDirective(Loc, Identifiers);
      Directive = Actions.ActOnDeclStmt(Res, Loc, Tok.getLocation());
    }
    SkipUntil(tok::annot_pragma_openmp_end);
    break;
  case OMPD_parallel:
  case OMPD_simd: {
    ConsumeToken();

    Actions.StartOpenMPDSABlock(DKind, DirName, Actions.getCurScope());

    while (Tok.isNot(tok::annot_pragma_openmp_end)) {
      OpenMPClauseKind CKind = Tok.isAnnotation()
                                   ? OMPC_unknown
                                   : getOpenMPClauseKind(PP.getSpelling(Tok));
      OMPClause *Clause =
          ParseOpenMPClause(DKind, CKind, !FirstClauses[CKind].getInt());
      FirstClauses[CKind].setInt(true);
      if (Clause) {
        FirstClauses[CKind].setPointer(Clause);
        Clauses.push_back(Clause);
      }

      // Skip ',' if any.
      if (Tok.is(tok::comma))
        ConsumeToken();
    }
    // End location of the directive.
    EndLoc = Tok.getLocation();
    // Consume final annot_pragma_openmp_end.
    ConsumeToken();

    StmtResult AssociatedStmt;
    bool CreateDirective = true;
    ParseScope OMPDirectiveScope(this, ScopeFlags);
    {
      // The body is a block scope like in Lambdas and Blocks.
      Sema::CompoundScopeRAII CompoundScope(Actions);
      Actions.ActOnOpenMPRegionStart(DKind, Loc, getCurScope());
      Actions.ActOnStartOfCompoundStmt();
      // Parse statement
      AssociatedStmt = ParseStatement();
      Actions.ActOnFinishOfCompoundStmt();
      if (!AssociatedStmt.isUsable()) {
        Actions.ActOnCapturedRegionError();
        CreateDirective = false;
      } else {
        AssociatedStmt = Actions.ActOnCapturedRegionEnd(AssociatedStmt.get());
        CreateDirective = AssociatedStmt.isUsable();
      }
    }
    if (CreateDirective)
      Directive = Actions.ActOnOpenMPExecutableDirective(
          DKind, Clauses, AssociatedStmt.get(), Loc, EndLoc);

    // Exit scope.
    Actions.EndOpenMPDSABlock(Directive.get());
    OMPDirectiveScope.Exit();
    break;
  }
  case OMPD_unknown:
    Diag(Tok, diag::err_omp_unknown_directive);
    SkipUntil(tok::annot_pragma_openmp_end);
    break;
  case OMPD_task:
    Diag(Tok, diag::err_omp_unexpected_directive)
        << getOpenMPDirectiveName(DKind);
    SkipUntil(tok::annot_pragma_openmp_end);
    break;
  }
  return Directive;
}

/// \brief Parses list of simple variables for '#pragma omp threadprivate'
/// directive.
///
///   simple-variable-list:
///         '(' id-expression {, id-expression} ')'
///
bool Parser::ParseOpenMPSimpleVarList(OpenMPDirectiveKind Kind,
                                      SmallVectorImpl<Expr *> &VarList,
                                      bool AllowScopeSpecifier) {
  VarList.clear();
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPDirectiveName(Kind)))
    return true;
  bool IsCorrect = true;
  bool NoIdentIsFound = true;

  // Read tokens while ')' or annot_pragma_openmp_end is not found.
  while (Tok.isNot(tok::r_paren) && Tok.isNot(tok::annot_pragma_openmp_end)) {
    CXXScopeSpec SS;
    SourceLocation TemplateKWLoc;
    UnqualifiedId Name;
    // Read var name.
    Token PrevTok = Tok;
    NoIdentIsFound = false;

    if (AllowScopeSpecifier && getLangOpts().CPlusPlus &&
        ParseOptionalCXXScopeSpecifier(SS, ParsedType(), false)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    } else if (ParseUnqualifiedId(SS, false, false, false, ParsedType(),
                                  TemplateKWLoc, Name)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    } else if (Tok.isNot(tok::comma) && Tok.isNot(tok::r_paren) &&
               Tok.isNot(tok::annot_pragma_openmp_end)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
      Diag(PrevTok.getLocation(), diag::err_expected)
          << tok::identifier
          << SourceRange(PrevTok.getLocation(), PrevTokLocation);
    } else {
      DeclarationNameInfo NameInfo = Actions.GetNameFromUnqualifiedId(Name);
      ExprResult Res =
          Actions.ActOnOpenMPIdExpression(getCurScope(), SS, NameInfo);
      if (Res.isUsable())
        VarList.push_back(Res.get());
    }
    // Consume ','.
    if (Tok.is(tok::comma)) {
      ConsumeToken();
    }
  }

  if (NoIdentIsFound) {
    Diag(Tok, diag::err_expected) << tok::identifier;
    IsCorrect = false;
  }

  // Parse ')'.
  IsCorrect = !T.consumeClose() && IsCorrect;

  return !IsCorrect && VarList.empty();
}

/// \brief Parsing of OpenMP clauses.
///
///    clause:
///       if-clause | num_threads-clause | safelen-clause | default-clause |
///       private-clause | firstprivate-clause | shared-clause | linear-clause |
///       aligned-clause | collapse-clause
///
OMPClause *Parser::ParseOpenMPClause(OpenMPDirectiveKind DKind,
                                     OpenMPClauseKind CKind, bool FirstClause) {
  OMPClause *Clause = nullptr;
  bool ErrorFound = false;
  // Check if clause is allowed for the given directive.
  if (CKind != OMPC_unknown && !isAllowedClauseForDirective(DKind, CKind)) {
    Diag(Tok, diag::err_omp_unexpected_clause) << getOpenMPClauseName(CKind)
                                               << getOpenMPDirectiveName(DKind);
    ErrorFound = true;
  }

  switch (CKind) {
  case OMPC_if:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_collapse:
    // OpenMP [2.5, Restrictions]
    //  At most one if clause can appear on the directive.
    //  At most one num_threads clause can appear on the directive.
    // OpenMP [2.8.1, simd construct, Restrictions]
    //  Only one safelen  clause can appear on a simd directive.
    //  Only one collapse clause can appear on a simd directive.
    if (!FirstClause) {
      Diag(Tok, diag::err_omp_more_one_clause) << getOpenMPDirectiveName(DKind)
                                               << getOpenMPClauseName(CKind);
    }

    Clause = ParseOpenMPSingleExprClause(CKind);
    break;
  case OMPC_default:
  case OMPC_proc_bind:
    // OpenMP [2.14.3.1, Restrictions]
    //  Only a single default clause may be specified on a parallel, task or
    //  teams directive.
    // OpenMP [2.5, parallel Construct, Restrictions]
    //  At most one proc_bind clause can appear on the directive.
    if (!FirstClause) {
      Diag(Tok, diag::err_omp_more_one_clause) << getOpenMPDirectiveName(DKind)
                                               << getOpenMPClauseName(CKind);
    }

    Clause = ParseOpenMPSimpleClause(CKind);
    break;
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_shared:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
    Clause = ParseOpenMPVarListClause(CKind);
    break;
  case OMPC_unknown:
    Diag(Tok, diag::warn_omp_extra_tokens_at_eol)
        << getOpenMPDirectiveName(DKind);
    SkipUntil(tok::annot_pragma_openmp_end, StopBeforeMatch);
    break;
  case OMPC_threadprivate:
    Diag(Tok, diag::err_omp_unexpected_clause) << getOpenMPClauseName(CKind)
                                               << getOpenMPDirectiveName(DKind);
    SkipUntil(tok::comma, tok::annot_pragma_openmp_end, StopBeforeMatch);
    break;
  }
  return ErrorFound ? nullptr : Clause;
}

/// \brief Parsing of OpenMP clauses with single expressions like 'if',
/// 'collapse', 'safelen', 'num_threads', 'simdlen', 'num_teams' or
/// 'thread_limit'.
///
///    if-clause:
///      'if' '(' expression ')'
///
///    num_threads-clause:
///      'num_threads' '(' expression ')'
///
///    safelen-clause:
///      'safelen' '(' expression ')'
///
///    collapse-clause:
///      'collapse' '(' expression ')'
///
OMPClause *Parser::ParseOpenMPSingleExprClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPClauseName(Kind)))
    return nullptr;

  ExprResult LHS(ParseCastExpression(false, false, NotTypeCast));
  ExprResult Val(ParseRHSOfBinaryExpression(LHS, prec::Conditional));

  // Parse ')'.
  T.consumeClose();

  if (Val.isInvalid())
    return nullptr;

  return Actions.ActOnOpenMPSingleExprClause(
      Kind, Val.get(), Loc, T.getOpenLocation(), T.getCloseLocation());
}

/// \brief Parsing of simple OpenMP clauses like 'default' or 'proc_bind'.
///
///    default-clause:
///         'default' '(' 'none' | 'shared' ')
///
///    proc_bind-clause:
///         'proc_bind' '(' 'master' | 'close' | 'spread' ')
///
OMPClause *Parser::ParseOpenMPSimpleClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPClauseName(Kind)))
    return nullptr;

  unsigned Type = getOpenMPSimpleClauseType(
      Kind, Tok.isAnnotation() ? "" : PP.getSpelling(Tok));
  SourceLocation TypeLoc = Tok.getLocation();
  if (Tok.isNot(tok::r_paren) && Tok.isNot(tok::comma) &&
      Tok.isNot(tok::annot_pragma_openmp_end))
    ConsumeAnyToken();

  // Parse ')'.
  T.consumeClose();

  return Actions.ActOnOpenMPSimpleClause(Kind, Type, TypeLoc, LOpen, Loc,
                                         Tok.getLocation());
}

/// \brief Parsing of OpenMP clause 'private', 'firstprivate',
/// 'shared', 'copyin', or 'reduction'.
///
///    private-clause:
///       'private' '(' list ')'
///    firstprivate-clause:
///       'firstprivate' '(' list ')'
///    shared-clause:
///       'shared' '(' list ')'
///    linear-clause:
///       'linear' '(' list [ ':' linear-step ] ')'
///    aligned-clause:
///       'aligned' '(' list [ ':' alignment ] ')'
///
OMPClause *Parser::ParseOpenMPVarListClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  SourceLocation ColonLoc = SourceLocation();
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPClauseName(Kind)))
    return nullptr;

  SmallVector<Expr *, 5> Vars;
  bool IsComma = true;
  const bool MayHaveTail = (Kind == OMPC_linear || Kind == OMPC_aligned);
  while (IsComma || (Tok.isNot(tok::r_paren) && Tok.isNot(tok::colon) &&
                     Tok.isNot(tok::annot_pragma_openmp_end))) {
    ColonProtectionRAIIObject ColonRAII(*this, MayHaveTail);
    // Parse variable
    ExprResult VarExpr = ParseAssignmentExpression();
    if (VarExpr.isUsable()) {
      Vars.push_back(VarExpr.get());
    } else {
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    }
    // Skip ',' if any
    IsComma = Tok.is(tok::comma);
    if (IsComma)
      ConsumeToken();
    else if (Tok.isNot(tok::r_paren) &&
             Tok.isNot(tok::annot_pragma_openmp_end) &&
             (!MayHaveTail || Tok.isNot(tok::colon)))
      Diag(Tok, diag::err_omp_expected_punc) << getOpenMPClauseName(Kind);
  }

  // Parse ':' linear-step (or ':' alignment).
  Expr *TailExpr = nullptr;
  const bool MustHaveTail = MayHaveTail && Tok.is(tok::colon);
  if (MustHaveTail) {
    ColonLoc = Tok.getLocation();
    ConsumeToken();
    ExprResult Tail = ParseAssignmentExpression();
    if (Tail.isUsable())
      TailExpr = Tail.get();
    else
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
  }

  // Parse ')'.
  T.consumeClose();
  if (Vars.empty() || (MustHaveTail && !TailExpr))
    return nullptr;

  return Actions.ActOnOpenMPVarListClause(Kind, Vars, TailExpr, Loc, LOpen,
                                          ColonLoc, Tok.getLocation());
}

