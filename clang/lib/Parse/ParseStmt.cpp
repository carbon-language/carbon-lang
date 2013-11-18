//===--- ParseStmt.cpp - Statement and Block Parser -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Statement and Block portions of the Parser
// interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/TypoCorrection.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.8: Statements and Blocks.
//===----------------------------------------------------------------------===//

/// \brief Parse a standalone statement (for instance, as the body of an 'if',
/// 'while', or 'for').
StmtResult Parser::ParseStatement(SourceLocation *TrailingElseLoc) {
  StmtResult Res;

  // We may get back a null statement if we found a #pragma. Keep going until
  // we get an actual statement.
  do {
    StmtVector Stmts;
    Res = ParseStatementOrDeclaration(Stmts, true, TrailingElseLoc);
  } while (!Res.isInvalid() && !Res.get());

  return Res;
}

/// ParseStatementOrDeclaration - Read 'statement' or 'declaration'.
///       StatementOrDeclaration:
///         statement
///         declaration
///
///       statement:
///         labeled-statement
///         compound-statement
///         expression-statement
///         selection-statement
///         iteration-statement
///         jump-statement
/// [C++]   declaration-statement
/// [C++]   try-block
/// [MS]    seh-try-block
/// [OBC]   objc-throw-statement
/// [OBC]   objc-try-catch-statement
/// [OBC]   objc-synchronized-statement
/// [GNU]   asm-statement
/// [OMP]   openmp-construct             [TODO]
///
///       labeled-statement:
///         identifier ':' statement
///         'case' constant-expression ':' statement
///         'default' ':' statement
///
///       selection-statement:
///         if-statement
///         switch-statement
///
///       iteration-statement:
///         while-statement
///         do-statement
///         for-statement
///
///       expression-statement:
///         expression[opt] ';'
///
///       jump-statement:
///         'goto' identifier ';'
///         'continue' ';'
///         'break' ';'
///         'return' expression[opt] ';'
/// [GNU]   'goto' '*' expression ';'
///
/// [OBC] objc-throw-statement:
/// [OBC]   '@' 'throw' expression ';'
/// [OBC]   '@' 'throw' ';'
///
StmtResult
Parser::ParseStatementOrDeclaration(StmtVector &Stmts, bool OnlyStatement,
                                    SourceLocation *TrailingElseLoc) {

  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  ParsedAttributesWithRange Attrs(AttrFactory);
  MaybeParseCXX11Attributes(Attrs, 0, /*MightBeObjCMessageSend*/ true);

  StmtResult Res = ParseStatementOrDeclarationAfterAttributes(Stmts,
                                 OnlyStatement, TrailingElseLoc, Attrs);

  assert((Attrs.empty() || Res.isInvalid() || Res.isUsable()) &&
         "attributes on empty statement");

  if (Attrs.empty() || Res.isInvalid())
    return Res;

  return Actions.ProcessStmtAttributes(Res.get(), Attrs.getList(), Attrs.Range);
}

namespace {
class StatementFilterCCC : public CorrectionCandidateCallback {
public:
  StatementFilterCCC(Token nextTok) : NextToken(nextTok) {
    WantTypeSpecifiers = nextTok.is(tok::l_paren) || nextTok.is(tok::less) ||
                         nextTok.is(tok::identifier) || nextTok.is(tok::star) ||
                         nextTok.is(tok::amp) || nextTok.is(tok::l_square);
    WantExpressionKeywords = nextTok.is(tok::l_paren) ||
                             nextTok.is(tok::identifier) ||
                             nextTok.is(tok::arrow) || nextTok.is(tok::period);
    WantRemainingKeywords = nextTok.is(tok::l_paren) || nextTok.is(tok::semi) ||
                            nextTok.is(tok::identifier) ||
                            nextTok.is(tok::l_brace);
    WantCXXNamedCasts = false;
  }

  virtual bool ValidateCandidate(const TypoCorrection &candidate) {
    if (FieldDecl *FD = candidate.getCorrectionDeclAs<FieldDecl>())
      return !candidate.getCorrectionSpecifier() || isa<ObjCIvarDecl>(FD);
    if (NextToken.is(tok::equal))
      return candidate.getCorrectionDeclAs<VarDecl>();
    if (NextToken.is(tok::period) &&
        candidate.getCorrectionDeclAs<NamespaceDecl>())
      return false;
    return CorrectionCandidateCallback::ValidateCandidate(candidate);
  }

private:
  Token NextToken;
};
}

StmtResult
Parser::ParseStatementOrDeclarationAfterAttributes(StmtVector &Stmts,
          bool OnlyStatement, SourceLocation *TrailingElseLoc,
          ParsedAttributesWithRange &Attrs) {
  const char *SemiError = 0;
  StmtResult Res;

  // Cases in this switch statement should fall through if the parser expects
  // the token to end in a semicolon (in which case SemiError should be set),
  // or they directly 'return;' if not.
Retry:
  tok::TokenKind Kind  = Tok.getKind();
  SourceLocation AtLoc;
  switch (Kind) {
  case tok::at: // May be a @try or @throw statement
    {
      ProhibitAttributes(Attrs); // TODO: is it correct?
      AtLoc = ConsumeToken();  // consume @
      return ParseObjCAtStatement(AtLoc);
    }

  case tok::code_completion:
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Statement);
    cutOffParsing();
    return StmtError();

  case tok::identifier: {
    Token Next = NextToken();
    if (Next.is(tok::colon)) { // C99 6.8.1: labeled-statement
      // identifier ':' statement
      return ParseLabeledStatement(Attrs);
    }

    // Look up the identifier, and typo-correct it to a keyword if it's not
    // found.
    if (Next.isNot(tok::coloncolon)) {
      // Try to limit which sets of keywords should be included in typo
      // correction based on what the next token is.
      StatementFilterCCC Validator(Next);
      if (TryAnnotateName(/*IsAddressOfOperand*/false, &Validator)
            == ANK_Error) {
        // Handle errors here by skipping up to the next semicolon or '}', and
        // eat the semicolon if that's what stopped us.
        SkipUntil(tok::r_brace, StopAtSemi | StopBeforeMatch);
        if (Tok.is(tok::semi))
          ConsumeToken();
        return StmtError();
      }

      // If the identifier was typo-corrected, try again.
      if (Tok.isNot(tok::identifier))
        goto Retry;
    }

    // Fall through
  }

  default: {
    if ((getLangOpts().CPlusPlus || !OnlyStatement) && isDeclarationStatement()) {
      SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
      DeclGroupPtrTy Decl = ParseDeclaration(Stmts, Declarator::BlockContext,
                                             DeclEnd, Attrs);
      return Actions.ActOnDeclStmt(Decl, DeclStart, DeclEnd);
    }

    if (Tok.is(tok::r_brace)) {
      Diag(Tok, diag::err_expected_statement);
      return StmtError();
    }

    return ParseExprStatement();
  }

  case tok::kw_case:                // C99 6.8.1: labeled-statement
    return ParseCaseStatement();
  case tok::kw_default:             // C99 6.8.1: labeled-statement
    return ParseDefaultStatement();

  case tok::l_brace:                // C99 6.8.2: compound-statement
    return ParseCompoundStatement();
  case tok::semi: {                 // C99 6.8.3p3: expression[opt] ';'
    bool HasLeadingEmptyMacro = Tok.hasLeadingEmptyMacro();
    return Actions.ActOnNullStmt(ConsumeToken(), HasLeadingEmptyMacro);
  }

  case tok::kw_if:                  // C99 6.8.4.1: if-statement
    return ParseIfStatement(TrailingElseLoc);
  case tok::kw_switch:              // C99 6.8.4.2: switch-statement
    return ParseSwitchStatement(TrailingElseLoc);

  case tok::kw_while:               // C99 6.8.5.1: while-statement
    return ParseWhileStatement(TrailingElseLoc);
  case tok::kw_do:                  // C99 6.8.5.2: do-statement
    Res = ParseDoStatement();
    SemiError = "do/while";
    break;
  case tok::kw_for:                 // C99 6.8.5.3: for-statement
    return ParseForStatement(TrailingElseLoc);

  case tok::kw_goto:                // C99 6.8.6.1: goto-statement
    Res = ParseGotoStatement();
    SemiError = "goto";
    break;
  case tok::kw_continue:            // C99 6.8.6.2: continue-statement
    Res = ParseContinueStatement();
    SemiError = "continue";
    break;
  case tok::kw_break:               // C99 6.8.6.3: break-statement
    Res = ParseBreakStatement();
    SemiError = "break";
    break;
  case tok::kw_return:              // C99 6.8.6.4: return-statement
    Res = ParseReturnStatement();
    SemiError = "return";
    break;

  case tok::kw_asm: {
    ProhibitAttributes(Attrs);
    bool msAsm = false;
    Res = ParseAsmStatement(msAsm);
    Res = Actions.ActOnFinishFullStmt(Res.get());
    if (msAsm) return Res;
    SemiError = "asm";
    break;
  }

  case tok::kw_try:                 // C++ 15: try-block
    return ParseCXXTryBlock();

  case tok::kw___try:
    ProhibitAttributes(Attrs); // TODO: is it correct?
    return ParseSEHTryBlock();

  case tok::annot_pragma_vis:
    ProhibitAttributes(Attrs);
    HandlePragmaVisibility();
    return StmtEmpty();

  case tok::annot_pragma_pack:
    ProhibitAttributes(Attrs);
    HandlePragmaPack();
    return StmtEmpty();

  case tok::annot_pragma_msstruct:
    ProhibitAttributes(Attrs);
    HandlePragmaMSStruct();
    return StmtEmpty();

  case tok::annot_pragma_align:
    ProhibitAttributes(Attrs);
    HandlePragmaAlign();
    return StmtEmpty();

  case tok::annot_pragma_weak:
    ProhibitAttributes(Attrs);
    HandlePragmaWeak();
    return StmtEmpty();

  case tok::annot_pragma_weakalias:
    ProhibitAttributes(Attrs);
    HandlePragmaWeakAlias();
    return StmtEmpty();

  case tok::annot_pragma_redefine_extname:
    ProhibitAttributes(Attrs);
    HandlePragmaRedefineExtname();
    return StmtEmpty();

  case tok::annot_pragma_fp_contract:
    ProhibitAttributes(Attrs);
    Diag(Tok, diag::err_pragma_fp_contract_scope);
    ConsumeToken();
    return StmtError();

  case tok::annot_pragma_opencl_extension:
    ProhibitAttributes(Attrs);
    HandlePragmaOpenCLExtension();
    return StmtEmpty();

  case tok::annot_pragma_captured:
    ProhibitAttributes(Attrs);
    return HandlePragmaCaptured();

  case tok::annot_pragma_openmp:
    ProhibitAttributes(Attrs);
    return ParseOpenMPDeclarativeOrExecutableDirective();

  }

  // If we reached this code, the statement must end in a semicolon.
  if (Tok.is(tok::semi)) {
    ConsumeToken();
  } else if (!Res.isInvalid()) {
    // If the result was valid, then we do want to diagnose this.  Use
    // ExpectAndConsume to emit the diagnostic, even though we know it won't
    // succeed.
    ExpectAndConsume(tok::semi, diag::err_expected_semi_after_stmt, SemiError);
    // Skip until we see a } or ;, but don't eat it.
    SkipUntil(tok::r_brace, StopAtSemi | StopBeforeMatch);
  }

  return Res;
}

/// \brief Parse an expression statement.
StmtResult Parser::ParseExprStatement() {
  // If a case keyword is missing, this is where it should be inserted.
  Token OldToken = Tok;

  // expression[opt] ';'
  ExprResult Expr(ParseExpression());
  if (Expr.isInvalid()) {
    // If the expression is invalid, skip ahead to the next semicolon or '}'.
    // Not doing this opens us up to the possibility of infinite loops if
    // ParseExpression does not consume any tokens.
    SkipUntil(tok::r_brace, StopAtSemi | StopBeforeMatch);
    if (Tok.is(tok::semi))
      ConsumeToken();
    return Actions.ActOnExprStmtError();
  }

  if (Tok.is(tok::colon) && getCurScope()->isSwitchScope() &&
      Actions.CheckCaseExpression(Expr.get())) {
    // If a constant expression is followed by a colon inside a switch block,
    // suggest a missing case keyword.
    Diag(OldToken, diag::err_expected_case_before_expression)
      << FixItHint::CreateInsertion(OldToken.getLocation(), "case ");

    // Recover parsing as a case statement.
    return ParseCaseStatement(/*MissingCase=*/true, Expr);
  }

  // Otherwise, eat the semicolon.
  ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
  return Actions.ActOnExprStmt(Expr);
}

StmtResult Parser::ParseSEHTryBlock() {
  assert(Tok.is(tok::kw___try) && "Expected '__try'");
  SourceLocation Loc = ConsumeToken();
  return ParseSEHTryBlockCommon(Loc);
}

/// ParseSEHTryBlockCommon
///
/// seh-try-block:
///   '__try' compound-statement seh-handler
///
/// seh-handler:
///   seh-except-block
///   seh-finally-block
///
StmtResult Parser::ParseSEHTryBlockCommon(SourceLocation TryLoc) {
  if(Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok,diag::err_expected_lbrace));

  StmtResult TryBlock(ParseCompoundStatement());
  if(TryBlock.isInvalid())
    return TryBlock;

  StmtResult Handler;
  if (Tok.is(tok::identifier) &&
      Tok.getIdentifierInfo() == getSEHExceptKeyword()) {
    SourceLocation Loc = ConsumeToken();
    Handler = ParseSEHExceptBlock(Loc);
  } else if (Tok.is(tok::kw___finally)) {
    SourceLocation Loc = ConsumeToken();
    Handler = ParseSEHFinallyBlock(Loc);
  } else {
    return StmtError(Diag(Tok,diag::err_seh_expected_handler));
  }

  if(Handler.isInvalid())
    return Handler;

  return Actions.ActOnSEHTryBlock(false /* IsCXXTry */,
                                  TryLoc,
                                  TryBlock.take(),
                                  Handler.take());
}

/// ParseSEHExceptBlock - Handle __except
///
/// seh-except-block:
///   '__except' '(' seh-filter-expression ')' compound-statement
///
StmtResult Parser::ParseSEHExceptBlock(SourceLocation ExceptLoc) {
  PoisonIdentifierRAIIObject raii(Ident__exception_code, false),
    raii2(Ident___exception_code, false),
    raii3(Ident_GetExceptionCode, false);

  if(ExpectAndConsume(tok::l_paren,diag::err_expected_lparen))
    return StmtError();

  ParseScope ExpectScope(this, Scope::DeclScope | Scope::ControlScope);

  if (getLangOpts().Borland) {
    Ident__exception_info->setIsPoisoned(false);
    Ident___exception_info->setIsPoisoned(false);
    Ident_GetExceptionInfo->setIsPoisoned(false);
  }
  ExprResult FilterExpr(ParseExpression());

  if (getLangOpts().Borland) {
    Ident__exception_info->setIsPoisoned(true);
    Ident___exception_info->setIsPoisoned(true);
    Ident_GetExceptionInfo->setIsPoisoned(true);
  }

  if(FilterExpr.isInvalid())
    return StmtError();

  if(ExpectAndConsume(tok::r_paren,diag::err_expected_rparen))
    return StmtError();

  StmtResult Block(ParseCompoundStatement());

  if(Block.isInvalid())
    return Block;

  return Actions.ActOnSEHExceptBlock(ExceptLoc, FilterExpr.take(), Block.take());
}

/// ParseSEHFinallyBlock - Handle __finally
///
/// seh-finally-block:
///   '__finally' compound-statement
///
StmtResult Parser::ParseSEHFinallyBlock(SourceLocation FinallyBlock) {
  PoisonIdentifierRAIIObject raii(Ident__abnormal_termination, false),
    raii2(Ident___abnormal_termination, false),
    raii3(Ident_AbnormalTermination, false);

  StmtResult Block(ParseCompoundStatement());
  if(Block.isInvalid())
    return Block;

  return Actions.ActOnSEHFinallyBlock(FinallyBlock,Block.take());
}

/// ParseLabeledStatement - We have an identifier and a ':' after it.
///
///       labeled-statement:
///         identifier ':' statement
/// [GNU]   identifier ':' attributes[opt] statement
///
StmtResult Parser::ParseLabeledStatement(ParsedAttributesWithRange &attrs) {
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  Token IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.

  assert(Tok.is(tok::colon) && "Not a label!");

  // identifier ':' statement
  SourceLocation ColonLoc = ConsumeToken();

  // Read label attributes, if present.
  StmtResult SubStmt;
  if (Tok.is(tok::kw___attribute)) {
    ParsedAttributesWithRange TempAttrs(AttrFactory);
    ParseGNUAttributes(TempAttrs);

    // In C++, GNU attributes only apply to the label if they are followed by a
    // semicolon, to disambiguate label attributes from attributes on a labeled
    // declaration.
    //
    // This doesn't quite match what GCC does; if the attribute list is empty
    // and followed by a semicolon, GCC will reject (it appears to parse the
    // attributes as part of a statement in that case). That looks like a bug.
    if (!getLangOpts().CPlusPlus || Tok.is(tok::semi))
      attrs.takeAllFrom(TempAttrs);
    else if (isDeclarationStatement()) {
      StmtVector Stmts;
      // FIXME: We should do this whether or not we have a declaration
      // statement, but that doesn't work correctly (because ProhibitAttributes
      // can't handle GNU attributes), so only call it in the one case where
      // GNU attributes are allowed.
      SubStmt = ParseStatementOrDeclarationAfterAttributes(
          Stmts, /*OnlyStmts*/ true, 0, TempAttrs);
      if (!TempAttrs.empty() && !SubStmt.isInvalid())
        SubStmt = Actions.ProcessStmtAttributes(
            SubStmt.get(), TempAttrs.getList(), TempAttrs.Range);
    } else {
      Diag(Tok, diag::err_expected_semi_after) << "__attribute__";
    }
  }

  // If we've not parsed a statement yet, parse one now.
  if (!SubStmt.isInvalid() && !SubStmt.isUsable())
    SubStmt = ParseStatement();

  // Broken substmt shouldn't prevent the label from being added to the AST.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);

  LabelDecl *LD = Actions.LookupOrCreateLabel(IdentTok.getIdentifierInfo(),
                                              IdentTok.getLocation());
  if (AttributeList *Attrs = attrs.getList()) {
    Actions.ProcessDeclAttributeList(Actions.CurScope, LD, Attrs);
    attrs.clear();
  }

  return Actions.ActOnLabelStmt(IdentTok.getLocation(), LD, ColonLoc,
                                SubStmt.get());
}

/// ParseCaseStatement
///       labeled-statement:
///         'case' constant-expression ':' statement
/// [GNU]   'case' constant-expression '...' constant-expression ':' statement
///
StmtResult Parser::ParseCaseStatement(bool MissingCase, ExprResult Expr) {
  assert((MissingCase || Tok.is(tok::kw_case)) && "Not a case stmt!");

  // It is very very common for code to contain many case statements recursively
  // nested, as in (but usually without indentation):
  //  case 1:
  //    case 2:
  //      case 3:
  //         case 4:
  //           case 5: etc.
  //
  // Parsing this naively works, but is both inefficient and can cause us to run
  // out of stack space in our recursive descent parser.  As a special case,
  // flatten this recursion into an iterative loop.  This is complex and gross,
  // but all the grossness is constrained to ParseCaseStatement (and some
  // weirdness in the actions), so this is just local grossness :).

  // TopLevelCase - This is the highest level we have parsed.  'case 1' in the
  // example above.
  StmtResult TopLevelCase(true);

  // DeepestParsedCaseStmt - This is the deepest statement we have parsed, which
  // gets updated each time a new case is parsed, and whose body is unset so
  // far.  When parsing 'case 4', this is the 'case 3' node.
  Stmt *DeepestParsedCaseStmt = 0;

  // While we have case statements, eat and stack them.
  SourceLocation ColonLoc;
  do {
    SourceLocation CaseLoc = MissingCase ? Expr.get()->getExprLoc() :
                                           ConsumeToken();  // eat the 'case'.

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteCase(getCurScope());
      cutOffParsing();
      return StmtError();
    }

    /// We don't want to treat 'case x : y' as a potential typo for 'case x::y'.
    /// Disable this form of error recovery while we're parsing the case
    /// expression.
    ColonProtectionRAIIObject ColonProtection(*this);

    ExprResult LHS(MissingCase ? Expr : ParseConstantExpression());
    MissingCase = false;
    if (LHS.isInvalid()) {
      SkipUntil(tok::colon, StopAtSemi);
      return StmtError();
    }

    // GNU case range extension.
    SourceLocation DotDotDotLoc;
    ExprResult RHS;
    if (Tok.is(tok::ellipsis)) {
      Diag(Tok, diag::ext_gnu_case_range);
      DotDotDotLoc = ConsumeToken();

      RHS = ParseConstantExpression();
      if (RHS.isInvalid()) {
        SkipUntil(tok::colon, StopAtSemi);
        return StmtError();
      }
    }

    ColonProtection.restore();

    if (Tok.is(tok::colon)) {
      ColonLoc = ConsumeToken();

    // Treat "case blah;" as a typo for "case blah:".
    } else if (Tok.is(tok::semi)) {
      ColonLoc = ConsumeToken();
      Diag(ColonLoc, diag::err_expected_colon_after) << "'case'"
        << FixItHint::CreateReplacement(ColonLoc, ":");
    } else {
      SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
      Diag(ExpectedLoc, diag::err_expected_colon_after) << "'case'"
        << FixItHint::CreateInsertion(ExpectedLoc, ":");
      ColonLoc = ExpectedLoc;
    }

    StmtResult Case =
      Actions.ActOnCaseStmt(CaseLoc, LHS.get(), DotDotDotLoc,
                            RHS.get(), ColonLoc);

    // If we had a sema error parsing this case, then just ignore it and
    // continue parsing the sub-stmt.
    if (Case.isInvalid()) {
      if (TopLevelCase.isInvalid())  // No parsed case stmts.
        return ParseStatement();
      // Otherwise, just don't add it as a nested case.
    } else {
      // If this is the first case statement we parsed, it becomes TopLevelCase.
      // Otherwise we link it into the current chain.
      Stmt *NextDeepest = Case.get();
      if (TopLevelCase.isInvalid())
        TopLevelCase = Case;
      else
        Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, Case.get());
      DeepestParsedCaseStmt = NextDeepest;
    }

    // Handle all case statements.
  } while (Tok.is(tok::kw_case));

  assert(!TopLevelCase.isInvalid() && "Should have parsed at least one case!");

  // If we found a non-case statement, start by parsing it.
  StmtResult SubStmt;

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Nicely diagnose the common error "switch (X) { case 4: }", which is
    // not valid.
    SourceLocation AfterColonLoc = PP.getLocForEndOfToken(ColonLoc);
    Diag(AfterColonLoc, diag::err_label_end_of_compound_statement)
      << FixItHint::CreateInsertion(AfterColonLoc, " ;");
    SubStmt = true;
  }

  // Broken sub-stmt shouldn't prevent forming the case statement properly.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(SourceLocation());

  // Install the body into the most deeply-nested case.
  Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, SubStmt.get());

  // Return the top level parsed statement tree.
  return TopLevelCase;
}

/// ParseDefaultStatement
///       labeled-statement:
///         'default' ':' statement
/// Note that this does not parse the 'statement' at the end.
///
StmtResult Parser::ParseDefaultStatement() {
  assert(Tok.is(tok::kw_default) && "Not a default stmt!");
  SourceLocation DefaultLoc = ConsumeToken();  // eat the 'default'.

  SourceLocation ColonLoc;
  if (Tok.is(tok::colon)) {
    ColonLoc = ConsumeToken();

  // Treat "default;" as a typo for "default:".
  } else if (Tok.is(tok::semi)) {
    ColonLoc = ConsumeToken();
    Diag(ColonLoc, diag::err_expected_colon_after) << "'default'"
      << FixItHint::CreateReplacement(ColonLoc, ":");
  } else {
    SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
    Diag(ExpectedLoc, diag::err_expected_colon_after) << "'default'"
      << FixItHint::CreateInsertion(ExpectedLoc, ":");
    ColonLoc = ExpectedLoc;
  }

  StmtResult SubStmt;

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Diagnose the common error "switch (X) {... default: }", which is
    // not valid.
    SourceLocation AfterColonLoc = PP.getLocForEndOfToken(ColonLoc);
    Diag(AfterColonLoc, diag::err_label_end_of_compound_statement)
      << FixItHint::CreateInsertion(AfterColonLoc, " ;");
    SubStmt = true;
  }

  // Broken sub-stmt shouldn't prevent forming the case statement properly.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);

  return Actions.ActOnDefaultStmt(DefaultLoc, ColonLoc,
                                  SubStmt.get(), getCurScope());
}

StmtResult Parser::ParseCompoundStatement(bool isStmtExpr) {
  return ParseCompoundStatement(isStmtExpr, Scope::DeclScope);
}

/// ParseCompoundStatement - Parse a "{}" block.
///
///       compound-statement: [C99 6.8.2]
///         { block-item-list[opt] }
/// [GNU]   { label-declarations block-item-list } [TODO]
///
///       block-item-list:
///         block-item
///         block-item-list block-item
///
///       block-item:
///         declaration
/// [GNU]   '__extension__' declaration
///         statement
/// [OMP]   openmp-directive            [TODO]
///
/// [GNU] label-declarations:
/// [GNU]   label-declaration
/// [GNU]   label-declarations label-declaration
///
/// [GNU] label-declaration:
/// [GNU]   '__label__' identifier-list ';'
///
/// [OMP] openmp-directive:             [TODO]
/// [OMP]   barrier-directive
/// [OMP]   flush-directive
///
StmtResult Parser::ParseCompoundStatement(bool isStmtExpr,
                                          unsigned ScopeFlags) {
  assert(Tok.is(tok::l_brace) && "Not a compount stmt!");

  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  ParseScope CompoundScope(this, ScopeFlags);

  // Parse the statements in the body.
  return ParseCompoundStatementBody(isStmtExpr);
}

/// Parse any pragmas at the start of the compound expression. We handle these
/// separately since some pragmas (FP_CONTRACT) must appear before any C
/// statement in the compound, but may be intermingled with other pragmas.
void Parser::ParseCompoundStatementLeadingPragmas() {
  bool checkForPragmas = true;
  while (checkForPragmas) {
    switch (Tok.getKind()) {
    case tok::annot_pragma_vis:
      HandlePragmaVisibility();
      break;
    case tok::annot_pragma_pack:
      HandlePragmaPack();
      break;
    case tok::annot_pragma_msstruct:
      HandlePragmaMSStruct();
      break;
    case tok::annot_pragma_align:
      HandlePragmaAlign();
      break;
    case tok::annot_pragma_weak:
      HandlePragmaWeak();
      break;
    case tok::annot_pragma_weakalias:
      HandlePragmaWeakAlias();
      break;
    case tok::annot_pragma_redefine_extname:
      HandlePragmaRedefineExtname();
      break;
    case tok::annot_pragma_opencl_extension:
      HandlePragmaOpenCLExtension();
      break;
    case tok::annot_pragma_fp_contract:
      HandlePragmaFPContract();
      break;
    default:
      checkForPragmas = false;
      break;
    }
  }

}

/// ParseCompoundStatementBody - Parse a sequence of statements and invoke the
/// ActOnCompoundStmt action.  This expects the '{' to be the current token, and
/// consume the '}' at the end of the block.  It does not manipulate the scope
/// stack.
StmtResult Parser::ParseCompoundStatementBody(bool isStmtExpr) {
  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(),
                                Tok.getLocation(),
                                "in compound statement ('{}')");

  // Record the state of the FP_CONTRACT pragma, restore on leaving the
  // compound statement.
  Sema::FPContractStateRAII SaveFPContractState(Actions);

  InMessageExpressionRAIIObject InMessage(*this, false);
  BalancedDelimiterTracker T(*this, tok::l_brace);
  if (T.consumeOpen())
    return StmtError();

  Sema::CompoundScopeRAII CompoundScope(Actions);

  // Parse any pragmas at the beginning of the compound statement.
  ParseCompoundStatementLeadingPragmas();

  StmtVector Stmts;

  // "__label__ X, Y, Z;" is the GNU "Local Label" extension.  These are
  // only allowed at the start of a compound stmt regardless of the language.
  while (Tok.is(tok::kw___label__)) {
    SourceLocation LabelLoc = ConsumeToken();

    SmallVector<Decl *, 8> DeclsInGroup;
    while (1) {
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        break;
      }

      IdentifierInfo *II = Tok.getIdentifierInfo();
      SourceLocation IdLoc = ConsumeToken();
      DeclsInGroup.push_back(Actions.LookupOrCreateLabel(II, IdLoc, LabelLoc));

      if (!Tok.is(tok::comma))
        break;
      ConsumeToken();
    }

    DeclSpec DS(AttrFactory);
    DeclGroupPtrTy Res =
        Actions.FinalizeDeclaratorGroup(getCurScope(), DS, DeclsInGroup);
    StmtResult R = Actions.ActOnDeclStmt(Res, LabelLoc, Tok.getLocation());

    ExpectAndConsumeSemi(diag::err_expected_semi_declaration);
    if (R.isUsable())
      Stmts.push_back(R.release());
  }

  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    if (Tok.is(tok::annot_pragma_unused)) {
      HandlePragmaUnused();
      continue;
    }

    if (getLangOpts().MicrosoftExt && (Tok.is(tok::kw___if_exists) ||
        Tok.is(tok::kw___if_not_exists))) {
      ParseMicrosoftIfExistsStatement(Stmts);
      continue;
    }

    StmtResult R;
    if (Tok.isNot(tok::kw___extension__)) {
      R = ParseStatementOrDeclaration(Stmts, false);
    } else {
      // __extension__ can start declarations and it can also be a unary
      // operator for expressions.  Consume multiple __extension__ markers here
      // until we can determine which is which.
      // FIXME: This loses extension expressions in the AST!
      SourceLocation ExtLoc = ConsumeToken();
      while (Tok.is(tok::kw___extension__))
        ConsumeToken();

      ParsedAttributesWithRange attrs(AttrFactory);
      MaybeParseCXX11Attributes(attrs, 0, /*MightBeObjCMessageSend*/ true);

      // If this is the start of a declaration, parse it as such.
      if (isDeclarationStatement()) {
        // __extension__ silences extension warnings in the subdeclaration.
        // FIXME: Save the __extension__ on the decl as a node somehow?
        ExtensionRAIIObject O(Diags);

        SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
        DeclGroupPtrTy Res = ParseDeclaration(Stmts,
                                              Declarator::BlockContext, DeclEnd,
                                              attrs);
        R = Actions.ActOnDeclStmt(Res, DeclStart, DeclEnd);
      } else {
        // Otherwise this was a unary __extension__ marker.
        ExprResult Res(ParseExpressionWithLeadingExtension(ExtLoc));

        if (Res.isInvalid()) {
          SkipUntil(tok::semi);
          continue;
        }

        // FIXME: Use attributes?
        // Eat the semicolon at the end of stmt and convert the expr into a
        // statement.
        ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
        R = Actions.ActOnExprStmt(Res);
      }
    }

    if (R.isUsable())
      Stmts.push_back(R.release());
  }

  SourceLocation CloseLoc = Tok.getLocation();

  // We broke out of the while loop because we found a '}' or EOF.
  if (!T.consumeClose())
    // Recover by creating a compound statement with what we parsed so far,
    // instead of dropping everything and returning StmtError();
    CloseLoc = T.getCloseLocation();

  return Actions.ActOnCompoundStmt(T.getOpenLocation(), CloseLoc,
                                   Stmts, isStmtExpr);
}

/// ParseParenExprOrCondition:
/// [C  ]     '(' expression ')'
/// [C++]     '(' condition ')'       [not allowed if OnlyAllowCondition=true]
///
/// This function parses and performs error recovery on the specified condition
/// or expression (depending on whether we're in C++ or C mode).  This function
/// goes out of its way to recover well.  It returns true if there was a parser
/// error (the right paren couldn't be found), which indicates that the caller
/// should try to recover harder.  It returns false if the condition is
/// successfully parsed.  Note that a successful parse can still have semantic
/// errors in the condition.
bool Parser::ParseParenExprOrCondition(ExprResult &ExprResult,
                                       Decl *&DeclResult,
                                       SourceLocation Loc,
                                       bool ConvertToBoolean) {
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  if (getLangOpts().CPlusPlus)
    ParseCXXCondition(ExprResult, DeclResult, Loc, ConvertToBoolean);
  else {
    ExprResult = ParseExpression();
    DeclResult = 0;

    // If required, convert to a boolean value.
    if (!ExprResult.isInvalid() && ConvertToBoolean)
      ExprResult
        = Actions.ActOnBooleanCondition(getCurScope(), Loc, ExprResult.get());
  }

  // If the parser was confused by the condition and we don't have a ')', try to
  // recover by skipping ahead to a semi and bailing out.  If condexp is
  // semantically invalid but we have well formed code, keep going.
  if (ExprResult.isInvalid() && !DeclResult && Tok.isNot(tok::r_paren)) {
    SkipUntil(tok::semi);
    // Skipping may have stopped if it found the containing ')'.  If so, we can
    // continue parsing the if statement.
    if (Tok.isNot(tok::r_paren))
      return true;
  }

  // Otherwise the condition is valid or the rparen is present.
  T.consumeClose();

  // Check for extraneous ')'s to catch things like "if (foo())) {".  We know
  // that all callers are looking for a statement after the condition, so ")"
  // isn't valid.
  while (Tok.is(tok::r_paren)) {
    Diag(Tok, diag::err_extraneous_rparen_in_condition)
      << FixItHint::CreateRemoval(Tok.getLocation());
    ConsumeParen();
  }

  return false;
}


/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
/// [C++]   'if' '(' condition ')' statement
/// [C++]   'if' '(' condition ')' statement 'else' statement
///
StmtResult Parser::ParseIfStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_if) && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "if";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.4p3 - In C99, the if statement is a block.  This is not
  // the case for C90.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  ParseScope IfScope(this, Scope::DeclScope | Scope::ControlScope, C99orCXX);

  // Parse the condition.
  ExprResult CondExp;
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(CondExp, CondVar, IfLoc, true))
    return StmtError();

  FullExprArg FullCondExp(Actions.MakeFullExpr(CondExp.get(), IfLoc));

  // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.4p1:
  // The substatement in a selection-statement (each substatement, in the else
  // form of the if statement) implicitly defines a local scope.
  //
  // For C++ we create a scope for the condition and a new scope for
  // substatements because:
  // -When the 'then' scope exits, we want the condition declaration to still be
  //    active for the 'else' scope too.
  // -Sema will detect name clashes by considering declarations of a
  //    'ControlScope' as part of its direct subscope.
  // -If we wanted the condition and substatement to be in the same scope, we
  //    would have to notify ParseStatement not to create a new scope. It's
  //    simpler to let it create a new scope.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the 'then' stmt.
  SourceLocation ThenStmtLoc = Tok.getLocation();

  SourceLocation InnerStatementTrailingElseLoc;
  StmtResult ThenStmt(ParseStatement(&InnerStatementTrailingElseLoc));

  // Pop the 'if' scope if needed.
  InnerScope.Exit();

  // If it has an else, parse it.
  SourceLocation ElseLoc;
  SourceLocation ElseStmtLoc;
  StmtResult ElseStmt;

  if (Tok.is(tok::kw_else)) {
    if (TrailingElseLoc)
      *TrailingElseLoc = Tok.getLocation();

    ElseLoc = ConsumeToken();
    ElseStmtLoc = Tok.getLocation();

    // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
    // there is no compound stmt.  C90 does not have this clause.  We only do
    // this if the body isn't a compound statement to avoid push/pop in common
    // cases.
    //
    // C++ 6.4p1:
    // The substatement in a selection-statement (each substatement, in the else
    // form of the if statement) implicitly defines a local scope.
    //
    ParseScope InnerScope(this, Scope::DeclScope,
                          C99orCXX && Tok.isNot(tok::l_brace));

    ElseStmt = ParseStatement();

    // Pop the 'else' scope if needed.
    InnerScope.Exit();
  } else if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteAfterIf(getCurScope());
    cutOffParsing();
    return StmtError();
  } else if (InnerStatementTrailingElseLoc.isValid()) {
    Diag(InnerStatementTrailingElseLoc, diag::warn_dangling_else);
  }

  IfScope.Exit();

  // If the then or else stmt is invalid and the other is valid (and present),
  // make turn the invalid one into a null stmt to avoid dropping the other
  // part.  If both are invalid, return error.
  if ((ThenStmt.isInvalid() && ElseStmt.isInvalid()) ||
      (ThenStmt.isInvalid() && ElseStmt.get() == 0) ||
      (ThenStmt.get() == 0  && ElseStmt.isInvalid())) {
    // Both invalid, or one is invalid and other is non-present: return error.
    return StmtError();
  }

  // Now if either are invalid, replace with a ';'.
  if (ThenStmt.isInvalid())
    ThenStmt = Actions.ActOnNullStmt(ThenStmtLoc);
  if (ElseStmt.isInvalid())
    ElseStmt = Actions.ActOnNullStmt(ElseStmtLoc);

  return Actions.ActOnIfStmt(IfLoc, FullCondExp, CondVar, ThenStmt.get(),
                             ElseLoc, ElseStmt.get());
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
/// [C++]   'switch' '(' condition ')' statement
StmtResult Parser::ParseSwitchStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_switch) && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "switch";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.4p3 - In C99, the switch statement is a block.  This is
  // not the case for C90.  Start the switch scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  unsigned ScopeFlags = Scope::BreakScope | Scope::SwitchScope;
  if (C99orCXX)
    ScopeFlags |= Scope::DeclScope | Scope::ControlScope;
  ParseScope SwitchScope(this, ScopeFlags);

  // Parse the condition.
  ExprResult Cond;
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(Cond, CondVar, SwitchLoc, false))
    return StmtError();

  StmtResult Switch
    = Actions.ActOnStartOfSwitchStmt(SwitchLoc, Cond.get(), CondVar);

  if (Switch.isInvalid()) {
    // Skip the switch body.
    // FIXME: This is not optimal recovery, but parsing the body is more
    // dangerous due to the presence of case and default statements, which
    // will have no place to connect back with the switch.
    if (Tok.is(tok::l_brace)) {
      ConsumeBrace();
      SkipUntil(tok::r_brace);
    } else
      SkipUntil(tok::semi);
    return Switch;
  }

  // C99 6.8.4p3 - In C99, the body of the switch statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.4p1:
  // The substatement in a selection-statement (each substatement, in the else
  // form of the if statement) implicitly defines a local scope.
  //
  // See comments in ParseIfStatement for why we create a scope for the
  // condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the scopes.
  InnerScope.Exit();
  SwitchScope.Exit();

  if (Body.isInvalid()) {
    // FIXME: Remove the case statement list from the Switch statement.

    // Put the synthesized null statement on the same line as the end of switch
    // condition.
    SourceLocation SynthesizedNullStmtLocation = Cond.get()->getLocEnd();
    Body = Actions.ActOnNullStmt(SynthesizedNullStmtLocation);
  }

  return Actions.ActOnFinishSwitchStmt(SwitchLoc, Switch.get(), Body.get());
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
/// [C++]   'while' '(' condition ')' statement
StmtResult Parser::ParseWhileStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_while) && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "while";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.5p5 - In C99, the while statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  unsigned ScopeFlags;
  if (C99orCXX)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
                 Scope::DeclScope  | Scope::ControlScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;
  ParseScope WhileScope(this, ScopeFlags);

  // Parse the condition.
  ExprResult Cond;
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(Cond, CondVar, WhileLoc, true))
    return StmtError();

  FullExprArg FullCond(Actions.MakeFullExpr(Cond.get(), WhileLoc));

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  // See comments in ParseIfStatement for why we create a scope for the
  // condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the body scope if needed.
  InnerScope.Exit();
  WhileScope.Exit();

  if ((Cond.isInvalid() && !CondVar) || Body.isInvalid())
    return StmtError();

  return Actions.ActOnWhileStmt(WhileLoc, FullCond, CondVar, Body.get());
}

/// ParseDoStatement
///       do-statement: [C99 6.8.5.2]
///         'do' statement 'while' '(' expression ')' ';'
/// Note: this lets the caller parse the end ';'.
StmtResult Parser::ParseDoStatement() {
  assert(Tok.is(tok::kw_do) && "Not a do stmt!");
  SourceLocation DoLoc = ConsumeToken();  // eat the 'do'.

  // C99 6.8.5p5 - In C99, the do statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  unsigned ScopeFlags;
  if (getLangOpts().C99)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope | Scope::DeclScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;

  ParseScope DoScope(this, ScopeFlags);

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause. We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        (getLangOpts().C99 || getLangOpts().CPlusPlus) &&
                        Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  if (Tok.isNot(tok::kw_while)) {
    if (!Body.isInvalid()) {
      Diag(Tok, diag::err_expected_while);
      Diag(DoLoc, diag::note_matching) << "do";
      SkipUntil(tok::semi, StopBeforeMatch);
    }
    return StmtError();
  }
  SourceLocation WhileLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "do/while";
    SkipUntil(tok::semi, StopBeforeMatch);
    return StmtError();
  }

  // Parse the parenthesized expression.
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  // A do-while expression is not a condition, so can't have attributes.
  DiagnoseAndSkipCXX11Attributes();

  ExprResult Cond = ParseExpression();
  T.consumeClose();
  DoScope.Exit();

  if (Cond.isInvalid() || Body.isInvalid())
    return StmtError();

  return Actions.ActOnDoStmt(DoLoc, Body.get(), WhileLoc, T.getOpenLocation(),
                             Cond.get(), T.getCloseLocation());
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
/// [C++]   'for' '(' for-init-statement condition[opt] ';' expression[opt] ')'
/// [C++]       statement
/// [C++0x] 'for' '(' for-range-declaration : for-range-initializer ) statement
/// [OBJC2] 'for' '(' declaration 'in' expr ')' statement
/// [OBJC2] 'for' '(' expr 'in' expr ')' statement
///
/// [C++] for-init-statement:
/// [C++]   expression-statement
/// [C++]   simple-declaration
///
/// [C++0x] for-range-declaration:
/// [C++0x]   attribute-specifier-seq[opt] type-specifier-seq declarator
/// [C++0x] for-range-initializer:
/// [C++0x]   expression
/// [C++0x]   braced-init-list            [TODO]
StmtResult Parser::ParseForStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_for) && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "for";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXXorObjC = getLangOpts().C99 || getLangOpts().CPlusPlus ||
    getLangOpts().ObjC1;

  // C99 6.8.5p5 - In C99, the for statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  // C++ 6.5.3p1:
  // Names declared in the for-init-statement are in the same declarative-region
  // as those declared in the condition.
  //
  unsigned ScopeFlags;
  if (C99orCXXorObjC)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
                 Scope::DeclScope  | Scope::ControlScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;

  ParseScope ForScope(this, ScopeFlags);

  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  ExprResult Value;

  bool ForEach = false, ForRange = false;
  StmtResult FirstPart;
  bool SecondPartIsInvalid = false;
  FullExprArg SecondPart(Actions);
  ExprResult Collection;
  ForRangeInit ForRangeInit;
  FullExprArg ThirdPart(Actions);
  Decl *SecondVar = 0;

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(),
                                     C99orCXXorObjC? Sema::PCC_ForInit
                                                   : Sema::PCC_Expression);
    cutOffParsing();
    return StmtError();
  }

  ParsedAttributesWithRange attrs(AttrFactory);
  MaybeParseCXX11Attributes(attrs);

  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (;
    ProhibitAttributes(attrs);
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isForInitDeclaration()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!C99orCXXorObjC)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);

    // In C++0x, "for (T NS:a" might not be a typo for ::
    bool MightBeForRangeStmt = getLangOpts().CPlusPlus;
    ColonProtectionRAIIObject ColonProtection(*this, MightBeForRangeStmt);

    SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
    StmtVector Stmts;
    DeclGroupPtrTy DG = ParseSimpleDeclaration(Stmts, Declarator::ForContext,
                                               DeclEnd, attrs, false,
                                               MightBeForRangeStmt ?
                                                 &ForRangeInit : 0);
    FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());

    if (ForRangeInit.ParsedForRangeDecl()) {
      Diag(ForRangeInit.ColonLoc, getLangOpts().CPlusPlus11 ?
           diag::warn_cxx98_compat_for_range : diag::ext_for_range);

      ForRange = true;
    } else if (Tok.is(tok::semi)) {  // for (int x = 4;
      ConsumeToken();
    } else if ((ForEach = isTokIdentifier_in())) {
      Actions.ActOnForEachDeclStmt(DG);
      // ObjC: for (id x in expr)
      ConsumeToken(); // consume 'in'

      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DG);
        cutOffParsing();
        return StmtError();
      }
      Collection = ParseExpression();
    } else {
      Diag(Tok, diag::err_expected_semi_for);
    }
  } else {
    ProhibitAttributes(attrs);
    Value = ParseExpression();

    ForEach = isTokIdentifier_in();

    // Turn the expression into a stmt.
    if (!Value.isInvalid()) {
      if (ForEach)
        FirstPart = Actions.ActOnForEachLValueExpr(Value.get());
      else
        FirstPart = Actions.ActOnExprStmt(Value);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (ForEach) {
      ConsumeToken(); // consume 'in'

      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DeclGroupPtrTy());
        cutOffParsing();
        return StmtError();
      }
      Collection = ParseExpression();
    } else if (getLangOpts().CPlusPlus11 && Tok.is(tok::colon) && FirstPart.get()) {
      // User tried to write the reasonable, but ill-formed, for-range-statement
      //   for (expr : expr) { ... }
      Diag(Tok, diag::err_for_range_expected_decl)
        << FirstPart.get()->getSourceRange();
      SkipUntil(tok::r_paren, StopBeforeMatch);
      SecondPartIsInvalid = true;
    } else {
      if (!Value.isInvalid()) {
        Diag(Tok, diag::err_expected_semi_for);
      } else {
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, StopAtSemi | StopBeforeMatch);
        if (Tok.is(tok::semi))
          ConsumeToken();
      }
    }
  }
  if (!ForEach && !ForRange) {
    assert(!SecondPart.get() && "Shouldn't have a second expression yet.");
    // Parse the second part of the for specifier.
    if (Tok.is(tok::semi)) {  // for (...;;
      // no second part.
    } else if (Tok.is(tok::r_paren)) {
      // missing both semicolons.
    } else {
      ExprResult Second;
      if (getLangOpts().CPlusPlus)
        ParseCXXCondition(Second, SecondVar, ForLoc, true);
      else {
        Second = ParseExpression();
        if (!Second.isInvalid())
          Second = Actions.ActOnBooleanCondition(getCurScope(), ForLoc,
                                                 Second.get());
      }
      SecondPartIsInvalid = Second.isInvalid();
      SecondPart = Actions.MakeFullExpr(Second.get(), ForLoc);
    }

    if (Tok.isNot(tok::semi)) {
      if (!SecondPartIsInvalid || SecondVar)
        Diag(Tok, diag::err_expected_semi_for);
      else
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, StopAtSemi | StopBeforeMatch);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    }

    // Parse the third part of the for specifier.
    if (Tok.isNot(tok::r_paren)) {   // for (...;...;)
      ExprResult Third = ParseExpression();
      // FIXME: The C++11 standard doesn't actually say that this is a
      // discarded-value expression, but it clearly should be.
      ThirdPart = Actions.MakeFullDiscardedValueExpr(Third.take());
    }
  }
  // Match the ')'.
  T.consumeClose();

  // We need to perform most of the semantic analysis for a C++0x for-range
  // statememt before parsing the body, in order to be able to deduce the type
  // of an auto-typed loop variable.
  StmtResult ForRangeStmt;
  StmtResult ForEachStmt;

  if (ForRange) {
    ForRangeStmt = Actions.ActOnCXXForRangeStmt(ForLoc, FirstPart.take(),
                                                ForRangeInit.ColonLoc,
                                                ForRangeInit.RangeExpr.get(),
                                                T.getCloseLocation(),
                                                Sema::BFRK_Build);


  // Similarly, we need to do the semantic analysis for a for-range
  // statement immediately in order to close over temporaries correctly.
  } else if (ForEach) {
    ForEachStmt = Actions.ActOnObjCForCollectionStmt(ForLoc,
                                                     FirstPart.take(),
                                                     Collection.take(),
                                                     T.getCloseLocation());
  }

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  // See comments in ParseIfStatement for why we create a scope for
  // for-init-statement/condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXXorObjC && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the body scope if needed.
  InnerScope.Exit();

  // Leave the for-scope.
  ForScope.Exit();

  if (Body.isInvalid())
    return StmtError();

  if (ForEach)
   return Actions.FinishObjCForCollectionStmt(ForEachStmt.take(),
                                              Body.take());

  if (ForRange)
    return Actions.FinishCXXForRangeStmt(ForRangeStmt.take(), Body.take());

  return Actions.ActOnForStmt(ForLoc, T.getOpenLocation(), FirstPart.take(),
                              SecondPart, SecondVar, ThirdPart,
                              T.getCloseLocation(), Body.take());
}

/// ParseGotoStatement
///       jump-statement:
///         'goto' identifier ';'
/// [GNU]   'goto' '*' expression ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseGotoStatement() {
  assert(Tok.is(tok::kw_goto) && "Not a goto stmt!");
  SourceLocation GotoLoc = ConsumeToken();  // eat the 'goto'.

  StmtResult Res;
  if (Tok.is(tok::identifier)) {
    LabelDecl *LD = Actions.LookupOrCreateLabel(Tok.getIdentifierInfo(),
                                                Tok.getLocation());
    Res = Actions.ActOnGotoStmt(GotoLoc, Tok.getLocation(), LD);
    ConsumeToken();
  } else if (Tok.is(tok::star)) {
    // GNU indirect goto extension.
    Diag(Tok, diag::ext_gnu_indirect_goto);
    SourceLocation StarLoc = ConsumeToken();
    ExprResult R(ParseExpression());
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, StopBeforeMatch);
      return StmtError();
    }
    Res = Actions.ActOnIndirectGotoStmt(GotoLoc, StarLoc, R.take());
  } else {
    Diag(Tok, diag::err_expected_ident);
    return StmtError();
  }

  return Res;
}

/// ParseContinueStatement
///       jump-statement:
///         'continue' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseContinueStatement() {
  SourceLocation ContinueLoc = ConsumeToken();  // eat the 'continue'.
  return Actions.ActOnContinueStmt(ContinueLoc, getCurScope());
}

/// ParseBreakStatement
///       jump-statement:
///         'break' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseBreakStatement() {
  SourceLocation BreakLoc = ConsumeToken();  // eat the 'break'.
  return Actions.ActOnBreakStmt(BreakLoc, getCurScope());
}

/// ParseReturnStatement
///       jump-statement:
///         'return' expression[opt] ';'
StmtResult Parser::ParseReturnStatement() {
  assert(Tok.is(tok::kw_return) && "Not a return stmt!");
  SourceLocation ReturnLoc = ConsumeToken();  // eat the 'return'.

  ExprResult R;
  if (Tok.isNot(tok::semi)) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteReturn(getCurScope());
      cutOffParsing();
      return StmtError();
    }

    if (Tok.is(tok::l_brace) && getLangOpts().CPlusPlus) {
      R = ParseInitializer();
      if (R.isUsable())
        Diag(R.get()->getLocStart(), getLangOpts().CPlusPlus11 ?
             diag::warn_cxx98_compat_generalized_initializer_lists :
             diag::ext_generalized_initializer_lists)
          << R.get()->getSourceRange();
    } else
        R = ParseExpression();
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, StopBeforeMatch);
      return StmtError();
    }
  }
  return Actions.ActOnReturnStmt(ReturnLoc, R.take());
}

namespace {
  class ClangAsmParserCallback : public llvm::MCAsmParserSemaCallback {
    Parser &TheParser;
    SourceLocation AsmLoc;
    StringRef AsmString;

    /// The tokens we streamed into AsmString and handed off to MC.
    ArrayRef<Token> AsmToks;

    /// The offset of each token in AsmToks within AsmString.
    ArrayRef<unsigned> AsmTokOffsets;

  public:
    ClangAsmParserCallback(Parser &P, SourceLocation Loc,
                           StringRef AsmString,
                           ArrayRef<Token> Toks,
                           ArrayRef<unsigned> Offsets)
      : TheParser(P), AsmLoc(Loc), AsmString(AsmString),
        AsmToks(Toks), AsmTokOffsets(Offsets) {
      assert(AsmToks.size() == AsmTokOffsets.size());
    }

    void *LookupInlineAsmIdentifier(StringRef &LineBuf,
                                    InlineAsmIdentifierInfo &Info,
                                    bool IsUnevaluatedContext) {
      // Collect the desired tokens.
      SmallVector<Token, 16> LineToks;
      const Token *FirstOrigToken = 0;
      findTokensForString(LineBuf, LineToks, FirstOrigToken);

      unsigned NumConsumedToks;
      ExprResult Result =
        TheParser.ParseMSAsmIdentifier(LineToks, NumConsumedToks, &Info,
                                       IsUnevaluatedContext);

      // If we consumed the entire line, tell MC that.
      // Also do this if we consumed nothing as a way of reporting failure.
      if (NumConsumedToks == 0 || NumConsumedToks == LineToks.size()) {
        // By not modifying LineBuf, we're implicitly consuming it all.

      // Otherwise, consume up to the original tokens.
      } else {
        assert(FirstOrigToken && "not using original tokens?");

        // Since we're using original tokens, apply that offset.
        assert(FirstOrigToken[NumConsumedToks].getLocation()
                  == LineToks[NumConsumedToks].getLocation());
        unsigned FirstIndex = FirstOrigToken - AsmToks.begin();
        unsigned LastIndex = FirstIndex + NumConsumedToks - 1;

        // The total length we've consumed is the relative offset
        // of the last token we consumed plus its length.
        unsigned TotalOffset = (AsmTokOffsets[LastIndex]
                                + AsmToks[LastIndex].getLength()
                                - AsmTokOffsets[FirstIndex]);
        LineBuf = LineBuf.substr(0, TotalOffset);
      }

      // Initialize the "decl" with the lookup result.
      Info.OpDecl = static_cast<void*>(Result.take());
      return Info.OpDecl;
    }

    bool LookupInlineAsmField(StringRef Base, StringRef Member,
                              unsigned &Offset) {
      return TheParser.getActions().LookupInlineAsmField(Base, Member,
                                                         Offset, AsmLoc);
    }

    static void DiagHandlerCallback(const llvm::SMDiagnostic &D,
                                    void *Context) {
      ((ClangAsmParserCallback*) Context)->handleDiagnostic(D);
    }

  private:
    /// Collect the appropriate tokens for the given string.
    void findTokensForString(StringRef Str, SmallVectorImpl<Token> &TempToks,
                             const Token *&FirstOrigToken) const {
      // For now, assert that the string we're working with is a substring
      // of what we gave to MC.  This lets us use the original tokens.
      assert(!std::less<const char*>()(Str.begin(), AsmString.begin()) &&
             !std::less<const char*>()(AsmString.end(), Str.end()));

      // Try to find a token whose offset matches the first token.
      unsigned FirstCharOffset = Str.begin() - AsmString.begin();
      const unsigned *FirstTokOffset
        = std::lower_bound(AsmTokOffsets.begin(), AsmTokOffsets.end(),
                           FirstCharOffset);

      // For now, assert that the start of the string exactly
      // corresponds to the start of a token.
      assert(*FirstTokOffset == FirstCharOffset);

      // Use all the original tokens for this line.  (We assume the
      // end of the line corresponds cleanly to a token break.)
      unsigned FirstTokIndex = FirstTokOffset - AsmTokOffsets.begin();
      FirstOrigToken = &AsmToks[FirstTokIndex];
      unsigned LastCharOffset = Str.end() - AsmString.begin();
      for (unsigned i = FirstTokIndex, e = AsmTokOffsets.size(); i != e; ++i) {
        if (AsmTokOffsets[i] >= LastCharOffset) break;
        TempToks.push_back(AsmToks[i]);
      }
    }

    void handleDiagnostic(const llvm::SMDiagnostic &D) {
      // Compute an offset into the inline asm buffer.
      // FIXME: This isn't right if .macro is involved (but hopefully, no
      // real-world code does that).
      const llvm::SourceMgr &LSM = *D.getSourceMgr();
      const llvm::MemoryBuffer *LBuf =
        LSM.getMemoryBuffer(LSM.FindBufferContainingLoc(D.getLoc()));
      unsigned Offset = D.getLoc().getPointer() - LBuf->getBufferStart();

      // Figure out which token that offset points into.
      const unsigned *TokOffsetPtr =
        std::lower_bound(AsmTokOffsets.begin(), AsmTokOffsets.end(), Offset);
      unsigned TokIndex = TokOffsetPtr - AsmTokOffsets.begin();
      unsigned TokOffset = *TokOffsetPtr;

      // If we come up with an answer which seems sane, use it; otherwise,
      // just point at the __asm keyword.
      // FIXME: Assert the answer is sane once we handle .macro correctly.
      SourceLocation Loc = AsmLoc;
      if (TokIndex < AsmToks.size()) {
        const Token &Tok = AsmToks[TokIndex];
        Loc = Tok.getLocation();
        Loc = Loc.getLocWithOffset(Offset - TokOffset);
      }
      TheParser.Diag(Loc, diag::err_inline_ms_asm_parsing)
        << D.getMessage();
    }
  };
}

/// Parse an identifier in an MS-style inline assembly block.
///
/// \param CastInfo - a void* so that we don't have to teach Parser.h
///   about the actual type.
ExprResult Parser::ParseMSAsmIdentifier(llvm::SmallVectorImpl<Token> &LineToks,
                                        unsigned &NumLineToksConsumed,
                                        void *CastInfo,
                                        bool IsUnevaluatedContext) {
  llvm::InlineAsmIdentifierInfo &Info =
    *(llvm::InlineAsmIdentifierInfo *) CastInfo;

  // Push a fake token on the end so that we don't overrun the token
  // stream.  We use ';' because it expression-parsing should never
  // overrun it.
  const tok::TokenKind EndOfStream = tok::semi;
  Token EndOfStreamTok;
  EndOfStreamTok.startToken();
  EndOfStreamTok.setKind(EndOfStream);
  LineToks.push_back(EndOfStreamTok);

  // Also copy the current token over.
  LineToks.push_back(Tok);

  PP.EnterTokenStream(LineToks.begin(),
                      LineToks.size(),
                      /*disable macros*/ true,
                      /*owns tokens*/ false);

  // Clear the current token and advance to the first token in LineToks.
  ConsumeAnyToken();

  // Parse an optional scope-specifier if we're in C++.
  CXXScopeSpec SS;
  if (getLangOpts().CPlusPlus) {
    ParseOptionalCXXScopeSpecifier(SS, ParsedType(), /*EnteringContext=*/false);
  }

  // Require an identifier here.
  SourceLocation TemplateKWLoc;
  UnqualifiedId Id;
  bool Invalid = ParseUnqualifiedId(SS,
                                    /*EnteringContext=*/false,
                                    /*AllowDestructorName=*/false,
                                    /*AllowConstructorName=*/false,
                                    /*ObjectType=*/ ParsedType(),
                                    TemplateKWLoc,
                                    Id);

  // If we've run into the poison token we inserted before, or there
  // was a parsing error, then claim the entire line.
  if (Invalid || Tok.is(EndOfStream)) {
    NumLineToksConsumed = LineToks.size() - 2;

    // Otherwise, claim up to the start of the next token.
  } else {
    // Figure out how many tokens we are into LineToks.
    unsigned LineIndex = 0;
    while (LineToks[LineIndex].getLocation() != Tok.getLocation()) {
      LineIndex++;
      assert(LineIndex < LineToks.size() - 2); // we added two extra tokens
    }

    NumLineToksConsumed = LineIndex;
  }
      
  // Finally, restore the old parsing state by consuming all the
  // tokens we staged before, implicitly killing off the
  // token-lexer we pushed.
  for (unsigned n = LineToks.size() - 2 - NumLineToksConsumed; n != 0; --n) {
    ConsumeAnyToken();
  }
  ConsumeToken(EndOfStream);

  // Leave LineToks in its original state.
  LineToks.pop_back();
  LineToks.pop_back();

  // Perform the lookup.
  return Actions.LookupInlineAsmIdentifier(SS, TemplateKWLoc, Id, Info,
                                           IsUnevaluatedContext);
}

/// Turn a sequence of our tokens back into a string that we can hand
/// to the MC asm parser.
static bool buildMSAsmString(Preprocessor &PP,
                             SourceLocation AsmLoc,
                             ArrayRef<Token> AsmToks,
                             SmallVectorImpl<unsigned> &TokOffsets,
                             SmallString<512> &Asm) {
  assert (!AsmToks.empty() && "Didn't expect an empty AsmToks!");

  // Is this the start of a new assembly statement?
  bool isNewStatement = true;

  for (unsigned i = 0, e = AsmToks.size(); i < e; ++i) {
    const Token &Tok = AsmToks[i];

    // Start each new statement with a newline and a tab.
    if (!isNewStatement &&
        (Tok.is(tok::kw_asm) || Tok.isAtStartOfLine())) {
      Asm += "\n\t";
      isNewStatement = true;
    }

    // Preserve the existence of leading whitespace except at the
    // start of a statement.
    if (!isNewStatement && Tok.hasLeadingSpace())
      Asm += ' ';

    // Remember the offset of this token.
    TokOffsets.push_back(Asm.size());

    // Don't actually write '__asm' into the assembly stream.
    if (Tok.is(tok::kw_asm)) {
      // Complain about __asm at the end of the stream.
      if (i + 1 == e) {
        PP.Diag(AsmLoc, diag::err_asm_empty);
        return true;
      }

      continue;
    }

    // Append the spelling of the token.
    SmallString<32> SpellingBuffer;
    bool SpellingInvalid = false;
    Asm += PP.getSpelling(Tok, SpellingBuffer, &SpellingInvalid);
    assert(!SpellingInvalid && "spelling was invalid after correct parse?");

    // We are no longer at the start of a statement.
    isNewStatement = false;
  }

  // Ensure that the buffer is null-terminated.
  Asm.push_back('\0');
  Asm.pop_back();

  assert(TokOffsets.size() == AsmToks.size());
  return false;
}

/// ParseMicrosoftAsmStatement. When -fms-extensions/-fasm-blocks is enabled,
/// this routine is called to collect the tokens for an MS asm statement.
///
/// [MS]  ms-asm-statement:
///         ms-asm-block
///         ms-asm-block ms-asm-statement
///
/// [MS]  ms-asm-block:
///         '__asm' ms-asm-line '\n'
///         '__asm' '{' ms-asm-instruction-block[opt] '}' ';'[opt]
///
/// [MS]  ms-asm-instruction-block
///         ms-asm-line
///         ms-asm-line '\n' ms-asm-instruction-block
///
StmtResult Parser::ParseMicrosoftAsmStatement(SourceLocation AsmLoc) {
  SourceManager &SrcMgr = PP.getSourceManager();
  SourceLocation EndLoc = AsmLoc;
  SmallVector<Token, 4> AsmToks;

  bool InBraces = false;
  unsigned short savedBraceCount = 0;
  bool InAsmComment = false;
  FileID FID;
  unsigned LineNo = 0;
  unsigned NumTokensRead = 0;
  SourceLocation LBraceLoc;

  if (Tok.is(tok::l_brace)) {
    // Braced inline asm: consume the opening brace.
    InBraces = true;
    savedBraceCount = BraceCount;
    EndLoc = LBraceLoc = ConsumeBrace();
    ++NumTokensRead;
  } else {
    // Single-line inline asm; compute which line it is on.
    std::pair<FileID, unsigned> ExpAsmLoc =
      SrcMgr.getDecomposedExpansionLoc(EndLoc);
    FID = ExpAsmLoc.first;
    LineNo = SrcMgr.getLineNumber(FID, ExpAsmLoc.second);
  }

  SourceLocation TokLoc = Tok.getLocation();
  do {
    // If we hit EOF, we're done, period.
    if (Tok.is(tok::eof))
      break;

    if (!InAsmComment && Tok.is(tok::semi)) {
      // A semicolon in an asm is the start of a comment.
      InAsmComment = true;
      if (InBraces) {
        // Compute which line the comment is on.
        std::pair<FileID, unsigned> ExpSemiLoc =
          SrcMgr.getDecomposedExpansionLoc(TokLoc);
        FID = ExpSemiLoc.first;
        LineNo = SrcMgr.getLineNumber(FID, ExpSemiLoc.second);
      }
    } else if (!InBraces || InAsmComment) {
      // If end-of-line is significant, check whether this token is on a
      // new line.
      std::pair<FileID, unsigned> ExpLoc =
        SrcMgr.getDecomposedExpansionLoc(TokLoc);
      if (ExpLoc.first != FID ||
          SrcMgr.getLineNumber(ExpLoc.first, ExpLoc.second) != LineNo) {
        // If this is a single-line __asm, we're done.
        if (!InBraces)
          break;
        // We're no longer in a comment.
        InAsmComment = false;
      } else if (!InAsmComment && Tok.is(tok::r_brace)) {
        // Single-line asm always ends when a closing brace is seen.
        // FIXME: This is compatible with Apple gcc's -fasm-blocks; what
        // does MSVC do here?
        break;
      }
    }
    if (!InAsmComment && InBraces && Tok.is(tok::r_brace) &&
        BraceCount == (savedBraceCount + 1)) {
      // Consume the closing brace, and finish
      EndLoc = ConsumeBrace();
      break;
    }

    // Consume the next token; make sure we don't modify the brace count etc.
    // if we are in a comment.
    EndLoc = TokLoc;
    if (InAsmComment)
      PP.Lex(Tok);
    else {
      AsmToks.push_back(Tok);
      ConsumeAnyToken();
    }
    TokLoc = Tok.getLocation();
    ++NumTokensRead;
  } while (1);

  if (InBraces && BraceCount != savedBraceCount) {
    // __asm without closing brace (this can happen at EOF).
    Diag(Tok, diag::err_expected_rbrace);
    Diag(LBraceLoc, diag::note_matching) << "{";
    return StmtError();
  } else if (NumTokensRead == 0) {
    // Empty __asm.
    Diag(Tok, diag::err_expected_lbrace);
    return StmtError();
  }

  // Okay, prepare to use MC to parse the assembly.
  SmallVector<StringRef, 4> ConstraintRefs;
  SmallVector<Expr*, 4> Exprs;
  SmallVector<StringRef, 4> ClobberRefs;

  // We need an actual supported target.
  llvm::Triple TheTriple = Actions.Context.getTargetInfo().getTriple();
  llvm::Triple::ArchType ArchTy = TheTriple.getArch();
  const std::string &TT = TheTriple.getTriple();
  const llvm::Target *TheTarget = 0;
  bool UnsupportedArch = (ArchTy != llvm::Triple::x86 &&
                          ArchTy != llvm::Triple::x86_64);
  if (UnsupportedArch) {
    Diag(AsmLoc, diag::err_msasm_unsupported_arch) << TheTriple.getArchName();
  } else {
    std::string Error;
    TheTarget = llvm::TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget)
      Diag(AsmLoc, diag::err_msasm_unable_to_create_target) << Error;
  }

  // If we don't support assembly, or the assembly is empty, we don't
  // need to instantiate the AsmParser, etc.
  if (!TheTarget || AsmToks.empty()) {
    return Actions.ActOnMSAsmStmt(AsmLoc, LBraceLoc, AsmToks, StringRef(),
                                  /*NumOutputs*/ 0, /*NumInputs*/ 0,
                                  ConstraintRefs, ClobberRefs, Exprs, EndLoc);
  }

  // Expand the tokens into a string buffer.
  SmallString<512> AsmString;
  SmallVector<unsigned, 8> TokOffsets;
  if (buildMSAsmString(PP, AsmLoc, AsmToks, TokOffsets, AsmString))
    return StmtError();

  OwningPtr<llvm::MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  OwningPtr<llvm::MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TT));
  // Get the instruction descriptor.
  const llvm::MCInstrInfo *MII = TheTarget->createMCInstrInfo(); 
  OwningPtr<llvm::MCObjectFileInfo> MOFI(new llvm::MCObjectFileInfo());
  OwningPtr<llvm::MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(TT, "", ""));

  llvm::SourceMgr TempSrcMgr;
  llvm::MCContext Ctx(MAI.get(), MRI.get(), MOFI.get(), &TempSrcMgr);
  llvm::MemoryBuffer *Buffer =
    llvm::MemoryBuffer::getMemBuffer(AsmString, "<MS inline asm>");

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  TempSrcMgr.AddNewSourceBuffer(Buffer, llvm::SMLoc());

  OwningPtr<llvm::MCStreamer> Str(createNullStreamer(Ctx));
  OwningPtr<llvm::MCAsmParser>
    Parser(createMCAsmParser(TempSrcMgr, Ctx, *Str.get(), *MAI));
  OwningPtr<llvm::MCTargetAsmParser>
    TargetParser(TheTarget->createMCAsmParser(*STI, *Parser, *MII));

  llvm::MCInstPrinter *IP =
    TheTarget->createMCInstPrinter(1, *MAI, *MII, *MRI, *STI);

  // Change to the Intel dialect.
  Parser->setAssemblerDialect(1);
  Parser->setTargetParser(*TargetParser.get());
  Parser->setParsingInlineAsm(true);
  TargetParser->setParsingInlineAsm(true);

  ClangAsmParserCallback Callback(*this, AsmLoc, AsmString,
                                  AsmToks, TokOffsets);
  TargetParser->setSemaCallback(&Callback);
  TempSrcMgr.setDiagHandler(ClangAsmParserCallback::DiagHandlerCallback,
                            &Callback);

  unsigned NumOutputs;
  unsigned NumInputs;
  std::string AsmStringIR;
  SmallVector<std::pair<void *, bool>, 4> OpExprs;
  SmallVector<std::string, 4> Constraints;
  SmallVector<std::string, 4> Clobbers;
  if (Parser->parseMSInlineAsm(AsmLoc.getPtrEncoding(), AsmStringIR,
                               NumOutputs, NumInputs, OpExprs, Constraints,
                               Clobbers, MII, IP, Callback))
    return StmtError();

  // Build the vector of clobber StringRefs.
  unsigned NumClobbers = Clobbers.size();
  ClobberRefs.resize(NumClobbers);
  for (unsigned i = 0; i != NumClobbers; ++i)
    ClobberRefs[i] = StringRef(Clobbers[i]);

  // Recast the void pointers and build the vector of constraint StringRefs.
  unsigned NumExprs = NumOutputs + NumInputs;
  ConstraintRefs.resize(NumExprs);
  Exprs.resize(NumExprs);
  for (unsigned i = 0, e = NumExprs; i != e; ++i) {
    Expr *OpExpr = static_cast<Expr *>(OpExprs[i].first);
    if (!OpExpr)
      return StmtError();

    // Need address of variable.
    if (OpExprs[i].second)
      OpExpr = Actions.BuildUnaryOp(getCurScope(), AsmLoc, UO_AddrOf, OpExpr)
        .take();

    ConstraintRefs[i] = StringRef(Constraints[i]);
    Exprs[i] = OpExpr;
  }

  // FIXME: We should be passing source locations for better diagnostics.
  return Actions.ActOnMSAsmStmt(AsmLoc, LBraceLoc, AsmToks, AsmStringIR,
                                NumOutputs, NumInputs,
                                ConstraintRefs, ClobberRefs, Exprs, EndLoc);
}

/// ParseAsmStatement - Parse a GNU extended asm statement.
///       asm-statement:
///         gnu-asm-statement
///         ms-asm-statement
///
/// [GNU] gnu-asm-statement:
///         'asm' type-qualifier[opt] '(' asm-argument ')' ';'
///
/// [GNU] asm-argument:
///         asm-string-literal
///         asm-string-literal ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///                 ':' asm-clobbers
///
/// [GNU] asm-clobbers:
///         asm-string-literal
///         asm-clobbers ',' asm-string-literal
///
StmtResult Parser::ParseAsmStatement(bool &msAsm) {
  assert(Tok.is(tok::kw_asm) && "Not an asm stmt");
  SourceLocation AsmLoc = ConsumeToken();

  if (getLangOpts().AsmBlocks && Tok.isNot(tok::l_paren) &&
      !isTypeQualifier()) {
    msAsm = true;
    return ParseMicrosoftAsmStatement(AsmLoc);
  }
  DeclSpec DS(AttrFactory);
  SourceLocation Loc = Tok.getLocation();
  ParseTypeQualifierListOpt(DS, true, false);

  // GNU asms accept, but warn, about type-qualifiers other than volatile.
  if (DS.getTypeQualifiers() & DeclSpec::TQ_const)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "const";
  if (DS.getTypeQualifiers() & DeclSpec::TQ_restrict)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "restrict";
  // FIXME: Once GCC supports _Atomic, check whether it permits it here.
  if (DS.getTypeQualifiers() & DeclSpec::TQ_atomic)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "_Atomic";

  // Remember if this was a volatile asm.
  bool isVolatile = DS.getTypeQualifiers() & DeclSpec::TQ_volatile;
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "asm";
    SkipUntil(tok::r_paren, StopAtSemi);
    return StmtError();
  }
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  ExprResult AsmString(ParseAsmStringLiteral());
  if (AsmString.isInvalid()) {
    // Consume up to and including the closing paren.
    T.skipToEnd();
    return StmtError();
  }

  SmallVector<IdentifierInfo *, 4> Names;
  ExprVector Constraints;
  ExprVector Exprs;
  ExprVector Clobbers;

  if (Tok.is(tok::r_paren)) {
    // We have a simple asm expression like 'asm("foo")'.
    T.consumeClose();
    return Actions.ActOnGCCAsmStmt(AsmLoc, /*isSimple*/ true, isVolatile,
                                   /*NumOutputs*/ 0, /*NumInputs*/ 0, 0,
                                   Constraints, Exprs, AsmString.take(),
                                   Clobbers, T.getCloseLocation());
  }

  // Parse Outputs, if present.
  bool AteExtraColon = false;
  if (Tok.is(tok::colon) || Tok.is(tok::coloncolon)) {
    // In C++ mode, parse "::" like ": :".
    AteExtraColon = Tok.is(tok::coloncolon);
    ConsumeToken();

    if (!AteExtraColon &&
        ParseAsmOperandsOpt(Names, Constraints, Exprs))
      return StmtError();
  }

  unsigned NumOutputs = Names.size();

  // Parse Inputs, if present.
  if (AteExtraColon ||
      Tok.is(tok::colon) || Tok.is(tok::coloncolon)) {
    // In C++ mode, parse "::" like ": :".
    if (AteExtraColon)
      AteExtraColon = false;
    else {
      AteExtraColon = Tok.is(tok::coloncolon);
      ConsumeToken();
    }

    if (!AteExtraColon &&
        ParseAsmOperandsOpt(Names, Constraints, Exprs))
      return StmtError();
  }

  assert(Names.size() == Constraints.size() &&
         Constraints.size() == Exprs.size() &&
         "Input operand size mismatch!");

  unsigned NumInputs = Names.size() - NumOutputs;

  // Parse the clobbers, if present.
  if (AteExtraColon || Tok.is(tok::colon)) {
    if (!AteExtraColon)
      ConsumeToken();

    // Parse the asm-string list for clobbers if present.
    if (Tok.isNot(tok::r_paren)) {
      while (1) {
        ExprResult Clobber(ParseAsmStringLiteral());

        if (Clobber.isInvalid())
          break;

        Clobbers.push_back(Clobber.release());

        if (Tok.isNot(tok::comma)) break;
        ConsumeToken();
      }
    }
  }

  T.consumeClose();
  return Actions.ActOnGCCAsmStmt(AsmLoc, false, isVolatile, NumOutputs,
                                 NumInputs, Names.data(), Constraints, Exprs,
                                 AsmString.take(), Clobbers,
                                 T.getCloseLocation());
}

/// ParseAsmOperands - Parse the asm-operands production as used by
/// asm-statement, assuming the leading ':' token was eaten.
///
/// [GNU] asm-operands:
///         asm-operand
///         asm-operands ',' asm-operand
///
/// [GNU] asm-operand:
///         asm-string-literal '(' expression ')'
///         '[' identifier ']' asm-string-literal '(' expression ')'
///
//
// FIXME: Avoid unnecessary std::string trashing.
bool Parser::ParseAsmOperandsOpt(SmallVectorImpl<IdentifierInfo *> &Names,
                                 SmallVectorImpl<Expr *> &Constraints,
                                 SmallVectorImpl<Expr *> &Exprs) {
  // 'asm-operands' isn't present?
  if (!isTokenStringLiteral() && Tok.isNot(tok::l_square))
    return false;

  while (1) {
    // Read the [id] if present.
    if (Tok.is(tok::l_square)) {
      BalancedDelimiterTracker T(*this, tok::l_square);
      T.consumeOpen();

      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::r_paren, StopAtSemi);
        return true;
      }

      IdentifierInfo *II = Tok.getIdentifierInfo();
      ConsumeToken();

      Names.push_back(II);
      T.consumeClose();
    } else
      Names.push_back(0);

    ExprResult Constraint(ParseAsmStringLiteral());
    if (Constraint.isInvalid()) {
        SkipUntil(tok::r_paren, StopAtSemi);
        return true;
    }
    Constraints.push_back(Constraint.release());

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_expected_lparen_after) << "asm operand";
      SkipUntil(tok::r_paren, StopAtSemi);
      return true;
    }

    // Read the parenthesized expression.
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();
    ExprResult Res(ParseExpression());
    T.consumeClose();
    if (Res.isInvalid()) {
      SkipUntil(tok::r_paren, StopAtSemi);
      return true;
    }
    Exprs.push_back(Res.release());
    // Eat the comma and continue parsing if it exists.
    if (Tok.isNot(tok::comma)) return false;
    ConsumeToken();
  }
}

Decl *Parser::ParseFunctionStatementBody(Decl *Decl, ParseScope &BodyScope) {
  assert(Tok.is(tok::l_brace));
  SourceLocation LBraceLoc = Tok.getLocation();

  if (SkipFunctionBodies && (!Decl || Actions.canSkipFunctionBody(Decl)) &&
      trySkippingFunctionBody()) {
    BodyScope.Exit();
    return Actions.ActOnSkippedFunctionBody(Decl);
  }

  PrettyDeclStackTraceEntry CrashInfo(Actions, Decl, LBraceLoc,
                                      "parsing function body");

  // Do not enter a scope for the brace, as the arguments are in the same scope
  // (the function body) as the body itself.  Instead, just read the statement
  // list and put it into a CompoundStmt for safe keeping.
  StmtResult FnBody(ParseCompoundStatementBody());

  // If the function body could not be parsed, make a bogus compoundstmt.
  if (FnBody.isInvalid()) {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc, None, false);
  }

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.take());
}

/// ParseFunctionTryBlock - Parse a C++ function-try-block.
///
///       function-try-block:
///         'try' ctor-initializer[opt] compound-statement handler-seq
///
Decl *Parser::ParseFunctionTryBlock(Decl *Decl, ParseScope &BodyScope) {
  assert(Tok.is(tok::kw_try) && "Expected 'try'");
  SourceLocation TryLoc = ConsumeToken();

  PrettyDeclStackTraceEntry CrashInfo(Actions, Decl, TryLoc,
                                      "parsing function try block");

  // Constructor initializer list?
  if (Tok.is(tok::colon))
    ParseConstructorInitializer(Decl);
  else
    Actions.ActOnDefaultCtorInitializers(Decl);

  if (SkipFunctionBodies && Actions.canSkipFunctionBody(Decl) &&
      trySkippingFunctionBody()) {
    BodyScope.Exit();
    return Actions.ActOnSkippedFunctionBody(Decl);
  }

  SourceLocation LBraceLoc = Tok.getLocation();
  StmtResult FnBody(ParseCXXTryBlockCommon(TryLoc, /*FnTry*/true));
  // If we failed to parse the try-catch, we just give the function an empty
  // compound statement as the body.
  if (FnBody.isInvalid()) {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc, None, false);
  }

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.take());
}

bool Parser::trySkippingFunctionBody() {
  assert(Tok.is(tok::l_brace));
  assert(SkipFunctionBodies &&
         "Should only be called when SkipFunctionBodies is enabled");

  if (!PP.isCodeCompletionEnabled()) {
    ConsumeBrace();
    SkipUntil(tok::r_brace);
    return true;
  }

  // We're in code-completion mode. Skip parsing for all function bodies unless
  // the body contains the code-completion point.
  TentativeParsingAction PA(*this);
  ConsumeBrace();
  if (SkipUntil(tok::r_brace, StopAtCodeCompletion)) {
    PA.Commit();
    return true;
  }

  PA.Revert();
  return false;
}

/// ParseCXXTryBlock - Parse a C++ try-block.
///
///       try-block:
///         'try' compound-statement handler-seq
///
StmtResult Parser::ParseCXXTryBlock() {
  assert(Tok.is(tok::kw_try) && "Expected 'try'");

  SourceLocation TryLoc = ConsumeToken();
  return ParseCXXTryBlockCommon(TryLoc);
}

/// ParseCXXTryBlockCommon - Parse the common part of try-block and
/// function-try-block.
///
///       try-block:
///         'try' compound-statement handler-seq
///
///       function-try-block:
///         'try' ctor-initializer[opt] compound-statement handler-seq
///
///       handler-seq:
///         handler handler-seq[opt]
///
///       [Borland] try-block:
///         'try' compound-statement seh-except-block
///         'try' compound-statment  seh-finally-block
///
StmtResult Parser::ParseCXXTryBlockCommon(SourceLocation TryLoc, bool FnTry) {
  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));
  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?

  StmtResult TryBlock(ParseCompoundStatement(/*isStmtExpr=*/false,
                      Scope::DeclScope | Scope::TryScope |
                        (FnTry ? Scope::FnTryCatchScope : 0)));
  if (TryBlock.isInvalid())
    return TryBlock;

  // Borland allows SEH-handlers with 'try'

  if ((Tok.is(tok::identifier) &&
       Tok.getIdentifierInfo() == getSEHExceptKeyword()) ||
      Tok.is(tok::kw___finally)) {
    // TODO: Factor into common return ParseSEHHandlerCommon(...)
    StmtResult Handler;
    if(Tok.getIdentifierInfo() == getSEHExceptKeyword()) {
      SourceLocation Loc = ConsumeToken();
      Handler = ParseSEHExceptBlock(Loc);
    }
    else {
      SourceLocation Loc = ConsumeToken();
      Handler = ParseSEHFinallyBlock(Loc);
    }
    if(Handler.isInvalid())
      return Handler;

    return Actions.ActOnSEHTryBlock(true /* IsCXXTry */,
                                    TryLoc,
                                    TryBlock.take(),
                                    Handler.take());
  }
  else {
    StmtVector Handlers;

    // C++11 attributes can't appear here, despite this context seeming
    // statement-like.
    DiagnoseAndSkipCXX11Attributes();

    if (Tok.isNot(tok::kw_catch))
      return StmtError(Diag(Tok, diag::err_expected_catch));
    while (Tok.is(tok::kw_catch)) {
      StmtResult Handler(ParseCXXCatchBlock(FnTry));
      if (!Handler.isInvalid())
        Handlers.push_back(Handler.release());
    }
    // Don't bother creating the full statement if we don't have any usable
    // handlers.
    if (Handlers.empty())
      return StmtError();

    return Actions.ActOnCXXTryBlock(TryLoc, TryBlock.take(), Handlers);
  }
}

/// ParseCXXCatchBlock - Parse a C++ catch block, called handler in the standard
///
///   handler:
///     'catch' '(' exception-declaration ')' compound-statement
///
///   exception-declaration:
///     attribute-specifier-seq[opt] type-specifier-seq declarator
///     attribute-specifier-seq[opt] type-specifier-seq abstract-declarator[opt]
///     '...'
///
StmtResult Parser::ParseCXXCatchBlock(bool FnCatch) {
  assert(Tok.is(tok::kw_catch) && "Expected 'catch'");

  SourceLocation CatchLoc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen))
    return StmtError();

  // C++ 3.3.2p3:
  // The name in a catch exception-declaration is local to the handler and
  // shall not be redeclared in the outermost block of the handler.
  ParseScope CatchScope(this, Scope::DeclScope | Scope::ControlScope |
                          (FnCatch ? Scope::FnTryCatchScope : 0));

  // exception-declaration is equivalent to '...' or a parameter-declaration
  // without default arguments.
  Decl *ExceptionDecl = 0;
  if (Tok.isNot(tok::ellipsis)) {
    ParsedAttributesWithRange Attributes(AttrFactory);
    MaybeParseCXX11Attributes(Attributes);

    DeclSpec DS(AttrFactory);
    DS.takeAttributesFrom(Attributes);

    if (ParseCXXTypeSpecifierSeq(DS))
      return StmtError();

    Declarator ExDecl(DS, Declarator::CXXCatchContext);
    ParseDeclarator(ExDecl);
    ExceptionDecl = Actions.ActOnExceptionDeclarator(getCurScope(), ExDecl);
  } else
    ConsumeToken();

  T.consumeClose();
  if (T.getCloseLocation().isInvalid())
    return StmtError();

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));

  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?
  StmtResult Block(ParseCompoundStatement());
  if (Block.isInvalid())
    return Block;

  return Actions.ActOnCXXCatchBlock(CatchLoc, ExceptionDecl, Block.take());
}

void Parser::ParseMicrosoftIfExistsStatement(StmtVector &Stmts) {
  IfExistsCondition Result;
  if (ParseMicrosoftIfExistsCondition(Result))
    return;

  // Handle dependent statements by parsing the braces as a compound statement.
  // This is not the same behavior as Visual C++, which don't treat this as a
  // compound statement, but for Clang's type checking we can't have anything
  // inside these braces escaping to the surrounding code.
  if (Result.Behavior == IEB_Dependent) {
    if (!Tok.is(tok::l_brace)) {
      Diag(Tok, diag::err_expected_lbrace);
      return;
    }

    StmtResult Compound = ParseCompoundStatement();
    if (Compound.isInvalid())
      return;

    StmtResult DepResult = Actions.ActOnMSDependentExistsStmt(Result.KeywordLoc,
                                                              Result.IsIfExists,
                                                              Result.SS,
                                                              Result.Name,
                                                              Compound.get());
    if (DepResult.isUsable())
      Stmts.push_back(DepResult.get());
    return;
  }

  BalancedDelimiterTracker Braces(*this, tok::l_brace);
  if (Braces.consumeOpen()) {
    Diag(Tok, diag::err_expected_lbrace);
    return;
  }

  switch (Result.Behavior) {
  case IEB_Parse:
    // Parse the statements below.
    break;

  case IEB_Dependent:
    llvm_unreachable("Dependent case handled above");

  case IEB_Skip:
    Braces.skipToEnd();
    return;
  }

  // Condition is true, parse the statements.
  while (Tok.isNot(tok::r_brace)) {
    StmtResult R = ParseStatementOrDeclaration(Stmts, false);
    if (R.isUsable())
      Stmts.push_back(R.release());
  }
  Braces.consumeClose();
}
