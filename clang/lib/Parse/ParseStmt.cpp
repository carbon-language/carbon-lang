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
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.8: Statements and Blocks.
//===----------------------------------------------------------------------===//

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
Parser::OwningStmtResult
Parser::ParseStatementOrDeclaration(bool OnlyStatement) {
  const char *SemiError = 0;
  OwningStmtResult Res(Actions);

  CXX0XAttributeList Attr;
  if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier())
    Attr = ParseCXX0XAttributes();

  // Cases in this switch statement should fall through if the parser expects
  // the token to end in a semicolon (in which case SemiError should be set),
  // or they directly 'return;' if not.
  tok::TokenKind Kind  = Tok.getKind();
  SourceLocation AtLoc;
  switch (Kind) {
  case tok::at: // May be a @try or @throw statement
    {
      AtLoc = ConsumeToken();  // consume @
      return ParseObjCAtStatement(AtLoc);
    }

  case tok::code_completion:
    Actions.CodeCompleteOrdinaryName(CurScope, Action::CCC_Statement);
    ConsumeToken();
    return ParseStatementOrDeclaration(OnlyStatement);
      
  case tok::identifier:
    if (NextToken().is(tok::colon)) { // C99 6.8.1: labeled-statement
      // identifier ':' statement
      return ParseLabeledStatement(Attr.AttrList);
    }
    // PASS THROUGH.

  default: {
    if ((getLang().CPlusPlus || !OnlyStatement) && isDeclarationStatement()) {
      SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
      DeclGroupPtrTy Decl = ParseDeclaration(Declarator::BlockContext, DeclEnd,
                                             Attr);
      return Actions.ActOnDeclStmt(Decl, DeclStart, DeclEnd);
    }

    if (Tok.is(tok::r_brace)) {
      Diag(Tok, diag::err_expected_statement);
      return StmtError();
    }

    // FIXME: Use the attributes
    // expression[opt] ';'
    OwningExprResult Expr(ParseExpression());
    if (Expr.isInvalid()) {
      // If the expression is invalid, skip ahead to the next semicolon.  Not
      // doing this opens us up to the possibility of infinite loops if
      // ParseExpression does not consume any tokens.
      SkipUntil(tok::semi);
      return StmtError();
    }
    // Otherwise, eat the semicolon.
    ExpectAndConsume(tok::semi, diag::err_expected_semi_after_expr);
    return Actions.ActOnExprStmt(Actions.MakeFullExpr(Expr));
  }

  case tok::kw_case:                // C99 6.8.1: labeled-statement
    return ParseCaseStatement(Attr.AttrList);
  case tok::kw_default:             // C99 6.8.1: labeled-statement
    return ParseDefaultStatement(Attr.AttrList);

  case tok::l_brace:                // C99 6.8.2: compound-statement
    return ParseCompoundStatement(Attr.AttrList);
  case tok::semi:                   // C99 6.8.3p3: expression[opt] ';'
    return Actions.ActOnNullStmt(ConsumeToken());

  case tok::kw_if:                  // C99 6.8.4.1: if-statement
    return ParseIfStatement(Attr.AttrList);
  case tok::kw_switch:              // C99 6.8.4.2: switch-statement
    return ParseSwitchStatement(Attr.AttrList);

  case tok::kw_while:               // C99 6.8.5.1: while-statement
    return ParseWhileStatement(Attr.AttrList);
  case tok::kw_do:                  // C99 6.8.5.2: do-statement
    Res = ParseDoStatement(Attr.AttrList);
    SemiError = "do/while";
    break;
  case tok::kw_for:                 // C99 6.8.5.3: for-statement
    return ParseForStatement(Attr.AttrList);

  case tok::kw_goto:                // C99 6.8.6.1: goto-statement
    Res = ParseGotoStatement(Attr.AttrList);
    SemiError = "goto";
    break;
  case tok::kw_continue:            // C99 6.8.6.2: continue-statement
    Res = ParseContinueStatement(Attr.AttrList);
    SemiError = "continue";
    break;
  case tok::kw_break:               // C99 6.8.6.3: break-statement
    Res = ParseBreakStatement(Attr.AttrList);
    SemiError = "break";
    break;
  case tok::kw_return:              // C99 6.8.6.4: return-statement
    Res = ParseReturnStatement(Attr.AttrList);
    SemiError = "return";
    break;

  case tok::kw_asm: {
    if (Attr.HasAttr)
      Diag(Attr.Range.getBegin(), diag::err_attributes_not_allowed)
        << Attr.Range;
    bool msAsm = false;
    Res = ParseAsmStatement(msAsm);
    if (msAsm) return move(Res);
    SemiError = "asm";
    break;
  }

  case tok::kw_try:                 // C++ 15: try-block
    return ParseCXXTryBlock(Attr.AttrList);
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
    SkipUntil(tok::r_brace, true, true);
  }

  return move(Res);
}

/// ParseLabeledStatement - We have an identifier and a ':' after it.
///
///       labeled-statement:
///         identifier ':' statement
/// [GNU]   identifier ':' attributes[opt] statement
///
Parser::OwningStmtResult Parser::ParseLabeledStatement(AttributeList *Attr) {
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  Token IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.

  assert(Tok.is(tok::colon) && "Not a label!");

  // identifier ':' statement
  SourceLocation ColonLoc = ConsumeToken();

  // Read label attributes, if present.
  if (Tok.is(tok::kw___attribute))
    Attr = addAttributeLists(Attr, ParseGNUAttributes());

  OwningStmtResult SubStmt(ParseStatement());

  // Broken substmt shouldn't prevent the label from being added to the AST.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);

  return Actions.ActOnLabelStmt(IdentTok.getLocation(),
                                IdentTok.getIdentifierInfo(),
                                ColonLoc, move(SubStmt));
}

/// ParseCaseStatement
///       labeled-statement:
///         'case' constant-expression ':' statement
/// [GNU]   'case' constant-expression '...' constant-expression ':' statement
///
Parser::OwningStmtResult Parser::ParseCaseStatement(AttributeList *Attr) {
  assert(Tok.is(tok::kw_case) && "Not a case stmt!");
  // FIXME: Use attributes?

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
  // wierdness in the actions), so this is just local grossness :).

  // TopLevelCase - This is the highest level we have parsed.  'case 1' in the
  // example above.
  OwningStmtResult TopLevelCase(Actions, true);

  // DeepestParsedCaseStmt - This is the deepest statement we have parsed, which
  // gets updated each time a new case is parsed, and whose body is unset so
  // far.  When parsing 'case 4', this is the 'case 3' node.
  StmtTy *DeepestParsedCaseStmt = 0;

  // While we have case statements, eat and stack them.
  do {
    SourceLocation CaseLoc = ConsumeToken();  // eat the 'case'.

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteCase(CurScope);
      ConsumeToken();
    }
    
    /// We don't want to treat 'case x : y' as a potential typo for 'case x::y'.
    /// Disable this form of error recovery while we're parsing the case
    /// expression.
    ColonProtectionRAIIObject ColonProtection(*this);
    
    OwningExprResult LHS(ParseConstantExpression());
    if (LHS.isInvalid()) {
      SkipUntil(tok::colon);
      return StmtError();
    }

    // GNU case range extension.
    SourceLocation DotDotDotLoc;
    OwningExprResult RHS(Actions);
    if (Tok.is(tok::ellipsis)) {
      Diag(Tok, diag::ext_gnu_case_range);
      DotDotDotLoc = ConsumeToken();

      RHS = ParseConstantExpression();
      if (RHS.isInvalid()) {
        SkipUntil(tok::colon);
        return StmtError();
      }
    }
    
    ColonProtection.restore();

    if (Tok.isNot(tok::colon)) {
      Diag(Tok, diag::err_expected_colon_after) << "'case'";
      SkipUntil(tok::colon);
      return StmtError();
    }

    SourceLocation ColonLoc = ConsumeToken();

    OwningStmtResult Case =
      Actions.ActOnCaseStmt(CaseLoc, move(LHS), DotDotDotLoc,
                            move(RHS), ColonLoc);

    // If we had a sema error parsing this case, then just ignore it and
    // continue parsing the sub-stmt.
    if (Case.isInvalid()) {
      if (TopLevelCase.isInvalid())  // No parsed case stmts.
        return ParseStatement();
      // Otherwise, just don't add it as a nested case.
    } else {
      // If this is the first case statement we parsed, it becomes TopLevelCase.
      // Otherwise we link it into the current chain.
      StmtTy *NextDeepest = Case.get();
      if (TopLevelCase.isInvalid())
        TopLevelCase = move(Case);
      else
        Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, move(Case));
      DeepestParsedCaseStmt = NextDeepest;
    }

    // Handle all case statements.
  } while (Tok.is(tok::kw_case));

  assert(!TopLevelCase.isInvalid() && "Should have parsed at least one case!");

  // If we found a non-case statement, start by parsing it.
  OwningStmtResult SubStmt(Actions);

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Nicely diagnose the common error "switch (X) { case 4: }", which is
    // not valid.
    // FIXME: add insertion hint.
    Diag(Tok, diag::err_label_end_of_compound_statement);
    SubStmt = true;
  }

  // Broken sub-stmt shouldn't prevent forming the case statement properly.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(SourceLocation());

  // Install the body into the most deeply-nested case.
  Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, move(SubStmt));

  // Return the top level parsed statement tree.
  return move(TopLevelCase);
}

/// ParseDefaultStatement
///       labeled-statement:
///         'default' ':' statement
/// Note that this does not parse the 'statement' at the end.
///
Parser::OwningStmtResult Parser::ParseDefaultStatement(AttributeList *Attr) {
  //FIXME: Use attributes?
  assert(Tok.is(tok::kw_default) && "Not a default stmt!");
  SourceLocation DefaultLoc = ConsumeToken();  // eat the 'default'.

  if (Tok.isNot(tok::colon)) {
    Diag(Tok, diag::err_expected_colon_after) << "'default'";
    SkipUntil(tok::colon);
    return StmtError();
  }

  SourceLocation ColonLoc = ConsumeToken();

  // Diagnose the common error "switch (X) {... default: }", which is not valid.
  if (Tok.is(tok::r_brace)) {
    Diag(Tok, diag::err_label_end_of_compound_statement);
    return StmtError();
  }

  OwningStmtResult SubStmt(ParseStatement());
  if (SubStmt.isInvalid())
    return StmtError();

  return Actions.ActOnDefaultStmt(DefaultLoc, ColonLoc,
                                  move(SubStmt), CurScope);
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
Parser::OwningStmtResult Parser::ParseCompoundStatement(AttributeList *Attr,
                                                        bool isStmtExpr) {
  //FIXME: Use attributes?
  assert(Tok.is(tok::l_brace) && "Not a compount stmt!");

  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  ParseScope CompoundScope(this, Scope::DeclScope);

  // Parse the statements in the body.
  return ParseCompoundStatementBody(isStmtExpr);
}


/// ParseCompoundStatementBody - Parse a sequence of statements and invoke the
/// ActOnCompoundStmt action.  This expects the '{' to be the current token, and
/// consume the '}' at the end of the block.  It does not manipulate the scope
/// stack.
Parser::OwningStmtResult Parser::ParseCompoundStatementBody(bool isStmtExpr) {
  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(),
                                Tok.getLocation(),
                                "in compound statement ('{}')");

  SourceLocation LBraceLoc = ConsumeBrace();  // eat the '{'.

  // TODO: "__label__ X, Y, Z;" is the GNU "Local Label" extension.  These are
  // only allowed at the start of a compound stmt regardless of the language.

  typedef StmtVector StmtsTy;
  StmtsTy Stmts(Actions);
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    OwningStmtResult R(Actions);
    if (Tok.isNot(tok::kw___extension__)) {
      R = ParseStatementOrDeclaration(false);
    } else {
      // __extension__ can start declarations and it can also be a unary
      // operator for expressions.  Consume multiple __extension__ markers here
      // until we can determine which is which.
      // FIXME: This loses extension expressions in the AST!
      SourceLocation ExtLoc = ConsumeToken();
      while (Tok.is(tok::kw___extension__))
        ConsumeToken();

      CXX0XAttributeList Attr;
      if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier())
        Attr = ParseCXX0XAttributes();

      // If this is the start of a declaration, parse it as such.
      if (isDeclarationStatement()) {
        // __extension__ silences extension warnings in the subdeclaration.
        // FIXME: Save the __extension__ on the decl as a node somehow?
        ExtensionRAIIObject O(Diags);

        SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
        DeclGroupPtrTy Res = ParseDeclaration(Declarator::BlockContext, DeclEnd,
                                              Attr);
        R = Actions.ActOnDeclStmt(Res, DeclStart, DeclEnd);
      } else {
        // Otherwise this was a unary __extension__ marker.
        OwningExprResult Res(ParseExpressionWithLeadingExtension(ExtLoc));

        if (Res.isInvalid()) {
          SkipUntil(tok::semi);
          continue;
        }

        // FIXME: Use attributes?
        // Eat the semicolon at the end of stmt and convert the expr into a
        // statement.
        ExpectAndConsume(tok::semi, diag::err_expected_semi_after_expr);
        R = Actions.ActOnExprStmt(Actions.MakeFullExpr(Res));
      }
    }

    if (R.isUsable())
      Stmts.push_back(R.release());
  }

  // We broke out of the while loop because we found a '}' or EOF.
  if (Tok.isNot(tok::r_brace)) {
    Diag(Tok, diag::err_expected_rbrace);
    return StmtError();
  }

  SourceLocation RBraceLoc = ConsumeBrace();
  return Actions.ActOnCompoundStmt(LBraceLoc, RBraceLoc, move_arg(Stmts),
                                   isStmtExpr);
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
bool Parser::ParseParenExprOrCondition(OwningExprResult &ExprResult,
                                       DeclPtrTy &DeclResult) {
  bool ParseError = false;
  
  SourceLocation LParenLoc = ConsumeParen();
  if (getLang().CPlusPlus) 
    ParseError = ParseCXXCondition(ExprResult, DeclResult);
  else {
    ExprResult = ParseExpression();
    DeclResult = DeclPtrTy();
  }

  // If the parser was confused by the condition and we don't have a ')', try to
  // recover by skipping ahead to a semi and bailing out.  If condexp is
  // semantically invalid but we have well formed code, keep going.
  if (ExprResult.isInvalid() && !DeclResult.get() && Tok.isNot(tok::r_paren)) {
    SkipUntil(tok::semi);
    // Skipping may have stopped if it found the containing ')'.  If so, we can
    // continue parsing the if statement.
    if (Tok.isNot(tok::r_paren))
      return true;
  }

  // Otherwise the condition is valid or the rparen is present.
  MatchRHSPunctuation(tok::r_paren, LParenLoc);
  return false;
}


/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
/// [C++]   'if' '(' condition ')' statement
/// [C++]   'if' '(' condition ')' statement 'else' statement
///
Parser::OwningStmtResult Parser::ParseIfStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  assert(Tok.is(tok::kw_if) && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "if";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

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
  OwningExprResult CondExp(Actions);
  DeclPtrTy CondVar;
  if (ParseParenExprOrCondition(CondExp, CondVar))
    return StmtError();

  FullExprArg FullCondExp(Actions.MakeFullExpr(CondExp));

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
  OwningStmtResult ThenStmt(ParseStatement());

  // Pop the 'if' scope if needed.
  InnerScope.Exit();

  // If it has an else, parse it.
  SourceLocation ElseLoc;
  SourceLocation ElseStmtLoc;
  OwningStmtResult ElseStmt(Actions);

  if (Tok.is(tok::kw_else)) {
    ElseLoc = ConsumeToken();

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

    bool WithinElse = CurScope->isWithinElse();
    CurScope->setWithinElse(true);
    ElseStmtLoc = Tok.getLocation();
    ElseStmt = ParseStatement();
    CurScope->setWithinElse(WithinElse);

    // Pop the 'else' scope if needed.
    InnerScope.Exit();
  }

  IfScope.Exit();

  // If the condition was invalid, discard the if statement.  We could recover
  // better by replacing it with a valid expr, but don't do that yet.
  if (CondExp.isInvalid() && !CondVar.get())
    return StmtError();

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

  return Actions.ActOnIfStmt(IfLoc, FullCondExp, CondVar, move(ThenStmt),
                             ElseLoc, move(ElseStmt));
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
/// [C++]   'switch' '(' condition ')' statement
Parser::OwningStmtResult Parser::ParseSwitchStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  assert(Tok.is(tok::kw_switch) && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "switch";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

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
  unsigned ScopeFlags = Scope::BreakScope;
  if (C99orCXX)
    ScopeFlags |= Scope::DeclScope | Scope::ControlScope;
  ParseScope SwitchScope(this, ScopeFlags);

  // Parse the condition.
  OwningExprResult Cond(Actions);
  DeclPtrTy CondVar;
  if (ParseParenExprOrCondition(Cond, CondVar))
    return StmtError();

  FullExprArg FullCond(Actions.MakeFullExpr(Cond));
  
  OwningStmtResult Switch = Actions.ActOnStartOfSwitchStmt(FullCond, CondVar);

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
  OwningStmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  if (Body.isInvalid()) {
    Body = Actions.ActOnNullStmt(Tok.getLocation());
    // FIXME: Remove the case statement list from the Switch statement.
  }

  SwitchScope.Exit();

  if (Cond.isInvalid() && !CondVar.get())
    return StmtError();

  return Actions.ActOnFinishSwitchStmt(SwitchLoc, move(Switch), move(Body));
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
/// [C++]   'while' '(' condition ')' statement
Parser::OwningStmtResult Parser::ParseWhileStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  assert(Tok.is(tok::kw_while) && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "while";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

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
  OwningExprResult Cond(Actions);
  DeclPtrTy CondVar;
  if (ParseParenExprOrCondition(Cond, CondVar))
    return StmtError();

  FullExprArg FullCond(Actions.MakeFullExpr(Cond));

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
  OwningStmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();
  WhileScope.Exit();

  if ((Cond.isInvalid() && !CondVar.get()) || Body.isInvalid())
    return StmtError();

  return Actions.ActOnWhileStmt(WhileLoc, FullCond, CondVar, move(Body));
}

/// ParseDoStatement
///       do-statement: [C99 6.8.5.2]
///         'do' statement 'while' '(' expression ')' ';'
/// Note: this lets the caller parse the end ';'.
Parser::OwningStmtResult Parser::ParseDoStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  assert(Tok.is(tok::kw_do) && "Not a do stmt!");
  SourceLocation DoLoc = ConsumeToken();  // eat the 'do'.

  // C99 6.8.5p5 - In C99, the do statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  unsigned ScopeFlags;
  if (getLang().C99)
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
                        (getLang().C99 || getLang().CPlusPlus) &&
                        Tok.isNot(tok::l_brace));

  // Read the body statement.
  OwningStmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  if (Tok.isNot(tok::kw_while)) {
    if (!Body.isInvalid()) {
      Diag(Tok, diag::err_expected_while);
      Diag(DoLoc, diag::note_matching) << "do";
      SkipUntil(tok::semi, false, true);
    }
    return StmtError();
  }
  SourceLocation WhileLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "do/while";
    SkipUntil(tok::semi, false, true);
    return StmtError();
  }

  // Parse the parenthesized condition.
  SourceLocation LPLoc = ConsumeParen();
  OwningExprResult Cond = ParseExpression();
  SourceLocation RPLoc = MatchRHSPunctuation(tok::r_paren, LPLoc);
  DoScope.Exit();

  if (Cond.isInvalid() || Body.isInvalid())
    return StmtError();

  return Actions.ActOnDoStmt(DoLoc, move(Body), WhileLoc, LPLoc,
                             move(Cond), RPLoc);
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
/// [C++]   'for' '(' for-init-statement condition[opt] ';' expression[opt] ')'
/// [C++]       statement
/// [OBJC2] 'for' '(' declaration 'in' expr ')' statement
/// [OBJC2] 'for' '(' expr 'in' expr ')' statement
///
/// [C++] for-init-statement:
/// [C++]   expression-statement
/// [C++]   simple-declaration
///
Parser::OwningStmtResult Parser::ParseForStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  assert(Tok.is(tok::kw_for) && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "for";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXXorObjC = getLang().C99 || getLang().CPlusPlus || getLang().ObjC1;

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

  SourceLocation LParenLoc = ConsumeParen();
  OwningExprResult Value(Actions);

  bool ForEach = false;
  OwningStmtResult FirstPart(Actions);
  OwningExprResult SecondPart(Actions), ThirdPart(Actions);
  DeclPtrTy SecondVar;
  
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(CurScope, 
                                     C99orCXXorObjC? Action::CCC_ForInit
                                                   : Action::CCC_Expression);
    ConsumeToken();
  }
  
  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (;
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isSimpleDeclaration()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!C99orCXXorObjC)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);

    AttributeList *AttrList = 0;
    if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier())
      AttrList = ParseCXX0XAttributes().AttrList;

    SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
    DeclGroupPtrTy DG = ParseSimpleDeclaration(Declarator::ForContext, DeclEnd,
                                               AttrList);
    FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());

    if (Tok.is(tok::semi)) {  // for (int x = 4;
      ConsumeToken();
    } else if ((ForEach = isTokIdentifier_in())) {
      Actions.ActOnForEachDeclStmt(DG);
      // ObjC: for (id x in expr)
      ConsumeToken(); // consume 'in'
      SecondPart = ParseExpression();
    } else {
      Diag(Tok, diag::err_expected_semi_for);
      SkipUntil(tok::semi);
    }
  } else {
    Value = ParseExpression();

    // Turn the expression into a stmt.
    if (!Value.isInvalid())
      FirstPart = Actions.ActOnExprStmt(Actions.MakeFullExpr(Value));

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if ((ForEach = isTokIdentifier_in())) {
      ConsumeToken(); // consume 'in'
      SecondPart = ParseExpression();
    } else {
      if (!Value.isInvalid()) Diag(Tok, diag::err_expected_semi_for);
      SkipUntil(tok::semi);
    }
  }
  if (!ForEach) {
    assert(!SecondPart.get() && "Shouldn't have a second expression yet.");
    // Parse the second part of the for specifier.
    if (Tok.is(tok::semi)) {  // for (...;;
      // no second part.
    } else {
      if (getLang().CPlusPlus)
        ParseCXXCondition(SecondPart, SecondVar);
      else
        SecondPart = ParseExpression();
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else {
      if (!SecondPart.isInvalid() || SecondVar.get()) 
        Diag(Tok, diag::err_expected_semi_for);
      SkipUntil(tok::semi);
    }

    // Parse the third part of the for specifier.
    if (Tok.isNot(tok::r_paren))    // for (...;...;)
      ThirdPart = ParseExpression();
  }
  // Match the ')'.
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

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
  OwningStmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  // Leave the for-scope.
  ForScope.Exit();

  if (Body.isInvalid())
    return StmtError();

  if (!ForEach)
    return Actions.ActOnForStmt(ForLoc, LParenLoc, move(FirstPart),
                                Actions.MakeFullExpr(SecondPart), SecondVar,
                                Actions.MakeFullExpr(ThirdPart), RParenLoc, 
                                move(Body));

  return Actions.ActOnObjCForCollectionStmt(ForLoc, LParenLoc,
                                            move(FirstPart),
                                            move(SecondPart),
                                            RParenLoc, move(Body));
}

/// ParseGotoStatement
///       jump-statement:
///         'goto' identifier ';'
/// [GNU]   'goto' '*' expression ';'
///
/// Note: this lets the caller parse the end ';'.
///
Parser::OwningStmtResult Parser::ParseGotoStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  assert(Tok.is(tok::kw_goto) && "Not a goto stmt!");
  SourceLocation GotoLoc = ConsumeToken();  // eat the 'goto'.

  OwningStmtResult Res(Actions);
  if (Tok.is(tok::identifier)) {
    Res = Actions.ActOnGotoStmt(GotoLoc, Tok.getLocation(),
                                Tok.getIdentifierInfo());
    ConsumeToken();
  } else if (Tok.is(tok::star)) {
    // GNU indirect goto extension.
    Diag(Tok, diag::ext_gnu_indirect_goto);
    SourceLocation StarLoc = ConsumeToken();
    OwningExprResult R(ParseExpression());
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return StmtError();
    }
    Res = Actions.ActOnIndirectGotoStmt(GotoLoc, StarLoc, move(R));
  } else {
    Diag(Tok, diag::err_expected_ident);
    return StmtError();
  }

  return move(Res);
}

/// ParseContinueStatement
///       jump-statement:
///         'continue' ';'
///
/// Note: this lets the caller parse the end ';'.
///
Parser::OwningStmtResult Parser::ParseContinueStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  SourceLocation ContinueLoc = ConsumeToken();  // eat the 'continue'.
  return Actions.ActOnContinueStmt(ContinueLoc, CurScope);
}

/// ParseBreakStatement
///       jump-statement:
///         'break' ';'
///
/// Note: this lets the caller parse the end ';'.
///
Parser::OwningStmtResult Parser::ParseBreakStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  SourceLocation BreakLoc = ConsumeToken();  // eat the 'break'.
  return Actions.ActOnBreakStmt(BreakLoc, CurScope);
}

/// ParseReturnStatement
///       jump-statement:
///         'return' expression[opt] ';'
Parser::OwningStmtResult Parser::ParseReturnStatement(AttributeList *Attr) {
  // FIXME: Use attributes?
  assert(Tok.is(tok::kw_return) && "Not a return stmt!");
  SourceLocation ReturnLoc = ConsumeToken();  // eat the 'return'.

  OwningExprResult R(Actions);
  if (Tok.isNot(tok::semi)) {
    R = ParseExpression();
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return StmtError();
    }
  }
  return Actions.ActOnReturnStmt(ReturnLoc, move(R));
}

/// FuzzyParseMicrosoftAsmStatement. When -fms-extensions is enabled, this
/// routine is called to skip/ignore tokens that comprise the MS asm statement.
Parser::OwningStmtResult Parser::FuzzyParseMicrosoftAsmStatement() {
  if (Tok.is(tok::l_brace)) {
    unsigned short savedBraceCount = BraceCount;
    do {
      ConsumeAnyToken();
    } while (BraceCount > savedBraceCount && Tok.isNot(tok::eof));
  } else {
    // From the MS website: If used without braces, the __asm keyword means
    // that the rest of the line is an assembly-language statement.
    SourceManager &SrcMgr = PP.getSourceManager();
    SourceLocation TokLoc = Tok.getLocation();
    unsigned LineNo = SrcMgr.getInstantiationLineNumber(TokLoc);
    do {
      ConsumeAnyToken();
      TokLoc = Tok.getLocation();
    } while ((SrcMgr.getInstantiationLineNumber(TokLoc) == LineNo) &&
             Tok.isNot(tok::r_brace) && Tok.isNot(tok::semi) &&
             Tok.isNot(tok::eof));
  }
  llvm::SmallVector<std::string, 4> Names;
  Token t;
  t.setKind(tok::string_literal);
  t.setLiteralData("\"FIXME: not done\"");
  t.clearFlag(Token::NeedsCleaning);
  t.setLength(17);
  OwningExprResult AsmString(Actions.ActOnStringLiteral(&t, 1));
  ExprVector Constraints(Actions);
  ExprVector Exprs(Actions);
  ExprVector Clobbers(Actions);
  return Actions.ActOnAsmStmt(Tok.getLocation(), true, true, 0, 0, Names.data(),
                              move_arg(Constraints), move_arg(Exprs),
                              move(AsmString), move_arg(Clobbers),
                              Tok.getLocation(), true);
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
/// [MS]  ms-asm-statement:
///         '__asm' assembly-instruction ';'[opt]
///         '__asm' '{' assembly-instruction-list '}' ';'[opt]
///
/// [MS]  assembly-instruction-list:
///         assembly-instruction ';'[opt]
///         assembly-instruction-list ';' assembly-instruction ';'[opt]
///
Parser::OwningStmtResult Parser::ParseAsmStatement(bool &msAsm) {
  assert(Tok.is(tok::kw_asm) && "Not an asm stmt");
  SourceLocation AsmLoc = ConsumeToken();

  if (getLang().Microsoft && Tok.isNot(tok::l_paren) && !isTypeQualifier()) {
    msAsm = true;
    return FuzzyParseMicrosoftAsmStatement();
  }
  DeclSpec DS;
  SourceLocation Loc = Tok.getLocation();
  ParseTypeQualifierListOpt(DS, true, false);

  // GNU asms accept, but warn, about type-qualifiers other than volatile.
  if (DS.getTypeQualifiers() & DeclSpec::TQ_const)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "const";
  if (DS.getTypeQualifiers() & DeclSpec::TQ_restrict)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "restrict";

  // Remember if this was a volatile asm.
  bool isVolatile = DS.getTypeQualifiers() & DeclSpec::TQ_volatile;
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "asm";
    SkipUntil(tok::r_paren);
    return StmtError();
  }
  Loc = ConsumeParen();

  OwningExprResult AsmString(ParseAsmStringLiteral());
  if (AsmString.isInvalid())
    return StmtError();

  llvm::SmallVector<std::string, 4> Names;
  ExprVector Constraints(Actions);
  ExprVector Exprs(Actions);
  ExprVector Clobbers(Actions);

  if (Tok.is(tok::r_paren)) {
    // We have a simple asm expression like 'asm("foo")'.
    SourceLocation RParenLoc = ConsumeParen();
    return Actions.ActOnAsmStmt(AsmLoc, /*isSimple*/ true, isVolatile,
                                /*NumOutputs*/ 0, /*NumInputs*/ 0, 0, 
                                move_arg(Constraints), move_arg(Exprs),
                                move(AsmString), move_arg(Clobbers),
                                RParenLoc);
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

    // Parse the asm-string list for clobbers.
    while (1) {
      OwningExprResult Clobber(ParseAsmStringLiteral());

      if (Clobber.isInvalid())
        break;

      Clobbers.push_back(Clobber.release());

      if (Tok.isNot(tok::comma)) break;
      ConsumeToken();
    }
  }

  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, Loc);
  return Actions.ActOnAsmStmt(AsmLoc, false, isVolatile,
                              NumOutputs, NumInputs, Names.data(),
                              move_arg(Constraints), move_arg(Exprs),
                              move(AsmString), move_arg(Clobbers),
                              RParenLoc);
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
bool Parser::ParseAsmOperandsOpt(llvm::SmallVectorImpl<std::string> &Names,
                                 llvm::SmallVectorImpl<ExprTy*> &Constraints,
                                 llvm::SmallVectorImpl<ExprTy*> &Exprs) {
  // 'asm-operands' isn't present?
  if (!isTokenStringLiteral() && Tok.isNot(tok::l_square))
    return false;

  while (1) {
    // Read the [id] if present.
    if (Tok.is(tok::l_square)) {
      SourceLocation Loc = ConsumeBracket();

      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::r_paren);
        return true;
      }

      IdentifierInfo *II = Tok.getIdentifierInfo();
      ConsumeToken();

      Names.push_back(II->getName());
      MatchRHSPunctuation(tok::r_square, Loc);
    } else
      Names.push_back(std::string());

    OwningExprResult Constraint(ParseAsmStringLiteral());
    if (Constraint.isInvalid()) {
        SkipUntil(tok::r_paren);
        return true;
    }
    Constraints.push_back(Constraint.release());

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_expected_lparen_after) << "asm operand";
      SkipUntil(tok::r_paren);
      return true;
    }

    // Read the parenthesized expression.
    SourceLocation OpenLoc = ConsumeParen();
    OwningExprResult Res(ParseExpression());
    MatchRHSPunctuation(tok::r_paren, OpenLoc);
    if (Res.isInvalid()) {
      SkipUntil(tok::r_paren);
      return true;
    }
    Exprs.push_back(Res.release());
    // Eat the comma and continue parsing if it exists.
    if (Tok.isNot(tok::comma)) return false;
    ConsumeToken();
  }

  return true;
}

Parser::DeclPtrTy Parser::ParseFunctionStatementBody(DeclPtrTy Decl) {
  assert(Tok.is(tok::l_brace));
  SourceLocation LBraceLoc = Tok.getLocation();

  PrettyStackTraceActionsDecl CrashInfo(Decl, LBraceLoc, Actions,
                                        PP.getSourceManager(),
                                        "parsing function body");

  // Do not enter a scope for the brace, as the arguments are in the same scope
  // (the function body) as the body itself.  Instead, just read the statement
  // list and put it into a CompoundStmt for safe keeping.
  OwningStmtResult FnBody(ParseCompoundStatementBody());

  // If the function body could not be parsed, make a bogus compoundstmt.
  if (FnBody.isInvalid())
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc,
                                       MultiStmtArg(Actions), false);

  return Actions.ActOnFinishFunctionBody(Decl, move(FnBody));
}

/// ParseFunctionTryBlock - Parse a C++ function-try-block.
///
///       function-try-block:
///         'try' ctor-initializer[opt] compound-statement handler-seq
///
Parser::DeclPtrTy Parser::ParseFunctionTryBlock(DeclPtrTy Decl) {
  assert(Tok.is(tok::kw_try) && "Expected 'try'");
  SourceLocation TryLoc = ConsumeToken();

  PrettyStackTraceActionsDecl CrashInfo(Decl, TryLoc, Actions,
                                        PP.getSourceManager(),
                                        "parsing function try block");

  // Constructor initializer list?
  if (Tok.is(tok::colon))
    ParseConstructorInitializer(Decl);

  SourceLocation LBraceLoc = Tok.getLocation();
  OwningStmtResult FnBody(ParseCXXTryBlockCommon(TryLoc));
  // If we failed to parse the try-catch, we just give the function an empty
  // compound statement as the body.
  if (FnBody.isInvalid())
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc,
                                       MultiStmtArg(Actions), false);

  return Actions.ActOnFinishFunctionBody(Decl, move(FnBody));
}

/// ParseCXXTryBlock - Parse a C++ try-block.
///
///       try-block:
///         'try' compound-statement handler-seq
///
Parser::OwningStmtResult Parser::ParseCXXTryBlock(AttributeList* Attr) {
  // FIXME: Add attributes?
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
Parser::OwningStmtResult Parser::ParseCXXTryBlockCommon(SourceLocation TryLoc) {
  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));
  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?
  OwningStmtResult TryBlock(ParseCompoundStatement(0));
  if (TryBlock.isInvalid())
    return move(TryBlock);

  StmtVector Handlers(Actions);
  if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier()) {
    CXX0XAttributeList Attr = ParseCXX0XAttributes();
    Diag(Attr.Range.getBegin(), diag::err_attributes_not_allowed)
      << Attr.Range;
  }
  if (Tok.isNot(tok::kw_catch))
    return StmtError(Diag(Tok, diag::err_expected_catch));
  while (Tok.is(tok::kw_catch)) {
    OwningStmtResult Handler(ParseCXXCatchBlock());
    if (!Handler.isInvalid())
      Handlers.push_back(Handler.release());
  }
  // Don't bother creating the full statement if we don't have any usable
  // handlers.
  if (Handlers.empty())
    return StmtError();

  return Actions.ActOnCXXTryBlock(TryLoc, move(TryBlock), move_arg(Handlers));
}

/// ParseCXXCatchBlock - Parse a C++ catch block, called handler in the standard
///
///       handler:
///         'catch' '(' exception-declaration ')' compound-statement
///
///       exception-declaration:
///         type-specifier-seq declarator
///         type-specifier-seq abstract-declarator
///         type-specifier-seq
///         '...'
///
Parser::OwningStmtResult Parser::ParseCXXCatchBlock() {
  assert(Tok.is(tok::kw_catch) && "Expected 'catch'");

  SourceLocation CatchLoc = ConsumeToken();

  SourceLocation LParenLoc = Tok.getLocation();
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen))
    return StmtError();

  // C++ 3.3.2p3:
  // The name in a catch exception-declaration is local to the handler and
  // shall not be redeclared in the outermost block of the handler.
  ParseScope CatchScope(this, Scope::DeclScope | Scope::ControlScope);

  // exception-declaration is equivalent to '...' or a parameter-declaration
  // without default arguments.
  DeclPtrTy ExceptionDecl;
  if (Tok.isNot(tok::ellipsis)) {
    DeclSpec DS;
    if (ParseCXXTypeSpecifierSeq(DS))
      return StmtError();
    Declarator ExDecl(DS, Declarator::CXXCatchContext);
    ParseDeclarator(ExDecl);
    ExceptionDecl = Actions.ActOnExceptionDeclarator(CurScope, ExDecl);
  } else
    ConsumeToken();

  if (MatchRHSPunctuation(tok::r_paren, LParenLoc).isInvalid())
    return StmtError();

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));

  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?
  OwningStmtResult Block(ParseCompoundStatement(0));
  if (Block.isInvalid())
    return move(Block);

  return Actions.ActOnCXXCatchBlock(CatchLoc, ExceptionDecl, move(Block));
}
