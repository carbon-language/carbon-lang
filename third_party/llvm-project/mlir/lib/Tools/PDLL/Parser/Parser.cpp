//===- Parser.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "Lexer.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Diagnostic.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/AST/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ScopedPrinter.h"
#include <string>

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

namespace {
class Parser {
public:
  Parser(ast::Context &ctx, llvm::SourceMgr &sourceMgr)
      : ctx(ctx), lexer(sourceMgr, ctx.getDiagEngine()),
        curToken(lexer.lexToken()), curDeclScope(nullptr),
        valueTy(ast::ValueType::get(ctx)),
        valueRangeTy(ast::ValueRangeType::get(ctx)),
        typeTy(ast::TypeType::get(ctx)),
        typeRangeTy(ast::TypeRangeType::get(ctx)) {}

  /// Try to parse a new module. Returns nullptr in the case of failure.
  FailureOr<ast::Module *> parseModule();

private:
  /// The current context of the parser. It allows for the parser to know a bit
  /// about the construct it is nested within during parsing. This is used
  /// specifically to provide additional verification during parsing, e.g. to
  /// prevent using rewrites within a match context, matcher constraints within
  /// a rewrite section, etc.
  enum class ParserContext {
    /// The parser is in the global context.
    Global,
    /// The parser is currently within a Constraint, which disallows all types
    /// of rewrites (e.g. `erase`, `replace`, calls to Rewrites, etc.).
    Constraint,
    /// The parser is currently within the matcher portion of a Pattern, which
    /// is allows a terminal operation rewrite statement but no other rewrite
    /// transformations.
    PatternMatch,
    /// The parser is currently within a Rewrite, which disallows calls to
    /// constraints, requires operation expressions to have names, etc.
    Rewrite,
  };

  //===--------------------------------------------------------------------===//
  // Parsing
  //===--------------------------------------------------------------------===//

  /// Push a new decl scope onto the lexer.
  ast::DeclScope *pushDeclScope() {
    ast::DeclScope *newScope =
        new (scopeAllocator.Allocate()) ast::DeclScope(curDeclScope);
    return (curDeclScope = newScope);
  }
  void pushDeclScope(ast::DeclScope *scope) { curDeclScope = scope; }

  /// Pop the last decl scope from the lexer.
  void popDeclScope() { curDeclScope = curDeclScope->getParentScope(); }

  /// Parse the body of an AST module.
  LogicalResult parseModuleBody(SmallVector<ast::Decl *> &decls);

  /// Try to convert the given expression to `type`. Returns failure and emits
  /// an error if a conversion is not viable. On failure, `noteAttachFn` is
  /// invoked to attach notes to the emitted error diagnostic. On success,
  /// `expr` is updated to the expression used to convert to `type`.
  LogicalResult convertExpressionTo(
      ast::Expr *&expr, ast::Type type,
      function_ref<void(ast::Diagnostic &diag)> noteAttachFn = {});

  /// Given an operation expression, convert it to a Value or ValueRange
  /// typed expression.
  ast::Expr *convertOpToValue(const ast::Expr *opExpr);

  //===--------------------------------------------------------------------===//
  // Directives

  LogicalResult parseDirective(SmallVector<ast::Decl *> &decls);
  LogicalResult parseInclude(SmallVector<ast::Decl *> &decls);

  //===--------------------------------------------------------------------===//
  // Decls

  /// This structure contains the set of pattern metadata that may be parsed.
  struct ParsedPatternMetadata {
    Optional<uint16_t> benefit;
    bool hasBoundedRecursion = false;
  };

  FailureOr<ast::Decl *> parseTopLevelDecl();
  FailureOr<ast::NamedAttributeDecl *> parseNamedAttributeDecl();

  /// Parse an argument variable as part of the signature of a
  /// UserConstraintDecl or UserRewriteDecl.
  FailureOr<ast::VariableDecl *> parseArgumentDecl();

  /// Parse a result variable as part of the signature of a UserConstraintDecl
  /// or UserRewriteDecl.
  FailureOr<ast::VariableDecl *> parseResultDecl(unsigned resultNum);

  /// Parse a UserConstraintDecl. `isInline` signals if the constraint is being
  /// defined in a non-global context.
  FailureOr<ast::UserConstraintDecl *>
  parseUserConstraintDecl(bool isInline = false);

  /// Parse an inline UserConstraintDecl. An inline decl is one defined in a
  /// non-global context, such as within a Pattern/Constraint/etc.
  FailureOr<ast::UserConstraintDecl *> parseInlineUserConstraintDecl();

  /// Parse a PDLL (i.e. non-native) UserRewriteDecl whose body is defined using
  /// PDLL constructs.
  FailureOr<ast::UserConstraintDecl *> parseUserPDLLConstraintDecl(
      const ast::Name &name, bool isInline,
      ArrayRef<ast::VariableDecl *> arguments, ast::DeclScope *argumentScope,
      ArrayRef<ast::VariableDecl *> results, ast::Type resultType);

  /// Parse a parseUserRewriteDecl. `isInline` signals if the rewrite is being
  /// defined in a non-global context.
  FailureOr<ast::UserRewriteDecl *> parseUserRewriteDecl(bool isInline = false);

  /// Parse an inline UserRewriteDecl. An inline decl is one defined in a
  /// non-global context, such as within a Pattern/Rewrite/etc.
  FailureOr<ast::UserRewriteDecl *> parseInlineUserRewriteDecl();

  /// Parse a PDLL (i.e. non-native) UserRewriteDecl whose body is defined using
  /// PDLL constructs.
  FailureOr<ast::UserRewriteDecl *> parseUserPDLLRewriteDecl(
      const ast::Name &name, bool isInline,
      ArrayRef<ast::VariableDecl *> arguments, ast::DeclScope *argumentScope,
      ArrayRef<ast::VariableDecl *> results, ast::Type resultType);

  /// Parse either a UserConstraintDecl or UserRewriteDecl. These decls have
  /// effectively the same syntax, and only differ on slight semantics (given
  /// the different parsing contexts).
  template <typename T, typename ParseUserPDLLDeclFnT>
  FailureOr<T *> parseUserConstraintOrRewriteDecl(
      ParseUserPDLLDeclFnT &&parseUserPDLLFn, ParserContext declContext,
      StringRef anonymousNamePrefix, bool isInline);

  /// Parse a native (i.e. non-PDLL) UserConstraintDecl or UserRewriteDecl.
  /// These decls have effectively the same syntax.
  template <typename T>
  FailureOr<T *> parseUserNativeConstraintOrRewriteDecl(
      const ast::Name &name, bool isInline,
      ArrayRef<ast::VariableDecl *> arguments,
      ArrayRef<ast::VariableDecl *> results, ast::Type resultType);

  /// Parse the functional signature (i.e. the arguments and results) of a
  /// UserConstraintDecl or UserRewriteDecl.
  LogicalResult parseUserConstraintOrRewriteSignature(
      SmallVectorImpl<ast::VariableDecl *> &arguments,
      SmallVectorImpl<ast::VariableDecl *> &results,
      ast::DeclScope *&argumentScope, ast::Type &resultType);

  /// Validate the return (which if present is specified by bodyIt) of a
  /// UserConstraintDecl or UserRewriteDecl.
  LogicalResult validateUserConstraintOrRewriteReturn(
      StringRef declType, ast::CompoundStmt *body,
      ArrayRef<ast::Stmt *>::iterator bodyIt,
      ArrayRef<ast::Stmt *>::iterator bodyE,
      ArrayRef<ast::VariableDecl *> results, ast::Type &resultType);

  FailureOr<ast::CompoundStmt *>
  parseLambdaBody(function_ref<LogicalResult(ast::Stmt *&)> processStatementFn,
                  bool expectTerminalSemicolon = true);
  FailureOr<ast::CompoundStmt *> parsePatternLambdaBody();
  FailureOr<ast::Decl *> parsePatternDecl();
  LogicalResult parsePatternDeclMetadata(ParsedPatternMetadata &metadata);

  /// Check to see if a decl has already been defined with the given name, if
  /// one has emit and error and return failure. Returns success otherwise.
  LogicalResult checkDefineNamedDecl(const ast::Name &name);

  /// Try to define a variable decl with the given components, returns the
  /// variable on success.
  FailureOr<ast::VariableDecl *>
  defineVariableDecl(StringRef name, SMRange nameLoc, ast::Type type,
                     ast::Expr *initExpr,
                     ArrayRef<ast::ConstraintRef> constraints);
  FailureOr<ast::VariableDecl *>
  defineVariableDecl(StringRef name, SMRange nameLoc, ast::Type type,
                     ArrayRef<ast::ConstraintRef> constraints);

  /// Parse the constraint reference list for a variable decl.
  LogicalResult parseVariableDeclConstraintList(
      SmallVectorImpl<ast::ConstraintRef> &constraints);

  /// Parse the expression used within a type constraint, e.g. Attr<type-expr>.
  FailureOr<ast::Expr *> parseTypeConstraintExpr();

  /// Try to parse a single reference to a constraint. `typeConstraint` is the
  /// location of a previously parsed type constraint for the entity that will
  /// be constrained by the parsed constraint. `existingConstraints` are any
  /// existing constraints that have already been parsed for the same entity
  /// that will be constrained by this constraint. `allowInlineTypeConstraints`
  /// allows the use of inline Type constraints, e.g. `Value<valueType: Type>`.
  FailureOr<ast::ConstraintRef>
  parseConstraint(Optional<SMRange> &typeConstraint,
                  ArrayRef<ast::ConstraintRef> existingConstraints,
                  bool allowInlineTypeConstraints);

  /// Try to parse the constraint for a UserConstraintDecl/UserRewriteDecl
  /// argument or result variable. The constraints for these variables do not
  /// allow inline type constraints, and only permit a single constraint.
  FailureOr<ast::ConstraintRef> parseArgOrResultConstraint();

  //===--------------------------------------------------------------------===//
  // Exprs

  FailureOr<ast::Expr *> parseExpr();

  /// Identifier expressions.
  FailureOr<ast::Expr *> parseAttributeExpr();
  FailureOr<ast::Expr *> parseCallExpr(ast::Expr *parentExpr);
  FailureOr<ast::Expr *> parseDeclRefExpr(StringRef name, SMRange loc);
  FailureOr<ast::Expr *> parseIdentifierExpr();
  FailureOr<ast::Expr *> parseInlineConstraintLambdaExpr();
  FailureOr<ast::Expr *> parseInlineRewriteLambdaExpr();
  FailureOr<ast::Expr *> parseMemberAccessExpr(ast::Expr *parentExpr);
  FailureOr<ast::OpNameDecl *> parseOperationName(bool allowEmptyName = false);
  FailureOr<ast::OpNameDecl *> parseWrappedOperationName(bool allowEmptyName);
  FailureOr<ast::Expr *> parseOperationExpr();
  FailureOr<ast::Expr *> parseTupleExpr();
  FailureOr<ast::Expr *> parseTypeExpr();
  FailureOr<ast::Expr *> parseUnderscoreExpr();

  //===--------------------------------------------------------------------===//
  // Stmts

  FailureOr<ast::Stmt *> parseStmt(bool expectTerminalSemicolon = true);
  FailureOr<ast::CompoundStmt *> parseCompoundStmt();
  FailureOr<ast::EraseStmt *> parseEraseStmt();
  FailureOr<ast::LetStmt *> parseLetStmt();
  FailureOr<ast::ReplaceStmt *> parseReplaceStmt();
  FailureOr<ast::ReturnStmt *> parseReturnStmt();
  FailureOr<ast::RewriteStmt *> parseRewriteStmt();

  //===--------------------------------------------------------------------===//
  // Creation+Analysis
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Decls

  /// Try to extract a callable from the given AST node. Returns nullptr on
  /// failure.
  ast::CallableDecl *tryExtractCallableDecl(ast::Node *node);

  /// Try to create a pattern decl with the given components, returning the
  /// Pattern on success.
  FailureOr<ast::PatternDecl *>
  createPatternDecl(SMRange loc, const ast::Name *name,
                    const ParsedPatternMetadata &metadata,
                    ast::CompoundStmt *body);

  /// Build the result type for a UserConstraintDecl/UserRewriteDecl given a set
  /// of results, defined as part of the signature.
  ast::Type
  createUserConstraintRewriteResultType(ArrayRef<ast::VariableDecl *> results);

  /// Create a PDLL (i.e. non-native) UserConstraintDecl or UserRewriteDecl.
  template <typename T>
  FailureOr<T *> createUserPDLLConstraintOrRewriteDecl(
      const ast::Name &name, ArrayRef<ast::VariableDecl *> arguments,
      ArrayRef<ast::VariableDecl *> results, ast::Type resultType,
      ast::CompoundStmt *body);

  /// Try to create a variable decl with the given components, returning the
  /// Variable on success.
  FailureOr<ast::VariableDecl *>
  createVariableDecl(StringRef name, SMRange loc, ast::Expr *initializer,
                     ArrayRef<ast::ConstraintRef> constraints);

  /// Create a variable for an argument or result defined as part of the
  /// signature of a UserConstraintDecl/UserRewriteDecl.
  FailureOr<ast::VariableDecl *>
  createArgOrResultVariableDecl(StringRef name, SMRange loc,
                                const ast::ConstraintRef &constraint);

  /// Validate the constraints used to constraint a variable decl.
  /// `inferredType` is the type of the variable inferred by the constraints
  /// within the list, and is updated to the most refined type as determined by
  /// the constraints. Returns success if the constraint list is valid, failure
  /// otherwise.
  LogicalResult
  validateVariableConstraints(ArrayRef<ast::ConstraintRef> constraints,
                              ast::Type &inferredType);
  /// Validate a single reference to a constraint. `inferredType` contains the
  /// currently inferred variabled type and is refined within the type defined
  /// by the constraint. Returns success if the constraint is valid, failure
  /// otherwise. If `allowNonCoreConstraints` is true, then complex (e.g. user
  /// defined constraints) may be used with the variable.
  LogicalResult validateVariableConstraint(const ast::ConstraintRef &ref,
                                           ast::Type &inferredType,
                                           bool allowNonCoreConstraints = true);
  LogicalResult validateTypeConstraintExpr(const ast::Expr *typeExpr);
  LogicalResult validateTypeRangeConstraintExpr(const ast::Expr *typeExpr);

  //===--------------------------------------------------------------------===//
  // Exprs

  FailureOr<ast::CallExpr *>
  createCallExpr(SMRange loc, ast::Expr *parentExpr,
                 MutableArrayRef<ast::Expr *> arguments);
  FailureOr<ast::DeclRefExpr *> createDeclRefExpr(SMRange loc, ast::Decl *decl);
  FailureOr<ast::DeclRefExpr *>
  createInlineVariableExpr(ast::Type type, StringRef name, SMRange loc,
                           ArrayRef<ast::ConstraintRef> constraints);
  FailureOr<ast::MemberAccessExpr *>
  createMemberAccessExpr(ast::Expr *parentExpr, StringRef name, SMRange loc);

  /// Validate the member access `name` into the given parent expression. On
  /// success, this also returns the type of the member accessed.
  FailureOr<ast::Type> validateMemberAccess(ast::Expr *parentExpr,
                                            StringRef name, SMRange loc);
  FailureOr<ast::OperationExpr *>
  createOperationExpr(SMRange loc, const ast::OpNameDecl *name,
                      MutableArrayRef<ast::Expr *> operands,
                      MutableArrayRef<ast::NamedAttributeDecl *> attributes,
                      MutableArrayRef<ast::Expr *> results);
  LogicalResult
  validateOperationOperands(SMRange loc, Optional<StringRef> name,
                            MutableArrayRef<ast::Expr *> operands);
  LogicalResult validateOperationResults(SMRange loc, Optional<StringRef> name,
                                         MutableArrayRef<ast::Expr *> results);
  LogicalResult
  validateOperationOperandsOrResults(SMRange loc, Optional<StringRef> name,
                                     MutableArrayRef<ast::Expr *> values,
                                     ast::Type singleTy, ast::Type rangeTy);
  FailureOr<ast::TupleExpr *> createTupleExpr(SMRange loc,
                                              ArrayRef<ast::Expr *> elements,
                                              ArrayRef<StringRef> elementNames);

  //===--------------------------------------------------------------------===//
  // Stmts

  FailureOr<ast::EraseStmt *> createEraseStmt(SMRange loc, ast::Expr *rootOp);
  FailureOr<ast::ReplaceStmt *>
  createReplaceStmt(SMRange loc, ast::Expr *rootOp,
                    MutableArrayRef<ast::Expr *> replValues);
  FailureOr<ast::RewriteStmt *>
  createRewriteStmt(SMRange loc, ast::Expr *rootOp,
                    ast::CompoundStmt *rewriteBody);

  //===--------------------------------------------------------------------===//
  // Lexer Utilities
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(Token::Kind kind) {
    if (curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    curToken = lexer.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is. This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(Token::Kind kind) {
    assert(curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// Reset the lexer to the location at the given position.
  void resetToken(SMRange tokLoc) {
    lexer.resetPointer(tokLoc.Start.getPointer());
    curToken = lexer.lexToken();
  }

  /// Consume the specified token if present and return success. On failure,
  /// output a diagnostic and return failure.
  LogicalResult parseToken(Token::Kind kind, const Twine &msg) {
    if (curToken.getKind() != kind)
      return emitError(curToken.getLoc(), msg);
    consumeToken();
    return success();
  }
  LogicalResult emitError(SMRange loc, const Twine &msg) {
    lexer.emitError(loc, msg);
    return failure();
  }
  LogicalResult emitError(const Twine &msg) {
    return emitError(curToken.getLoc(), msg);
  }
  LogicalResult emitErrorAndNote(SMRange loc, const Twine &msg, SMRange noteLoc,
                                 const Twine &note) {
    lexer.emitErrorAndNote(loc, msg, noteLoc, note);
    return failure();
  }

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The owning AST context.
  ast::Context &ctx;

  /// The lexer of this parser.
  Lexer lexer;

  /// The current token within the lexer.
  Token curToken;

  /// The most recently defined decl scope.
  ast::DeclScope *curDeclScope;
  llvm::SpecificBumpPtrAllocator<ast::DeclScope> scopeAllocator;

  /// The current context of the parser.
  ParserContext parserContext = ParserContext::Global;

  /// Cached types to simplify verification and expression creation.
  ast::Type valueTy, valueRangeTy;
  ast::Type typeTy, typeRangeTy;

  /// A counter used when naming anonymous constraints and rewrites.
  unsigned anonymousDeclNameCounter = 0;
};
} // namespace

FailureOr<ast::Module *> Parser::parseModule() {
  SMLoc moduleLoc = curToken.getStartLoc();
  pushDeclScope();

  // Parse the top-level decls of the module.
  SmallVector<ast::Decl *> decls;
  if (failed(parseModuleBody(decls)))
    return popDeclScope(), failure();

  popDeclScope();
  return ast::Module::create(ctx, moduleLoc, decls);
}

LogicalResult Parser::parseModuleBody(SmallVector<ast::Decl *> &decls) {
  while (curToken.isNot(Token::eof)) {
    if (curToken.is(Token::directive)) {
      if (failed(parseDirective(decls)))
        return failure();
      continue;
    }

    FailureOr<ast::Decl *> decl = parseTopLevelDecl();
    if (failed(decl))
      return failure();
    decls.push_back(*decl);
  }
  return success();
}

ast::Expr *Parser::convertOpToValue(const ast::Expr *opExpr) {
  return ast::AllResultsMemberAccessExpr::create(ctx, opExpr->getLoc(), opExpr,
                                                 valueRangeTy);
}

LogicalResult Parser::convertExpressionTo(
    ast::Expr *&expr, ast::Type type,
    function_ref<void(ast::Diagnostic &diag)> noteAttachFn) {
  ast::Type exprType = expr->getType();
  if (exprType == type)
    return success();

  auto emitConvertError = [&]() -> ast::InFlightDiagnostic {
    ast::InFlightDiagnostic diag = ctx.getDiagEngine().emitError(
        expr->getLoc(), llvm::formatv("unable to convert expression of type "
                                      "`{0}` to the expected type of "
                                      "`{1}`",
                                      exprType, type));
    if (noteAttachFn)
      noteAttachFn(*diag);
    return diag;
  };

  if (auto exprOpType = exprType.dyn_cast<ast::OperationType>()) {
    // Two operation types are compatible if they have the same name, or if the
    // expected type is more general.
    if (auto opType = type.dyn_cast<ast::OperationType>()) {
      if (opType.getName())
        return emitConvertError();
      return success();
    }

    // An operation can always convert to a ValueRange.
    if (type == valueRangeTy) {
      expr = ast::AllResultsMemberAccessExpr::create(ctx, expr->getLoc(), expr,
                                                     valueRangeTy);
      return success();
    }

    // Allow conversion to a single value by constraining the result range.
    if (type == valueTy) {
      expr = ast::AllResultsMemberAccessExpr::create(ctx, expr->getLoc(), expr,
                                                     valueTy);
      return success();
    }
    return emitConvertError();
  }

  // FIXME: Decide how to allow/support converting a single result to multiple,
  // and multiple to a single result. For now, we just allow Single->Range,
  // but this isn't something really supported in the PDL dialect. We should
  // figure out some way to support both.
  if ((exprType == valueTy || exprType == valueRangeTy) &&
      (type == valueTy || type == valueRangeTy))
    return success();
  if ((exprType == typeTy || exprType == typeRangeTy) &&
      (type == typeTy || type == typeRangeTy))
    return success();

  // Handle tuple types.
  if (auto exprTupleType = exprType.dyn_cast<ast::TupleType>()) {
    auto tupleType = type.dyn_cast<ast::TupleType>();
    if (!tupleType || tupleType.size() != exprTupleType.size())
      return emitConvertError();

    // Build a new tuple expression using each of the elements of the current
    // tuple.
    SmallVector<ast::Expr *> newExprs;
    for (unsigned i = 0, e = exprTupleType.size(); i < e; ++i) {
      newExprs.push_back(ast::MemberAccessExpr::create(
          ctx, expr->getLoc(), expr, llvm::to_string(i),
          exprTupleType.getElementTypes()[i]));

      auto diagFn = [&](ast::Diagnostic &diag) {
        diag.attachNote(llvm::formatv("when converting element #{0} of `{1}`",
                                      i, exprTupleType));
        if (noteAttachFn)
          noteAttachFn(diag);
      };
      if (failed(convertExpressionTo(newExprs.back(),
                                     tupleType.getElementTypes()[i], diagFn)))
        return failure();
    }
    expr = ast::TupleExpr::create(ctx, expr->getLoc(), newExprs,
                                  tupleType.getElementNames());
    return success();
  }

  return emitConvertError();
}

//===----------------------------------------------------------------------===//
// Directives

LogicalResult Parser::parseDirective(SmallVector<ast::Decl *> &decls) {
  StringRef directive = curToken.getSpelling();
  if (directive == "#include")
    return parseInclude(decls);

  return emitError("unknown directive `" + directive + "`");
}

LogicalResult Parser::parseInclude(SmallVector<ast::Decl *> &decls) {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::directive);

  // Parse the file being included.
  if (!curToken.isString())
    return emitError(loc,
                     "expected string file name after `include` directive");
  SMRange fileLoc = curToken.getLoc();
  std::string filenameStr = curToken.getStringValue();
  StringRef filename = filenameStr;
  consumeToken();

  // Check the type of include. If ending with `.pdll`, this is another pdl file
  // to be parsed along with the current module.
  if (filename.endswith(".pdll")) {
    if (failed(lexer.pushInclude(filename)))
      return emitError(fileLoc,
                       "unable to open include file `" + filename + "`");

    // If we added the include successfully, parse it into the current module.
    // Make sure to save the current token so that we can restore it when we
    // finish parsing the nested file.
    Token oldToken = curToken;
    curToken = lexer.lexToken();
    LogicalResult result = parseModuleBody(decls);
    curToken = oldToken;
    return result;
  }

  return emitError(fileLoc, "expected include filename to end with `.pdll`");
}

//===----------------------------------------------------------------------===//
// Decls

FailureOr<ast::Decl *> Parser::parseTopLevelDecl() {
  FailureOr<ast::Decl *> decl;
  switch (curToken.getKind()) {
  case Token::kw_Constraint:
    decl = parseUserConstraintDecl();
    break;
  case Token::kw_Pattern:
    decl = parsePatternDecl();
    break;
  case Token::kw_Rewrite:
    decl = parseUserRewriteDecl();
    break;
  default:
    return emitError("expected top-level declaration, such as a `Pattern`");
  }
  if (failed(decl))
    return failure();

  // If the decl has a name, add it to the current scope.
  if (const ast::Name *name = (*decl)->getName()) {
    if (failed(checkDefineNamedDecl(*name)))
      return failure();
    curDeclScope->add(*decl);
  }
  return decl;
}

FailureOr<ast::NamedAttributeDecl *> Parser::parseNamedAttributeDecl() {
  std::string attrNameStr;
  if (curToken.isString())
    attrNameStr = curToken.getStringValue();
  else if (curToken.is(Token::identifier) || curToken.isKeyword())
    attrNameStr = curToken.getSpelling().str();
  else
    return emitError("expected identifier or string attribute name");
  const auto &name = ast::Name::create(ctx, attrNameStr, curToken.getLoc());
  consumeToken();

  // Check for a value of the attribute.
  ast::Expr *attrValue = nullptr;
  if (consumeIf(Token::equal)) {
    FailureOr<ast::Expr *> attrExpr = parseExpr();
    if (failed(attrExpr))
      return failure();
    attrValue = *attrExpr;
  } else {
    // If there isn't a concrete value, create an expression representing a
    // UnitAttr.
    attrValue = ast::AttributeExpr::create(ctx, name.getLoc(), "unit");
  }

  return ast::NamedAttributeDecl::create(ctx, name, attrValue);
}

FailureOr<ast::CompoundStmt *> Parser::parseLambdaBody(
    function_ref<LogicalResult(ast::Stmt *&)> processStatementFn,
    bool expectTerminalSemicolon) {
  consumeToken(Token::equal_arrow);

  // Parse the single statement of the lambda body.
  SMLoc bodyStartLoc = curToken.getStartLoc();
  pushDeclScope();
  FailureOr<ast::Stmt *> singleStatement = parseStmt(expectTerminalSemicolon);
  bool failedToParse =
      failed(singleStatement) || failed(processStatementFn(*singleStatement));
  popDeclScope();
  if (failedToParse)
    return failure();

  SMRange bodyLoc(bodyStartLoc, curToken.getStartLoc());
  return ast::CompoundStmt::create(ctx, bodyLoc, *singleStatement);
}

FailureOr<ast::VariableDecl *> Parser::parseArgumentDecl() {
  // Ensure that the argument is named.
  if (curToken.isNot(Token::identifier) && !curToken.isDependentKeyword())
    return emitError("expected identifier argument name");

  // Parse the argument similarly to a normal variable.
  StringRef name = curToken.getSpelling();
  SMRange nameLoc = curToken.getLoc();
  consumeToken();

  if (failed(
          parseToken(Token::colon, "expected `:` before argument constraint")))
    return failure();

  FailureOr<ast::ConstraintRef> cst = parseArgOrResultConstraint();
  if (failed(cst))
    return failure();

  return createArgOrResultVariableDecl(name, nameLoc, *cst);
}

FailureOr<ast::VariableDecl *> Parser::parseResultDecl(unsigned resultNum) {
  // Check to see if this result is named.
  if (curToken.is(Token::identifier) || curToken.isDependentKeyword()) {
    // Check to see if this name actually refers to a Constraint.
    ast::Decl *existingDecl = curDeclScope->lookup(curToken.getSpelling());
    if (isa_and_nonnull<ast::ConstraintDecl>(existingDecl)) {
      // If yes, and this is a Rewrite, give a nice error message as non-Core
      // constraints are not supported on Rewrite results.
      if (parserContext == ParserContext::Rewrite) {
        return emitError(
            "`Rewrite` results are only permitted to use core constraints, "
            "such as `Attr`, `Op`, `Type`, `TypeRange`, `Value`, `ValueRange`");
      }

      // Otherwise, parse this as an unnamed result variable.
    } else {
      // If it wasn't a constraint, parse the result similarly to a variable. If
      // there is already an existing decl, we will emit an error when defining
      // this variable later.
      StringRef name = curToken.getSpelling();
      SMRange nameLoc = curToken.getLoc();
      consumeToken();

      if (failed(parseToken(Token::colon,
                            "expected `:` before result constraint")))
        return failure();

      FailureOr<ast::ConstraintRef> cst = parseArgOrResultConstraint();
      if (failed(cst))
        return failure();

      return createArgOrResultVariableDecl(name, nameLoc, *cst);
    }
  }

  // If it isn't named, we parse the constraint directly and create an unnamed
  // result variable.
  FailureOr<ast::ConstraintRef> cst = parseArgOrResultConstraint();
  if (failed(cst))
    return failure();

  return createArgOrResultVariableDecl("", cst->referenceLoc, *cst);
}

FailureOr<ast::UserConstraintDecl *>
Parser::parseUserConstraintDecl(bool isInline) {
  // Constraints and rewrites have very similar formats, dispatch to a shared
  // interface for parsing.
  return parseUserConstraintOrRewriteDecl<ast::UserConstraintDecl>(
      [&](auto &&...args) {
        return this->parseUserPDLLConstraintDecl(args...);
      },
      ParserContext::Constraint, "constraint", isInline);
}

FailureOr<ast::UserConstraintDecl *> Parser::parseInlineUserConstraintDecl() {
  FailureOr<ast::UserConstraintDecl *> decl =
      parseUserConstraintDecl(/*isInline=*/true);
  if (failed(decl) || failed(checkDefineNamedDecl((*decl)->getName())))
    return failure();

  curDeclScope->add(*decl);
  return decl;
}

FailureOr<ast::UserConstraintDecl *> Parser::parseUserPDLLConstraintDecl(
    const ast::Name &name, bool isInline,
    ArrayRef<ast::VariableDecl *> arguments, ast::DeclScope *argumentScope,
    ArrayRef<ast::VariableDecl *> results, ast::Type resultType) {
  // Push the argument scope back onto the list, so that the body can
  // reference arguments.
  pushDeclScope(argumentScope);

  // Parse the body of the constraint. The body is either defined as a compound
  // block, i.e. `{ ... }`, or a lambda body, i.e. `=> <expr>`.
  ast::CompoundStmt *body;
  if (curToken.is(Token::equal_arrow)) {
    FailureOr<ast::CompoundStmt *> bodyResult = parseLambdaBody(
        [&](ast::Stmt *&stmt) -> LogicalResult {
          ast::Expr *stmtExpr = dyn_cast<ast::Expr>(stmt);
          if (!stmtExpr) {
            return emitError(stmt->getLoc(),
                             "expected `Constraint` lambda body to contain a "
                             "single expression");
          }
          stmt = ast::ReturnStmt::create(ctx, stmt->getLoc(), stmtExpr);
          return success();
        },
        /*expectTerminalSemicolon=*/!isInline);
    if (failed(bodyResult))
      return failure();
    body = *bodyResult;
  } else {
    FailureOr<ast::CompoundStmt *> bodyResult = parseCompoundStmt();
    if (failed(bodyResult))
      return failure();
    body = *bodyResult;

    // Verify the structure of the body.
    auto bodyIt = body->begin(), bodyE = body->end();
    for (; bodyIt != bodyE; ++bodyIt)
      if (isa<ast::ReturnStmt>(*bodyIt))
        break;
    if (failed(validateUserConstraintOrRewriteReturn(
            "Constraint", body, bodyIt, bodyE, results, resultType)))
      return failure();
  }
  popDeclScope();

  return createUserPDLLConstraintOrRewriteDecl<ast::UserConstraintDecl>(
      name, arguments, results, resultType, body);
}

FailureOr<ast::UserRewriteDecl *> Parser::parseUserRewriteDecl(bool isInline) {
  // Constraints and rewrites have very similar formats, dispatch to a shared
  // interface for parsing.
  return parseUserConstraintOrRewriteDecl<ast::UserRewriteDecl>(
      [&](auto &&...args) { return this->parseUserPDLLRewriteDecl(args...); },
      ParserContext::Rewrite, "rewrite", isInline);
}

FailureOr<ast::UserRewriteDecl *> Parser::parseInlineUserRewriteDecl() {
  FailureOr<ast::UserRewriteDecl *> decl =
      parseUserRewriteDecl(/*isInline=*/true);
  if (failed(decl) || failed(checkDefineNamedDecl((*decl)->getName())))
    return failure();

  curDeclScope->add(*decl);
  return decl;
}

FailureOr<ast::UserRewriteDecl *> Parser::parseUserPDLLRewriteDecl(
    const ast::Name &name, bool isInline,
    ArrayRef<ast::VariableDecl *> arguments, ast::DeclScope *argumentScope,
    ArrayRef<ast::VariableDecl *> results, ast::Type resultType) {
  // Push the argument scope back onto the list, so that the body can
  // reference arguments.
  curDeclScope = argumentScope;
  ast::CompoundStmt *body;
  if (curToken.is(Token::equal_arrow)) {
    FailureOr<ast::CompoundStmt *> bodyResult = parseLambdaBody(
        [&](ast::Stmt *&statement) -> LogicalResult {
          if (isa<ast::OpRewriteStmt>(statement))
            return success();

          ast::Expr *statementExpr = dyn_cast<ast::Expr>(statement);
          if (!statementExpr) {
            return emitError(
                statement->getLoc(),
                "expected `Rewrite` lambda body to contain a single expression "
                "or an operation rewrite statement; such as `erase`, "
                "`replace`, or `rewrite`");
          }
          statement =
              ast::ReturnStmt::create(ctx, statement->getLoc(), statementExpr);
          return success();
        },
        /*expectTerminalSemicolon=*/!isInline);
    if (failed(bodyResult))
      return failure();
    body = *bodyResult;
  } else {
    FailureOr<ast::CompoundStmt *> bodyResult = parseCompoundStmt();
    if (failed(bodyResult))
      return failure();
    body = *bodyResult;
  }
  popDeclScope();

  // Verify the structure of the body.
  auto bodyIt = body->begin(), bodyE = body->end();
  for (; bodyIt != bodyE; ++bodyIt)
    if (isa<ast::ReturnStmt>(*bodyIt))
      break;
  if (failed(validateUserConstraintOrRewriteReturn("Rewrite", body, bodyIt,
                                                   bodyE, results, resultType)))
    return failure();
  return createUserPDLLConstraintOrRewriteDecl<ast::UserRewriteDecl>(
      name, arguments, results, resultType, body);
}

template <typename T, typename ParseUserPDLLDeclFnT>
FailureOr<T *> Parser::parseUserConstraintOrRewriteDecl(
    ParseUserPDLLDeclFnT &&parseUserPDLLFn, ParserContext declContext,
    StringRef anonymousNamePrefix, bool isInline) {
  SMRange loc = curToken.getLoc();
  consumeToken();
  llvm::SaveAndRestore<ParserContext> saveCtx(parserContext, declContext);

  // Parse the name of the decl.
  const ast::Name *name = nullptr;
  if (curToken.isNot(Token::identifier)) {
    // Only inline decls can be un-named. Inline decls are similar to "lambdas"
    // in C++, so being unnamed is fine.
    if (!isInline)
      return emitError("expected identifier name");

    // Create a unique anonymous name to use, as the name for this decl is not
    // important.
    std::string anonName =
        llvm::formatv("<anonymous_{0}_{1}>", anonymousNamePrefix,
                      anonymousDeclNameCounter++)
            .str();
    name = &ast::Name::create(ctx, anonName, loc);
  } else {
    // If a name was provided, we can use it directly.
    name = &ast::Name::create(ctx, curToken.getSpelling(), curToken.getLoc());
    consumeToken(Token::identifier);
  }

  // Parse the functional signature of the decl.
  SmallVector<ast::VariableDecl *> arguments, results;
  ast::DeclScope *argumentScope;
  ast::Type resultType;
  if (failed(parseUserConstraintOrRewriteSignature(arguments, results,
                                                   argumentScope, resultType)))
    return failure();

  // Check to see which type of constraint this is. If the constraint contains a
  // compound body, this is a PDLL decl.
  if (curToken.isAny(Token::l_brace, Token::equal_arrow))
    return parseUserPDLLFn(*name, isInline, arguments, argumentScope, results,
                           resultType);

  // Otherwise, this is a native decl.
  return parseUserNativeConstraintOrRewriteDecl<T>(*name, isInline, arguments,
                                                   results, resultType);
}

template <typename T>
FailureOr<T *> Parser::parseUserNativeConstraintOrRewriteDecl(
    const ast::Name &name, bool isInline,
    ArrayRef<ast::VariableDecl *> arguments,
    ArrayRef<ast::VariableDecl *> results, ast::Type resultType) {
  // If followed by a string, the native code body has also been specified.
  std::string codeStrStorage;
  Optional<StringRef> optCodeStr;
  if (curToken.isString()) {
    codeStrStorage = curToken.getStringValue();
    optCodeStr = codeStrStorage;
    consumeToken();
  } else if (isInline) {
    return emitError(name.getLoc(),
                     "external declarations must be declared in global scope");
  }
  if (failed(parseToken(Token::semicolon,
                        "expected `;` after native declaration")))
    return failure();
  return T::createNative(ctx, name, arguments, results, optCodeStr, resultType);
}

LogicalResult Parser::parseUserConstraintOrRewriteSignature(
    SmallVectorImpl<ast::VariableDecl *> &arguments,
    SmallVectorImpl<ast::VariableDecl *> &results,
    ast::DeclScope *&argumentScope, ast::Type &resultType) {
  // Parse the argument list of the decl.
  if (failed(parseToken(Token::l_paren, "expected `(` to start argument list")))
    return failure();

  argumentScope = pushDeclScope();
  if (curToken.isNot(Token::r_paren)) {
    do {
      FailureOr<ast::VariableDecl *> argument = parseArgumentDecl();
      if (failed(argument))
        return failure();
      arguments.emplace_back(*argument);
    } while (consumeIf(Token::comma));
  }
  popDeclScope();
  if (failed(parseToken(Token::r_paren, "expected `)` to end argument list")))
    return failure();

  // Parse the results of the decl.
  pushDeclScope();
  if (consumeIf(Token::arrow)) {
    auto parseResultFn = [&]() -> LogicalResult {
      FailureOr<ast::VariableDecl *> result = parseResultDecl(results.size());
      if (failed(result))
        return failure();
      results.emplace_back(*result);
      return success();
    };

    // Check for a list of results.
    if (consumeIf(Token::l_paren)) {
      do {
        if (failed(parseResultFn()))
          return failure();
      } while (consumeIf(Token::comma));
      if (failed(parseToken(Token::r_paren, "expected `)` to end result list")))
        return failure();

      // Otherwise, there is only one result.
    } else if (failed(parseResultFn())) {
      return failure();
    }
  }
  popDeclScope();

  // Compute the result type of the decl.
  resultType = createUserConstraintRewriteResultType(results);

  // Verify that results are only named if there are more than one.
  if (results.size() == 1 && !results.front()->getName().getName().empty()) {
    return emitError(
        results.front()->getLoc(),
        "cannot create a single-element tuple with an element label");
  }
  return success();
}

LogicalResult Parser::validateUserConstraintOrRewriteReturn(
    StringRef declType, ast::CompoundStmt *body,
    ArrayRef<ast::Stmt *>::iterator bodyIt,
    ArrayRef<ast::Stmt *>::iterator bodyE,
    ArrayRef<ast::VariableDecl *> results, ast::Type &resultType) {
  // Handle if a `return` was provided.
  if (bodyIt != bodyE) {
    // Emit an error if we have trailing statements after the return.
    if (std::next(bodyIt) != bodyE) {
      return emitError(
          (*std::next(bodyIt))->getLoc(),
          llvm::formatv("`return` terminated the `{0}` body, but found "
                        "trailing statements afterwards",
                        declType));
    }

    // Otherwise if a return wasn't provided, check that no results are
    // expected.
  } else if (!results.empty()) {
    return emitError(
        {body->getLoc().End, body->getLoc().End},
        llvm::formatv("missing return in a `{0}` expected to return `{1}`",
                      declType, resultType));
  }
  return success();
}

FailureOr<ast::CompoundStmt *> Parser::parsePatternLambdaBody() {
  return parseLambdaBody([&](ast::Stmt *&statement) -> LogicalResult {
    if (isa<ast::OpRewriteStmt>(statement))
      return success();
    return emitError(
        statement->getLoc(),
        "expected Pattern lambda body to contain a single operation "
        "rewrite statement, such as `erase`, `replace`, or `rewrite`");
  });
}

FailureOr<ast::Decl *> Parser::parsePatternDecl() {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_Pattern);
  llvm::SaveAndRestore<ParserContext> saveCtx(parserContext,
                                              ParserContext::PatternMatch);

  // Check for an optional identifier for the pattern name.
  const ast::Name *name = nullptr;
  if (curToken.is(Token::identifier)) {
    name = &ast::Name::create(ctx, curToken.getSpelling(), curToken.getLoc());
    consumeToken(Token::identifier);
  }

  // Parse any pattern metadata.
  ParsedPatternMetadata metadata;
  if (consumeIf(Token::kw_with) && failed(parsePatternDeclMetadata(metadata)))
    return failure();

  // Parse the pattern body.
  ast::CompoundStmt *body;

  // Handle a lambda body.
  if (curToken.is(Token::equal_arrow)) {
    FailureOr<ast::CompoundStmt *> bodyResult = parsePatternLambdaBody();
    if (failed(bodyResult))
      return failure();
    body = *bodyResult;
  } else {
    if (curToken.isNot(Token::l_brace))
      return emitError("expected `{` or `=>` to start pattern body");
    FailureOr<ast::CompoundStmt *> bodyResult = parseCompoundStmt();
    if (failed(bodyResult))
      return failure();
    body = *bodyResult;

    // Verify the body of the pattern.
    auto bodyIt = body->begin(), bodyE = body->end();
    for (; bodyIt != bodyE; ++bodyIt) {
      if (isa<ast::ReturnStmt>(*bodyIt)) {
        return emitError((*bodyIt)->getLoc(),
                         "`return` statements are only permitted within a "
                         "`Constraint` or `Rewrite` body");
      }
      // Break when we've found the rewrite statement.
      if (isa<ast::OpRewriteStmt>(*bodyIt))
        break;
    }
    if (bodyIt == bodyE) {
      return emitError(loc,
                       "expected Pattern body to terminate with an operation "
                       "rewrite statement, such as `erase`");
    }
    if (std::next(bodyIt) != bodyE) {
      return emitError((*std::next(bodyIt))->getLoc(),
                       "Pattern body was terminated by an operation "
                       "rewrite statement, but found trailing statements");
    }
  }

  return createPatternDecl(loc, name, metadata, body);
}

LogicalResult
Parser::parsePatternDeclMetadata(ParsedPatternMetadata &metadata) {
  Optional<SMRange> benefitLoc;
  Optional<SMRange> hasBoundedRecursionLoc;

  do {
    if (curToken.isNot(Token::identifier))
      return emitError("expected pattern metadata identifier");
    StringRef metadataStr = curToken.getSpelling();
    SMRange metadataLoc = curToken.getLoc();
    consumeToken(Token::identifier);

    // Parse the benefit metadata: benefit(<integer-value>)
    if (metadataStr == "benefit") {
      if (benefitLoc) {
        return emitErrorAndNote(metadataLoc,
                                "pattern benefit has already been specified",
                                *benefitLoc, "see previous definition here");
      }
      if (failed(parseToken(Token::l_paren,
                            "expected `(` before pattern benefit")))
        return failure();

      uint16_t benefitValue = 0;
      if (curToken.isNot(Token::integer))
        return emitError("expected integral pattern benefit");
      if (curToken.getSpelling().getAsInteger(/*Radix=*/10, benefitValue))
        return emitError(
            "expected pattern benefit to fit within a 16-bit integer");
      consumeToken(Token::integer);

      metadata.benefit = benefitValue;
      benefitLoc = metadataLoc;

      if (failed(
              parseToken(Token::r_paren, "expected `)` after pattern benefit")))
        return failure();
      continue;
    }

    // Parse the bounded recursion metadata: recursion
    if (metadataStr == "recursion") {
      if (hasBoundedRecursionLoc) {
        return emitErrorAndNote(
            metadataLoc,
            "pattern recursion metadata has already been specified",
            *hasBoundedRecursionLoc, "see previous definition here");
      }
      metadata.hasBoundedRecursion = true;
      hasBoundedRecursionLoc = metadataLoc;
      continue;
    }

    return emitError(metadataLoc, "unknown pattern metadata");
  } while (consumeIf(Token::comma));

  return success();
}

FailureOr<ast::Expr *> Parser::parseTypeConstraintExpr() {
  consumeToken(Token::less);

  FailureOr<ast::Expr *> typeExpr = parseExpr();
  if (failed(typeExpr) ||
      failed(parseToken(Token::greater,
                        "expected `>` after variable type constraint")))
    return failure();
  return typeExpr;
}

LogicalResult Parser::checkDefineNamedDecl(const ast::Name &name) {
  assert(curDeclScope && "defining decl outside of a decl scope");
  if (ast::Decl *lastDecl = curDeclScope->lookup(name.getName())) {
    return emitErrorAndNote(
        name.getLoc(), "`" + name.getName() + "` has already been defined",
        lastDecl->getName()->getLoc(), "see previous definition here");
  }
  return success();
}

FailureOr<ast::VariableDecl *>
Parser::defineVariableDecl(StringRef name, SMRange nameLoc, ast::Type type,
                           ast::Expr *initExpr,
                           ArrayRef<ast::ConstraintRef> constraints) {
  assert(curDeclScope && "defining variable outside of decl scope");
  const ast::Name &nameDecl = ast::Name::create(ctx, name, nameLoc);

  // If the name of the variable indicates a special variable, we don't add it
  // to the scope. This variable is local to the definition point.
  if (name.empty() || name == "_") {
    return ast::VariableDecl::create(ctx, nameDecl, type, initExpr,
                                     constraints);
  }
  if (failed(checkDefineNamedDecl(nameDecl)))
    return failure();

  auto *varDecl =
      ast::VariableDecl::create(ctx, nameDecl, type, initExpr, constraints);
  curDeclScope->add(varDecl);
  return varDecl;
}

FailureOr<ast::VariableDecl *>
Parser::defineVariableDecl(StringRef name, SMRange nameLoc, ast::Type type,
                           ArrayRef<ast::ConstraintRef> constraints) {
  return defineVariableDecl(name, nameLoc, type, /*initExpr=*/nullptr,
                            constraints);
}

LogicalResult Parser::parseVariableDeclConstraintList(
    SmallVectorImpl<ast::ConstraintRef> &constraints) {
  Optional<SMRange> typeConstraint;
  auto parseSingleConstraint = [&] {
    FailureOr<ast::ConstraintRef> constraint = parseConstraint(
        typeConstraint, constraints, /*allowInlineTypeConstraints=*/true);
    if (failed(constraint))
      return failure();
    constraints.push_back(*constraint);
    return success();
  };

  // Check to see if this is a single constraint, or a list.
  if (!consumeIf(Token::l_square))
    return parseSingleConstraint();

  do {
    if (failed(parseSingleConstraint()))
      return failure();
  } while (consumeIf(Token::comma));
  return parseToken(Token::r_square, "expected `]` after constraint list");
}

FailureOr<ast::ConstraintRef>
Parser::parseConstraint(Optional<SMRange> &typeConstraint,
                        ArrayRef<ast::ConstraintRef> existingConstraints,
                        bool allowInlineTypeConstraints) {
  auto parseTypeConstraint = [&](ast::Expr *&typeExpr) -> LogicalResult {
    if (!allowInlineTypeConstraints) {
      return emitError(
          curToken.getLoc(),
          "inline `Attr`, `Value`, and `ValueRange` type constraints are not "
          "permitted on arguments or results");
    }
    if (typeConstraint)
      return emitErrorAndNote(
          curToken.getLoc(),
          "the type of this variable has already been constrained",
          *typeConstraint, "see previous constraint location here");
    FailureOr<ast::Expr *> constraintExpr = parseTypeConstraintExpr();
    if (failed(constraintExpr))
      return failure();
    typeExpr = *constraintExpr;
    typeConstraint = typeExpr->getLoc();
    return success();
  };

  SMRange loc = curToken.getLoc();
  switch (curToken.getKind()) {
  case Token::kw_Attr: {
    consumeToken(Token::kw_Attr);

    // Check for a type constraint.
    ast::Expr *typeExpr = nullptr;
    if (curToken.is(Token::less) && failed(parseTypeConstraint(typeExpr)))
      return failure();
    return ast::ConstraintRef(
        ast::AttrConstraintDecl::create(ctx, loc, typeExpr), loc);
  }
  case Token::kw_Op: {
    consumeToken(Token::kw_Op);

    // Parse an optional operation name. If the name isn't provided, this refers
    // to "any" operation.
    FailureOr<ast::OpNameDecl *> opName =
        parseWrappedOperationName(/*allowEmptyName=*/true);
    if (failed(opName))
      return failure();

    return ast::ConstraintRef(ast::OpConstraintDecl::create(ctx, loc, *opName),
                              loc);
  }
  case Token::kw_Type:
    consumeToken(Token::kw_Type);
    return ast::ConstraintRef(ast::TypeConstraintDecl::create(ctx, loc), loc);
  case Token::kw_TypeRange:
    consumeToken(Token::kw_TypeRange);
    return ast::ConstraintRef(ast::TypeRangeConstraintDecl::create(ctx, loc),
                              loc);
  case Token::kw_Value: {
    consumeToken(Token::kw_Value);

    // Check for a type constraint.
    ast::Expr *typeExpr = nullptr;
    if (curToken.is(Token::less) && failed(parseTypeConstraint(typeExpr)))
      return failure();

    return ast::ConstraintRef(
        ast::ValueConstraintDecl::create(ctx, loc, typeExpr), loc);
  }
  case Token::kw_ValueRange: {
    consumeToken(Token::kw_ValueRange);

    // Check for a type constraint.
    ast::Expr *typeExpr = nullptr;
    if (curToken.is(Token::less) && failed(parseTypeConstraint(typeExpr)))
      return failure();

    return ast::ConstraintRef(
        ast::ValueRangeConstraintDecl::create(ctx, loc, typeExpr), loc);
  }

  case Token::kw_Constraint: {
    // Handle an inline constraint.
    FailureOr<ast::UserConstraintDecl *> decl = parseInlineUserConstraintDecl();
    if (failed(decl))
      return failure();
    return ast::ConstraintRef(*decl, loc);
  }
  case Token::identifier: {
    StringRef constraintName = curToken.getSpelling();
    consumeToken(Token::identifier);

    // Lookup the referenced constraint.
    ast::Decl *cstDecl = curDeclScope->lookup<ast::Decl>(constraintName);
    if (!cstDecl) {
      return emitError(loc, "unknown reference to constraint `" +
                                constraintName + "`");
    }

    // Handle a reference to a proper constraint.
    if (auto *cst = dyn_cast<ast::ConstraintDecl>(cstDecl))
      return ast::ConstraintRef(cst, loc);

    return emitErrorAndNote(
        loc, "invalid reference to non-constraint", cstDecl->getLoc(),
        "see the definition of `" + constraintName + "` here");
  }
  default:
    break;
  }
  return emitError(loc, "expected identifier constraint");
}

FailureOr<ast::ConstraintRef> Parser::parseArgOrResultConstraint() {
  Optional<SMRange> typeConstraint;
  return parseConstraint(typeConstraint, /*existingConstraints=*/llvm::None,
                         /*allowInlineTypeConstraints=*/false);
}

//===----------------------------------------------------------------------===//
// Exprs

FailureOr<ast::Expr *> Parser::parseExpr() {
  if (curToken.is(Token::underscore))
    return parseUnderscoreExpr();

  // Parse the LHS expression.
  FailureOr<ast::Expr *> lhsExpr;
  switch (curToken.getKind()) {
  case Token::kw_attr:
    lhsExpr = parseAttributeExpr();
    break;
  case Token::kw_Constraint:
    lhsExpr = parseInlineConstraintLambdaExpr();
    break;
  case Token::identifier:
    lhsExpr = parseIdentifierExpr();
    break;
  case Token::kw_op:
    lhsExpr = parseOperationExpr();
    break;
  case Token::kw_Rewrite:
    lhsExpr = parseInlineRewriteLambdaExpr();
    break;
  case Token::kw_type:
    lhsExpr = parseTypeExpr();
    break;
  case Token::l_paren:
    lhsExpr = parseTupleExpr();
    break;
  default:
    return emitError("expected expression");
  }
  if (failed(lhsExpr))
    return failure();

  // Check for an operator expression.
  while (true) {
    switch (curToken.getKind()) {
    case Token::dot:
      lhsExpr = parseMemberAccessExpr(*lhsExpr);
      break;
    case Token::l_paren:
      lhsExpr = parseCallExpr(*lhsExpr);
      break;
    default:
      return lhsExpr;
    }
    if (failed(lhsExpr))
      return failure();
  }
}

FailureOr<ast::Expr *> Parser::parseAttributeExpr() {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_attr);

  // If we aren't followed by a `<`, the `attr` keyword is treated as a normal
  // identifier.
  if (!consumeIf(Token::less)) {
    resetToken(loc);
    return parseIdentifierExpr();
  }

  if (!curToken.isString())
    return emitError("expected string literal containing MLIR attribute");
  std::string attrExpr = curToken.getStringValue();
  consumeToken();

  if (failed(
          parseToken(Token::greater, "expected `>` after attribute literal")))
    return failure();
  return ast::AttributeExpr::create(ctx, loc, attrExpr);
}

FailureOr<ast::Expr *> Parser::parseCallExpr(ast::Expr *parentExpr) {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::l_paren);

  // Parse the arguments of the call.
  SmallVector<ast::Expr *> arguments;
  if (curToken.isNot(Token::r_paren)) {
    do {
      FailureOr<ast::Expr *> argument = parseExpr();
      if (failed(argument))
        return failure();
      arguments.push_back(*argument);
    } while (consumeIf(Token::comma));
  }
  loc.End = curToken.getEndLoc();
  if (failed(parseToken(Token::r_paren, "expected `)` after argument list")))
    return failure();

  return createCallExpr(loc, parentExpr, arguments);
}

FailureOr<ast::Expr *> Parser::parseDeclRefExpr(StringRef name, SMRange loc) {
  ast::Decl *decl = curDeclScope->lookup(name);
  if (!decl)
    return emitError(loc, "undefined reference to `" + name + "`");

  return createDeclRefExpr(loc, decl);
}

FailureOr<ast::Expr *> Parser::parseIdentifierExpr() {
  StringRef name = curToken.getSpelling();
  SMRange nameLoc = curToken.getLoc();
  consumeToken();

  // Check to see if this is a decl ref expression that defines a variable
  // inline.
  if (consumeIf(Token::colon)) {
    SmallVector<ast::ConstraintRef> constraints;
    if (failed(parseVariableDeclConstraintList(constraints)))
      return failure();
    ast::Type type;
    if (failed(validateVariableConstraints(constraints, type)))
      return failure();
    return createInlineVariableExpr(type, name, nameLoc, constraints);
  }

  return parseDeclRefExpr(name, nameLoc);
}

FailureOr<ast::Expr *> Parser::parseInlineConstraintLambdaExpr() {
  FailureOr<ast::UserConstraintDecl *> decl = parseInlineUserConstraintDecl();
  if (failed(decl))
    return failure();

  return ast::DeclRefExpr::create(ctx, (*decl)->getLoc(), *decl,
                                  ast::ConstraintType::get(ctx));
}

FailureOr<ast::Expr *> Parser::parseInlineRewriteLambdaExpr() {
  FailureOr<ast::UserRewriteDecl *> decl = parseInlineUserRewriteDecl();
  if (failed(decl))
    return failure();

  return ast::DeclRefExpr::create(ctx, (*decl)->getLoc(), *decl,
                                  ast::RewriteType::get(ctx));
}

FailureOr<ast::Expr *> Parser::parseMemberAccessExpr(ast::Expr *parentExpr) {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::dot);

  // Parse the member name.
  Token memberNameTok = curToken;
  if (memberNameTok.isNot(Token::identifier, Token::integer) &&
      !memberNameTok.isKeyword())
    return emitError(loc, "expected identifier or numeric member name");
  StringRef memberName = memberNameTok.getSpelling();
  consumeToken();

  return createMemberAccessExpr(parentExpr, memberName, loc);
}

FailureOr<ast::OpNameDecl *> Parser::parseOperationName(bool allowEmptyName) {
  SMRange loc = curToken.getLoc();

  // Handle the case of an no operation name.
  if (curToken.isNot(Token::identifier) && !curToken.isKeyword()) {
    if (allowEmptyName)
      return ast::OpNameDecl::create(ctx, SMRange());
    return emitError("expected dialect namespace");
  }
  StringRef name = curToken.getSpelling();
  consumeToken();

  // Otherwise, this is a literal operation name.
  if (failed(parseToken(Token::dot, "expected `.` after dialect namespace")))
    return failure();

  if (curToken.isNot(Token::identifier) && !curToken.isKeyword())
    return emitError("expected operation name after dialect namespace");

  name = StringRef(name.data(), name.size() + 1);
  do {
    name = StringRef(name.data(), name.size() + curToken.getSpelling().size());
    loc.End = curToken.getEndLoc();
    consumeToken();
  } while (curToken.isAny(Token::identifier, Token::dot) ||
           curToken.isKeyword());
  return ast::OpNameDecl::create(ctx, ast::Name::create(ctx, name, loc));
}

FailureOr<ast::OpNameDecl *>
Parser::parseWrappedOperationName(bool allowEmptyName) {
  if (!consumeIf(Token::less))
    return ast::OpNameDecl::create(ctx, SMRange());

  FailureOr<ast::OpNameDecl *> opNameDecl = parseOperationName(allowEmptyName);
  if (failed(opNameDecl))
    return failure();

  if (failed(parseToken(Token::greater, "expected `>` after operation name")))
    return failure();
  return opNameDecl;
}

FailureOr<ast::Expr *> Parser::parseOperationExpr() {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_op);

  // If it isn't followed by a `<`, the `op` keyword is treated as a normal
  // identifier.
  if (curToken.isNot(Token::less)) {
    resetToken(loc);
    return parseIdentifierExpr();
  }

  // Parse the operation name. The name may be elided, in which case the
  // operation refers to "any" operation(i.e. a difference between `MyOp` and
  // `Operation*`). Operation names within a rewrite context must be named.
  bool allowEmptyName = parserContext != ParserContext::Rewrite;
  FailureOr<ast::OpNameDecl *> opNameDecl =
      parseWrappedOperationName(allowEmptyName);
  if (failed(opNameDecl))
    return failure();

  // Check for the optional list of operands.
  SmallVector<ast::Expr *> operands;
  if (consumeIf(Token::l_paren)) {
    do {
      FailureOr<ast::Expr *> operand = parseExpr();
      if (failed(operand))
        return failure();
      operands.push_back(*operand);
    } while (consumeIf(Token::comma));

    if (failed(parseToken(Token::r_paren,
                          "expected `)` after operation operand list")))
      return failure();
  }

  // Check for the optional list of attributes.
  SmallVector<ast::NamedAttributeDecl *> attributes;
  if (consumeIf(Token::l_brace)) {
    do {
      FailureOr<ast::NamedAttributeDecl *> decl = parseNamedAttributeDecl();
      if (failed(decl))
        return failure();
      attributes.emplace_back(*decl);
    } while (consumeIf(Token::comma));

    if (failed(parseToken(Token::r_brace,
                          "expected `}` after operation attribute list")))
      return failure();
  }

  // Check for the optional list of result types.
  SmallVector<ast::Expr *> resultTypes;
  if (consumeIf(Token::arrow)) {
    if (failed(parseToken(Token::l_paren,
                          "expected `(` before operation result type list")))
      return failure();

    do {
      FailureOr<ast::Expr *> resultTypeExpr = parseExpr();
      if (failed(resultTypeExpr))
        return failure();
      resultTypes.push_back(*resultTypeExpr);
    } while (consumeIf(Token::comma));

    if (failed(parseToken(Token::r_paren,
                          "expected `)` after operation result type list")))
      return failure();
  }

  return createOperationExpr(loc, *opNameDecl, operands, attributes,
                             resultTypes);
}

FailureOr<ast::Expr *> Parser::parseTupleExpr() {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::l_paren);

  DenseMap<StringRef, SMRange> usedNames;
  SmallVector<StringRef> elementNames;
  SmallVector<ast::Expr *> elements;
  if (curToken.isNot(Token::r_paren)) {
    do {
      // Check for the optional element name assignment before the value.
      StringRef elementName;
      if (curToken.is(Token::identifier) || curToken.isDependentKeyword()) {
        Token elementNameTok = curToken;
        consumeToken();

        // The element name is only present if followed by an `=`.
        if (consumeIf(Token::equal)) {
          elementName = elementNameTok.getSpelling();

          // Check to see if this name is already used.
          auto elementNameIt =
              usedNames.try_emplace(elementName, elementNameTok.getLoc());
          if (!elementNameIt.second) {
            return emitErrorAndNote(
                elementNameTok.getLoc(),
                llvm::formatv("duplicate tuple element label `{0}`",
                              elementName),
                elementNameIt.first->getSecond(),
                "see previous label use here");
          }
        } else {
          // Otherwise, we treat this as part of an expression so reset the
          // lexer.
          resetToken(elementNameTok.getLoc());
        }
      }
      elementNames.push_back(elementName);

      // Parse the tuple element value.
      FailureOr<ast::Expr *> element = parseExpr();
      if (failed(element))
        return failure();
      elements.push_back(*element);
    } while (consumeIf(Token::comma));
  }
  loc.End = curToken.getEndLoc();
  if (failed(
          parseToken(Token::r_paren, "expected `)` after tuple element list")))
    return failure();
  return createTupleExpr(loc, elements, elementNames);
}

FailureOr<ast::Expr *> Parser::parseTypeExpr() {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_type);

  // If we aren't followed by a `<`, the `type` keyword is treated as a normal
  // identifier.
  if (!consumeIf(Token::less)) {
    resetToken(loc);
    return parseIdentifierExpr();
  }

  if (!curToken.isString())
    return emitError("expected string literal containing MLIR type");
  std::string attrExpr = curToken.getStringValue();
  consumeToken();

  if (failed(parseToken(Token::greater, "expected `>` after type literal")))
    return failure();
  return ast::TypeExpr::create(ctx, loc, attrExpr);
}

FailureOr<ast::Expr *> Parser::parseUnderscoreExpr() {
  StringRef name = curToken.getSpelling();
  SMRange nameLoc = curToken.getLoc();
  consumeToken(Token::underscore);

  // Underscore expressions require a constraint list.
  if (failed(parseToken(Token::colon, "expected `:` after `_` variable")))
    return failure();

  // Parse the constraints for the expression.
  SmallVector<ast::ConstraintRef> constraints;
  if (failed(parseVariableDeclConstraintList(constraints)))
    return failure();

  ast::Type type;
  if (failed(validateVariableConstraints(constraints, type)))
    return failure();
  return createInlineVariableExpr(type, name, nameLoc, constraints);
}

//===----------------------------------------------------------------------===//
// Stmts

FailureOr<ast::Stmt *> Parser::parseStmt(bool expectTerminalSemicolon) {
  FailureOr<ast::Stmt *> stmt;
  switch (curToken.getKind()) {
  case Token::kw_erase:
    stmt = parseEraseStmt();
    break;
  case Token::kw_let:
    stmt = parseLetStmt();
    break;
  case Token::kw_replace:
    stmt = parseReplaceStmt();
    break;
  case Token::kw_return:
    stmt = parseReturnStmt();
    break;
  case Token::kw_rewrite:
    stmt = parseRewriteStmt();
    break;
  default:
    stmt = parseExpr();
    break;
  }
  if (failed(stmt) ||
      (expectTerminalSemicolon &&
       failed(parseToken(Token::semicolon, "expected `;` after statement"))))
    return failure();
  return stmt;
}

FailureOr<ast::CompoundStmt *> Parser::parseCompoundStmt() {
  SMLoc startLoc = curToken.getStartLoc();
  consumeToken(Token::l_brace);

  // Push a new block scope and parse any nested statements.
  pushDeclScope();
  SmallVector<ast::Stmt *> statements;
  while (curToken.isNot(Token::r_brace)) {
    FailureOr<ast::Stmt *> statement = parseStmt();
    if (failed(statement))
      return popDeclScope(), failure();
    statements.push_back(*statement);
  }
  popDeclScope();

  // Consume the end brace.
  SMRange location(startLoc, curToken.getEndLoc());
  consumeToken(Token::r_brace);

  return ast::CompoundStmt::create(ctx, location, statements);
}

FailureOr<ast::EraseStmt *> Parser::parseEraseStmt() {
  if (parserContext == ParserContext::Constraint)
    return emitError("`erase` cannot be used within a Constraint");
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_erase);

  // Parse the root operation expression.
  FailureOr<ast::Expr *> rootOp = parseExpr();
  if (failed(rootOp))
    return failure();

  return createEraseStmt(loc, *rootOp);
}

FailureOr<ast::LetStmt *> Parser::parseLetStmt() {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_let);

  // Parse the name of the new variable.
  SMRange varLoc = curToken.getLoc();
  if (curToken.isNot(Token::identifier) && !curToken.isDependentKeyword()) {
    // `_` is a reserved variable name.
    if (curToken.is(Token::underscore)) {
      return emitError(varLoc,
                       "`_` may only be used to define \"inline\" variables");
    }
    return emitError(varLoc,
                     "expected identifier after `let` to name a new variable");
  }
  StringRef varName = curToken.getSpelling();
  consumeToken();

  // Parse the optional set of constraints.
  SmallVector<ast::ConstraintRef> constraints;
  if (consumeIf(Token::colon) &&
      failed(parseVariableDeclConstraintList(constraints)))
    return failure();

  // Parse the optional initializer expression.
  ast::Expr *initializer = nullptr;
  if (consumeIf(Token::equal)) {
    FailureOr<ast::Expr *> initOrFailure = parseExpr();
    if (failed(initOrFailure))
      return failure();
    initializer = *initOrFailure;

    // Check that the constraints are compatible with having an initializer,
    // e.g. type constraints cannot be used with initializers.
    for (ast::ConstraintRef constraint : constraints) {
      LogicalResult result =
          TypeSwitch<const ast::Node *, LogicalResult>(constraint.constraint)
              .Case<ast::AttrConstraintDecl, ast::ValueConstraintDecl,
                    ast::ValueRangeConstraintDecl>([&](const auto *cst) {
                if (auto *typeConstraintExpr = cst->getTypeExpr()) {
                  return this->emitError(
                      constraint.referenceLoc,
                      "type constraints are not permitted on variables with "
                      "initializers");
                }
                return success();
              })
              .Default(success());
      if (failed(result))
        return failure();
    }
  }

  FailureOr<ast::VariableDecl *> varDecl =
      createVariableDecl(varName, varLoc, initializer, constraints);
  if (failed(varDecl))
    return failure();
  return ast::LetStmt::create(ctx, loc, *varDecl);
}

FailureOr<ast::ReplaceStmt *> Parser::parseReplaceStmt() {
  if (parserContext == ParserContext::Constraint)
    return emitError("`replace` cannot be used within a Constraint");
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_replace);

  // Parse the root operation expression.
  FailureOr<ast::Expr *> rootOp = parseExpr();
  if (failed(rootOp))
    return failure();

  if (failed(
          parseToken(Token::kw_with, "expected `with` after root operation")))
    return failure();

  // The replacement portion of this statement is within a rewrite context.
  llvm::SaveAndRestore<ParserContext> saveCtx(parserContext,
                                              ParserContext::Rewrite);

  // Parse the replacement values.
  SmallVector<ast::Expr *> replValues;
  if (consumeIf(Token::l_paren)) {
    if (consumeIf(Token::r_paren)) {
      return emitError(
          loc, "expected at least one replacement value, consider using "
               "`erase` if no replacement values are desired");
    }

    do {
      FailureOr<ast::Expr *> replExpr = parseExpr();
      if (failed(replExpr))
        return failure();
      replValues.emplace_back(*replExpr);
    } while (consumeIf(Token::comma));

    if (failed(parseToken(Token::r_paren,
                          "expected `)` after replacement values")))
      return failure();
  } else {
    FailureOr<ast::Expr *> replExpr = parseExpr();
    if (failed(replExpr))
      return failure();
    replValues.emplace_back(*replExpr);
  }

  return createReplaceStmt(loc, *rootOp, replValues);
}

FailureOr<ast::ReturnStmt *> Parser::parseReturnStmt() {
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_return);

  // Parse the result value.
  FailureOr<ast::Expr *> resultExpr = parseExpr();
  if (failed(resultExpr))
    return failure();

  return ast::ReturnStmt::create(ctx, loc, *resultExpr);
}

FailureOr<ast::RewriteStmt *> Parser::parseRewriteStmt() {
  if (parserContext == ParserContext::Constraint)
    return emitError("`rewrite` cannot be used within a Constraint");
  SMRange loc = curToken.getLoc();
  consumeToken(Token::kw_rewrite);

  // Parse the root operation.
  FailureOr<ast::Expr *> rootOp = parseExpr();
  if (failed(rootOp))
    return failure();

  if (failed(parseToken(Token::kw_with, "expected `with` before rewrite body")))
    return failure();

  if (curToken.isNot(Token::l_brace))
    return emitError("expected `{` to start rewrite body");

  // The rewrite body of this statement is within a rewrite context.
  llvm::SaveAndRestore<ParserContext> saveCtx(parserContext,
                                              ParserContext::Rewrite);

  FailureOr<ast::CompoundStmt *> rewriteBody = parseCompoundStmt();
  if (failed(rewriteBody))
    return failure();

  // Verify the rewrite body.
  for (const ast::Stmt *stmt : (*rewriteBody)->getChildren()) {
    if (isa<ast::ReturnStmt>(stmt)) {
      return emitError(stmt->getLoc(),
                       "`return` statements are only permitted within a "
                       "`Constraint` or `Rewrite` body");
    }
  }

  return createRewriteStmt(loc, *rootOp, *rewriteBody);
}

//===----------------------------------------------------------------------===//
// Creation+Analysis
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Decls

ast::CallableDecl *Parser::tryExtractCallableDecl(ast::Node *node) {
  // Unwrap reference expressions.
  if (auto *init = dyn_cast<ast::DeclRefExpr>(node))
    node = init->getDecl();
  return dyn_cast<ast::CallableDecl>(node);
}

FailureOr<ast::PatternDecl *>
Parser::createPatternDecl(SMRange loc, const ast::Name *name,
                          const ParsedPatternMetadata &metadata,
                          ast::CompoundStmt *body) {
  return ast::PatternDecl::create(ctx, loc, name, metadata.benefit,
                                  metadata.hasBoundedRecursion, body);
}

ast::Type Parser::createUserConstraintRewriteResultType(
    ArrayRef<ast::VariableDecl *> results) {
  // Single result decls use the type of the single result.
  if (results.size() == 1)
    return results[0]->getType();

  // Multiple results use a tuple type, with the types and names grabbed from
  // the result variable decls.
  auto resultTypes = llvm::map_range(
      results, [&](const auto *result) { return result->getType(); });
  auto resultNames = llvm::map_range(
      results, [&](const auto *result) { return result->getName().getName(); });
  return ast::TupleType::get(ctx, llvm::to_vector(resultTypes),
                             llvm::to_vector(resultNames));
}

template <typename T>
FailureOr<T *> Parser::createUserPDLLConstraintOrRewriteDecl(
    const ast::Name &name, ArrayRef<ast::VariableDecl *> arguments,
    ArrayRef<ast::VariableDecl *> results, ast::Type resultType,
    ast::CompoundStmt *body) {
  if (!body->getChildren().empty()) {
    if (auto *retStmt = dyn_cast<ast::ReturnStmt>(body->getChildren().back())) {
      ast::Expr *resultExpr = retStmt->getResultExpr();

      // Process the result of the decl. If no explicit signature results
      // were provided, check for return type inference. Otherwise, check that
      // the return expression can be converted to the expected type.
      if (results.empty())
        resultType = resultExpr->getType();
      else if (failed(convertExpressionTo(resultExpr, resultType)))
        return failure();
      else
        retStmt->setResultExpr(resultExpr);
    }
  }
  return T::createPDLL(ctx, name, arguments, results, body, resultType);
}

FailureOr<ast::VariableDecl *>
Parser::createVariableDecl(StringRef name, SMRange loc, ast::Expr *initializer,
                           ArrayRef<ast::ConstraintRef> constraints) {
  // The type of the variable, which is expected to be inferred by either a
  // constraint or an initializer expression.
  ast::Type type;
  if (failed(validateVariableConstraints(constraints, type)))
    return failure();

  if (initializer) {
    // Update the variable type based on the initializer, or try to convert the
    // initializer to the existing type.
    if (!type)
      type = initializer->getType();
    else if (ast::Type mergedType = type.refineWith(initializer->getType()))
      type = mergedType;
    else if (failed(convertExpressionTo(initializer, type)))
      return failure();

    // Otherwise, if there is no initializer check that the type has already
    // been resolved from the constraint list.
  } else if (!type) {
    return emitErrorAndNote(
        loc, "unable to infer type for variable `" + name + "`", loc,
        "the type of a variable must be inferable from the constraint "
        "list or the initializer");
  }

  // Constraint types cannot be used when defining variables.
  if (type.isa<ast::ConstraintType, ast::RewriteType>()) {
    return emitError(
        loc, llvm::formatv("unable to define variable of `{0}` type", type));
  }

  // Try to define a variable with the given name.
  FailureOr<ast::VariableDecl *> varDecl =
      defineVariableDecl(name, loc, type, initializer, constraints);
  if (failed(varDecl))
    return failure();

  return *varDecl;
}

FailureOr<ast::VariableDecl *>
Parser::createArgOrResultVariableDecl(StringRef name, SMRange loc,
                                      const ast::ConstraintRef &constraint) {
  // Constraint arguments may apply more complex constraints via the arguments.
  bool allowNonCoreConstraints = parserContext == ParserContext::Constraint;
  ast::Type argType;
  if (failed(validateVariableConstraint(constraint, argType,
                                        allowNonCoreConstraints)))
    return failure();
  return defineVariableDecl(name, loc, argType, constraint);
}

LogicalResult
Parser::validateVariableConstraints(ArrayRef<ast::ConstraintRef> constraints,
                                    ast::Type &inferredType) {
  for (const ast::ConstraintRef &ref : constraints)
    if (failed(validateVariableConstraint(ref, inferredType)))
      return failure();
  return success();
}

LogicalResult Parser::validateVariableConstraint(const ast::ConstraintRef &ref,
                                                 ast::Type &inferredType,
                                                 bool allowNonCoreConstraints) {
  ast::Type constraintType;
  if (const auto *cst = dyn_cast<ast::AttrConstraintDecl>(ref.constraint)) {
    if (const ast::Expr *typeExpr = cst->getTypeExpr()) {
      if (failed(validateTypeConstraintExpr(typeExpr)))
        return failure();
    }
    constraintType = ast::AttributeType::get(ctx);
  } else if (const auto *cst =
                 dyn_cast<ast::OpConstraintDecl>(ref.constraint)) {
    constraintType = ast::OperationType::get(ctx, cst->getName());
  } else if (isa<ast::TypeConstraintDecl>(ref.constraint)) {
    constraintType = typeTy;
  } else if (isa<ast::TypeRangeConstraintDecl>(ref.constraint)) {
    constraintType = typeRangeTy;
  } else if (const auto *cst =
                 dyn_cast<ast::ValueConstraintDecl>(ref.constraint)) {
    if (const ast::Expr *typeExpr = cst->getTypeExpr()) {
      if (failed(validateTypeConstraintExpr(typeExpr)))
        return failure();
    }
    constraintType = valueTy;
  } else if (const auto *cst =
                 dyn_cast<ast::ValueRangeConstraintDecl>(ref.constraint)) {
    if (const ast::Expr *typeExpr = cst->getTypeExpr()) {
      if (failed(validateTypeRangeConstraintExpr(typeExpr)))
        return failure();
    }
    constraintType = valueRangeTy;
  } else if (const auto *cst =
                 dyn_cast<ast::UserConstraintDecl>(ref.constraint)) {
    if (!allowNonCoreConstraints) {
      return emitError(ref.referenceLoc,
                       "`Rewrite` arguments and results are only permitted to "
                       "use core constraints, such as `Attr`, `Op`, `Type`, "
                       "`TypeRange`, `Value`, `ValueRange`");
    }

    ArrayRef<ast::VariableDecl *> inputs = cst->getInputs();
    if (inputs.size() != 1) {
      return emitErrorAndNote(ref.referenceLoc,
                              "`Constraint`s applied via a variable constraint "
                              "list must take a single input, but got " +
                                  Twine(inputs.size()),
                              cst->getLoc(),
                              "see definition of constraint here");
    }
    constraintType = inputs.front()->getType();
  } else {
    llvm_unreachable("unknown constraint type");
  }

  // Check that the constraint type is compatible with the current inferred
  // type.
  if (!inferredType) {
    inferredType = constraintType;
  } else if (ast::Type mergedTy = inferredType.refineWith(constraintType)) {
    inferredType = mergedTy;
  } else {
    return emitError(ref.referenceLoc,
                     llvm::formatv("constraint type `{0}` is incompatible "
                                   "with the previously inferred type `{1}`",
                                   constraintType, inferredType));
  }
  return success();
}

LogicalResult Parser::validateTypeConstraintExpr(const ast::Expr *typeExpr) {
  ast::Type typeExprType = typeExpr->getType();
  if (typeExprType != typeTy) {
    return emitError(typeExpr->getLoc(),
                     "expected expression of `Type` in type constraint");
  }
  return success();
}

LogicalResult
Parser::validateTypeRangeConstraintExpr(const ast::Expr *typeExpr) {
  ast::Type typeExprType = typeExpr->getType();
  if (typeExprType != typeRangeTy) {
    return emitError(typeExpr->getLoc(),
                     "expected expression of `TypeRange` in type constraint");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Exprs

FailureOr<ast::CallExpr *>
Parser::createCallExpr(SMRange loc, ast::Expr *parentExpr,
                       MutableArrayRef<ast::Expr *> arguments) {
  ast::Type parentType = parentExpr->getType();

  ast::CallableDecl *callableDecl = tryExtractCallableDecl(parentExpr);
  if (!callableDecl) {
    return emitError(loc,
                     llvm::formatv("expected a reference to a callable "
                                   "`Constraint` or `Rewrite`, but got: `{0}`",
                                   parentType));
  }
  if (parserContext == ParserContext::Rewrite) {
    if (isa<ast::UserConstraintDecl>(callableDecl))
      return emitError(
          loc, "unable to invoke `Constraint` within a rewrite section");
  } else if (isa<ast::UserRewriteDecl>(callableDecl)) {
    return emitError(loc, "unable to invoke `Rewrite` within a match section");
  }

  // Verify the arguments of the call.
  /// Handle size mismatch.
  ArrayRef<ast::VariableDecl *> callArgs = callableDecl->getInputs();
  if (callArgs.size() != arguments.size()) {
    return emitErrorAndNote(
        loc,
        llvm::formatv("invalid number of arguments for {0} call; expected "
                      "{1}, but got {2}",
                      callableDecl->getCallableType(), callArgs.size(),
                      arguments.size()),
        callableDecl->getLoc(),
        llvm::formatv("see the definition of {0} here",
                      callableDecl->getName()->getName()));
  }

  /// Handle argument type mismatch.
  auto attachDiagFn = [&](ast::Diagnostic &diag) {
    diag.attachNote(llvm::formatv("see the definition of `{0}` here",
                                  callableDecl->getName()->getName()),
                    callableDecl->getLoc());
  };
  for (auto it : llvm::zip(callArgs, arguments)) {
    if (failed(convertExpressionTo(std::get<1>(it), std::get<0>(it)->getType(),
                                   attachDiagFn)))
      return failure();
  }

  return ast::CallExpr::create(ctx, loc, parentExpr, arguments,
                               callableDecl->getResultType());
}

FailureOr<ast::DeclRefExpr *> Parser::createDeclRefExpr(SMRange loc,
                                                        ast::Decl *decl) {
  // Check the type of decl being referenced.
  ast::Type declType;
  if (isa<ast::ConstraintDecl>(decl))
    declType = ast::ConstraintType::get(ctx);
  else if (isa<ast::UserRewriteDecl>(decl))
    declType = ast::RewriteType::get(ctx);
  else if (auto *varDecl = dyn_cast<ast::VariableDecl>(decl))
    declType = varDecl->getType();
  else
    return emitError(loc, "invalid reference to `" +
                              decl->getName()->getName() + "`");

  return ast::DeclRefExpr::create(ctx, loc, decl, declType);
}

FailureOr<ast::DeclRefExpr *>
Parser::createInlineVariableExpr(ast::Type type, StringRef name, SMRange loc,
                                 ArrayRef<ast::ConstraintRef> constraints) {
  FailureOr<ast::VariableDecl *> decl =
      defineVariableDecl(name, loc, type, constraints);
  if (failed(decl))
    return failure();
  return ast::DeclRefExpr::create(ctx, loc, *decl, type);
}

FailureOr<ast::MemberAccessExpr *>
Parser::createMemberAccessExpr(ast::Expr *parentExpr, StringRef name,
                               SMRange loc) {
  // Validate the member name for the given parent expression.
  FailureOr<ast::Type> memberType = validateMemberAccess(parentExpr, name, loc);
  if (failed(memberType))
    return failure();

  return ast::MemberAccessExpr::create(ctx, loc, parentExpr, name, *memberType);
}

FailureOr<ast::Type> Parser::validateMemberAccess(ast::Expr *parentExpr,
                                                  StringRef name, SMRange loc) {
  ast::Type parentType = parentExpr->getType();
  if (parentType.isa<ast::OperationType>()) {
    if (name == ast::AllResultsMemberAccessExpr::getMemberName())
      return valueRangeTy;
  } else if (auto tupleType = parentType.dyn_cast<ast::TupleType>()) {
    // Handle indexed results.
    unsigned index = 0;
    if (llvm::isDigit(name[0]) && !name.getAsInteger(/*Radix=*/10, index) &&
        index < tupleType.size()) {
      return tupleType.getElementTypes()[index];
    }

    // Handle named results.
    auto elementNames = tupleType.getElementNames();
    const auto *it = llvm::find(elementNames, name);
    if (it != elementNames.end())
      return tupleType.getElementTypes()[it - elementNames.begin()];
  }
  return emitError(
      loc,
      llvm::formatv("invalid member access `{0}` on expression of type `{1}`",
                    name, parentType));
}

FailureOr<ast::OperationExpr *> Parser::createOperationExpr(
    SMRange loc, const ast::OpNameDecl *name,
    MutableArrayRef<ast::Expr *> operands,
    MutableArrayRef<ast::NamedAttributeDecl *> attributes,
    MutableArrayRef<ast::Expr *> results) {
  Optional<StringRef> opNameRef = name->getName();

  // Verify the inputs operands.
  if (failed(validateOperationOperands(loc, opNameRef, operands)))
    return failure();

  // Verify the attribute list.
  for (ast::NamedAttributeDecl *attr : attributes) {
    // Check for an attribute type, or a type awaiting resolution.
    ast::Type attrType = attr->getValue()->getType();
    if (!attrType.isa<ast::AttributeType>()) {
      return emitError(
          attr->getValue()->getLoc(),
          llvm::formatv("expected `Attr` expression, but got `{0}`", attrType));
    }
  }

  // Verify the result types.
  if (failed(validateOperationResults(loc, opNameRef, results)))
    return failure();

  return ast::OperationExpr::create(ctx, loc, name, operands, results,
                                    attributes);
}

LogicalResult
Parser::validateOperationOperands(SMRange loc, Optional<StringRef> name,
                                  MutableArrayRef<ast::Expr *> operands) {
  return validateOperationOperandsOrResults(loc, name, operands, valueTy,
                                            valueRangeTy);
}

LogicalResult
Parser::validateOperationResults(SMRange loc, Optional<StringRef> name,
                                 MutableArrayRef<ast::Expr *> results) {
  return validateOperationOperandsOrResults(loc, name, results, typeTy,
                                            typeRangeTy);
}

LogicalResult Parser::validateOperationOperandsOrResults(
    SMRange loc, Optional<StringRef> name, MutableArrayRef<ast::Expr *> values,
    ast::Type singleTy, ast::Type rangeTy) {
  // All operation types accept a single range parameter.
  if (values.size() == 1) {
    if (failed(convertExpressionTo(values[0], rangeTy)))
      return failure();
    return success();
  }

  // Otherwise, accept the value groups as they have been defined and just
  // ensure they are one of the expected types.
  for (ast::Expr *&valueExpr : values) {
    ast::Type valueExprType = valueExpr->getType();

    // Check if this is one of the expected types.
    if (valueExprType == rangeTy || valueExprType == singleTy)
      continue;

    // If the operand is an Operation, allow converting to a Value or
    // ValueRange. This situations arises quite often with nested operation
    // expressions: `op<my_dialect.foo>(op<my_dialect.bar>)`
    if (singleTy == valueTy) {
      if (valueExprType.isa<ast::OperationType>()) {
        valueExpr = convertOpToValue(valueExpr);
        continue;
      }
    }

    return emitError(
        valueExpr->getLoc(),
        llvm::formatv(
            "expected `{0}` or `{1}` convertible expression, but got `{2}`",
            singleTy, rangeTy, valueExprType));
  }
  return success();
}

FailureOr<ast::TupleExpr *>
Parser::createTupleExpr(SMRange loc, ArrayRef<ast::Expr *> elements,
                        ArrayRef<StringRef> elementNames) {
  for (const ast::Expr *element : elements) {
    ast::Type eleTy = element->getType();
    if (eleTy.isa<ast::ConstraintType, ast::RewriteType, ast::TupleType>()) {
      return emitError(
          element->getLoc(),
          llvm::formatv("unable to build a tuple with `{0}` element", eleTy));
    }
  }
  return ast::TupleExpr::create(ctx, loc, elements, elementNames);
}

//===----------------------------------------------------------------------===//
// Stmts

FailureOr<ast::EraseStmt *> Parser::createEraseStmt(SMRange loc,
                                                    ast::Expr *rootOp) {
  // Check that root is an Operation.
  ast::Type rootType = rootOp->getType();
  if (!rootType.isa<ast::OperationType>())
    return emitError(rootOp->getLoc(), "expected `Op` expression");

  return ast::EraseStmt::create(ctx, loc, rootOp);
}

FailureOr<ast::ReplaceStmt *>
Parser::createReplaceStmt(SMRange loc, ast::Expr *rootOp,
                          MutableArrayRef<ast::Expr *> replValues) {
  // Check that root is an Operation.
  ast::Type rootType = rootOp->getType();
  if (!rootType.isa<ast::OperationType>()) {
    return emitError(
        rootOp->getLoc(),
        llvm::formatv("expected `Op` expression, but got `{0}`", rootType));
  }

  // If there are multiple replacement values, we implicitly convert any Op
  // expressions to the value form.
  bool shouldConvertOpToValues = replValues.size() > 1;
  for (ast::Expr *&replExpr : replValues) {
    ast::Type replType = replExpr->getType();

    // Check that replExpr is an Operation, Value, or ValueRange.
    if (replType.isa<ast::OperationType>()) {
      if (shouldConvertOpToValues)
        replExpr = convertOpToValue(replExpr);
      continue;
    }

    if (replType != valueTy && replType != valueRangeTy) {
      return emitError(replExpr->getLoc(),
                       llvm::formatv("expected `Op`, `Value` or `ValueRange` "
                                     "expression, but got `{0}`",
                                     replType));
    }
  }

  return ast::ReplaceStmt::create(ctx, loc, rootOp, replValues);
}

FailureOr<ast::RewriteStmt *>
Parser::createRewriteStmt(SMRange loc, ast::Expr *rootOp,
                          ast::CompoundStmt *rewriteBody) {
  // Check that root is an Operation.
  ast::Type rootType = rootOp->getType();
  if (!rootType.isa<ast::OperationType>()) {
    return emitError(
        rootOp->getLoc(),
        llvm::formatv("expected `Op` expression, but got `{0}`", rootType));
  }

  return ast::RewriteStmt::create(ctx, loc, rootOp, rewriteBody);
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

FailureOr<ast::Module *> mlir::pdll::parsePDLAST(ast::Context &ctx,
                                                 llvm::SourceMgr &sourceMgr) {
  Parser parser(ctx, sourceMgr);
  return parser.parseModule();
}
