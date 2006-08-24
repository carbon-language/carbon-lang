//===--- Action.h - Parser Action Interface ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Action and EmptyAction interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_ACTION_H
#define LLVM_CLANG_PARSE_ACTION_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"

namespace llvm {
namespace clang {
  // Semantic.
  class Declarator;
  // Parse.
  class Scope;
  class Action;
  // Lex.
  class IdentifierInfo;
  class LexerToken;

/// Action - As the parser reads the input file and recognizes the productions
/// of the grammar, it invokes methods on this class to turn the parsed input
/// into something useful: e.g. a parse tree.
///
/// The callback methods that this class provides are phrased as actions that
/// the parser has just done or is about to do when the method is called.  They
/// are not requests that the actions module do the specified action.
///
/// All of the methods here are optional except isTypedefName(), which must be
/// specified in order for the parse to complete accurately.  The EmptyAction
/// class does this bare-minimum of tracking to implement this functionality.
class Action {
public:
  /// Out-of-line virtual destructor to provide home for this class.
  virtual ~Action();
  
  // Types - Though these don't actually enforce strong typing, they document
  // what types are required to be identical for the actions.
  typedef void ExprTy;
  typedef void DeclTy;
  
  /// ExprResult - This structure is used while parsing/acting on expressions.
  /// It encapsulates both the expression object returned by the action, plus
  /// a sense of whether or not it is valid.
  struct ExprResult {
    ExprTy *Val;
    bool isInvalid;
    
    ExprResult(bool Invalid = false) : Val(0), isInvalid(Invalid) {}
    template<typename ActualExprTy>
    ExprResult(ActualExprTy *val) : Val(val), isInvalid(false) {}
    
    const ExprResult &operator=(ExprTy *RHS) {
      Val = RHS;
      isInvalid = false;
      return *this;
    }
  };
  
  //===--------------------------------------------------------------------===//
  // Symbol Table Tracking Callbacks.
  //===--------------------------------------------------------------------===//
  
  /// isTypedefName - Return true if the specified identifier is a typedef name
  /// in the current scope.
  virtual bool isTypedefName(const IdentifierInfo &II, Scope *S) const = 0;
  
  /// ParseDeclarator - This callback is invoked when a declarator is parsed and
  /// 'Init' specifies the initializer if any.  This is for things like:
  /// "int X = 4" or "typedef int foo".
  virtual void ParseDeclarator(SourceLocation Loc, Scope *S, Declarator &D,
                               ExprTy *Init) {}
  
  /// PopScope - This callback is called immediately before the specified scope
  /// is popped and deleted.
  virtual void PopScope(SourceLocation Loc, Scope *S) {}
  
  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks.
  //===--------------------------------------------------------------------===//
  
  // Primary Expressions.
  virtual ExprResult ParseSimplePrimaryExpr(const LexerToken &Tok) { return 0; }
  virtual ExprResult ParseIntegerConstant(const LexerToken &Tok) { return 0; }
  virtual ExprResult ParseFloatingConstant(const LexerToken &Tok) { return 0; }

  virtual ExprResult ParseParenExpr(SourceLocation L, SourceLocation R,
                                    ExprTy *Val) {
    return Val;  // Default impl returns operand.
  }

  // Postfix Expressions.
  virtual ExprResult ParsePostfixUnaryOp(const LexerToken &Tok, ExprTy *Input) {
    return 0;
  }
  virtual ExprResult ParseArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                                             ExprTy *Idx, SourceLocation RLoc) {
    return 0;
  }
  virtual ExprResult ParseMemberReferenceExpr(ExprTy *Base,SourceLocation OpLoc,
                                              tok::TokenKind OpKind,
                                              SourceLocation MemberLoc,
                                              IdentifierInfo &Member) {
    return 0;
  }
  
  /// ParseCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.  There are guaranteed to be one fewer commas than arguments,
  /// unless there are zero arguments.
  virtual ExprResult ParseCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
                                   ExprTy **Args, unsigned NumArgs,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc) {
    return 0;
  }
  
  // Unary Operators.  'Tok' is the token for the operator.
  virtual ExprResult ParseUnaryOp(const LexerToken &Tok, ExprTy *Input) {
    return 0;
  }

  virtual ExprResult ParseBinOp(const LexerToken &Tok,
                                ExprTy *LHS, ExprTy *RHS) {
    return 0;
  }

  /// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual ExprResult ParseConditionalOp(SourceLocation QuestionLoc, 
                                        SourceLocation ColonLoc,
                                        ExprTy *Cond, ExprTy *LHS, ExprTy *RHS){
    return 0;
  }
};


/// EmptyAction - This is a simple (bare-minimum) implementation of the Action
/// class, which only keeps track of which typedefs are in-scope.  This class is
/// useful to subclass if clients want to implement some actions without having
/// to reimplement all of the scoping rules.
class EmptyAction : public Action {
public:
  /// isTypedefName - This looks at the IdentifierInfo::FETokenInfo field to
  /// determine whether the name is a typedef or not in this scope.
  virtual bool isTypedefName(const IdentifierInfo &II, Scope *S) const;
  
  /// ParseDeclarator - If this is a typedef declarator, we modify the
  /// IdentifierInfo::FETokenInfo field to keep track of this fact, until S is
  /// popped.
  virtual void ParseDeclarator(SourceLocation Loc, Scope *S, Declarator &D,
                               ExprTy *Init);
  
  /// PopScope - When a scope is popped, if any typedefs are now out-of-scope,
  /// they are removed from the IdentifierInfo::FETokenInfo field.
  virtual void PopScope(SourceLocation Loc, Scope *S);
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
