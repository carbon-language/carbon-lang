//===--- ScopeInfo.h - Information about a semantic context -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines FunctionScopeInfo and BlockScopeInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SCOPE_INFO_H
#define LLVM_CLANG_SEMA_SCOPE_INFO_H

#include "clang/AST/Type.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class BlockDecl;
class IdentifierInfo;
class LabelDecl;
class ReturnStmt;
class Scope;
class SwitchStmt;

namespace sema {

class PossiblyUnreachableDiag {
public:
  PartialDiagnostic PD;
  SourceLocation Loc;
  const Stmt *stmt;
  
  PossiblyUnreachableDiag(const PartialDiagnostic &PD, SourceLocation Loc,
                          const Stmt *stmt)
    : PD(PD), Loc(Loc), stmt(stmt) {}
};
    
/// \brief Retains information about a function, method, or block that is
/// currently being parsed.
class FunctionScopeInfo {
protected:
  enum ScopeKind {
    SK_Function,
    SK_Block,
    SK_Lambda
  };
  
public:
  /// \brief What kind of scope we are describing.
  ///
  ScopeKind Kind;

  /// \brief Whether this function contains a VLA, @try, try, C++
  /// initializer, or anything else that can't be jumped past.
  bool HasBranchProtectedScope;

  /// \brief Whether this function contains any switches or direct gotos.
  bool HasBranchIntoScope;

  /// \brief Whether this function contains any indirect gotos.
  bool HasIndirectGoto;

  /// \brief Used to determine if errors occurred in this function or block.
  DiagnosticErrorTrap ErrorTrap;

  /// SwitchStack - This is the current set of active switch statements in the
  /// block.
  SmallVector<SwitchStmt*, 8> SwitchStack;

  /// \brief The list of return statements that occur within the function or
  /// block, if there is any chance of applying the named return value
  /// optimization.
  SmallVector<ReturnStmt*, 4> Returns;
  
  /// \brief A list of PartialDiagnostics created but delayed within the
  /// current function scope.  These diagnostics are vetted for reachability
  /// prior to being emitted.
  SmallVector<PossiblyUnreachableDiag, 4> PossiblyUnreachableDiags;

  void setHasBranchIntoScope() {
    HasBranchIntoScope = true;
  }

  void setHasBranchProtectedScope() {
    HasBranchProtectedScope = true;
  }

  void setHasIndirectGoto() {
    HasIndirectGoto = true;
  }

  bool NeedsScopeChecking() const {
    return HasIndirectGoto ||
          (HasBranchProtectedScope && HasBranchIntoScope);
  }
  
  FunctionScopeInfo(DiagnosticsEngine &Diag)
    : Kind(SK_Function),
      HasBranchProtectedScope(false),
      HasBranchIntoScope(false),
      HasIndirectGoto(false),
      ErrorTrap(Diag) { }

  virtual ~FunctionScopeInfo();

  /// \brief Clear out the information in this function scope, making it
  /// suitable for reuse.
  void Clear();

  static bool classof(const FunctionScopeInfo *FSI) { return true; }
};

class CapturingScopeInfo : public FunctionScopeInfo {
public:
  enum ImplicitCaptureStyle {
    ImpCap_None, ImpCap_LambdaByval, ImpCap_LambdaByref, ImpCap_Block
  };

  ImplicitCaptureStyle ImpCaptureStyle;

  class Capture {
    // There are two categories of capture: capturing 'this', and capturing
    // local variables.  There are three ways to capture a local variable:
    // capture by copy in the C++11 sense, capture by reference
    // in the C++11 sense, and __block capture.  Lambdas explicitly specify
    // capture by copy or capture by reference.  For blocks, __block capture
    // applies to variables with that annotation, variables of reference type
    // are captured by reference, and other variables are captured by copy.
    enum CaptureKind {
      Cap_This, Cap_ByCopy, Cap_ByRef, Cap_Block
    };

    // The variable being captured (if we are not capturing 'this'),
    // and misc bits descibing the capture.
    llvm::PointerIntPair<VarDecl*, 2, CaptureKind> VarAndKind;

    // Expression to initialize a field of the given type, and whether this
    // is a nested capture; the expression is only required if we are
    // capturing ByVal and the variable's type has a non-trivial
    // copy constructor.
    llvm::PointerIntPair<Expr*, 1, bool> CopyExprAndNested;

    /// \brief The source location at which the first capture occurred..
    SourceLocation Loc;
    
  public:
    Capture(VarDecl *Var, bool block, bool byRef, bool isNested, 
            SourceLocation Loc, Expr *Cpy)
      : VarAndKind(Var, block ? Cap_Block : byRef ? Cap_ByRef : Cap_ByCopy),
        CopyExprAndNested(Cpy, isNested) {}

    enum IsThisCapture { ThisCapture };
    Capture(IsThisCapture, bool isNested, SourceLocation Loc)
      : VarAndKind(0, Cap_This), CopyExprAndNested(0, isNested), Loc(Loc) {
    }

    bool isThisCapture() const { return VarAndKind.getInt() == Cap_This; }
    bool isVariableCapture() const { return !isThisCapture(); }
    bool isCopyCapture() const { return VarAndKind.getInt() == Cap_ByCopy; }
    bool isReferenceCapture() const { return VarAndKind.getInt() == Cap_ByRef; }
    bool isBlockCapture() const { return VarAndKind.getInt() == Cap_Block; }
    bool isNested() { return CopyExprAndNested.getInt(); }

    VarDecl *getVariable() const {
      return VarAndKind.getPointer();
    }
    
    /// \brief Retrieve the location at which this variable was captured.
    SourceLocation getLocation() const { return Loc; }
    
    Expr *getCopyExpr() const {
      return CopyExprAndNested.getPointer();
    }
  };

  CapturingScopeInfo(DiagnosticsEngine &Diag, ImplicitCaptureStyle Style)
    : FunctionScopeInfo(Diag), ImpCaptureStyle(Style), CXXThisCaptureIndex(0),
      HasImplicitReturnType(false)
     {}

  /// CaptureMap - A map of captured variables to (index+1) into Captures.
  llvm::DenseMap<VarDecl*, unsigned> CaptureMap;

  /// CXXThisCaptureIndex - The (index+1) of the capture of 'this';
  /// zero if 'this' is not captured.
  unsigned CXXThisCaptureIndex;

  /// Captures - The captures.
  SmallVector<Capture, 4> Captures;

  /// \brief - Whether the target type of return statements in this context
  /// is deduced (e.g. a lambda or block with omitted return type).
  bool HasImplicitReturnType;

  /// ReturnType - The target type of return statements in this context,
  /// or null if unknown.
  QualType ReturnType;

  void AddCapture(VarDecl *Var, bool isBlock, bool isByref, bool isNested,
                  SourceLocation Loc, Expr *Cpy) {
    Captures.push_back(Capture(Var, isBlock, isByref, isNested, Loc, Cpy));
    CaptureMap[Var] = Captures.size();
  }

  void AddThisCapture(bool isNested, SourceLocation Loc) {
    Captures.push_back(Capture(Capture::ThisCapture, isNested, Loc));
    CXXThisCaptureIndex = Captures.size();
  }

  /// \brief Determine whether the C++ 'this' is captured.
  bool isCXXThisCaptured() const { return CXXThisCaptureIndex != 0; }
  
  /// \brief Retrieve the capture of C++ 'this', if it has been captured.
  Capture &getCXXThisCapture() {
    assert(isCXXThisCaptured() && "this has not been captured");
    return Captures[CXXThisCaptureIndex - 1];
  }
  
  /// \brief Determine whether the given variable has been captured.
  bool isCaptured(VarDecl *Var) const {
    return CaptureMap.count(Var) > 0;
  }
  
  /// \brief Retrieve the capture of the given variable, if it has been
  /// captured already.
  Capture &getCapture(VarDecl *Var) {
    assert(isCaptured(Var) && "Variable has not been captured");
    return Captures[CaptureMap[Var] - 1];
  }

  const Capture &getCapture(VarDecl *Var) const {
    llvm::DenseMap<VarDecl*, unsigned>::const_iterator Known
      = CaptureMap.find(Var);
    assert(Known != CaptureMap.end() && "Variable has not been captured");
    return Captures[Known->second - 1];
  }

  static bool classof(const FunctionScopeInfo *FSI) { 
    return FSI->Kind == SK_Block || FSI->Kind == SK_Lambda; 
  }
  static bool classof(const CapturingScopeInfo *BSI) { return true; }
};

/// \brief Retains information about a block that is currently being parsed.
class BlockScopeInfo : public CapturingScopeInfo {
public:
  BlockDecl *TheDecl;
  
  /// TheScope - This is the scope for the block itself, which contains
  /// arguments etc.
  Scope *TheScope;

  /// BlockType - The function type of the block, if one was given.
  /// Its return type may be BuiltinType::Dependent.
  QualType FunctionType;

  BlockScopeInfo(DiagnosticsEngine &Diag, Scope *BlockScope, BlockDecl *Block)
    : CapturingScopeInfo(Diag, ImpCap_Block), TheDecl(Block),
      TheScope(BlockScope)
  {
    Kind = SK_Block;
  }

  virtual ~BlockScopeInfo();

  static bool classof(const FunctionScopeInfo *FSI) { 
    return FSI->Kind == SK_Block; 
  }
  static bool classof(const BlockScopeInfo *BSI) { return true; }
};

class LambdaScopeInfo : public CapturingScopeInfo {
public:
  /// \brief The class that describes the lambda.
  CXXRecordDecl *Lambda;
  
  /// \brief The number of captures in the \c Captures list that are 
  /// explicit captures.
  unsigned NumExplicitCaptures;

  bool Mutable;

  LambdaScopeInfo(DiagnosticsEngine &Diag, CXXRecordDecl *Lambda)
    : CapturingScopeInfo(Diag, ImpCap_None), Lambda(Lambda),
      NumExplicitCaptures(0), Mutable(false)
  {
    Kind = SK_Lambda;
  }

  virtual ~LambdaScopeInfo();

  /// \brief Note when 
  void finishedExplicitCaptures() {
    NumExplicitCaptures = Captures.size();
  }
  
  static bool classof(const FunctionScopeInfo *FSI) { 
    return FSI->Kind == SK_Lambda; 
  }
  static bool classof(const LambdaScopeInfo *BSI) { return true; }

};

}
}

#endif
