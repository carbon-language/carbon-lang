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
#include "llvm/ADT/SetVector.h"

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
    enum CaptureKind {
      Cap_This, Cap_ByVal, Cap_ByRef
    };

    // The variable being captured (if we are not capturing 'this'),
    // and misc bits descibing the capture.
    llvm::PointerIntPair<VarDecl*, 2, CaptureKind> VarAndKind;

    // Expression to initialize a field of the given type, and whether this
    // is a nested capture; the expression is only required if we are
    // capturing ByVal and the variable's type has a non-trivial
    // copy constructor.
    llvm::PointerIntPair<Expr*, 1, bool> CopyExprAndNested;

  public:
    Capture(VarDecl *Var, bool isByref, bool isNested, Expr *Cpy)
      : VarAndKind(Var, isByref ? Cap_ByRef : Cap_ByVal),
        CopyExprAndNested(Cpy, isNested) {}

    enum IsThisCapture { ThisCapture };
    Capture(IsThisCapture, bool isNested)
      : VarAndKind(0, Cap_This),
        CopyExprAndNested(0, isNested) {
    }

    bool isThisCapture() const { return VarAndKind.getInt() == Cap_This; }
    bool isVariableCapture() const { return !isThisCapture(); }
    bool isCopyCapture() const { return VarAndKind.getInt() == Cap_ByVal; }
    bool isReferenceCapture() const { return VarAndKind.getInt() == Cap_ByRef; }
    bool isNested() { return CopyExprAndNested.getInt(); }

    VarDecl *getVariable() const {
      return VarAndKind.getPointer();
    }
    Expr *getCopyExpr() const {
      return CopyExprAndNested.getPointer();
    }
  };

  CapturingScopeInfo(DiagnosticsEngine &Diag, ImplicitCaptureStyle Style)
    : FunctionScopeInfo(Diag), ImpCaptureStyle(Style), CXXThisCaptureIndex(0)
     {}

  /// CaptureMap - A map of captured variables to (index+1) into Captures.
  llvm::DenseMap<VarDecl*, unsigned> CaptureMap;

  /// CXXThisCaptureIndex - The (index+1) of the capture of 'this';
  /// zero if 'this' is not captured.
  unsigned CXXThisCaptureIndex;

  /// Captures - The captures.
  SmallVector<Capture, 4> Captures;

  void AddCapture(VarDecl *Var, bool isByref, bool isNested, Expr *Cpy) {
    Captures.push_back(Capture(Var, isByref, isNested, Cpy));
    CaptureMap[Var] = Captures.size();
  }

  void AddThisCapture(bool isNested) {
    Captures.push_back(Capture(Capture::ThisCapture, isNested));
    CXXThisCaptureIndex = Captures.size();
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

  /// ReturnType - The return type of the block, or null if the block
  /// signature didn't provide an explicit return type.
  QualType ReturnType;

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

  /// \brief - Whether the return type of the lambda is implicit
  bool HasImplicitReturnType;

  /// ReturnType - The return type of the lambda, or null if unknown.
  QualType ReturnType;

  LambdaScopeInfo(DiagnosticsEngine &Diag, CXXRecordDecl *Lambda)
    : CapturingScopeInfo(Diag, ImpCap_None), Lambda(Lambda),
      NumExplicitCaptures(0), HasImplicitReturnType(false)
  {
    Kind = SK_Lambda;
  }

  virtual ~LambdaScopeInfo();

  static bool classof(const FunctionScopeInfo *FSI) { 
    return FSI->Kind == SK_Lambda; 
  }
  static bool classof(const LambdaScopeInfo *BSI) { return true; }

};

}
}

#endif
