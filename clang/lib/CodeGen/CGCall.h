//===----- CGCall.h - Encapsulate calling convention details ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGCALL_H
#define CLANG_CODEGEN_CGCALL_H

#include "CGValue.h"
#include "EHScopeStack.h"
#include "clang/AST/CanonicalType.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/IR/Value.h"

// FIXME: Restructure so we don't have to expose so much stuff.
#include "ABIInfo.h"

namespace llvm {
  class AttributeSet;
  class Function;
  class Type;
  class Value;
}

namespace clang {
  class ASTContext;
  class Decl;
  class FunctionDecl;
  class ObjCMethodDecl;
  class VarDecl;

namespace CodeGen {
  typedef SmallVector<llvm::AttributeSet, 8> AttributeListType;

  struct CallArg {
    RValue RV;
    QualType Ty;
    bool NeedsCopy;
    CallArg(RValue rv, QualType ty, bool needscopy)
    : RV(rv), Ty(ty), NeedsCopy(needscopy)
    { }
  };

  /// CallArgList - Type for representing both the value and type of
  /// arguments in a call.
  class CallArgList :
    public SmallVector<CallArg, 16> {
  public:
    struct Writeback {
      /// The original argument.  Note that the argument l-value
      /// is potentially null.
      LValue Source;

      /// The temporary alloca.
      llvm::Value *Temporary;

      /// A value to "use" after the writeback, or null.
      llvm::Value *ToUse;
    };

    struct CallArgCleanup {
      EHScopeStack::stable_iterator Cleanup;

      /// The "is active" insertion point.  This instruction is temporary and
      /// will be removed after insertion.
      llvm::Instruction *IsActiveIP;
    };

    void add(RValue rvalue, QualType type, bool needscopy = false) {
      push_back(CallArg(rvalue, type, needscopy));
    }

    void addFrom(const CallArgList &other) {
      insert(end(), other.begin(), other.end());
      Writebacks.insert(Writebacks.end(),
                        other.Writebacks.begin(), other.Writebacks.end());
    }

    void addWriteback(LValue srcLV, llvm::Value *temporary,
                      llvm::Value *toUse) {
      Writeback writeback;
      writeback.Source = srcLV;
      writeback.Temporary = temporary;
      writeback.ToUse = toUse;
      Writebacks.push_back(writeback);
    }

    bool hasWritebacks() const { return !Writebacks.empty(); }

    typedef SmallVectorImpl<Writeback>::const_iterator writeback_iterator;
    writeback_iterator writeback_begin() const { return Writebacks.begin(); }
    writeback_iterator writeback_end() const { return Writebacks.end(); }

    void addArgCleanupDeactivation(EHScopeStack::stable_iterator Cleanup,
                                   llvm::Instruction *IsActiveIP) {
      CallArgCleanup ArgCleanup;
      ArgCleanup.Cleanup = Cleanup;
      ArgCleanup.IsActiveIP = IsActiveIP;
      CleanupsToDeactivate.push_back(ArgCleanup);
    }

    ArrayRef<CallArgCleanup> getCleanupsToDeactivate() const {
      return CleanupsToDeactivate;
    }

  private:
    SmallVector<Writeback, 1> Writebacks;

    /// Deactivate these cleanups immediately before making the call.  This
    /// is used to cleanup objects that are owned by the callee once the call
    /// occurs.
    SmallVector<CallArgCleanup, 1> CleanupsToDeactivate;
  };

  /// FunctionArgList - Type for representing both the decl and type
  /// of parameters to a function. The decl must be either a
  /// ParmVarDecl or ImplicitParamDecl.
  class FunctionArgList : public SmallVector<const VarDecl*, 16> {
  };

  /// ReturnValueSlot - Contains the address where the return value of a 
  /// function can be stored, and whether the address is volatile or not.
  class ReturnValueSlot {
    llvm::PointerIntPair<llvm::Value *, 1, bool> Value;

  public:
    ReturnValueSlot() {}
    ReturnValueSlot(llvm::Value *Value, bool IsVolatile)
      : Value(Value, IsVolatile) {}

    bool isNull() const { return !getValue(); }
    
    bool isVolatile() const { return Value.getInt(); }
    llvm::Value *getValue() const { return Value.getPointer(); }
  };
  
}  // end namespace CodeGen
}  // end namespace clang

#endif
