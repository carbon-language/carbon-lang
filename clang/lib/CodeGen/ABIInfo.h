//===----- ABIInfo.h - ABI information access & encapsulation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_ABIINFO_H
#define LLVM_CLANG_LIB_CODEGEN_ABIINFO_H

#include "clang/AST/Type.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Type.h"

namespace llvm {
  class Value;
  class LLVMContext;
  class DataLayout;
}

namespace clang {
  class ASTContext;
  class TargetInfo;

  namespace CodeGen {
    class CGCXXABI;
    class CGFunctionInfo;
    class CodeGenFunction;
    class CodeGenTypes;
  }

  // FIXME: All of this stuff should be part of the target interface
  // somehow. It is currently here because it is not clear how to factor
  // the targets to support this, since the Targets currently live in a
  // layer below types n'stuff.


  /// ABIInfo - Target specific hooks for defining how a type should be
  /// passed or returned from functions.
  class ABIInfo {
  public:
    CodeGen::CodeGenTypes &CGT;
  protected:
    llvm::CallingConv::ID RuntimeCC;
  public:
    ABIInfo(CodeGen::CodeGenTypes &cgt)
      : CGT(cgt), RuntimeCC(llvm::CallingConv::C) {}

    virtual ~ABIInfo();

    CodeGen::CGCXXABI &getCXXABI() const;
    ASTContext &getContext() const;
    llvm::LLVMContext &getVMContext() const;
    const llvm::DataLayout &getDataLayout() const;
    const TargetInfo &getTarget() const;

    /// Return the calling convention to use for system runtime
    /// functions.
    llvm::CallingConv::ID getRuntimeCC() const {
      return RuntimeCC;
    }

    virtual void computeInfo(CodeGen::CGFunctionInfo &FI) const = 0;

    /// EmitVAArg - Emit the target dependent code to load a value of
    /// \arg Ty from the va_list pointed to by \arg VAListAddr.

    // FIXME: This is a gaping layering violation if we wanted to drop
    // the ABI information any lower than CodeGen. Of course, for
    // VAArg handling it has to be at this level; there is no way to
    // abstract this out.
    virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                   CodeGen::CodeGenFunction &CGF) const = 0;

    virtual bool isHomogeneousAggregateBaseType(QualType Ty) const;

    virtual bool isHomogeneousAggregateSmallEnough(const Type *Base,
                                                   uint64_t Members) const;

    bool isHomogeneousAggregate(QualType Ty, const Type *&Base,
                                uint64_t &Members) const;

  };
}  // end namespace clang

#endif
