//===--- ByteCodeEmitter.h - Instruction emitter for the VM ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the instruction emitters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_LINKEMITTER_H
#define LLVM_CLANG_AST_INTERP_LINKEMITTER_H

#include "ByteCodeGenError.h"
#include "Context.h"
#include "InterpStack.h"
#include "InterpState.h"
#include "PrimType.h"
#include "Program.h"
#include "Source.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace interp {
class Context;
class SourceInfo;
enum Opcode : uint32_t;

/// An emitter which links the program to bytecode for later use.
class ByteCodeEmitter {
protected:
  using LabelTy = uint32_t;
  using AddrTy = uintptr_t;
  using Local = Scope::Local;

public:
  /// Compiles the function into the module.
  llvm::Expected<Function *> compileFunc(const FunctionDecl *F);

protected:
  ByteCodeEmitter(Context &Ctx, Program &P) : Ctx(Ctx), P(P) {}

  virtual ~ByteCodeEmitter() {}

  /// Define a label.
  void emitLabel(LabelTy Label);
  /// Create a label.
  LabelTy getLabel() { return ++NextLabel; }

  /// Methods implemented by the compiler.
  virtual bool visitFunc(const FunctionDecl *E) = 0;
  virtual bool visitExpr(const Expr *E) = 0;
  virtual bool visitDecl(const VarDecl *E) = 0;

  /// Bails out if a given node cannot be compiled.
  bool bail(const Stmt *S) { return bail(S->getBeginLoc()); }
  bool bail(const Decl *D) { return bail(D->getBeginLoc()); }
  bool bail(const SourceLocation &Loc);

  /// Emits jumps.
  bool jumpTrue(const LabelTy &Label);
  bool jumpFalse(const LabelTy &Label);
  bool jump(const LabelTy &Label);
  bool fallthrough(const LabelTy &Label);

  /// Callback for local registration.
  Local createLocal(Descriptor *D);

  /// Parameter indices.
  llvm::DenseMap<const ParmVarDecl *, unsigned> Params;
  /// Local descriptors.
  llvm::SmallVector<SmallVector<Local, 8>, 2> Descriptors;

private:
  /// Current compilation context.
  Context &Ctx;
  /// Program to link to.
  Program &P;
  /// Index of the next available label.
  LabelTy NextLabel = 0;
  /// Offset of the next local variable.
  unsigned NextLocalOffset = 0;
  /// Location of a failure.
  llvm::Optional<SourceLocation> BailLocation;
  /// Label information for linker.
  llvm::DenseMap<LabelTy, unsigned> LabelOffsets;
  /// Location of label relocations.
  llvm::DenseMap<LabelTy, llvm::SmallVector<unsigned, 5>> LabelRelocs;
  /// Program code.
  std::vector<char> Code;
  /// Opcode to expression mapping.
  SourceMap SrcMap;

  /// Returns the offset for a jump or records a relocation.
  int32_t getOffset(LabelTy Label);

  /// Emits an opcode.
  template <typename... Tys>
  bool emitOp(Opcode Op, const Tys &... Args, const SourceInfo &L);

protected:
#define GET_LINK_PROTO
#include "Opcodes.inc"
#undef GET_LINK_PROTO
};

} // namespace interp
} // namespace clang

#endif
