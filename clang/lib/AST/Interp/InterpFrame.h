//===--- InterpFrame.h - Call Frame implementation for the VM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the class storing information about stack frames in the interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_INTERPFRAME_H
#define LLVM_CLANG_AST_INTERP_INTERPFRAME_H

#include "Frame.h"
#include "Pointer.h"
#include "Program.h"
#include "State.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace interp {
class Function;
class InterpState;

/// Frame storing local variables.
class InterpFrame final : public Frame {
public:
  /// The frame of the previous function.
  InterpFrame *Caller;

  /// Creates a new frame for a method call.
  InterpFrame(InterpState &S, Function *Func, InterpFrame *Caller,
              CodePtr RetPC, Pointer &&This);

  /// Destroys the frame, killing all live pointers to stack slots.
  ~InterpFrame();

  /// Invokes the destructors for a scope.
  void destroy(unsigned Idx);

  /// Pops the arguments off the stack.
  void popArgs();

  /// Describes the frame with arguments for diagnostic purposes.
  void describe(llvm::raw_ostream &OS) override;

  /// Returns the parent frame object.
  Frame *getCaller() const override;

  /// Returns the location of the call to the frame.
  SourceLocation getCallLocation() const override;

  /// Returns the caller.
  const FunctionDecl *getCallee() const override;

  /// Returns the current function.
  Function *getFunction() const { return Func; }

  /// Returns the offset on the stack at which the frame starts.
  size_t getFrameOffset() const { return FrameOffset; }

  /// Returns the value of a local variable.
  template <typename T> const T &getLocal(unsigned Offset) {
    return localRef<T>(Offset);
  }

  /// Mutates a local variable.
  template <typename T> void setLocal(unsigned Offset, const T &Value) {
    localRef<T>(Offset) = Value;
  }

  /// Returns a pointer to a local variables.
  Pointer getLocalPointer(unsigned Offset);

  /// Returns the value of an argument.
  template <typename T> const T &getParam(unsigned Offset) {
    auto Pt = Params.find(Offset);
    if (Pt == Params.end()) {
      return stackRef<T>(Offset);
    } else {
      return Pointer(reinterpret_cast<Block *>(Pt->second.get())).deref<T>();
    }
  }

  /// Mutates a local copy of a parameter.
  template <typename T> void setParam(unsigned Offset, const T &Value) {
     getParamPointer(Offset).deref<T>() = Value;
  }

  /// Returns a pointer to an argument - lazily creates a block.
  Pointer getParamPointer(unsigned Offset);

  /// Returns the 'this' pointer.
  const Pointer &getThis() const { return This; }

  /// Checks if the frame is a root frame - return should quit the interpreter.
  bool isRoot() const { return !Func; }

  /// Returns the PC of the frame's code start.
  CodePtr getPC() const { return Func->getCodeBegin(); }

  /// Returns the return address of the frame.
  CodePtr getRetPC() const { return RetPC; }

  /// Map a location to a source.
  virtual SourceInfo getSource(CodePtr PC) const;
  const Expr *getExpr(CodePtr PC) const;
  SourceLocation getLocation(CodePtr PC) const;

private:
  /// Returns an original argument from the stack.
  template <typename T> const T &stackRef(unsigned Offset) {
    return *reinterpret_cast<const T *>(Args - ArgSize + Offset);
  }

  /// Returns an offset to a local.
  template <typename T> T &localRef(unsigned Offset) {
    return *reinterpret_cast<T *>(Locals.get() + Offset);
  }

  /// Returns a pointer to a local's block.
  void *localBlock(unsigned Offset) {
    return Locals.get() + Offset - sizeof(Block);
  }

private:
  /// Reference to the interpreter state.
  InterpState &S;
  /// Reference to the function being executed.
  Function *Func;
  /// Current object pointer for methods.
  Pointer This;
  /// Return address.
  CodePtr RetPC;
  /// The size of all the arguments.
  const unsigned ArgSize;
  /// Pointer to the arguments in the callee's frame.
  char *Args = nullptr;
  /// Fixed, initial storage for known local variables.
  std::unique_ptr<char[]> Locals;
  /// Offset on the stack at entry.
  const size_t FrameOffset;
  /// Mapping from arg offsets to their argument blocks.
  llvm::DenseMap<unsigned, std::unique_ptr<char[]>> Params;
};

} // namespace interp
} // namespace clang

#endif
