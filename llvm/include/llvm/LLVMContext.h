//===-- llvm/LLVMContext.h - Class for managing "global" state --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVMContext, a container of "global" state in LLVM, such
// as the global type and constant uniquing tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLVMCONTEXT_H
#define LLVM_LLVMCONTEXT_H

namespace llvm {

class LLVMContextImpl;
class StringRef;
class Instruction;
template <typename T> class SmallVectorImpl;

/// This is an important class for using LLVM in a threaded context.  It
/// (opaquely) owns and manages the core "global" data of LLVM's core 
/// infrastructure, including the type and constant uniquing tables.
/// LLVMContext itself provides no locking guarantees, so you should be careful
/// to have one context per thread.
class LLVMContext {
  // DO NOT IMPLEMENT
  LLVMContext(LLVMContext&);
  void operator=(LLVMContext&);

public:
  LLVMContextImpl *const pImpl;
  LLVMContext();
  ~LLVMContext();
  
  // Pinned metadata names, which always have the same value.  This is a
  // compile-time performance optimization, not a correctness optimization.
  enum {
    MD_dbg = 1   // "dbg" -> 1.
  };
  
  /// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
  /// This ID is uniqued across modules in the current LLVMContext.
  unsigned getMDKindID(StringRef Name) const;
  
  /// getMDKindNames - Populate client supplied SmallVector with the name for
  /// custom metadata IDs registered in this LLVMContext.   ID #0 is not used,
  /// so it is filled in as an empty string.
  void getMDKindNames(SmallVectorImpl<StringRef> &Result) const;
  
  /// setInlineAsmDiagnosticHandler - This method sets a handler that is invoked
  /// when problems with inline asm are detected by the backend.  The first
  /// argument is a function pointer (of type SourceMgr::DiagHandlerTy) and the
  /// second is a context pointer that gets passed into the DiagHandler.
  ///
  /// LLVMContext doesn't take ownership or interpreter either of these
  /// pointers.
  void setInlineAsmDiagnosticHandler(void *DiagHandler, void *DiagContext = 0);

  /// getInlineAsmDiagnosticHandler - Return the diagnostic handler set by
  /// setInlineAsmDiagnosticHandler.
  void *getInlineAsmDiagnosticHandler() const;

  /// getInlineAsmDiagnosticContext - Return the diagnostic context set by
  /// setInlineAsmDiagnosticHandler.
  void *getInlineAsmDiagnosticContext() const;
  
  
  /// emitError - Emit an error message to the currently installed error handler
  /// with optional location information.  This function returns, so code should
  /// be prepared to drop the erroneous construct on the floor and "not crash".
  /// The generated code need not be correct.  The error message will be
  /// implicitly prefixed with "error: " and should not end with a ".".
  void emitError(unsigned LocCookie, StringRef ErrorStr);
  void emitError(const Instruction *I, StringRef ErrorStr);
  void emitError(StringRef ErrorStr);
};

/// getGlobalContext - Returns a global context.  This is for LLVM clients that
/// only care about operating on a single thread.
extern LLVMContext &getGlobalContext();

}

#endif
