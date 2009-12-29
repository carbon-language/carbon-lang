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
  
  /// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
  /// This ID is uniqued across modules in the current LLVMContext.
  unsigned getMDKindID(StringRef Name) const;
  
  /// getMDKindNames - Populate client supplied SmallVector with the name for
  /// custom metadata IDs registered in this LLVMContext.   ID #0 is not used,
  /// so it is filled in as an empty string.
  void getMDKindNames(SmallVectorImpl<StringRef> &Result) const;
};

/// getGlobalContext - Returns a global context.  This is for LLVM clients that
/// only care about operating on a single thread.
extern LLVMContext &getGlobalContext();

}

#endif
