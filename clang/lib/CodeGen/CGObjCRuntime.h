//===----- CGObjCRuntime.h - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for Objective-C code generation.  Concrete
// subclasses of this implement code generation for specific Objective-C
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_OBCJRUNTIME_H
#define CLANG_CODEGEN_OBCJRUNTIME_H

namespace llvm {
  class IRBuilder;
  class Constant;
  class Type;
  class Value;
  class Module;
  class Function;
}


namespace clang {
namespace CodeGen {

// Implements runtime-specific code generation functions
class CGObjCRuntime {
public:
  virtual ~CGObjCRuntime();
  
  /// Generate an Objective-C message send operation
  virtual llvm::Value *generateMessageSend(llvm::IRBuilder &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Sender,
                                           llvm::Value *Receiver,
                                           llvm::Value *Selector,
                                           llvm::Value** ArgV,
                                           unsigned ArgC) = 0;
  /// Generate the function required to register all Objective-C components in
  /// this compilation unit with the runtime library.
  virtual llvm::Function *ModuleInitFunction() { return 0; }
  /// Generate a function preamble for a method with the specified types
  virtual llvm::Function *MethodPreamble(const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isVarArg) = 0;
  /// If instance variable addresses are determined at runtime then this should
  /// return true, otherwise instance variables will be accessed directly from
  /// the structure.  If this returns true then @defs is invalid for this
  /// runtime and a warning should be generated.
  virtual bool LateBoundIVars() { return false; }
};

/// Creates an instance of an Objective-C runtime class.  
//TODO: This should include some way of selecting which runtime to target.
CGObjCRuntime *CreateObjCRuntime(llvm::Module &M,
                                 const llvm::Type *LLVMIntType,
                                 const llvm::Type *LLVMLongType);
}
}
#endif
