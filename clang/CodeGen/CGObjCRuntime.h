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
  class LLVMFoldingBuilder;
  class Constant;
  class Type;
  class Value;
  class Module;
}

namespace clang {
namespace CodeGen {

// Implements runtime-specific code generation functions
class CGObjCRuntime {
public:
  virtual ~CGObjCRuntime();
  
  // Generate an Objective-C message send operation
  virtual llvm::Value *generateMessageSend(llvm::LLVMFoldingBuilder &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Receiver,
                                           llvm::Constant *Selector,
                                           llvm::Value** ArgV,
                                           unsigned ArgC) = 0;
};

CGObjCRuntime *CreateObjCRuntime(llvm::Module &M);
}
}
#endif
