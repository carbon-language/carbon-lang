//===----- CGObjCRuntime.h - Interface to ObjC Runtimes ---------*- C++ -*-===//
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
#include "clang/Basic/IdentifierTable.h" // Selector
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/IRBuilder.h"
#include <string>

namespace llvm {
  class Constant;
  class Type;
  class Value;
  class Module;
  class Function;
}

namespace clang {
  class Selector;
  
namespace CodeGen {
  class CodeGenModule;

//FIXME Several methods should be pure virtual but aren't to avoid the
//partially-implemented subclass breaking.

/// Implements runtime-specific code generation functions.
class CGObjCRuntime {
  typedef llvm::IRBuilder<> BuilderType;

public:
  virtual ~CGObjCRuntime();
  
  /// Generate an Objective-C message send operation
  virtual llvm::Value *GenerateMessageSend(BuilderType &Builder,
                                           const llvm::Type *ReturnTy,
                                           // FIXME: This should be
                                           // dropped, it is unused
                                           // and generates a spurious
                                           // load.
                                           llvm::Value *Sender,
                                           llvm::Value *Receiver,
                                           Selector Sel,
                                           llvm::Value** ArgV,
                                           unsigned ArgC) =0;
  /// Generate the function required to register all Objective-C components in
  /// this compilation unit with the runtime library.
  virtual llvm::Function *ModuleInitFunction() =0;
  /// Get a selector for the specified name and type values
  virtual llvm::Value *GetSelector(BuilderType &Builder,
                                   Selector Sel) =0;
  /// Generate a constant string object
  virtual llvm::Constant *GenerateConstantString(const std::string &String) = 0;
  /// Generate a category.  A category contains a list of methods (and
  /// accompanying metadata) and a list of protocols.
  virtual void GenerateCategory(const char *ClassName, const char *CategoryName,
             const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
             const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
             const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
             const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
             const llvm::SmallVectorImpl<std::string> &Protocols) =0;
  /// Generate a class stucture for this class.
  virtual void GenerateClass(
             const char *ClassName,
             const char *SuperClassName,
             const int instanceSize,
             const llvm::SmallVectorImpl<llvm::Constant *>  &IvarNames,
             const llvm::SmallVectorImpl<llvm::Constant *>  &IvarTypes,
             const llvm::SmallVectorImpl<llvm::Constant *>  &IvarOffsets,
             const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
             const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
             const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
             const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
             const llvm::SmallVectorImpl<std::string> &Protocols) =0;
  virtual llvm::Value *GenerateMessageSendSuper(llvm::IRBuilder<true> &Builder,
                                                const llvm::Type *ReturnTy,
                                                // FIXME: This should
                                                // be dropped, it is
                                                // unused and
                                                // generates a
                                                // spurious load.
                                                llvm::Value *Sender,
                                                const char *SuperClassName,
                                                llvm::Value *Receiver,
                                                Selector Sel,
                                                llvm::Value** ArgV,
                                                unsigned ArgC) = 0;

  /// Emit the code to return the named protocol as an object, as in a
  /// @protocol expression.
  virtual llvm::Value *GenerateProtocolRef(llvm::IRBuilder<true> &Builder,
                                           const char *ProtocolName) = 0;

  /// Generate the named protocol.  Protocols contain method metadata but no 
  /// implementations. 
  virtual void GenerateProtocol(const char *ProtocolName,
    const llvm::SmallVectorImpl<std::string> &Protocols,
    const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
    const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes) =0;
  /// Generate a function preamble for a method with the specified types
  virtual llvm::Function *MethodPreamble(
                                         const std::string &ClassName,
                                         const std::string &CategoryName,
                                         const std::string &MethodName,
                                         const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isClassMethod,
                                         bool isVarArg) = 0;
  /// Look up the class for the specified name
  virtual llvm::Value *LookupClass(BuilderType &Builder, 
                                   llvm::Value *ClassName) =0;
  /// If instance variable addresses are determined at runtime then this should
  /// return true, otherwise instance variables will be accessed directly from
  /// the structure.  If this returns true then @defs is invalid for this
  /// runtime and a warning should be generated.
  virtual bool LateBoundIVars() { return false; }
};

/// Creates an instance of an Objective-C runtime class.  
//TODO: This should include some way of selecting which runtime to target.
CGObjCRuntime *CreateGNUObjCRuntime(CodeGenModule &CGM);
CGObjCRuntime *CreateMacObjCRuntime(CodeGenModule &CGM);
}
}
#endif
