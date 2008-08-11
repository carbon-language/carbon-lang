//===------- CGObjCMac.cpp - Interface to Apple Objective-C Runtime -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides Objective-C code generation targetting the Apple runtime.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "llvm/Module.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <map>

using namespace clang;

namespace {
class CGObjCMac : public CodeGen::CGObjCRuntime {
private:
  CodeGen::CodeGenModule &CGM;

public:
  CGObjCMac(CodeGen::CodeGenModule &cgm);
  virtual llvm::Constant *GenerateConstantString(const char *String, 
                                                 const size_t length);

  virtual llvm::Value *GenerateMessageSend(llvm::IRBuilder<> &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Sender,
                                           llvm::Value *Receiver,
                                           Selector Sel,
                                           llvm::Value** ArgV,
                                           unsigned ArgC);

  virtual llvm::Value *GenerateMessageSendSuper(llvm::IRBuilder<> &Builder,
                                                const llvm::Type *ReturnTy,
                                                llvm::Value *Sender,
                                                const char *SuperClassName,
                                                llvm::Value *Receiver,
                                                Selector Sel,
                                                llvm::Value** ArgV,
                                                unsigned ArgC);

  virtual llvm::Value *LookupClass(llvm::IRBuilder<> &Builder,
                                   llvm::Value *ClassName);

  virtual llvm::Value *GetSelector(llvm::IRBuilder<> &Builder, Selector Sel);
  
  virtual llvm::Function *MethodPreamble(const std::string &ClassName,
                                         const std::string &CategoryName,
                                         const std::string &MethodName,
                                         const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isClassMethod,
                                         bool isVarArg);

  virtual void GenerateCategory(const char *ClassName, const char *CategoryName,
           const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols);

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
           const llvm::SmallVectorImpl<std::string> &Protocols);

  virtual llvm::Value *GenerateProtocolRef(llvm::IRBuilder<> &Builder,
                                           const char *ProtocolName);

  virtual void GenerateProtocol(const char *ProtocolName,
      const llvm::SmallVectorImpl<std::string> &Protocols,
      const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
      const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
      const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
      const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes);

  virtual llvm::Function *ModuleInitFunction();
};
} // end anonymous namespace
 
CGObjCMac::CGObjCMac(CodeGen::CodeGenModule &cgm) : CGM(cgm) {
}

// This has to perform the lookup every time, since posing and related
// techniques can modify the name -> class mapping.
llvm::Value *CGObjCMac::LookupClass(llvm::IRBuilder<> &Builder,                                    
                                    llvm::Value *ClassName) {
  assert(0 && "Cannot lookup classes on Mac runtime.");
  return 0;
}

/// GetSelector - Return the pointer to the unique'd string for this selector.
llvm::Value *CGObjCMac::GetSelector(llvm::IRBuilder<> &Builder, Selector Sel) {
  assert(0 && "Cannot get selector on Mac runtime.");
  return 0;
}

/// Generate an NSConstantString object.
llvm::Constant *CGObjCMac::GenerateConstantString(const char *String, 
                                                  const size_t length) {
  assert(0 && "Cannot generate constant string for Mac runtime.");
  return 0;
}

/// Generates a message send where the super is the receiver.  This is
/// a message send to self with special delivery semantics indicating
/// which class's method should be called.
llvm::Value *CGObjCMac::GenerateMessageSendSuper(llvm::IRBuilder<> &Builder,
                                                 const llvm::Type *ReturnTy,
                                                 llvm::Value *Sender,
                                                 const char *SuperClassName,
                                                 llvm::Value *Receiver,
                                                 Selector Sel,
                                                 llvm::Value** ArgV,
                                                 unsigned ArgC) {
  assert(0 && "Cannot generate message send to super for Mac runtime.");
  return 0;
}

/// Generate code for a message send expression.  
llvm::Value *CGObjCMac::GenerateMessageSend(llvm::IRBuilder<> &Builder,
                                            const llvm::Type *ReturnTy,
                                            llvm::Value *Sender,
                                            llvm::Value *Receiver,
                                            Selector Sel,
                                            llvm::Value** ArgV,
                                            unsigned ArgC) {
  assert(0 && "Cannot generate message send for Mac runtime.");
  return 0;
}

llvm::Value *CGObjCMac::GenerateProtocolRef(llvm::IRBuilder<> &Builder, 
                                            const char *ProtocolName) {
  assert(0 && "Cannot get protocol reference on Mac runtime.");
  return 0;
}

void CGObjCMac::GenerateProtocol(const char *ProtocolName,
    const llvm::SmallVectorImpl<std::string> &Protocols,
    const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
    const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodNames,
    const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes) {
  assert(0 && "Cannot generate protocol for Mac runtime.");
}

void CGObjCMac::GenerateCategory(
           const char *ClassName,
           const char *CategoryName,
           const llvm::SmallVectorImpl<Selector>  &InstanceMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &InstanceMethodTypes,
           const llvm::SmallVectorImpl<Selector>  &ClassMethodSels,
           const llvm::SmallVectorImpl<llvm::Constant *>  &ClassMethodTypes,
           const llvm::SmallVectorImpl<std::string> &Protocols) {
  assert(0 && "Cannot generate category for Mac runtime.");
}

void CGObjCMac::GenerateClass(
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
           const llvm::SmallVectorImpl<std::string> &Protocols) {
  assert(0 && "Cannot generate class for Mac runtime.");
}

llvm::Function *CGObjCMac::ModuleInitFunction() { 
  return NULL;
}

llvm::Function *CGObjCMac::MethodPreamble(
                                         const std::string &ClassName,
                                         const std::string &CategoryName,
                                         const std::string &MethodName,
                                         const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isClassMethod,
                                         bool isVarArg) {
  assert(0 && "Cannot generate method preamble for Mac runtime.");
  return 0;
}

CodeGen::CGObjCRuntime *CodeGen::CreateMacObjCRuntime(CodeGen::CodeGenModule &CGM){
  return new CGObjCMac(CGM);
}
