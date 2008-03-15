//===------- CGObjCGNU.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides Objective-C code generation targetting the GNU runtime.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "llvm/Module.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/LLVMBuilder.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::CodeGen;
using namespace clang;

CGObjCRuntime::~CGObjCRuntime() {}

namespace {
class CGObjCGNU : public CGObjCRuntime {
private:
  llvm::Module &TheModule;
public:
  CGObjCGNU(llvm::Module &M) : TheModule(M) {};
  virtual llvm::Value *generateMessageSend(llvm::LLVMFoldingBuilder &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Receiver,
                                           llvm::Constant *Selector,
                                           llvm::Value** ArgV,
                                           unsigned ArgC);
};
} // end anonymous namespace

// Generate code for a message send expression on the GNU runtime.
// BIG FAT WARNING: Much of this code will need factoring out later.
// FIXME: This currently only handles id returns.  Other return types 
// need some explicit casting.
llvm::Value *CGObjCGNU::generateMessageSend(llvm::LLVMFoldingBuilder &Builder,
                                            const llvm::Type *ReturnTy,
                                            llvm::Value *Receiver,
                                            llvm::Constant *Selector,
                                            llvm::Value** ArgV,
                                            unsigned ArgC) {
  // Get the selector Type.
  const llvm::Type *PtrToInt8Ty = 
    llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  std::vector<const llvm::Type*> Str2(2, PtrToInt8Ty);
  const llvm::Type *SelStructTy = llvm::StructType::get(Str2);
  const llvm::Type *SelTy = llvm::PointerType::getUnqual(SelStructTy);

  // Look up the selector.
  // If we haven't got the selector lookup function, look it up now.
  // TODO: Factor this out and use it to implement @selector() too.
  llvm::Constant *SelFunction = 
    TheModule.getOrInsertFunction("sel_get_uid", SelTy, PtrToInt8Ty, NULL);
  // FIXME: Selectors should be statically cached, not looked up on every call.

  // TODO: Pull this out into the caller.
  llvm::Constant *Idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *Ops[] = {Idx0, Idx0};
  llvm::Value *SelStr = llvm::ConstantExpr::getGetElementPtr(Selector, Ops, 2);
  llvm::Value *cmd = Builder.CreateCall(SelFunction, &SelStr, &SelStr+1);

  // Look up the method implementation.
  std::vector<const llvm::Type*> impArgTypes;
  impArgTypes.push_back(Receiver->getType());
  impArgTypes.push_back(SelTy);
  
  // Avoid an explicit cast on the IMP by getting a version that has the right
  // return type.
  llvm::FunctionType *impType = llvm::FunctionType::get(ReturnTy, impArgTypes,
                                                        true);
  
  llvm::Constant *lookupFunction = 
     TheModule.getOrInsertFunction("objc_msg_lookup",
                                   llvm::PointerType::get(impType, 0),
                                   Receiver->getType(), SelTy, NULL);
  llvm::SmallVector<llvm::Value*, 16> lookupArgs;
  lookupArgs.push_back(Receiver);
  lookupArgs.push_back(cmd);
  llvm::Value *imp = Builder.CreateCall(lookupFunction,
                                        lookupArgs.begin(), lookupArgs.end());

  // Call the method.
  lookupArgs.insert(lookupArgs.end(), ArgV, ArgV+ArgC);
  return Builder.CreateCall(imp, lookupArgs.begin(), lookupArgs.end());
}

CGObjCRuntime * clang::CodeGen::CreateObjCRuntime(llvm::Module &M) {
  return new CGObjCGNU(M);
}
