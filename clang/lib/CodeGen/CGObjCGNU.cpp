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
#include "llvm/Support/IRBuilder.h"
#include "llvm/ADT/SmallVector.h"

namespace {
class CGObjCGNU : public clang::CodeGen::CGObjCRuntime {
private:
  llvm::Module &TheModule;
  const llvm::Type *SelectorTy;
  const llvm::Type *PtrToInt8Ty;
  const llvm::Type *IMPTy;
  const llvm::Type *IdTy;
  const llvm::Type *IntTy;
  const llvm::Type *PtrTy;
  const llvm::Type *LongTy;
  const llvm::Type *PtrToIntTy;
public:
  CGObjCGNU(llvm::Module &Mp,
    const llvm::Type *LLVMIntType,
    const llvm::Type *LLVMLongType);
  virtual llvm::Value *generateMessageSend(llvm::IRBuilder &Builder,
                                           const llvm::Type *ReturnTy,
                                           llvm::Value *Sender,
                                           llvm::Value *Receiver,
                                           llvm::Value *Selector,
                                           llvm::Value** ArgV,
                                           unsigned ArgC);
  llvm::Value *getSelector(llvm::IRBuilder &Builder,
      llvm::Value *SelName,
      llvm::Value *SelTypes);
  virtual llvm::Function *MethodPreamble(const llvm::Type *ReturnTy,
                                 const llvm::Type *SelfTy,
                                 const llvm::Type **ArgTy,
                                 unsigned ArgC,
                                 bool isVarArg);
};
} // end anonymous namespace

CGObjCGNU::CGObjCGNU(llvm::Module &M,
    const llvm::Type *LLVMIntType,
    const llvm::Type *LLVMLongType) : 
  TheModule(M),
  IntTy(LLVMIntType),
  LongTy(LLVMLongType)
{
  // C string type.  Used in lots of places.
  PtrToInt8Ty = 
    llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  // Get the selector Type.
  const llvm::Type *SelStructTy = llvm::StructType::get(
      PtrToInt8Ty,
      PtrToInt8Ty,
      NULL);
  SelectorTy = llvm::PointerType::getUnqual(SelStructTy);
  PtrToIntTy = llvm::PointerType::getUnqual(IntTy);
  PtrTy = PtrToInt8Ty;
 
  // Object type
  llvm::PATypeHolder OpaqueObjTy = llvm::OpaqueType::get();
  llvm::Type *OpaqueIdTy = llvm::PointerType::getUnqual(OpaqueObjTy);
  IdTy = llvm::StructType::get(OpaqueIdTy, NULL);
  llvm::cast<llvm::OpaqueType>(OpaqueObjTy.get())->refineAbstractTypeTo(IdTy);
  IdTy = llvm::cast<llvm::StructType>(OpaqueObjTy.get());
  IdTy = llvm::PointerType::getUnqual(IdTy);
 
  // IMP type
  std::vector<const llvm::Type*> IMPArgs;
  IMPArgs.push_back(IdTy);
  IMPArgs.push_back(SelectorTy);
  IMPTy = llvm::FunctionType::get(IdTy, IMPArgs, true);

}

/// Looks up the selector for the specified name / type pair.
// FIXME: Selectors should be statically cached, not looked up on every call.
llvm::Value *CGObjCGNU::getSelector(llvm::IRBuilder &Builder,
    llvm::Value *SelName,
    llvm::Value *SelTypes)
{
  // Look up the selector.
  llvm::Value *cmd;
  if(SelTypes == 0) {
    llvm::Constant *SelFunction = TheModule.getOrInsertFunction("sel_get_uid", 
        SelectorTy, 
        PtrToInt8Ty, 
        NULL);
    cmd = Builder.CreateCall(SelFunction, SelName);
  }
  else {
    llvm::Constant *SelFunction = 
      TheModule.getOrInsertFunction("sel_get_typed_uid",
          SelectorTy,
          PtrToInt8Ty,
          PtrToInt8Ty,
          NULL);
    llvm::Value *Args[] = { SelName, SelTypes };
    cmd = Builder.CreateCall(SelFunction, Args, Args+2);
  }
  return cmd;
}


/// Generate code for a message send expression on the GNU runtime.
// FIXME: Much of this code will need factoring out later.
// TODO: This should take a sender argument (pointer to self in the calling
// context)
llvm::Value *CGObjCGNU::generateMessageSend(llvm::IRBuilder &Builder,
                                            const llvm::Type *ReturnTy,
                                            llvm::Value *Sender,
                                            llvm::Value *Receiver,
                                            llvm::Value *Selector,
                                            llvm::Value** ArgV,
                                            unsigned ArgC) {
  llvm::Value *cmd = getSelector(Builder, Selector, 0);

  // Look up the method implementation.
  std::vector<const llvm::Type*> impArgTypes;
  impArgTypes.push_back(Receiver->getType());
  impArgTypes.push_back(SelectorTy);
  
  // Avoid an explicit cast on the IMP by getting a version that has the right
  // return type.
  llvm::FunctionType *impType = llvm::FunctionType::get(ReturnTy, impArgTypes,
                                                        true);
  
  llvm::Constant *lookupFunction = 
     TheModule.getOrInsertFunction("objc_msg_lookup",
                                   llvm::PointerType::getUnqual(impType),
                                   Receiver->getType(), SelectorTy, NULL);
  llvm::SmallVector<llvm::Value*, 16> lookupArgs;
  lookupArgs.push_back(Receiver);
  lookupArgs.push_back(cmd);
  llvm::Value *imp = Builder.CreateCall(lookupFunction,
                                        lookupArgs.begin(), lookupArgs.end());

  // Call the method.
  lookupArgs.insert(lookupArgs.end(), ArgV, ArgV+ArgC);
  return Builder.CreateCall(imp, lookupArgs.begin(), lookupArgs.end());
}

llvm::Function *CGObjCGNU::MethodPreamble(
                                         const llvm::Type *ReturnTy,
                                         const llvm::Type *SelfTy,
                                         const llvm::Type **ArgTy,
                                         unsigned ArgC,
                                         bool isVarArg) {
  std::vector<const llvm::Type*> Args;
  Args.push_back(SelfTy);
  Args.push_back(SelectorTy);
  Args.insert(Args.end(), ArgTy, ArgTy+ArgC);

  llvm::FunctionType *MethodTy = llvm::FunctionType::get(ReturnTy,
      Args,
      isVarArg);
  llvm::Function *Method = llvm::Function::Create(MethodTy,
      llvm::GlobalValue::InternalLinkage,
      ".objc.method",
      &TheModule);
  // Set the names of the hidden arguments
  llvm::Function::arg_iterator AI = Method->arg_begin();
  AI->setName("self");
  ++AI;
  AI->setName("_cmd");
  return Method;
}

clang::CodeGen::CGObjCRuntime *clang::CodeGen::CreateObjCRuntime(
    llvm::Module &M,
    const llvm::Type *LLVMIntType,
    const llvm::Type *LLVMLongType) {
  return new CGObjCGNU(M, LLVMIntType, LLVMLongType);
}
