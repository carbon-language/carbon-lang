//===- llvm/unittest/Linker/LinkModulesTest.cpp - IRBuilder tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker/Linker.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class LinkModuleTest : public testing::Test {
protected:
  virtual void SetUp() {
    M.reset(new Module("MyModule", Ctx));
    FunctionType *FTy = FunctionType::get(
        Type::getInt8PtrTy(Ctx), Type::getInt32Ty(Ctx), false /*=isVarArg*/);
    F = Function::Create(FTy, Function::ExternalLinkage, "ba_func", M.get());
    F->setCallingConv(CallingConv::C);

    EntryBB = BasicBlock::Create(Ctx, "entry", F);
    SwitchCase1BB = BasicBlock::Create(Ctx, "switch.case.1", F);
    SwitchCase2BB = BasicBlock::Create(Ctx, "switch.case.2", F);
    ExitBB = BasicBlock::Create(Ctx, "exit", F);

    ArrayType *AT = ArrayType::get(Type::getInt8PtrTy(Ctx), 3);

    GV = new GlobalVariable(*M.get(), AT, false /*=isConstant*/,
                            GlobalValue::InternalLinkage, 0, "switch.bas");

    // Global Initializer
    std::vector<Constant *> Init;
    Constant *SwitchCase1BA = BlockAddress::get(SwitchCase1BB);
    Init.push_back(SwitchCase1BA);

    Constant *SwitchCase2BA = BlockAddress::get(SwitchCase2BB);
    Init.push_back(SwitchCase2BA);

    ConstantInt *One = ConstantInt::get(Type::getInt32Ty(Ctx), 1);
    Constant *OnePtr = ConstantExpr::getCast(Instruction::IntToPtr, One,
                                             Type::getInt8PtrTy(Ctx));
    Init.push_back(OnePtr);

    GV->setInitializer(ConstantArray::get(AT, Init));
  }

  virtual void TearDown() { M.reset(); }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  GlobalVariable *GV;
  BasicBlock *EntryBB;
  BasicBlock *SwitchCase1BB;
  BasicBlock *SwitchCase2BB;
  BasicBlock *ExitBB;
};

TEST_F(LinkModuleTest, BlockAddress) {
  IRBuilder<> Builder(EntryBB);

  std::vector<Value *> GEPIndices;
  GEPIndices.push_back(ConstantInt::get(Type::getInt32Ty(Ctx), 0));
  GEPIndices.push_back(F->arg_begin());

  Value *GEP = Builder.CreateGEP(GV, GEPIndices, "switch.gep");
  Value *Load = Builder.CreateLoad(GEP, "switch.load");

  Builder.CreateRet(Load);

  Builder.SetInsertPoint(SwitchCase1BB);
  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(SwitchCase2BB);
  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRet(ConstantPointerNull::get(Type::getInt8PtrTy(Ctx)));

  Module *LinkedModule = new Module("MyModuleLinked", Ctx);
  Linker::LinkModules(LinkedModule, M.get(), Linker::PreserveSource, 0);

  // Delete the original module.
  M.reset();

  // Check that the global "@switch.bas" is well-formed.
  const GlobalVariable *LinkedGV = LinkedModule->getNamedGlobal("switch.bas");
  const Constant *Init = LinkedGV->getInitializer();

  // @switch.bas = internal global [3 x i8*]
  //   [i8* blockaddress(@ba_func, %switch.case.1),
  //    i8* blockaddress(@ba_func, %switch.case.2),
  //    i8* inttoptr (i32 1 to i8*)]

  ArrayType *AT = ArrayType::get(Type::getInt8PtrTy(Ctx), 3);
  EXPECT_EQ(AT, Init->getType());

  Value *Elem = Init->getOperand(0);
  ASSERT_TRUE(isa<BlockAddress>(Elem));
  EXPECT_EQ(cast<BlockAddress>(Elem)->getFunction(),
            LinkedModule->getFunction("ba_func"));
  EXPECT_EQ(cast<BlockAddress>(Elem)->getBasicBlock()->getParent(),
            LinkedModule->getFunction("ba_func"));

  Elem = Init->getOperand(1);
  ASSERT_TRUE(isa<BlockAddress>(Elem));
  EXPECT_EQ(cast<BlockAddress>(Elem)->getFunction(),
            LinkedModule->getFunction("ba_func"));
  EXPECT_EQ(cast<BlockAddress>(Elem)->getBasicBlock()->getParent(),
            LinkedModule->getFunction("ba_func"));

  delete LinkedModule;
}

TEST_F(LinkModuleTest, EmptyModule) {
  Module *InternalM = new Module("InternalModule", Ctx);
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx), Type::getInt8PtrTy(Ctx), false /*=isVarArgs*/);

  F = Function::Create(FTy, Function::InternalLinkage, "bar", InternalM);
  F->setCallingConv(CallingConv::C);

  BasicBlock *BB = BasicBlock::Create(Ctx, "", F);
  IRBuilder<> Builder(BB);
  Builder.CreateRetVoid();

  StructType *STy = StructType::create(Ctx, PointerType::get(FTy, 0));

  GlobalVariable *GV =
      new GlobalVariable(*InternalM, STy, false /*=isConstant*/,
                         GlobalValue::InternalLinkage, 0, "g");

  GV->setInitializer(ConstantStruct::get(STy, F));

  Module *EmptyM = new Module("EmptyModule1", Ctx);
  Linker::LinkModules(EmptyM, InternalM, Linker::PreserveSource, 0);

  delete EmptyM;
  EmptyM = new Module("EmptyModule2", Ctx);
  Linker::LinkModules(InternalM, EmptyM, Linker::PreserveSource, 0);

  delete EmptyM;
  delete InternalM;
}

} // end anonymous namespace
