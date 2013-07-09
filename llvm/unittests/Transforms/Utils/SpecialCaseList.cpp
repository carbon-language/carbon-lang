//===- SpecialCaseList.cpp - Unit tests for SpecialCaseList ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/Utils/SpecialCaseList.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class SpecialCaseListTest : public ::testing::Test {
protected:
  Function *makeFunction(StringRef Name, Module &M) {
    return Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                            GlobalValue::ExternalLinkage,
                            Name,
                            &M);
  }

  GlobalVariable *makeGlobal(StringRef Name, StringRef StructName, Module &M) {
    StructType *ST =
        StructType::create(StructName, Type::getInt32Ty(Ctx), (Type*)0);
    return new GlobalVariable(
        M, ST, false, GlobalValue::ExternalLinkage, 0, Name);
  }

  SpecialCaseList *makeSpecialCaseList(StringRef List) {
    OwningPtr<MemoryBuffer> MB(MemoryBuffer::getMemBuffer(List));
    return new SpecialCaseList(MB.get());
  }

  LLVMContext Ctx;
};

TEST_F(SpecialCaseListTest, ModuleIsIn) {
  Module M("hello", Ctx);
  Function *F = makeFunction("foo", M);
  GlobalVariable *GV = makeGlobal("bar", "t", M);

  OwningPtr<SpecialCaseList> SCL(makeSpecialCaseList("# This is a comment.\n"
                                                     "\n"
                                                     "src:hello\n"));
  EXPECT_TRUE(SCL->isIn(M));
  EXPECT_TRUE(SCL->isIn(*F));
  EXPECT_TRUE(SCL->isIn(*GV));

  SCL.reset(makeSpecialCaseList("src:he*o\n"));
  EXPECT_TRUE(SCL->isIn(M));
  EXPECT_TRUE(SCL->isIn(*F));
  EXPECT_TRUE(SCL->isIn(*GV));

  SCL.reset(makeSpecialCaseList("src:hi\n"));
  EXPECT_FALSE(SCL->isIn(M));
  EXPECT_FALSE(SCL->isIn(*F));
  EXPECT_FALSE(SCL->isIn(*GV));
}

TEST_F(SpecialCaseListTest, FunctionIsIn) {
  Module M("hello", Ctx);
  Function *Foo = makeFunction("foo", M);
  Function *Bar = makeFunction("bar", M);

  OwningPtr<SpecialCaseList> SCL(makeSpecialCaseList("fun:foo\n"));
  EXPECT_TRUE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));

  SCL.reset(makeSpecialCaseList("fun:b*\n"));
  EXPECT_FALSE(SCL->isIn(*Foo));
  EXPECT_TRUE(SCL->isIn(*Bar));

  SCL.reset(makeSpecialCaseList("fun:f*\n"
                                "fun:bar\n"));
  EXPECT_TRUE(SCL->isIn(*Foo));
  EXPECT_TRUE(SCL->isIn(*Bar));

  SCL.reset(makeSpecialCaseList("fun:foo=functional\n"));
  EXPECT_TRUE(SCL->isIn(*Foo, "functional"));
  StringRef Category;
  EXPECT_TRUE(SCL->findCategory(*Foo, Category));
  EXPECT_EQ("functional", Category);
  EXPECT_FALSE(SCL->isIn(*Bar, "functional"));
}

TEST_F(SpecialCaseListTest, GlobalIsIn) {
  Module M("hello", Ctx);
  GlobalVariable *Foo = makeGlobal("foo", "t1", M);
  GlobalVariable *Bar = makeGlobal("bar", "t2", M);

  OwningPtr<SpecialCaseList> SCL(makeSpecialCaseList("global:foo\n"));
  EXPECT_TRUE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));
  EXPECT_FALSE(SCL->isIn(*Foo, "init"));
  EXPECT_FALSE(SCL->isIn(*Bar, "init"));

  SCL.reset(makeSpecialCaseList("global:foo=init\n"));
  EXPECT_FALSE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));
  EXPECT_TRUE(SCL->isIn(*Foo, "init"));
  EXPECT_FALSE(SCL->isIn(*Bar, "init"));

  SCL.reset(makeSpecialCaseList("global-init:foo\n"));
  EXPECT_FALSE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));
  EXPECT_TRUE(SCL->isIn(*Foo, "init"));
  EXPECT_FALSE(SCL->isIn(*Bar, "init"));

  SCL.reset(makeSpecialCaseList("type:t2=init\n"));
  EXPECT_FALSE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));
  EXPECT_FALSE(SCL->isIn(*Foo, "init"));
  EXPECT_TRUE(SCL->isIn(*Bar, "init"));

  SCL.reset(makeSpecialCaseList("global-init-type:t2\n"));
  EXPECT_FALSE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));
  EXPECT_FALSE(SCL->isIn(*Foo, "init"));
  EXPECT_TRUE(SCL->isIn(*Bar, "init"));

  SCL.reset(makeSpecialCaseList("src:hello=init\n"));
  EXPECT_FALSE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));
  EXPECT_TRUE(SCL->isIn(*Foo, "init"));
  EXPECT_TRUE(SCL->isIn(*Bar, "init"));

  SCL.reset(makeSpecialCaseList("global-init-src:hello\n"));
  EXPECT_FALSE(SCL->isIn(*Foo));
  EXPECT_FALSE(SCL->isIn(*Bar));
  EXPECT_TRUE(SCL->isIn(*Foo, "init"));
  EXPECT_TRUE(SCL->isIn(*Bar, "init"));
}

}
