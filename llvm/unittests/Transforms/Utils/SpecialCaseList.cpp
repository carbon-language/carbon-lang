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

  GlobalAlias *makeAlias(StringRef Name, GlobalValue *Aliasee) {
    return new GlobalAlias(Aliasee->getType()->getElementType(),
                           GlobalValue::ExternalLinkage, Name, Aliasee,
                           Aliasee->getParent());
  }

  SpecialCaseList *makeSpecialCaseList(StringRef List, std::string &Error) {
    std::unique_ptr<MemoryBuffer> MB(MemoryBuffer::getMemBuffer(List));
    return SpecialCaseList::create(MB.get(), Error);
  }

  SpecialCaseList *makeSpecialCaseList(StringRef List) {
    std::string Error;
    SpecialCaseList *SCL = makeSpecialCaseList(List, Error);
    assert(SCL);
    assert(Error == "");
    return SCL;
  }

  LLVMContext Ctx;
};

TEST_F(SpecialCaseListTest, ModuleIsIn) {
  Module M("hello", Ctx);
  Function *F = makeFunction("foo", M);
  GlobalVariable *GV = makeGlobal("bar", "t", M);

  std::unique_ptr<SpecialCaseList> SCL(
      makeSpecialCaseList("# This is a comment.\n"
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

  std::unique_ptr<SpecialCaseList> SCL(makeSpecialCaseList("fun:foo\n"));
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
  EXPECT_FALSE(SCL->isIn(*Bar, "functional"));
}

TEST_F(SpecialCaseListTest, GlobalIsIn) {
  Module M("hello", Ctx);
  GlobalVariable *Foo = makeGlobal("foo", "t1", M);
  GlobalVariable *Bar = makeGlobal("bar", "t2", M);

  std::unique_ptr<SpecialCaseList> SCL(makeSpecialCaseList("global:foo\n"));
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

TEST_F(SpecialCaseListTest, AliasIsIn) {
  Module M("hello", Ctx);
  Function *Foo = makeFunction("foo", M);
  GlobalVariable *Bar = makeGlobal("bar", "t", M);
  GlobalAlias *FooAlias = makeAlias("fooalias", Foo);
  GlobalAlias *BarAlias = makeAlias("baralias", Bar);

  std::unique_ptr<SpecialCaseList> SCL(makeSpecialCaseList("fun:foo\n"));
  EXPECT_FALSE(SCL->isIn(*FooAlias));
  EXPECT_FALSE(SCL->isIn(*BarAlias));

  SCL.reset(makeSpecialCaseList("global:bar\n"));
  EXPECT_FALSE(SCL->isIn(*FooAlias));
  EXPECT_FALSE(SCL->isIn(*BarAlias));

  SCL.reset(makeSpecialCaseList("global:fooalias\n"));
  EXPECT_FALSE(SCL->isIn(*FooAlias));
  EXPECT_FALSE(SCL->isIn(*BarAlias));

  SCL.reset(makeSpecialCaseList("fun:fooalias\n"));
  EXPECT_TRUE(SCL->isIn(*FooAlias));
  EXPECT_FALSE(SCL->isIn(*BarAlias));

  SCL.reset(makeSpecialCaseList("global:baralias=init\n"));
  EXPECT_FALSE(SCL->isIn(*FooAlias, "init"));
  EXPECT_TRUE(SCL->isIn(*BarAlias, "init"));

  SCL.reset(makeSpecialCaseList("type:t=init\n"));
  EXPECT_FALSE(SCL->isIn(*FooAlias, "init"));
  EXPECT_TRUE(SCL->isIn(*BarAlias, "init"));

  SCL.reset(makeSpecialCaseList("fun:baralias=init\n"));
  EXPECT_FALSE(SCL->isIn(*FooAlias, "init"));
  EXPECT_FALSE(SCL->isIn(*BarAlias, "init"));
}

TEST_F(SpecialCaseListTest, Substring) {
  Module M("othello", Ctx);
  Function *F = makeFunction("tomfoolery", M);
  GlobalVariable *GV = makeGlobal("bartender", "t", M);
  GlobalAlias *GA1 = makeAlias("buffoonery", F);
  GlobalAlias *GA2 = makeAlias("foobar", GV);

  std::unique_ptr<SpecialCaseList> SCL(makeSpecialCaseList("src:hello\n"
                                                           "fun:foo\n"
                                                           "global:bar\n"));
  EXPECT_FALSE(SCL->isIn(M));
  EXPECT_FALSE(SCL->isIn(*F));
  EXPECT_FALSE(SCL->isIn(*GV));
  EXPECT_FALSE(SCL->isIn(*GA1));
  EXPECT_FALSE(SCL->isIn(*GA2));

  SCL.reset(makeSpecialCaseList("fun:*foo*\n"));
  EXPECT_TRUE(SCL->isIn(*F));
  EXPECT_TRUE(SCL->isIn(*GA1));
}

TEST_F(SpecialCaseListTest, InvalidSpecialCaseList) {
  std::string Error;
  EXPECT_EQ(0, makeSpecialCaseList("badline", Error));
  EXPECT_EQ("Malformed line 1: 'badline'", Error);
  EXPECT_EQ(0, makeSpecialCaseList("src:bad[a-", Error));
  EXPECT_EQ("Malformed regex in line 1: 'bad[a-': invalid character range",
            Error);
  EXPECT_EQ(0, makeSpecialCaseList("src:a.c\n"
                                   "fun:fun(a\n",
                                   Error));
  EXPECT_EQ("Malformed regex in line 2: 'fun(a': parentheses not balanced",
            Error);
  EXPECT_EQ(0, SpecialCaseList::create("unexisting", Error));
  EXPECT_EQ(0U, Error.find("Can't open file 'unexisting':"));
}

TEST_F(SpecialCaseListTest, EmptySpecialCaseList) {
  std::unique_ptr<SpecialCaseList> SCL(makeSpecialCaseList(""));
  Module M("foo", Ctx);
  EXPECT_FALSE(SCL->isIn(M));
}

}
