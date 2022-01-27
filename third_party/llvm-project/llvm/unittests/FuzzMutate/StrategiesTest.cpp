//===- InjectorIRStrategyTest.cpp - Tests for injector strategy -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr int Seed = 5;

namespace {

std::unique_ptr<IRMutator> createInjectorMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(
      std::make_unique<InjectorIRStrategy>(
          InjectorIRStrategy::getDefaultOps()));

  return std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

std::unique_ptr<IRMutator> createDeleterMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(std::make_unique<InstDeleterIRStrategy>());

  return std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

std::unique_ptr<IRMutator> createInstModifierMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(std::make_unique<InstModificationIRStrategy>());

  return std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

std::unique_ptr<Module> parseAssembly(
    const char *Assembly, LLVMContext &Context) {

  SMDiagnostic Error;
  std::unique_ptr<Module> M = parseAssemblyString(Assembly, Error, Context);

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  assert(M && !verifyModule(*M, &errs()));
  return M;
}

void IterateOnSource(StringRef Source, IRMutator &Mutator) {
  LLVMContext Ctx;

  for (int i = 0; i < 10; ++i) {
    auto M = parseAssembly(Source.data(), Ctx);
    ASSERT_TRUE(M && !verifyModule(*M, &errs()));

    Mutator.mutateModule(*M, Seed, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
  }
}

TEST(InjectorIRStrategyTest, EmptyModule) {
  // Test that we can inject into empty module

  LLVMContext Ctx;
  auto M = std::make_unique<Module>("M", Ctx);
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));

  auto Mutator = createInjectorMutator();
  ASSERT_TRUE(Mutator);

  Mutator->mutateModule(*M, Seed, 1, 1);
  EXPECT_TRUE(!verifyModule(*M, &errs()));
}

TEST(InstDeleterIRStrategyTest, EmptyFunction) {
  // Test that we don't crash even if we can't remove from one of the functions.

  StringRef Source = ""
      "define <8 x i32> @func1() {\n"
        "ret <8 x i32> undef\n"
      "}\n"
      "\n"
      "define i32 @func2() {\n"
        "%A9 = alloca i32\n"
        "%L6 = load i32, i32* %A9\n"
        "ret i32 %L6\n"
      "}\n";

  auto Mutator = createDeleterMutator();
  ASSERT_TRUE(Mutator);

  IterateOnSource(Source, *Mutator);
}

TEST(InstDeleterIRStrategyTest, PhiNodes) {
  // Test that inst deleter works correctly with the phi nodes.

  LLVMContext Ctx;
  StringRef Source = "\n\
      define i32 @earlyreturncrash(i32 %x) {\n\
      entry:\n\
        switch i32 %x, label %sw.epilog [\n\
          i32 1, label %sw.bb1\n\
        ]\n\
      \n\
      sw.bb1:\n\
        br label %sw.epilog\n\
      \n\
      sw.epilog:\n\
        %a.0 = phi i32 [ 7, %entry ],  [ 9, %sw.bb1 ]\n\
        %b.0 = phi i32 [ 10, %entry ], [ 4, %sw.bb1 ]\n\
        ret i32 %a.0\n\
      }";

  auto Mutator = createDeleterMutator();
  ASSERT_TRUE(Mutator);

  IterateOnSource(Source, *Mutator);
}

static void checkModifyNoUnsignedAndNoSignedWrap(StringRef Opc) {
  LLVMContext Ctx;
  std::string Source = std::string("\n\
      define i32 @test(i32 %x) {\n\
        %a = ") + Opc.str() +
                       std::string(" i32 %x, 10\n\
        ret i32 %a\n\
      }");

  auto Mutator = createInstModifierMutator();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  auto *AddI = &*F.begin()->begin();
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  bool FoundNUW = false;
  bool FoundNSW = false;
  for (int i = 0; i < 100; ++i) {
    Mutator->mutateModule(*M, Seed + i, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
    FoundNUW |= AddI->hasNoUnsignedWrap();
    FoundNSW |= AddI->hasNoSignedWrap();
  }

  // The mutator should have added nuw and nsw during some mutations.
  EXPECT_TRUE(FoundNUW);
  EXPECT_TRUE(FoundNSW);
}
TEST(InstModificationIRStrategyTest, Add) {
  checkModifyNoUnsignedAndNoSignedWrap("add");
}

TEST(InstModificationIRStrategyTest, Sub) {
  checkModifyNoUnsignedAndNoSignedWrap("sub");
}

TEST(InstModificationIRStrategyTest, Mul) {
  checkModifyNoUnsignedAndNoSignedWrap("mul");
}

TEST(InstModificationIRStrategyTest, Shl) {
  checkModifyNoUnsignedAndNoSignedWrap("shl");
}

TEST(InstModificationIRStrategyTest, ICmp) {
  LLVMContext Ctx;
  StringRef Source = "\n\
      define i1 @test(i32 %x) {\n\
        %a = icmp eq i32 %x, 10\n\
        ret i1 %a\n\
      }";

  auto Mutator = createInstModifierMutator();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  CmpInst *CI = cast<CmpInst>(&*F.begin()->begin());
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  bool FoundNE = false;
  for (int i = 0; i < 100; ++i) {
    Mutator->mutateModule(*M, Seed + i, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
    FoundNE |= CI->getPredicate() == CmpInst::ICMP_NE;
  }

  EXPECT_TRUE(FoundNE);
}

TEST(InstModificationIRStrategyTest, GEP) {
  LLVMContext Ctx;
  StringRef Source = "\n\
      define i32* @test(i32* %ptr) {\n\
        %gep = getelementptr i32, i32* %ptr, i32 10\n\
        ret i32* %gep\n\
      }";

  auto Mutator = createInstModifierMutator();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  GetElementPtrInst *GEP = cast<GetElementPtrInst>(&*F.begin()->begin());
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  bool FoundInbounds = false;
  for (int i = 0; i < 100; ++i) {
    Mutator->mutateModule(*M, Seed + i, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
    FoundInbounds |= GEP->isInBounds();
  }

  EXPECT_TRUE(FoundInbounds);
}
}
