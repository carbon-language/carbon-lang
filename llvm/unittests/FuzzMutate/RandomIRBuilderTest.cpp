//===- RandomIRBuilderTest.cpp - Tests for injector strategy --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr int Seed = 5;

namespace {

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

TEST(RandomIRBuilderTest, ShuffleVectorIncorrectOperands) {
  // Test that we don't create load instruction as a source for the shuffle
  // vector operation.

  LLVMContext Ctx;
  const char *Source =
      "define <2 x i32> @test(<2 x i1> %cond, <2 x i32> %a) {\n"
      "  %A = alloca <2 x i32>\n"
      "  %I = insertelement <2 x i32> %a, i32 1, i32 1\n"
      "  ret <2 x i32> undef\n"
      "}";
  auto M = parseAssembly(Source, Ctx);

  fuzzerop::OpDescriptor Descr = fuzzerop::shuffleVectorDescriptor(1);

  // Empty known types since we ShuffleVector descriptor doesn't care about them
  RandomIRBuilder IB(Seed, {});

  // Get first basic block of the first function
  Function &F = *M->begin();
  BasicBlock &BB = *F.begin();

  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);

  // Pick first and second sources
  SmallVector<Value *, 2> Srcs;
  ASSERT_TRUE(Descr.SourcePreds[0].matches(Srcs, Insts[1]));
  Srcs.push_back(Insts[1]);
  ASSERT_TRUE(Descr.SourcePreds[1].matches(Srcs, Insts[1]));
  Srcs.push_back(Insts[1]);

  // Create new source. Check that it always matches with the descriptor.
  // Run some iterations to account for random decisions.
  for (int i = 0; i < 10; ++i) {
    Value *LastSrc = IB.newSource(BB, Insts, Srcs, Descr.SourcePreds[2]);
    ASSERT_TRUE(Descr.SourcePreds[2].matches(Srcs, LastSrc));
  }
}

TEST(RandomIRBuilderTest, InsertValueIndexes) {
  // Check that we will generate correct indexes for the insertvalue operation

  LLVMContext Ctx;
  const char *Source =
      "%T = type {i8, i32, i64}\n"
      "define void @test() {\n"
      "  %A = alloca %T\n"
      "  %L = load %T, %T* %A"
      "  ret void\n"
      "}";
  auto M = parseAssembly(Source, Ctx);

  fuzzerop::OpDescriptor IVDescr = fuzzerop::insertValueDescriptor(1);

  std::vector<Type *> Types =
      {Type::getInt8Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);

  // Get first basic block of the first function
  Function &F = *M->begin();
  BasicBlock &BB = *F.begin();

  // Pick first source
  Instruction *Src = &*std::next(BB.begin());

  SmallVector<Value *, 2> Srcs(2);
  ASSERT_TRUE(IVDescr.SourcePreds[0].matches({}, Src));
  Srcs[0] = Src;

  // Generate constants for each of the types and check that we pick correct
  // index for the given type
  for (auto *T: Types) {
    // Loop to account for possible random decisions
    for (int i = 0; i < 10; ++i) {
      // Create value we want to insert. Only it's type matters.
      Srcs[1] = ConstantInt::get(T, 5);

      // Try to pick correct index
      Value *Src = IB.findOrCreateSource(
          BB, &*BB.begin(), Srcs, IVDescr.SourcePreds[2]);
      ASSERT_TRUE(IVDescr.SourcePreds[2].matches(Srcs, Src));
    }
  }
}

}
