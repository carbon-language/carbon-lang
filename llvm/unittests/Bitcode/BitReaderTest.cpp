//===- llvm/unittest/Bitcode/BitReaderTest.cpp - Tests for BitReader ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<Module> parseAssembly(const char *Assembly) {
  auto M = make_unique<Module>("Module", getGlobalContext());

  SMDiagnostic Error;
  bool Parsed =
      ParseAssemblyString(Assembly, M.get(), Error, M->getContext()) == M.get();

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  // A failure here means that the test itself is buggy.
  if (!Parsed)
    report_fatal_error(OS.str().c_str());

  return M;
}

static void writeModuleToBuffer(std::unique_ptr<Module> Mod,
                                SmallVectorImpl<char> &Buffer) {
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(Mod.get(), OS);
}

static std::unique_ptr<Module> getLazyModuleFromAssembly(LLVMContext &Context,
                                                         SmallString<1024> &Mem,
                                                         const char *Assembly) {
  writeModuleToBuffer(parseAssembly(Assembly), Mem);
  MemoryBuffer *Buffer = MemoryBuffer::getMemBuffer(Mem.str(), "test", false);
  ErrorOr<Module *> ModuleOrErr = getLazyBitcodeModule(Buffer, Context);
  return std::unique_ptr<Module>(ModuleOrErr.get());
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddr) { // PR11677
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "@table = constant i8* blockaddress(@func, %bb)\n"
                    "define void @func() {\n"
                    "  unreachable\n"
                    "bb:\n"
                    "  unreachable\n"
                    "}\n");
  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

} // end namespace
