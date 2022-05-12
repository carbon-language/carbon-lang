//===-- AssemblerTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "X86InstrInfo.h"

namespace llvm {
namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using X86::EAX;
using X86::MOV32ri;
using X86::MOV64ri32;
using X86::RAX;
using X86::XOR32rr;

class X86MachineFunctionGeneratorTest
    : public MachineFunctionGeneratorBaseTest {
protected:
  X86MachineFunctionGeneratorTest()
      : MachineFunctionGeneratorBaseTest("x86_64-unknown-linux", "haswell") {}

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    InitializeX86ExegesisTarget();
  }
};

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunction) {
  Check({}, MCInst(), 0xc3);
}

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunctionXOR32rr_X86) {
  Check({{EAX, APInt(32, 1)}},
        MCInstBuilder(XOR32rr).addReg(EAX).addReg(EAX).addReg(EAX),
        // mov eax, 1
        0xb8, 0x01, 0x00, 0x00, 0x00,
        // xor eax, eax
        0x31, 0xc0, 0xc3);
}

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunctionMOV64ri) {
  Check({}, MCInstBuilder(MOV64ri32).addReg(RAX).addImm(42), 0x48, 0xc7, 0xc0,
        0x2a, 0x00, 0x00, 0x00, 0xc3);
}

TEST_F(X86MachineFunctionGeneratorTest, DISABLED_JitFunctionMOV32ri) {
  Check({}, MCInstBuilder(MOV32ri).addReg(EAX).addImm(42), 0xb8, 0x2a, 0x00,
        0x00, 0x00, 0xc3);
}

} // namespace
} // namespace exegesis
} // namespace llvm
