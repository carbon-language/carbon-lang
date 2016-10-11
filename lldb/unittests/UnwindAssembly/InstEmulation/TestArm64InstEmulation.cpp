//===-- TestArm64InstEmulation.cpp ------------------------------------*- C++
//-*-===//

//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <vector>

#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"
#include "Utility/ARM64_DWARF_Registers.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/UnwindAssembly.h"

#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class TestArm64InstEmulation : public testing::Test {
public:
  //  static void SetUpTestCase() { }

  //  static void TearDownTestCase() { }

  //  virtual void SetUp() override { }

  //  virtual void TearDown() override { }

protected:
};

static void init() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
  EmulateInstructionARM64::Initialize();
}

static void terminate() {
  DisassemblerLLVMC::Terminate();
  EmulateInstructionARM64::Terminate();
}

TEST_F(TestArm64InstEmulation, TestSimpleFunction) {

  init();

  ArchSpec arch("arm64-apple-ios10", nullptr);
  UnwindAssemblyInstEmulation *engine =
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch));
  EXPECT_TRUE(engine != nullptr);
  if (engine == nullptr)
    return;

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

  // 'int main() { }' compiled for arm64-apple-macosx with clang
  uint8_t data[] = {
      0xfd, 0x7b, 0xbf, 0xa9, // 0xa9bf7bfd :  stp x29, x30, [sp, #-0x10]!
      0xfd, 0x03, 0x00, 0x91, // 0x910003fd :  mov x29, sp
      0xff, 0x43, 0x00, 0xd1, // 0xd10043ff :  sub sp, sp, #0x10

      0xbf, 0x03, 0x00, 0x91, // 0x910003bf :  mov sp, x29
      0xfd, 0x7b, 0xc1, 0xa8, // 0xa8c17bfd :  ldp x29, x30, [sp], #16
      0xc0, 0x03, 0x5f, 0xd6, // 0xd65f03c0 :  ret
  };

  // UnwindPlan we expect:

  // row[0]:    0: CFA=sp +0 =>
  // row[1]:    4: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:    8: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:   16: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[3]:   20: CFA=sp +0 => fp= <same> lr= <same>

  // But this is missing the setup of the frame pointer register (x29) and
  // restore of the same.  This is a bug in the instruction profiler --
  // it won't work if we have a stack frame with a variable length array, or
  // an alloca style call where the stack frame size is not fixed.  We need
  // to recognize the setup of the frame pointer register.

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), unwind_plan));

  // CFA=sp +0
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::fp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp +0 => fp= <same> lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(20);
  EXPECT_EQ(20, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  terminate();
}
