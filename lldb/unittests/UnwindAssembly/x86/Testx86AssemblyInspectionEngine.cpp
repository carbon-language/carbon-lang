//===-- Testx86AssemblyInspectionEngine.cpp ---------------------------*- C++
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

#include "Plugins/UnwindAssembly/x86/x86AssemblyInspectionEngine.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Symbol/UnwindPlan.h"

#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class Testx86AssemblyInspectionEngine : public testing::Test {
public:
  //  static void SetUpTestCase() { }

  //  static void TearDownTestCase() { }

  //  virtual void SetUp() override { }

  //  virtual void TearDown() override { }

protected:
};

// only defining the register names / numbers that the unwinder is actually
// using today

// names should match the constants below.  These will be the eRegisterKindLLDB
// register numbers.

const char *x86_64_reg_names[] = {"rax", "rcx", "rdx", "rsp", "rbp", "rsi",
                                  "rdi", "r8",  "r9",  "r10", "r11", "r12",
                                  "r13", "r14", "r15", "rip"};

enum x86_64_regs {
  k_rax = 0,
  k_rcx = 1,
  k_rdx = 2,
  k_rsp = 3,
  k_rbp = 4,
  k_rsi = 5,
  k_rdi = 6,
  k_r8 = 7,
  k_r9 = 8,
  k_r10 = 9,
  k_r11 = 10,
  k_r12 = 11,
  k_r13 = 12,
  k_r14 = 13,
  k_r15 = 14,
  k_rip = 15
};

// names should match the constants below.  These will be the eRegisterKindLLDB
// register numbers.

const char *i386_reg_names[] = {"eax", "ecx", "edx", "ebx", "esp",
                                "ebp", "esi", "edi", "eip"};

enum i386_regs {
  k_eax = 0,
  k_ecx = 1,
  k_edx = 2,
  k_ebx = 3,
  k_esp = 4,
  k_ebp = 5,
  k_esi = 6,
  k_edi = 7,
  k_eip = 8
};

std::unique_ptr<x86AssemblyInspectionEngine> Getx86_64Inspector() {

  ArchSpec arch("x86_64-apple-macosx", nullptr);
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  std::unique_ptr<x86AssemblyInspectionEngine> engine(
      new x86AssemblyInspectionEngine(arch));

  std::vector<x86AssemblyInspectionEngine::lldb_reg_info> lldb_regnums;
  int i = 0;
  for (const auto &name : x86_64_reg_names) {
    x86AssemblyInspectionEngine::lldb_reg_info ri;
    ri.name = name;
    ri.lldb_regnum = i++;
    lldb_regnums.push_back(ri);
  }

  engine->Initialize(lldb_regnums);
  return engine;
}

std::unique_ptr<x86AssemblyInspectionEngine> Geti386Inspector() {

  ArchSpec arch("i386-apple-macosx", nullptr);
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  std::unique_ptr<x86AssemblyInspectionEngine> engine(
      new x86AssemblyInspectionEngine(arch));

  std::vector<x86AssemblyInspectionEngine::lldb_reg_info> lldb_regnums;
  int i = 0;
  for (const auto &name : i386_reg_names) {
    x86AssemblyInspectionEngine::lldb_reg_info ri;
    ri.name = name;
    ri.lldb_regnum = i++;
    lldb_regnums.push_back(ri);
  }

  engine->Initialize(lldb_regnums);
  return engine;
}

TEST_F(Testx86AssemblyInspectionEngine, TestSimple64bitFrameFunction) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x55,             // offset 0 -- pushq %rbp
      0x48, 0x89, 0xe5, // offset 1 -- movq %rsp, %rbp
      0x31, 0xc0,       // offset 4 -- xorl %eax, %eax
      0x5d,             // offset 6 -- popq %rbp
      0xc3              // offset 7 -- retq
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Expect four unwind rows:
  // 0: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  // 1: CFA=rsp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  // 4: CFA=rbp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  // 7: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]

  EXPECT_TRUE(unwind_plan.GetInitialCFARegister() == k_rsp);
  EXPECT_TRUE(unwind_plan.GetUnwindPlanValidAtAllInstructions() ==
              eLazyBoolYes);
  EXPECT_TRUE(unwind_plan.GetSourcedFromCompiler() == eLazyBoolNo);

  UnwindPlan::Row::RegisterLocation regloc;

  // 0: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  UnwindPlan::RowSP row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_TRUE(row_sp->GetOffset() == 0);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 8);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -8);

  // 1: CFA=rsp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_TRUE(row_sp->GetOffset() == 1);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 16);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -8);

  // 4: CFA=rbp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_TRUE(row_sp->GetOffset() == 4);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rbp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 16);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -8);

  // 7: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_TRUE(row_sp->GetOffset() == 7);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 8);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -8);
}

TEST_F(Testx86AssemblyInspectionEngine, TestSimple32bitFrameFunction) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x55,       // offset 0 -- pushl %ebp
      0x89, 0xe5, // offset 1 -- movl %esp, %ebp
      0x31, 0xc0, // offset 3 -- xorl %eax, %eax
      0x5d,       // offset 5 -- popl %ebp
      0xc3        // offset 6 -- retl
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Expect four unwind rows:
  // 0: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  // 1: CFA=esp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  // 3: CFA=ebp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  // 6: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]

  EXPECT_TRUE(unwind_plan.GetInitialCFARegister() == k_esp);
  EXPECT_TRUE(unwind_plan.GetUnwindPlanValidAtAllInstructions() ==
              eLazyBoolYes);
  EXPECT_TRUE(unwind_plan.GetSourcedFromCompiler() == eLazyBoolNo);

  UnwindPlan::Row::RegisterLocation regloc;

  // offset 0 -- pushl %ebp
  UnwindPlan::RowSP row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_TRUE(row_sp->GetOffset() == 0);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 4);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -4);

  // 1: CFA=esp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_TRUE(row_sp->GetOffset() == 1);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 8);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -4);

  // 3: CFA=ebp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(3);
  EXPECT_TRUE(row_sp->GetOffset() == 3);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_ebp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 8);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -4);

  // 6: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(6);
  EXPECT_TRUE(row_sp->GetOffset() == 6);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_TRUE(row_sp->GetCFAValue().GetOffset() == 4);

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -4);
}
