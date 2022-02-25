//===- MachineInstrBundleIteratorTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineModuleSlotTracker.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class MachineMetadataTest : public testing::Test {
public:
  MachineMetadataTest() {}

protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;

  static void SetUpTestCase() {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override { M = std::make_unique<Module>("Dummy", Context); }

  void addHooks(ModuleSlotTracker &MST, const MachineOperand &MO) {
    // Setup hooks to assign slot numbers for the specified machine metadata.
    MST.setProcessHook([&MO](AbstractSlotTrackerStorage *AST, const Module *M,
                             bool ShouldInitializeAllMetadata) {
      if (ShouldInitializeAllMetadata) {
        if (MO.isMetadata())
          AST->createMetadataSlot(MO.getMetadata());
      }
    });
    MST.setProcessHook([&MO](AbstractSlotTrackerStorage *AST, const Function *F,
                             bool ShouldInitializeAllMetadata) {
      if (!ShouldInitializeAllMetadata) {
        if (MO.isMetadata())
          AST->createMetadataSlot(MO.getMetadata());
      }
    });
  }

  std::unique_ptr<LLVMTargetMachine>
  createTargetMachine(std::string TT, StringRef CPU, StringRef FS) {
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      return nullptr;
    TargetOptions Options;
    return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
        T->createTargetMachine(TT, CPU, FS, Options, None, None)));
  }

  std::unique_ptr<Module> parseMIR(const TargetMachine &TM, StringRef MIRCode,
                                   const char *FnName, MachineModuleInfo &MMI) {
    SMDiagnostic Diagnostic;
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return nullptr;

    std::unique_ptr<Module> Mod = MIR->parseIRModule();
    if (!Mod)
      return nullptr;

    Mod->setDataLayout(TM.createDataLayout());

    if (MIR->parseMachineFunctions(*Mod, MMI)) {
      M.reset();
      return nullptr;
    }

    return Mod;
  }
};

// Helper to dump the printer output into a string.
static std::string print(std::function<void(raw_ostream &OS)> PrintFn) {
  std::string Str;
  raw_string_ostream OS(Str);
  PrintFn(OS);
  OS.flush();
  return Str;
}

TEST_F(MachineMetadataTest, TrivialHook) {
  // Verify that post-process hook is invoked to assign slot numbers for
  // machine metadata.
  ASSERT_TRUE(M);

  // Create a MachineOperand with a metadata and print it.
  Metadata *MDS = MDString::get(Context, "foo");
  MDNode *Node = MDNode::get(Context, MDS);
  MachineOperand MO = MachineOperand::CreateMetadata(Node);

  // Checking some preconditions on the newly created
  // MachineOperand.
  ASSERT_TRUE(MO.isMetadata());
  ASSERT_EQ(MO.getMetadata(), Node);

  ModuleSlotTracker MST(M.get());
  addHooks(MST, MO);

  // Print a MachineOperand containing a metadata node.
  EXPECT_EQ("!0", print([&](raw_ostream &OS) {
              MO.print(OS, MST, LLT{}, /*OpIdx*/ ~0U, /*PrintDef=*/false,
                       /*IsStandalone=*/false,
                       /*ShouldPrintRegisterTies=*/false, /*TiedOperandIdx=*/0,
                       /*TRI=*/nullptr,
                       /*IntrinsicInfo=*/nullptr);
            }));
  // Print the definition of that metadata node.
  EXPECT_EQ("!0 = !{!\"foo\"}",
            print([&](raw_ostream &OS) { Node->print(OS, MST); }));
}

TEST_F(MachineMetadataTest, BasicHook) {
  // Verify that post-process hook is invoked to assign slot numbers for
  // machine metadata. When both LLVM IR and machine IR contain metadata,
  // ensure that machine metadata is always assigned after LLVM IR.
  ASSERT_TRUE(M);

  // Create a MachineOperand with a metadata and print it.
  Metadata *MachineMDS = MDString::get(Context, "foo");
  MDNode *MachineNode = MDNode::get(Context, MachineMDS);
  MachineOperand MO = MachineOperand::CreateMetadata(MachineNode);

  // Checking some preconditions on the newly created
  // MachineOperand.
  ASSERT_TRUE(MO.isMetadata());
  ASSERT_EQ(MO.getMetadata(), MachineNode);

  // Create metadata in LLVM IR.
  NamedMDNode *MD = M->getOrInsertNamedMetadata("namedmd");
  Metadata *MDS = MDString::get(Context, "bar");
  MDNode *Node = MDNode::get(Context, MDS);
  MD->addOperand(Node);

  ModuleSlotTracker MST(M.get());
  addHooks(MST, MO);

  // Print a MachineOperand containing a metadata node.
  EXPECT_EQ("!1", print([&](raw_ostream &OS) {
              MO.print(OS, MST, LLT{}, /*OpIdx*/ ~0U, /*PrintDef=*/false,
                       /*IsStandalone=*/false,
                       /*ShouldPrintRegisterTies=*/false, /*TiedOperandIdx=*/0,
                       /*TRI=*/nullptr,
                       /*IntrinsicInfo=*/nullptr);
            }));
  // Print the definition of these unnamed metadata nodes.
  EXPECT_EQ("!0 = !{!\"bar\"}",
            print([&](raw_ostream &OS) { Node->print(OS, MST); }));
  EXPECT_EQ("!1 = !{!\"foo\"}",
            print([&](raw_ostream &OS) { MachineNode->print(OS, MST); }));
}

static bool checkOutput(std::string CheckString, std::string Output) {
  auto CheckBuffer = MemoryBuffer::getMemBuffer(CheckString, "");
  auto OutputBuffer = MemoryBuffer::getMemBuffer(Output, "Output", false);

  SmallString<4096> CheckFileBuffer;
  FileCheckRequest Req;
  FileCheck FC(Req);
  StringRef CheckFileText =
      FC.CanonicalizeFile(*CheckBuffer.get(), CheckFileBuffer);

  SourceMgr SM;
  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(CheckFileText, "CheckFile"),
                        SMLoc());
  Regex PrefixRE = FC.buildCheckPrefixRegex();
  if (FC.readCheckFile(SM, CheckFileText, PrefixRE))
    return false;

  auto OutBuffer = OutputBuffer->getBuffer();
  SM.AddNewSourceBuffer(std::move(OutputBuffer), SMLoc());
  return FC.checkInput(SM, OutBuffer);
}

TEST_F(MachineMetadataTest, MMSlotTrackerAArch64) {
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    GTEST_SKIP();

  StringRef MIRString = R"MIR(
--- |
  define i32 @test0(i32* %p) {
    %r = load i32, i32* %p, align 4
    ret i32 %r
  }
...
---
name:            test0
liveins:
  - { reg: '$x0', virtual-reg: '%0' }
body:             |
  bb.0 (%ir-block.0):
    liveins: $x0

  %0:gpr64common = COPY $x0
  %1:gpr32 = LDRWui %0, 0 :: (load (s32) from %ir.p)
...
)MIR";

  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, "test0", MMI);
  ASSERT_TRUE(M);

  auto *MF = MMI.getMachineFunction(*M->getFunction("test0"));
  auto *MBB = MF->getBlockNumbered(0);

  auto &MI = MBB->back();
  ASSERT_TRUE(MI.hasOneMemOperand());

  // Create and attached scoped AA metadata on that instruction with one MMO.
  MDBuilder MDB(Context);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("domain");
  MDNode *Scope0 = MDB.createAnonymousAliasScope(Domain, "scope0");
  MDNode *Scope1 = MDB.createAnonymousAliasScope(Domain, "scope1");
  MDNode *Set0 = MDNode::get(Context, {Scope0});
  MDNode *Set1 = MDNode::get(Context, {Scope1});

  AAMDNodes AAInfo;
  AAInfo.TBAA = AAInfo.TBAAStruct = nullptr;
  AAInfo.Scope = Set0;
  AAInfo.NoAlias = Set1;

  auto *OldMMO = MI.memoperands().front();
  auto *NewMMO = MF->getMachineMemOperand(OldMMO, AAInfo);
  MI.setMemRefs(*MF, NewMMO);

  MachineModuleSlotTracker MST(MF);
  // Print that MI with new machine metadata, which slot numbers should be
  // assigned.
  EXPECT_EQ("%1:gpr32 = LDRWui %0, 0 :: (load (s32) from %ir.p, "
            "!alias.scope !0, !noalias !3)",
            print([&](raw_ostream &OS) {
              MI.print(OS, MST, /*IsStandalone=*/false, /*SkipOpers=*/false,
                       /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
            }));

  std::vector<const MDNode *> Generated{Domain, Scope0, Scope1, Set0, Set1};
  // Examine machine metadata collected. They should match ones
  // afore-generated.
  std::vector<const MDNode *> Collected;
  MachineModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMachineMDNodes(MDList);
  for (auto &MD : MDList)
    Collected.push_back(MD.second);

  std::sort(Generated.begin(), Generated.end());
  std::sort(Collected.begin(), Collected.end());
  EXPECT_EQ(Collected, Generated);

  // FileCheck the output from MIR printer.
  std::string Output = print([&](raw_ostream &OS) { printMIR(OS, *MF); });
  std::string CheckString = R"(
CHECK: machineMetadataNodes:
CHECK-DAG: ![[MMDOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"domain"}
CHECK-DAG: ![[MMSCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope0"}
CHECK-DAG: ![[MMSCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope1"}
CHECK-DAG: ![[MMSET0:[0-9]+]] = !{![[MMSCOPE0]]}
CHECK-DAG: ![[MMSET1:[0-9]+]] = !{![[MMSCOPE1]]}
CHECK: body:
CHECK: %1:gpr32 = LDRWui %0, 0 :: (load (s32) from %ir.p, !alias.scope ![[MMSET0]], !noalias ![[MMSET1]])
)";
  EXPECT_TRUE(checkOutput(CheckString, Output));
}

TEST_F(MachineMetadataTest, MMSlotTrackerX64) {
  auto TM = createTargetMachine(Triple::normalize("x86_64--"), "", "");
  if (!TM)
    GTEST_SKIP();

  StringRef MIRString = R"MIR(
--- |
  define i32 @test0(i32* %p) {
    %r = load i32, i32* %p, align 4
    ret i32 %r
  }
...
---
name:            test0
liveins:
  - { reg: '$rdi', virtual-reg: '%0' }
body:             |
  bb.0 (%ir-block.0):
    liveins: $rdi

  %0:gr64 = COPY $rdi
  %1:gr32 = MOV32rm %0, 1, $noreg, 0, $noreg :: (load (s32) from %ir.p)
...
)MIR";

  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, "test0", MMI);
  ASSERT_TRUE(M);

  auto *MF = MMI.getMachineFunction(*M->getFunction("test0"));
  auto *MBB = MF->getBlockNumbered(0);

  auto &MI = MBB->back();
  ASSERT_FALSE(MI.memoperands_empty());
  ASSERT_TRUE(MI.hasOneMemOperand());

  // Create and attached scoped AA metadata on that instruction with one MMO.
  MDBuilder MDB(Context);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("domain");
  MDNode *Scope0 = MDB.createAnonymousAliasScope(Domain, "scope0");
  MDNode *Scope1 = MDB.createAnonymousAliasScope(Domain, "scope1");
  MDNode *Set0 = MDNode::get(Context, {Scope0});
  MDNode *Set1 = MDNode::get(Context, {Scope1});

  AAMDNodes AAInfo;
  AAInfo.TBAA = AAInfo.TBAAStruct = nullptr;
  AAInfo.Scope = Set0;
  AAInfo.NoAlias = Set1;

  auto *OldMMO = MI.memoperands().front();
  auto *NewMMO = MF->getMachineMemOperand(OldMMO, AAInfo);
  MI.setMemRefs(*MF, NewMMO);

  MachineModuleSlotTracker MST(MF);
  // Print that MI with new machine metadata, which slot numbers should be
  // assigned.
  EXPECT_EQ("%1:gr32 = MOV32rm %0, 1, $noreg, 0, $noreg :: (load (s32) from %ir.p, "
            "!alias.scope !0, !noalias !3)",
            print([&](raw_ostream &OS) {
              MI.print(OS, MST, /*IsStandalone=*/false, /*SkipOpers=*/false,
                       /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
            }));

  std::vector<const MDNode *> Generated{Domain, Scope0, Scope1, Set0, Set1};
  // Examine machine metadata collected. They should match ones
  // afore-generated.
  std::vector<const MDNode *> Collected;
  MachineModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMachineMDNodes(MDList);
  for (auto &MD : MDList)
    Collected.push_back(MD.second);

  std::sort(Generated.begin(), Generated.end());
  std::sort(Collected.begin(), Collected.end());
  EXPECT_EQ(Collected, Generated);

  // FileCheck the output from MIR printer.
  std::string Output = print([&](raw_ostream &OS) { printMIR(OS, *MF); });
  std::string CheckString = R"(
CHECK: machineMetadataNodes:
CHECK-DAG: ![[MMDOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"domain"}
CHECK-DAG: ![[MMSCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope0"}
CHECK-DAG: ![[MMSCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope1"}
CHECK-DAG: ![[MMSET0:[0-9]+]] = !{![[MMSCOPE0]]}
CHECK-DAG: ![[MMSET1:[0-9]+]] = !{![[MMSCOPE1]]}
CHECK: body:
CHECK: %1:gr32 = MOV32rm %0, 1, $noreg, 0, $noreg :: (load (s32) from %ir.p, !alias.scope ![[MMSET0]], !noalias ![[MMSET1]])
)";
  EXPECT_TRUE(checkOutput(CheckString, Output));
}

TEST_F(MachineMetadataTest, MMSlotTrackerAMDGPU) {
  auto TM = createTargetMachine(Triple::normalize("amdgcn-amd-amdhsa"),
                                "gfx1010", "");
  if (!TM)
    GTEST_SKIP();

  StringRef MIRString = R"MIR(
--- |
  define i32 @test0(i32* %p) {
    %r = load i32, i32* %p, align 4
    ret i32 %r
  }
...
---
name:            test0
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$vgpr1', virtual-reg: '%1' }
  - { reg: '$sgpr30_sgpr31', virtual-reg: '%2' }
body:             |
  bb.0 (%ir-block.0):
    liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31

    %2:sreg_64 = COPY $sgpr30_sgpr31
    %1:vgpr_32 = COPY $vgpr1
    %0:vgpr_32 = COPY $vgpr0
    %8:vreg_64 = REG_SEQUENCE %0, %subreg.sub0, %1, %subreg.sub1
    %6:vreg_64 = COPY %8
    %5:vgpr_32 = FLAT_LOAD_DWORD killed %6, 0, 0, implicit $exec, implicit $flat_scr :: (load (s32) from %ir.p)
...
)MIR";

  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, "test0", MMI);
  ASSERT_TRUE(M);

  auto *MF = MMI.getMachineFunction(*M->getFunction("test0"));
  auto *MBB = MF->getBlockNumbered(0);

  auto &MI = MBB->back();
  ASSERT_FALSE(MI.memoperands_empty());
  ASSERT_TRUE(MI.hasOneMemOperand());

  // Create and attached scoped AA metadata on that instruction with one MMO.
  MDBuilder MDB(Context);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("domain");
  MDNode *Scope0 = MDB.createAnonymousAliasScope(Domain, "scope0");
  MDNode *Scope1 = MDB.createAnonymousAliasScope(Domain, "scope1");
  MDNode *Set0 = MDNode::get(Context, {Scope0});
  MDNode *Set1 = MDNode::get(Context, {Scope1});

  AAMDNodes AAInfo;
  AAInfo.TBAA = AAInfo.TBAAStruct = nullptr;
  AAInfo.Scope = Set0;
  AAInfo.NoAlias = Set1;

  auto *OldMMO = MI.memoperands().front();
  auto *NewMMO = MF->getMachineMemOperand(OldMMO, AAInfo);
  MI.setMemRefs(*MF, NewMMO);

  MachineModuleSlotTracker MST(MF);
  // Print that MI with new machine metadata, which slot numbers should be
  // assigned.
  EXPECT_EQ(
      "%5:vgpr_32 = FLAT_LOAD_DWORD killed %4, 0, 0, implicit $exec, implicit "
      "$flat_scr :: (load (s32) from %ir.p, !alias.scope !0, !noalias !3)",
      print([&](raw_ostream &OS) {
        MI.print(OS, MST, /*IsStandalone=*/false, /*SkipOpers=*/false,
                 /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
      }));

  std::vector<const MDNode *> Generated{Domain, Scope0, Scope1, Set0, Set1};
  // Examine machine metadata collected. They should match ones
  // afore-generated.
  std::vector<const MDNode *> Collected;
  MachineModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMachineMDNodes(MDList);
  for (auto &MD : MDList)
    Collected.push_back(MD.second);

  std::sort(Generated.begin(), Generated.end());
  std::sort(Collected.begin(), Collected.end());
  EXPECT_EQ(Collected, Generated);

  // FileCheck the output from MIR printer.
  std::string Output = print([&](raw_ostream &OS) { printMIR(OS, *MF); });
  std::string CheckString = R"(
CHECK: machineMetadataNodes:
CHECK-DAG: ![[MMDOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"domain"}
CHECK-DAG: ![[MMSCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope0"}
CHECK-DAG: ![[MMSCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope1"}
CHECK-DAG: ![[MMSET0:[0-9]+]] = !{![[MMSCOPE0]]}
CHECK-DAG: ![[MMSET1:[0-9]+]] = !{![[MMSCOPE1]]}
CHECK: body:
CHECK: %5:vgpr_32 = FLAT_LOAD_DWORD killed %4, 0, 0, implicit $exec, implicit $flat_scr :: (load (s32) from %ir.p, !alias.scope ![[MMSET0]], !noalias ![[MMSET1]])
)";
  EXPECT_TRUE(checkOutput(CheckString, Output));
}
