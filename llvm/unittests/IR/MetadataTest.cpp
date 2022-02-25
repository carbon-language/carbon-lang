//===- unittests/IR/MetadataTest.cpp - Metadata unit tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Metadata.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(ContextAndReplaceableUsesTest, FromContext) {
  LLVMContext Context;
  ContextAndReplaceableUses CRU(Context);
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_FALSE(CRU.hasReplaceableUses());
  EXPECT_FALSE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, FromReplaceableUses) {
  LLVMContext Context;
  ContextAndReplaceableUses CRU(std::make_unique<ReplaceableMetadataImpl>(Context));
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_TRUE(CRU.hasReplaceableUses());
  EXPECT_TRUE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, makeReplaceable) {
  LLVMContext Context;
  ContextAndReplaceableUses CRU(Context);
  CRU.makeReplaceable(std::make_unique<ReplaceableMetadataImpl>(Context));
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_TRUE(CRU.hasReplaceableUses());
  EXPECT_TRUE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, takeReplaceableUses) {
  LLVMContext Context;
  auto ReplaceableUses = std::make_unique<ReplaceableMetadataImpl>(Context);
  auto *Ptr = ReplaceableUses.get();
  ContextAndReplaceableUses CRU(std::move(ReplaceableUses));
  ReplaceableUses = CRU.takeReplaceableUses();
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_FALSE(CRU.hasReplaceableUses());
  EXPECT_FALSE(CRU.getReplaceableUses());
  EXPECT_EQ(Ptr, ReplaceableUses.get());
}

class MetadataTest : public testing::Test {
public:
  MetadataTest() : M("test", Context), Counter(0) {}

protected:
  LLVMContext Context;
  Module M;
  int Counter;

  MDNode *getNode() { return MDNode::get(Context, None); }
  MDNode *getNode(Metadata *MD) { return MDNode::get(Context, MD); }
  MDNode *getNode(Metadata *MD1, Metadata *MD2) {
    Metadata *MDs[] = {MD1, MD2};
    return MDNode::get(Context, MDs);
  }

  MDTuple *getTuple() { return MDTuple::getDistinct(Context, None); }
  DISubroutineType *getSubroutineType() {
    return DISubroutineType::getDistinct(Context, DINode::FlagZero, 0,
                                         getNode(nullptr));
  }
  DISubprogram *getSubprogram() {
    return DISubprogram::getDistinct(
        Context, nullptr, "", "", nullptr, 0, nullptr, 0, nullptr, 0, 0,
        DINode::FlagZero, DISubprogram::SPFlagZero, nullptr);
  }
  DIFile *getFile() {
    return DIFile::getDistinct(Context, "file.c", "/path/to/dir");
  }
  DICompileUnit *getUnit() {
    return DICompileUnit::getDistinct(
        Context, 1, getFile(), "clang", false, "-g", 2, "",
        DICompileUnit::FullDebug, getTuple(), getTuple(), getTuple(),
        getTuple(), getTuple(), 0, true, false,
        DICompileUnit::DebugNameTableKind::Default, false, "/", "");
  }
  DIType *getBasicType(StringRef Name) {
    return DIBasicType::get(Context, dwarf::DW_TAG_unspecified_type, Name);
  }
  DIType *getDerivedType() {
    return DIDerivedType::getDistinct(
        Context, dwarf::DW_TAG_pointer_type, "", nullptr, 0, nullptr,
        getBasicType("basictype"), 1, 2, 0, None, DINode::FlagZero);
  }
  Constant *getConstant() {
    return ConstantInt::get(Type::getInt32Ty(Context), Counter++);
  }
  ConstantAsMetadata *getConstantAsMetadata() {
    return ConstantAsMetadata::get(getConstant());
  }
  DIType *getCompositeType() {
    return DICompositeType::getDistinct(
        Context, dwarf::DW_TAG_structure_type, "", nullptr, 0, nullptr, nullptr,
        32, 32, 0, DINode::FlagZero, nullptr, 0, nullptr, nullptr, "");
  }
  Function *getFunction(StringRef Name) {
    return Function::Create(
        FunctionType::get(Type::getVoidTy(Context), None, false),
        Function::ExternalLinkage, Name, M);
  }
};
typedef MetadataTest MDStringTest;

// Test that construction of MDString with different value produces different
// MDString objects, even with the same string pointer and nulls in the string.
TEST_F(MDStringTest, CreateDifferent) {
  char x[3] = { 'f', 0, 'A' };
  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  x[2] = 'B';
  MDString *s2 = MDString::get(Context, StringRef(&x[0], 3));
  EXPECT_NE(s1, s2);
}

// Test that creation of MDStrings with the same string contents produces the
// same MDString object, even with different pointers.
TEST_F(MDStringTest, CreateSame) {
  char x[4] = { 'a', 'b', 'c', 'X' };
  char y[4] = { 'a', 'b', 'c', 'Y' };

  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  MDString *s2 = MDString::get(Context, StringRef(&y[0], 3));
  EXPECT_EQ(s1, s2);
}

// Test that MDString prints out the string we fed it.
TEST_F(MDStringTest, PrintingSimple) {
  char str[14] = "testing 1 2 3";
  MDString *s = MDString::get(Context, StringRef(&str[0], 13));
  strncpy(str, "aaaaaaaaaaaaa", 14);

  std::string Str;
  raw_string_ostream oss(Str);
  s->print(oss);
  EXPECT_STREQ("!\"testing 1 2 3\"", oss.str().c_str());
}

// Test printing of MDString with non-printable characters.
TEST_F(MDStringTest, PrintingComplex) {
  char str[5] = {0, '\n', '"', '\\', (char)-1};
  MDString *s = MDString::get(Context, StringRef(str+0, 5));
  std::string Str;
  raw_string_ostream oss(Str);
  s->print(oss);
  EXPECT_STREQ("!\"\\00\\0A\\22\\\\\\FF\"", oss.str().c_str());
}

typedef MetadataTest MDNodeTest;

// Test the two constructors, and containing other Constants.
TEST_F(MDNodeTest, Simple) {
  char x[3] = { 'a', 'b', 'c' };
  char y[3] = { '1', '2', '3' };

  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  MDString *s2 = MDString::get(Context, StringRef(&y[0], 3));
  ConstantAsMetadata *CI =
      ConstantAsMetadata::get(ConstantInt::get(Context, APInt(8, 0)));

  std::vector<Metadata *> V;
  V.push_back(s1);
  V.push_back(CI);
  V.push_back(s2);

  MDNode *n1 = MDNode::get(Context, V);
  Metadata *const c1 = n1;
  MDNode *n2 = MDNode::get(Context, c1);
  Metadata *const c2 = n2;
  MDNode *n3 = MDNode::get(Context, V);
  MDNode *n4 = MDNode::getIfExists(Context, V);
  MDNode *n5 = MDNode::getIfExists(Context, c1);
  MDNode *n6 = MDNode::getIfExists(Context, c2);
  EXPECT_NE(n1, n2);
  EXPECT_EQ(n1, n3);
  EXPECT_EQ(n4, n1);
  EXPECT_EQ(n5, n2);
  EXPECT_EQ(n6, (Metadata *)nullptr);

  EXPECT_EQ(3u, n1->getNumOperands());
  EXPECT_EQ(s1, n1->getOperand(0));
  EXPECT_EQ(CI, n1->getOperand(1));
  EXPECT_EQ(s2, n1->getOperand(2));

  EXPECT_EQ(1u, n2->getNumOperands());
  EXPECT_EQ(n1, n2->getOperand(0));
}

TEST_F(MDNodeTest, Delete) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(Context), 1);
  Instruction *I = new BitCastInst(C, Type::getInt32Ty(Context));

  Metadata *const V = LocalAsMetadata::get(I);
  MDNode *n = MDNode::get(Context, V);
  TrackingMDRef wvh(n);

  EXPECT_EQ(n, wvh);

  I->deleteValue();
}

TEST_F(MDNodeTest, SelfReference) {
  // !0 = !{!0}
  // !1 = !{!0}
  {
    auto Temp = MDNode::getTemporary(Context, None);
    Metadata *Args[] = {Temp.get()};
    MDNode *Self = MDNode::get(Context, Args);
    Self->replaceOperandWith(0, Self);
    ASSERT_EQ(Self, Self->getOperand(0));

    // Self-references should be distinct, so MDNode::get() should grab a
    // uniqued node that references Self, not Self.
    Args[0] = Self;
    MDNode *Ref1 = MDNode::get(Context, Args);
    MDNode *Ref2 = MDNode::get(Context, Args);
    EXPECT_NE(Self, Ref1);
    EXPECT_EQ(Ref1, Ref2);
  }

  // !0 = !{!0, !{}}
  // !1 = !{!0, !{}}
  {
    auto Temp = MDNode::getTemporary(Context, None);
    Metadata *Args[] = {Temp.get(), MDNode::get(Context, None)};
    MDNode *Self = MDNode::get(Context, Args);
    Self->replaceOperandWith(0, Self);
    ASSERT_EQ(Self, Self->getOperand(0));

    // Self-references should be distinct, so MDNode::get() should grab a
    // uniqued node that references Self, not Self itself.
    Args[0] = Self;
    MDNode *Ref1 = MDNode::get(Context, Args);
    MDNode *Ref2 = MDNode::get(Context, Args);
    EXPECT_NE(Self, Ref1);
    EXPECT_EQ(Ref1, Ref2);
  }
}

TEST_F(MDNodeTest, Print) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(Context), 7);
  MDString *S = MDString::get(Context, "foo");
  MDNode *N0 = getNode();
  MDNode *N1 = getNode(N0);
  MDNode *N2 = getNode(N0, N1);

  Metadata *Args[] = {ConstantAsMetadata::get(C), S, nullptr, N0, N1, N2};
  MDNode *N = MDNode::get(Context, Args);

  std::string Expected;
  {
    raw_string_ostream OS(Expected);
    OS << "<" << (void *)N << "> = !{";
    C->printAsOperand(OS);
    OS << ", ";
    S->printAsOperand(OS);
    OS << ", null";
    MDNode *Nodes[] = {N0, N1, N2};
    for (auto *Node : Nodes)
      OS << ", <" << (void *)Node << ">";
    OS << "}";
  }

  std::string Actual;
  {
    raw_string_ostream OS(Actual);
    N->print(OS);
  }

  EXPECT_EQ(Expected, Actual);
}

#define EXPECT_PRINTER_EQ(EXPECTED, PRINT)                                     \
  do {                                                                         \
    std::string Actual_;                                                       \
    raw_string_ostream OS(Actual_);                                            \
    PRINT;                                                                     \
    OS.flush();                                                                \
    std::string Expected_(EXPECTED);                                           \
    EXPECT_EQ(Expected_, Actual_);                                             \
  } while (false)

TEST_F(MDNodeTest, PrintTemporary) {
  MDNode *Arg = getNode();
  TempMDNode Temp = MDNode::getTemporary(Context, Arg);
  MDNode *N = getNode(Temp.get());
  Module M("test", Context);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("named");
  NMD->addOperand(N);

  EXPECT_PRINTER_EQ("!0 = !{!1}", N->print(OS, &M));
  EXPECT_PRINTER_EQ("!1 = <temporary!> !{!2}", Temp->print(OS, &M));
  EXPECT_PRINTER_EQ("!2 = !{}", Arg->print(OS, &M));

  // Cleanup.
  Temp->replaceAllUsesWith(Arg);
}

TEST_F(MDNodeTest, PrintFromModule) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(Context), 7);
  MDString *S = MDString::get(Context, "foo");
  MDNode *N0 = getNode();
  MDNode *N1 = getNode(N0);
  MDNode *N2 = getNode(N0, N1);

  Metadata *Args[] = {ConstantAsMetadata::get(C), S, nullptr, N0, N1, N2};
  MDNode *N = MDNode::get(Context, Args);
  Module M("test", Context);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("named");
  NMD->addOperand(N);

  std::string Expected;
  {
    raw_string_ostream OS(Expected);
    OS << "!0 = !{";
    C->printAsOperand(OS);
    OS << ", ";
    S->printAsOperand(OS);
    OS << ", null, !1, !2, !3}";
  }

  EXPECT_PRINTER_EQ(Expected, N->print(OS, &M));
}

TEST_F(MDNodeTest, PrintFromFunction) {
  Module M("test", Context);
  auto *FTy = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F0 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F0", &M);
  auto *F1 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F1", &M);
  auto *BB0 = BasicBlock::Create(Context, "entry", F0);
  auto *BB1 = BasicBlock::Create(Context, "entry", F1);
  auto *R0 = ReturnInst::Create(Context, BB0);
  auto *R1 = ReturnInst::Create(Context, BB1);
  auto *N0 = MDNode::getDistinct(Context, None);
  auto *N1 = MDNode::getDistinct(Context, None);
  R0->setMetadata("md", N0);
  R1->setMetadata("md", N1);

  EXPECT_PRINTER_EQ("!0 = distinct !{}", N0->print(OS, &M));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", N1->print(OS, &M));

  ModuleSlotTracker MST(&M);
  EXPECT_PRINTER_EQ("!0 = distinct !{}", N0->print(OS, MST));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", N1->print(OS, MST));
}

TEST_F(MDNodeTest, PrintFromMetadataAsValue) {
  Module M("test", Context);

  auto *Intrinsic =
      Function::Create(FunctionType::get(Type::getVoidTy(Context),
                                         Type::getMetadataTy(Context), false),
                       GlobalValue::ExternalLinkage, "llvm.intrinsic", &M);

  auto *FTy = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F0 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F0", &M);
  auto *F1 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F1", &M);
  auto *BB0 = BasicBlock::Create(Context, "entry", F0);
  auto *BB1 = BasicBlock::Create(Context, "entry", F1);
  auto *N0 = MDNode::getDistinct(Context, None);
  auto *N1 = MDNode::getDistinct(Context, None);
  auto *MAV0 = MetadataAsValue::get(Context, N0);
  auto *MAV1 = MetadataAsValue::get(Context, N1);
  CallInst::Create(Intrinsic, MAV0, "", BB0);
  CallInst::Create(Intrinsic, MAV1, "", BB1);

  EXPECT_PRINTER_EQ("!0 = distinct !{}", MAV0->print(OS));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", MAV1->print(OS));
  EXPECT_PRINTER_EQ("!0", MAV0->printAsOperand(OS, false));
  EXPECT_PRINTER_EQ("!1", MAV1->printAsOperand(OS, false));
  EXPECT_PRINTER_EQ("metadata !0", MAV0->printAsOperand(OS, true));
  EXPECT_PRINTER_EQ("metadata !1", MAV1->printAsOperand(OS, true));

  ModuleSlotTracker MST(&M);
  EXPECT_PRINTER_EQ("!0 = distinct !{}", MAV0->print(OS, MST));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", MAV1->print(OS, MST));
  EXPECT_PRINTER_EQ("!0", MAV0->printAsOperand(OS, false, MST));
  EXPECT_PRINTER_EQ("!1", MAV1->printAsOperand(OS, false, MST));
  EXPECT_PRINTER_EQ("metadata !0", MAV0->printAsOperand(OS, true, MST));
  EXPECT_PRINTER_EQ("metadata !1", MAV1->printAsOperand(OS, true, MST));
}

TEST_F(MDNodeTest, PrintWithDroppedCallOperand) {
  Module M("test", Context);

  auto *FTy = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F0 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F0", &M);
  auto *F1 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F1", &M);
  auto *BB0 = BasicBlock::Create(Context, "entry", F0);

  CallInst *CI0 = CallInst::Create(F1, "", BB0);
  CI0->dropAllReferences();

  auto *R0 = ReturnInst::Create(Context, BB0);
  auto *N0 = MDNode::getDistinct(Context, None);
  R0->setMetadata("md", N0);

  // Printing the metadata node would previously result in a failed assertion
  // due to the call instruction's dropped function operand.
  ModuleSlotTracker MST(&M);
  EXPECT_PRINTER_EQ("!0 = distinct !{}", N0->print(OS, MST));
}

TEST_F(MDNodeTest, PrintTree) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  {
    DIType *Type = getDerivedType();
    auto *Var = DILocalVariable::get(Context, Scope, "foo", File,
                                     /*LineNo=*/8, Type, /*ArgNo=*/2, Flags,
                                     /*Align=*/8, nullptr);
    std::string Expected;
    {
      raw_string_ostream SS(Expected);
      Var->print(SS);
      // indent level 1
      Scope->print((SS << "\n").indent(2));
      File->print((SS << "\n").indent(2));
      Type->print((SS << "\n").indent(2));
      // indent level 2
      auto *BaseType = cast<DIDerivedType>(Type)->getBaseType();
      BaseType->print((SS << "\n").indent(4));
    }

    EXPECT_PRINTER_EQ(Expected, Var->printTree(OS));
  }

  {
    // Test if printTree works correctly when there is
    // a cycle in the MDNode and its dependencies.
    //
    // We're trying to create type like this:
    // struct LinkedList {
    //   LinkedList *Head;
    // };
    auto *StructTy = cast<DICompositeType>(getCompositeType());
    DIType *PointerTy = DIDerivedType::getDistinct(
        Context, dwarf::DW_TAG_pointer_type, "", nullptr, 0, nullptr, StructTy,
        1, 2, 0, None, DINode::FlagZero);
    StructTy->replaceElements(MDTuple::get(Context, PointerTy));

    auto *Var = DILocalVariable::get(Context, Scope, "foo", File,
                                     /*LineNo=*/8, StructTy, /*ArgNo=*/2, Flags,
                                     /*Align=*/8, nullptr);
    std::string Expected;
    {
      raw_string_ostream SS(Expected);
      Var->print(SS);
      // indent level 1
      Scope->print((SS << "\n").indent(2));
      File->print((SS << "\n").indent(2));
      StructTy->print((SS << "\n").indent(2));
      // indent level 2
      StructTy->getRawElements()->print((SS << "\n").indent(4));
      // indent level 3
      auto Elements = StructTy->getElements();
      Elements[0]->print((SS << "\n").indent(6));
    }

    EXPECT_PRINTER_EQ(Expected, Var->printTree(OS));
  }
}
#undef EXPECT_PRINTER_EQ

TEST_F(MDNodeTest, NullOperand) {
  // metadata !{}
  MDNode *Empty = MDNode::get(Context, None);

  // metadata !{metadata !{}}
  Metadata *Ops[] = {Empty};
  MDNode *N = MDNode::get(Context, Ops);
  ASSERT_EQ(Empty, N->getOperand(0));

  // metadata !{metadata !{}} => metadata !{null}
  N->replaceOperandWith(0, nullptr);
  ASSERT_EQ(nullptr, N->getOperand(0));

  // metadata !{null}
  Ops[0] = nullptr;
  MDNode *NullOp = MDNode::get(Context, Ops);
  ASSERT_EQ(nullptr, NullOp->getOperand(0));
  EXPECT_EQ(N, NullOp);
}

TEST_F(MDNodeTest, DistinctOnUniquingCollision) {
  // !{}
  MDNode *Empty = MDNode::get(Context, None);
  ASSERT_TRUE(Empty->isResolved());
  EXPECT_FALSE(Empty->isDistinct());

  // !{!{}}
  Metadata *Wrapped1Ops[] = {Empty};
  MDNode *Wrapped1 = MDNode::get(Context, Wrapped1Ops);
  ASSERT_EQ(Empty, Wrapped1->getOperand(0));
  ASSERT_TRUE(Wrapped1->isResolved());
  EXPECT_FALSE(Wrapped1->isDistinct());

  // !{!{!{}}}
  Metadata *Wrapped2Ops[] = {Wrapped1};
  MDNode *Wrapped2 = MDNode::get(Context, Wrapped2Ops);
  ASSERT_EQ(Wrapped1, Wrapped2->getOperand(0));
  ASSERT_TRUE(Wrapped2->isResolved());
  EXPECT_FALSE(Wrapped2->isDistinct());

  // !{!{!{}}} => !{!{}}
  Wrapped2->replaceOperandWith(0, Empty);
  ASSERT_EQ(Empty, Wrapped2->getOperand(0));
  EXPECT_TRUE(Wrapped2->isDistinct());
  EXPECT_FALSE(Wrapped1->isDistinct());
}

TEST_F(MDNodeTest, UniquedOnDeletedOperand) {
  // temp !{}
  TempMDTuple T = MDTuple::getTemporary(Context, None);

  // !{temp !{}}
  Metadata *Ops[] = {T.get()};
  MDTuple *N = MDTuple::get(Context, Ops);

  // !{temp !{}} => !{null}
  T.reset();
  ASSERT_TRUE(N->isUniqued());
  Metadata *NullOps[] = {nullptr};
  ASSERT_EQ(N, MDTuple::get(Context, NullOps));
}

TEST_F(MDNodeTest, DistinctOnDeletedValueOperand) {
  // i1* @GV
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  ConstantAsMetadata *Op = ConstantAsMetadata::get(GV.get());

  // !{i1* @GV}
  Metadata *Ops[] = {Op};
  MDTuple *N = MDTuple::get(Context, Ops);

  // !{i1* @GV} => !{null}
  GV.reset();
  ASSERT_TRUE(N->isDistinct());
  ASSERT_EQ(nullptr, N->getOperand(0));
  Metadata *NullOps[] = {nullptr};
  ASSERT_NE(N, MDTuple::get(Context, NullOps));
}

TEST_F(MDNodeTest, getDistinct) {
  // !{}
  MDNode *Empty = MDNode::get(Context, None);
  ASSERT_TRUE(Empty->isResolved());
  ASSERT_FALSE(Empty->isDistinct());
  ASSERT_EQ(Empty, MDNode::get(Context, None));

  // distinct !{}
  MDNode *Distinct1 = MDNode::getDistinct(Context, None);
  MDNode *Distinct2 = MDNode::getDistinct(Context, None);
  EXPECT_TRUE(Distinct1->isResolved());
  EXPECT_TRUE(Distinct2->isDistinct());
  EXPECT_NE(Empty, Distinct1);
  EXPECT_NE(Empty, Distinct2);
  EXPECT_NE(Distinct1, Distinct2);

  // !{}
  ASSERT_EQ(Empty, MDNode::get(Context, None));
}

TEST_F(MDNodeTest, isUniqued) {
  MDNode *U = MDTuple::get(Context, None);
  MDNode *D = MDTuple::getDistinct(Context, None);
  auto T = MDTuple::getTemporary(Context, None);
  EXPECT_TRUE(U->isUniqued());
  EXPECT_FALSE(D->isUniqued());
  EXPECT_FALSE(T->isUniqued());
}

TEST_F(MDNodeTest, isDistinct) {
  MDNode *U = MDTuple::get(Context, None);
  MDNode *D = MDTuple::getDistinct(Context, None);
  auto T = MDTuple::getTemporary(Context, None);
  EXPECT_FALSE(U->isDistinct());
  EXPECT_TRUE(D->isDistinct());
  EXPECT_FALSE(T->isDistinct());
}

TEST_F(MDNodeTest, isTemporary) {
  MDNode *U = MDTuple::get(Context, None);
  MDNode *D = MDTuple::getDistinct(Context, None);
  auto T = MDTuple::getTemporary(Context, None);
  EXPECT_FALSE(U->isTemporary());
  EXPECT_FALSE(D->isTemporary());
  EXPECT_TRUE(T->isTemporary());
}

TEST_F(MDNodeTest, getDistinctWithUnresolvedOperands) {
  // temporary !{}
  auto Temp = MDTuple::getTemporary(Context, None);
  ASSERT_FALSE(Temp->isResolved());

  // distinct !{temporary !{}}
  Metadata *Ops[] = {Temp.get()};
  MDNode *Distinct = MDNode::getDistinct(Context, Ops);
  EXPECT_TRUE(Distinct->isResolved());
  EXPECT_EQ(Temp.get(), Distinct->getOperand(0));

  // temporary !{} => !{}
  MDNode *Empty = MDNode::get(Context, None);
  Temp->replaceAllUsesWith(Empty);
  EXPECT_EQ(Empty, Distinct->getOperand(0));
}

TEST_F(MDNodeTest, handleChangedOperandRecursion) {
  // !0 = !{}
  MDNode *N0 = MDNode::get(Context, None);

  // !1 = !{!3, null}
  auto Temp3 = MDTuple::getTemporary(Context, None);
  Metadata *Ops1[] = {Temp3.get(), nullptr};
  MDNode *N1 = MDNode::get(Context, Ops1);

  // !2 = !{!3, !0}
  Metadata *Ops2[] = {Temp3.get(), N0};
  MDNode *N2 = MDNode::get(Context, Ops2);

  // !3 = !{!2}
  Metadata *Ops3[] = {N2};
  MDNode *N3 = MDNode::get(Context, Ops3);
  Temp3->replaceAllUsesWith(N3);

  // !4 = !{!1}
  Metadata *Ops4[] = {N1};
  MDNode *N4 = MDNode::get(Context, Ops4);

  // Confirm that the cycle prevented RAUW from getting dropped.
  EXPECT_TRUE(N0->isResolved());
  EXPECT_FALSE(N1->isResolved());
  EXPECT_FALSE(N2->isResolved());
  EXPECT_FALSE(N3->isResolved());
  EXPECT_FALSE(N4->isResolved());

  // Create a couple of distinct nodes to observe what's going on.
  //
  // !5 = distinct !{!2}
  // !6 = distinct !{!3}
  Metadata *Ops5[] = {N2};
  MDNode *N5 = MDNode::getDistinct(Context, Ops5);
  Metadata *Ops6[] = {N3};
  MDNode *N6 = MDNode::getDistinct(Context, Ops6);

  // Mutate !2 to look like !1, causing a uniquing collision (and an RAUW).
  // This will ripple up, with !3 colliding with !4, and RAUWing.  Since !2
  // references !3, this can cause a re-entry of handleChangedOperand() when !3
  // is not ready for it.
  //
  // !2->replaceOperandWith(1, nullptr)
  // !2: !{!3, !0} => !{!3, null}
  // !2->replaceAllUsesWith(!1)
  // !3: !{!2] => !{!1}
  // !3->replaceAllUsesWith(!4)
  N2->replaceOperandWith(1, nullptr);

  // If all has gone well, N2 and N3 will have been RAUW'ed and deleted from
  // under us.  Just check that the other nodes are sane.
  //
  // !1 = !{!4, null}
  // !4 = !{!1}
  // !5 = distinct !{!1}
  // !6 = distinct !{!4}
  EXPECT_EQ(N4, N1->getOperand(0));
  EXPECT_EQ(N1, N4->getOperand(0));
  EXPECT_EQ(N1, N5->getOperand(0));
  EXPECT_EQ(N4, N6->getOperand(0));
}

TEST_F(MDNodeTest, replaceResolvedOperand) {
  // Check code for replacing one resolved operand with another.  If doing this
  // directly (via replaceOperandWith()) becomes illegal, change the operand to
  // a global value that gets RAUW'ed.
  //
  // Use a temporary node to keep N from being resolved.
  auto Temp = MDTuple::getTemporary(Context, None);
  Metadata *Ops[] = {nullptr, Temp.get()};

  MDNode *Empty = MDTuple::get(Context, ArrayRef<Metadata *>());
  MDNode *N = MDTuple::get(Context, Ops);
  EXPECT_EQ(nullptr, N->getOperand(0));
  ASSERT_FALSE(N->isResolved());

  // Check code for replacing resolved nodes.
  N->replaceOperandWith(0, Empty);
  EXPECT_EQ(Empty, N->getOperand(0));

  // Check code for adding another unresolved operand.
  N->replaceOperandWith(0, Temp.get());
  EXPECT_EQ(Temp.get(), N->getOperand(0));

  // Remove the references to Temp; required for teardown.
  Temp->replaceAllUsesWith(nullptr);
}

TEST_F(MDNodeTest, replaceWithUniqued) {
  auto *Empty = MDTuple::get(Context, None);
  MDTuple *FirstUniqued;
  {
    Metadata *Ops[] = {Empty};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Don't expect a collision.
    auto *Current = Temp.get();
    FirstUniqued = MDNode::replaceWithUniqued(std::move(Temp));
    EXPECT_TRUE(FirstUniqued->isUniqued());
    EXPECT_TRUE(FirstUniqued->isResolved());
    EXPECT_EQ(Current, FirstUniqued);
  }
  {
    Metadata *Ops[] = {Empty};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Should collide with Uniqued above this time.
    auto *Uniqued = MDNode::replaceWithUniqued(std::move(Temp));
    EXPECT_TRUE(Uniqued->isUniqued());
    EXPECT_TRUE(Uniqued->isResolved());
    EXPECT_EQ(FirstUniqued, Uniqued);
  }
  {
    auto Unresolved = MDTuple::getTemporary(Context, None);
    Metadata *Ops[] = {Unresolved.get()};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Shouldn't be resolved.
    auto *Uniqued = MDNode::replaceWithUniqued(std::move(Temp));
    EXPECT_TRUE(Uniqued->isUniqued());
    EXPECT_FALSE(Uniqued->isResolved());

    // Should be a different node.
    EXPECT_NE(FirstUniqued, Uniqued);

    // Should resolve when we update its node (note: be careful to avoid a
    // collision with any other nodes above).
    Uniqued->replaceOperandWith(0, nullptr);
    EXPECT_TRUE(Uniqued->isResolved());
  }
}

TEST_F(MDNodeTest, replaceWithUniquedResolvingOperand) {
  // temp !{}
  MDTuple *Op = MDTuple::getTemporary(Context, None).release();
  EXPECT_FALSE(Op->isResolved());

  // temp !{temp !{}}
  Metadata *Ops[] = {Op};
  MDTuple *N = MDTuple::getTemporary(Context, Ops).release();
  EXPECT_FALSE(N->isResolved());

  // temp !{temp !{}} => !{temp !{}}
  ASSERT_EQ(N, MDNode::replaceWithUniqued(TempMDTuple(N)));
  EXPECT_FALSE(N->isResolved());

  // !{temp !{}} => !{!{}}
  ASSERT_EQ(Op, MDNode::replaceWithUniqued(TempMDTuple(Op)));
  EXPECT_TRUE(Op->isResolved());
  EXPECT_TRUE(N->isResolved());
}

TEST_F(MDNodeTest, replaceWithUniquedDeletedOperand) {
  // i1* @GV
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  ConstantAsMetadata *Op = ConstantAsMetadata::get(GV.get());

  // temp !{i1* @GV}
  Metadata *Ops[] = {Op};
  MDTuple *N = MDTuple::getTemporary(Context, Ops).release();

  // temp !{i1* @GV} => !{i1* @GV}
  ASSERT_EQ(N, MDNode::replaceWithUniqued(TempMDTuple(N)));
  ASSERT_TRUE(N->isUniqued());

  // !{i1* @GV} => !{null}
  GV.reset();
  ASSERT_TRUE(N->isDistinct());
  ASSERT_EQ(nullptr, N->getOperand(0));
  Metadata *NullOps[] = {nullptr};
  ASSERT_NE(N, MDTuple::get(Context, NullOps));
}

TEST_F(MDNodeTest, replaceWithUniquedChangedOperand) {
  // i1* @GV
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  ConstantAsMetadata *Op = ConstantAsMetadata::get(GV.get());

  // temp !{i1* @GV}
  Metadata *Ops[] = {Op};
  MDTuple *N = MDTuple::getTemporary(Context, Ops).release();

  // temp !{i1* @GV} => !{i1* @GV}
  ASSERT_EQ(N, MDNode::replaceWithUniqued(TempMDTuple(N)));
  ASSERT_TRUE(N->isUniqued());

  // !{i1* @GV} => !{i1* @GV2}
  std::unique_ptr<GlobalVariable> GV2(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  GV->replaceAllUsesWith(GV2.get());
  ASSERT_TRUE(N->isUniqued());
  Metadata *NullOps[] = {ConstantAsMetadata::get(GV2.get())};
  ASSERT_EQ(N, MDTuple::get(Context, NullOps));
}

TEST_F(MDNodeTest, replaceWithDistinct) {
  {
    auto *Empty = MDTuple::get(Context, None);
    Metadata *Ops[] = {Empty};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Don't expect a collision.
    auto *Current = Temp.get();
    auto *Distinct = MDNode::replaceWithDistinct(std::move(Temp));
    EXPECT_TRUE(Distinct->isDistinct());
    EXPECT_TRUE(Distinct->isResolved());
    EXPECT_EQ(Current, Distinct);
  }
  {
    auto Unresolved = MDTuple::getTemporary(Context, None);
    Metadata *Ops[] = {Unresolved.get()};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Don't expect a collision.
    auto *Current = Temp.get();
    auto *Distinct = MDNode::replaceWithDistinct(std::move(Temp));
    EXPECT_TRUE(Distinct->isDistinct());
    EXPECT_TRUE(Distinct->isResolved());
    EXPECT_EQ(Current, Distinct);

    // Cleanup; required for teardown.
    Unresolved->replaceAllUsesWith(nullptr);
  }
}

TEST_F(MDNodeTest, replaceWithPermanent) {
  Metadata *Ops[] = {nullptr};
  auto Temp = MDTuple::getTemporary(Context, Ops);
  auto *T = Temp.get();

  // U is a normal, uniqued node that references T.
  auto *U = MDTuple::get(Context, T);
  EXPECT_TRUE(U->isUniqued());

  // Make Temp self-referencing.
  Temp->replaceOperandWith(0, T);

  // Try to uniquify Temp.  This should, despite the name in the API, give a
  // 'distinct' node, since self-references aren't allowed to be uniqued.
  //
  // Since it's distinct, N should have the same address as when it was a
  // temporary (i.e., be equal to T not U).
  auto *N = MDNode::replaceWithPermanent(std::move(Temp));
  EXPECT_EQ(N, T);
  EXPECT_TRUE(N->isDistinct());

  // U should be the canonical unique node with N as the argument.
  EXPECT_EQ(U, MDTuple::get(Context, N));
  EXPECT_TRUE(U->isUniqued());

  // This temporary should collide with U when replaced, but it should still be
  // uniqued.
  EXPECT_EQ(U, MDNode::replaceWithPermanent(MDTuple::getTemporary(Context, N)));
  EXPECT_TRUE(U->isUniqued());

  // This temporary should become a new uniqued node.
  auto Temp2 = MDTuple::getTemporary(Context, U);
  auto *V = Temp2.get();
  EXPECT_EQ(V, MDNode::replaceWithPermanent(std::move(Temp2)));
  EXPECT_TRUE(V->isUniqued());
  EXPECT_EQ(U, V->getOperand(0));
}

TEST_F(MDNodeTest, deleteTemporaryWithTrackingRef) {
  TrackingMDRef Ref;
  EXPECT_EQ(nullptr, Ref.get());
  {
    auto Temp = MDTuple::getTemporary(Context, None);
    Ref.reset(Temp.get());
    EXPECT_EQ(Temp.get(), Ref.get());
  }
  EXPECT_EQ(nullptr, Ref.get());
}

typedef MetadataTest DILocationTest;

TEST_F(DILocationTest, Overflow) {
  DISubprogram *N = getSubprogram();
  {
    DILocation *L = DILocation::get(Context, 2, 7, N);
    EXPECT_EQ(2u, L->getLine());
    EXPECT_EQ(7u, L->getColumn());
  }
  unsigned U16 = 1u << 16;
  {
    DILocation *L = DILocation::get(Context, UINT32_MAX, U16 - 1, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(U16 - 1, L->getColumn());
  }
  {
    DILocation *L = DILocation::get(Context, UINT32_MAX, U16, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(0u, L->getColumn());
  }
  {
    DILocation *L = DILocation::get(Context, UINT32_MAX, U16 + 1, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(0u, L->getColumn());
  }
}

TEST_F(DILocationTest, Merge) {
  DISubprogram *N = getSubprogram();
  DIScope *S = DILexicalBlock::get(Context, N, getFile(), 3, 4);

  {
    // Identical.
    auto *A = DILocation::get(Context, 2, 7, N);
    auto *B = DILocation::get(Context, 2, 7, N);
    auto *M = DILocation::getMergedLocation(A, B);
    EXPECT_EQ(2u, M->getLine());
    EXPECT_EQ(7u, M->getColumn());
    EXPECT_EQ(N, M->getScope());
  }

  {
    // Identical, different scopes.
    auto *A = DILocation::get(Context, 2, 7, N);
    auto *B = DILocation::get(Context, 2, 7, S);
    auto *M = DILocation::getMergedLocation(A, B);
    EXPECT_EQ(0u, M->getLine()); // FIXME: Should this be 2?
    EXPECT_EQ(0u, M->getColumn()); // FIXME: Should this be 7?
    EXPECT_EQ(N, M->getScope());
  }

  {
    // Different lines, same scopes.
    auto *A = DILocation::get(Context, 1, 6, N);
    auto *B = DILocation::get(Context, 2, 7, N);
    auto *M = DILocation::getMergedLocation(A, B);
    EXPECT_EQ(0u, M->getLine());
    EXPECT_EQ(0u, M->getColumn());
    EXPECT_EQ(N, M->getScope());
  }

  {
    // Twisty locations, all different, same function.
    auto *A = DILocation::get(Context, 1, 6, N);
    auto *B = DILocation::get(Context, 2, 7, S);
    auto *M = DILocation::getMergedLocation(A, B);
    EXPECT_EQ(0u, M->getLine());
    EXPECT_EQ(0u, M->getColumn());
    EXPECT_EQ(N, M->getScope());
  }

  {
    // Different function, same inlined-at.
    auto *F = getFile();
    auto *SP1 = DISubprogram::getDistinct(Context, F, "a", "a", F, 0, nullptr,
                                          0, nullptr, 0, 0, DINode::FlagZero,
                                          DISubprogram::SPFlagZero, nullptr);
    auto *SP2 = DISubprogram::getDistinct(Context, F, "b", "b", F, 0, nullptr,
                                          0, nullptr, 0, 0, DINode::FlagZero,
                                          DISubprogram::SPFlagZero, nullptr);

    auto *I = DILocation::get(Context, 2, 7, N);
    auto *A = DILocation::get(Context, 1, 6, SP1, I);
    auto *B = DILocation::get(Context, 2, 7, SP2, I);
    auto *M = DILocation::getMergedLocation(A, B);
    EXPECT_EQ(0u, M->getLine());
    EXPECT_EQ(0u, M->getColumn());
    EXPECT_TRUE(isa<DILocalScope>(M->getScope()));
    EXPECT_EQ(I, M->getInlinedAt());
  }

   {
    // Completely different.
    auto *I = DILocation::get(Context, 2, 7, N);
    auto *A = DILocation::get(Context, 1, 6, S, I);
    auto *B = DILocation::get(Context, 2, 7, getSubprogram());
    auto *M = DILocation::getMergedLocation(A, B);
    EXPECT_EQ(0u, M->getLine());
    EXPECT_EQ(0u, M->getColumn());
    EXPECT_TRUE(isa<DILocalScope>(M->getScope()));
    EXPECT_EQ(S, M->getScope());
    EXPECT_EQ(nullptr, M->getInlinedAt());
  }
}

TEST_F(DILocationTest, getDistinct) {
  MDNode *N = getSubprogram();
  DILocation *L0 = DILocation::getDistinct(Context, 2, 7, N);
  EXPECT_TRUE(L0->isDistinct());
  DILocation *L1 = DILocation::get(Context, 2, 7, N);
  EXPECT_FALSE(L1->isDistinct());
  EXPECT_EQ(L1, DILocation::get(Context, 2, 7, N));
}

TEST_F(DILocationTest, getTemporary) {
  MDNode *N = MDNode::get(Context, None);
  auto L = DILocation::getTemporary(Context, 2, 7, N);
  EXPECT_TRUE(L->isTemporary());
  EXPECT_FALSE(L->isResolved());
}

TEST_F(DILocationTest, cloneTemporary) {
  MDNode *N = MDNode::get(Context, None);
  auto L = DILocation::getTemporary(Context, 2, 7, N);
  EXPECT_TRUE(L->isTemporary());
  auto L2 = L->clone();
  EXPECT_TRUE(L2->isTemporary());
}

TEST_F(DILocationTest, discriminatorEncoding) {
  EXPECT_EQ(0U, DILocation::encodeDiscriminator(0, 0, 0).getValue());

  // Encode base discriminator as a component: lsb is 0, then the value.
  // The other components are all absent, so we leave all the other bits 0.
  EXPECT_EQ(2U, DILocation::encodeDiscriminator(1, 0, 0).getValue());

  // Base discriminator component is empty, so lsb is 1. Next component is not
  // empty, so its lsb is 0, then its value (1). Next component is empty.
  // So the bit pattern is 101.
  EXPECT_EQ(5U, DILocation::encodeDiscriminator(0, 1, 0).getValue());

  // First 2 components are empty, so the bit pattern is 11. Then the
  // next component - ending up with 1011.
  EXPECT_EQ(0xbU, DILocation::encodeDiscriminator(0, 0, 1).getValue());

  // The bit pattern for the first 2 components is 11. The next bit is 0,
  // because the last component is not empty. We have 29 bits usable for
  // encoding, but we cap it at 12 bits uniformously for all components. We
  // encode the last component over 14 bits.
  EXPECT_EQ(0xfffbU, DILocation::encodeDiscriminator(0, 0, 0xfff).getValue());

  EXPECT_EQ(0x102U, DILocation::encodeDiscriminator(1, 1, 0).getValue());

  EXPECT_EQ(0x13eU, DILocation::encodeDiscriminator(0x1f, 1, 0).getValue());

  EXPECT_EQ(0x87feU, DILocation::encodeDiscriminator(0x1ff, 1, 0).getValue());

  EXPECT_EQ(0x1f3eU, DILocation::encodeDiscriminator(0x1f, 0x1f, 0).getValue());

  EXPECT_EQ(0x3ff3eU,
            DILocation::encodeDiscriminator(0x1f, 0x1ff, 0).getValue());

  EXPECT_EQ(0x1ff87feU,
            DILocation::encodeDiscriminator(0x1ff, 0x1ff, 0).getValue());

  EXPECT_EQ(0xfff9f3eU,
            DILocation::encodeDiscriminator(0x1f, 0x1f, 0xfff).getValue());

  EXPECT_EQ(0xffc3ff3eU,
            DILocation::encodeDiscriminator(0x1f, 0x1ff, 0x1ff).getValue());

  EXPECT_EQ(0xffcf87feU,
            DILocation::encodeDiscriminator(0x1ff, 0x1f, 0x1ff).getValue());

  EXPECT_EQ(0xe1ff87feU,
            DILocation::encodeDiscriminator(0x1ff, 0x1ff, 7).getValue());
}

TEST_F(DILocationTest, discriminatorEncodingNegativeTests) {
  EXPECT_EQ(None, DILocation::encodeDiscriminator(0, 0, 0x1000));
  EXPECT_EQ(None, DILocation::encodeDiscriminator(0x1000, 0, 0));
  EXPECT_EQ(None, DILocation::encodeDiscriminator(0, 0x1000, 0));
  EXPECT_EQ(None, DILocation::encodeDiscriminator(0, 0, 0x1000));
  EXPECT_EQ(None, DILocation::encodeDiscriminator(0x1ff, 0x1ff, 8));
  EXPECT_EQ(None,
            DILocation::encodeDiscriminator(std::numeric_limits<uint32_t>::max(),
                                            std::numeric_limits<uint32_t>::max(),
                                            0));
}

TEST_F(DILocationTest, discriminatorSpecialCases) {
  // We don't test getCopyIdentifier here because the only way
  // to set it is by constructing an encoded discriminator using
  // encodeDiscriminator, which is already tested.
  auto L1 = DILocation::get(Context, 1, 2, getSubprogram());
  EXPECT_EQ(0U, L1->getBaseDiscriminator());
  EXPECT_EQ(1U, L1->getDuplicationFactor());

  EXPECT_EQ(L1, L1->cloneWithBaseDiscriminator(0).getValue());
  EXPECT_EQ(L1, L1->cloneByMultiplyingDuplicationFactor(0).getValue());
  EXPECT_EQ(L1, L1->cloneByMultiplyingDuplicationFactor(1).getValue());

  auto L2 = L1->cloneWithBaseDiscriminator(1).getValue();
  EXPECT_EQ(0U, L1->getBaseDiscriminator());
  EXPECT_EQ(1U, L1->getDuplicationFactor());

  EXPECT_EQ(1U, L2->getBaseDiscriminator());
  EXPECT_EQ(1U, L2->getDuplicationFactor());

  auto L3 = L2->cloneByMultiplyingDuplicationFactor(2).getValue();
  EXPECT_EQ(1U, L3->getBaseDiscriminator());
  EXPECT_EQ(2U, L3->getDuplicationFactor());

  EXPECT_EQ(L2, L2->cloneByMultiplyingDuplicationFactor(1).getValue());

  auto L4 = L3->cloneByMultiplyingDuplicationFactor(4).getValue();
  EXPECT_EQ(1U, L4->getBaseDiscriminator());
  EXPECT_EQ(8U, L4->getDuplicationFactor());

  auto L5 = L4->cloneWithBaseDiscriminator(2).getValue();
  EXPECT_EQ(2U, L5->getBaseDiscriminator());
  EXPECT_EQ(8U, L5->getDuplicationFactor());

  // Check extreme cases
  auto L6 = L1->cloneWithBaseDiscriminator(0xfff).getValue();
  EXPECT_EQ(0xfffU, L6->getBaseDiscriminator());
  EXPECT_EQ(0xfffU, L6->cloneByMultiplyingDuplicationFactor(0xfff)
                        .getValue()
                        ->getDuplicationFactor());

  // Check we return None for unencodable cases.
  EXPECT_EQ(None, L4->cloneWithBaseDiscriminator(0x1000));
  EXPECT_EQ(None, L4->cloneByMultiplyingDuplicationFactor(0x1000));
}


typedef MetadataTest GenericDINodeTest;

TEST_F(GenericDINodeTest, get) {
  StringRef Header = "header";
  auto *Empty = MDNode::get(Context, None);
  Metadata *Ops1[] = {Empty};
  auto *N = GenericDINode::get(Context, 15, Header, Ops1);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(2u, N->getNumOperands());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(MDString::get(Context, Header), N->getOperand(0));
  EXPECT_EQ(1u, N->getNumDwarfOperands());
  EXPECT_EQ(Empty, N->getDwarfOperand(0));
  EXPECT_EQ(Empty, N->getOperand(1));
  ASSERT_TRUE(N->isUniqued());

  EXPECT_EQ(N, GenericDINode::get(Context, 15, Header, Ops1));

  N->replaceOperandWith(1, nullptr);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(nullptr, N->getDwarfOperand(0));
  ASSERT_TRUE(N->isUniqued());

  Metadata *Ops2[] = {nullptr};
  EXPECT_EQ(N, GenericDINode::get(Context, 15, Header, Ops2));

  N->replaceDwarfOperandWith(0, Empty);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(Empty, N->getDwarfOperand(0));
  ASSERT_TRUE(N->isUniqued());
  EXPECT_EQ(N, GenericDINode::get(Context, 15, Header, Ops1));

  TempGenericDINode Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(GenericDINodeTest, getEmptyHeader) {
  // Canonicalize !"" to null.
  auto *N = GenericDINode::get(Context, 15, StringRef(), None);
  EXPECT_EQ(StringRef(), N->getHeader());
  EXPECT_EQ(nullptr, N->getOperand(0));
}

typedef MetadataTest DISubrangeTest;

TEST_F(DISubrangeTest, get) {
  auto *N = DISubrange::get(Context, 5, 7);
  auto Count = N->getCount();
  auto Lower = N->getLowerBound();
  EXPECT_EQ(dwarf::DW_TAG_subrange_type, N->getTag());
  ASSERT_TRUE(Count);
  ASSERT_TRUE(Count.is<ConstantInt*>());
  EXPECT_EQ(5, Count.get<ConstantInt*>()->getSExtValue());
  EXPECT_EQ(7, Lower.get<ConstantInt *>()->getSExtValue());
  EXPECT_EQ(N, DISubrange::get(Context, 5, 7));
  EXPECT_EQ(DISubrange::get(Context, 5, 0), DISubrange::get(Context, 5));

  TempDISubrange Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DISubrangeTest, getEmptyArray) {
  auto *N = DISubrange::get(Context, -1, 0);
  auto Count = N->getCount();
  auto Lower = N->getLowerBound();
  EXPECT_EQ(dwarf::DW_TAG_subrange_type, N->getTag());
  ASSERT_TRUE(Count);
  ASSERT_TRUE(Count.is<ConstantInt*>());
  EXPECT_EQ(-1, Count.get<ConstantInt*>()->getSExtValue());
  EXPECT_EQ(0, Lower.get<ConstantInt *>()->getSExtValue());
  EXPECT_EQ(N, DISubrange::get(Context, -1, 0));
}

TEST_F(DISubrangeTest, getVariableCount) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  auto *VlaExpr = DILocalVariable::get(Context, Scope, "vla_expr", File, 8,
                                       Type, 2, Flags, 8, nullptr);

  auto *N = DISubrange::get(Context, VlaExpr, 0);
  auto Count = N->getCount();
  auto Lower = N->getLowerBound();
  ASSERT_TRUE(Count);
  ASSERT_TRUE(Count.is<DIVariable*>());
  EXPECT_EQ(VlaExpr, Count.get<DIVariable*>());
  ASSERT_TRUE(isa<DIVariable>(N->getRawCountNode()));
  EXPECT_EQ(0, Lower.get<ConstantInt *>()->getSExtValue());
  EXPECT_EQ("vla_expr", Count.get<DIVariable*>()->getName());
  EXPECT_EQ(N, DISubrange::get(Context, VlaExpr, 0));
}

TEST_F(DISubrangeTest, fortranAllocatableInt) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  auto *LI = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), -10));
  auto *UI = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), 10));
  auto *SI = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), 4));
  auto *UIother = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), 20));
  auto *UVother = DILocalVariable::get(Context, Scope, "ubother", File, 8, Type,
                                       2, Flags, 8, nullptr);
  auto *UEother = DIExpression::get(Context, {5, 6});
  auto *LIZero = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), 0));
  auto *UIZero = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), 0));

  auto *N = DISubrange::get(Context, nullptr, LI, UI, SI);

  auto Lower = N->getLowerBound();
  ASSERT_TRUE(Lower);
  ASSERT_TRUE(Lower.is<ConstantInt *>());
  EXPECT_EQ(cast<ConstantInt>(LI->getValue()), Lower.get<ConstantInt *>());

  auto Upper = N->getUpperBound();
  ASSERT_TRUE(Upper);
  ASSERT_TRUE(Upper.is<ConstantInt *>());
  EXPECT_EQ(cast<ConstantInt>(UI->getValue()), Upper.get<ConstantInt *>());

  auto Stride = N->getStride();
  ASSERT_TRUE(Stride);
  ASSERT_TRUE(Stride.is<ConstantInt *>());
  EXPECT_EQ(cast<ConstantInt>(SI->getValue()), Stride.get<ConstantInt *>());

  EXPECT_EQ(N, DISubrange::get(Context, nullptr, LI, UI, SI));

  EXPECT_NE(N, DISubrange::get(Context, nullptr, LI, UIother, SI));
  EXPECT_NE(N, DISubrange::get(Context, nullptr, LI, UEother, SI));
  EXPECT_NE(N, DISubrange::get(Context, nullptr, LI, UVother, SI));

  auto *NZeroLower = DISubrange::get(Context, nullptr, LIZero, UI, SI);
  EXPECT_NE(NZeroLower, DISubrange::get(Context, nullptr, nullptr, UI, SI));

  auto *NZeroUpper = DISubrange::get(Context, nullptr, LI, UIZero, SI);
  EXPECT_NE(NZeroUpper, DISubrange::get(Context, nullptr, LI, nullptr, SI));
}

TEST_F(DISubrangeTest, fortranAllocatableVar) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  auto *LV =
      DILocalVariable::get(Context, Scope, "lb", File, 8, Type, 2, Flags, 8,
                           nullptr);
  auto *UV =
      DILocalVariable::get(Context, Scope, "ub", File, 8, Type, 2, Flags, 8,
                           nullptr);
  auto *SV =
      DILocalVariable::get(Context, Scope, "st", File, 8, Type, 2, Flags, 8,
                           nullptr);
  auto *SVother = DILocalVariable::get(Context, Scope, "stother", File, 8, Type,
                                       2, Flags, 8, nullptr);
  auto *SIother = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), 20));
  auto *SEother = DIExpression::get(Context, {5, 6});

  auto *N = DISubrange::get(Context, nullptr, LV, UV, SV);

  auto Lower = N->getLowerBound();
  ASSERT_TRUE(Lower);
  ASSERT_TRUE(Lower.is<DIVariable *>());
  EXPECT_EQ(LV, Lower.get<DIVariable *>());

  auto Upper = N->getUpperBound();
  ASSERT_TRUE(Upper);
  ASSERT_TRUE(Upper.is<DIVariable *>());
  EXPECT_EQ(UV, Upper.get<DIVariable *>());

  auto Stride = N->getStride();
  ASSERT_TRUE(Stride);
  ASSERT_TRUE(Stride.is<DIVariable *>());
  EXPECT_EQ(SV, Stride.get<DIVariable *>());

  EXPECT_EQ(N, DISubrange::get(Context, nullptr, LV, UV, SV));

  EXPECT_NE(N, DISubrange::get(Context, nullptr, LV, UV, SVother));
  EXPECT_NE(N, DISubrange::get(Context, nullptr, LV, UV, SEother));
  EXPECT_NE(N, DISubrange::get(Context, nullptr, LV, UV, SIother));
}

TEST_F(DISubrangeTest, fortranAllocatableExpr) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  auto *LE = DIExpression::get(Context, {1, 2});
  auto *UE = DIExpression::get(Context, {2, 3});
  auto *SE = DIExpression::get(Context, {3, 4});
  auto *LEother = DIExpression::get(Context, {5, 6});
  auto *LIother = ConstantAsMetadata::get(
      ConstantInt::getSigned(Type::getInt64Ty(Context), 20));
  auto *LVother = DILocalVariable::get(Context, Scope, "lbother", File, 8, Type,
                                       2, Flags, 8, nullptr);

  auto *N = DISubrange::get(Context, nullptr, LE, UE, SE);

  auto Lower = N->getLowerBound();
  ASSERT_TRUE(Lower);
  ASSERT_TRUE(Lower.is<DIExpression *>());
  EXPECT_EQ(LE, Lower.get<DIExpression *>());

  auto Upper = N->getUpperBound();
  ASSERT_TRUE(Upper);
  ASSERT_TRUE(Upper.is<DIExpression *>());
  EXPECT_EQ(UE, Upper.get<DIExpression *>());

  auto Stride = N->getStride();
  ASSERT_TRUE(Stride);
  ASSERT_TRUE(Stride.is<DIExpression *>());
  EXPECT_EQ(SE, Stride.get<DIExpression *>());

  EXPECT_EQ(N, DISubrange::get(Context, nullptr, LE, UE, SE));

  EXPECT_NE(N, DISubrange::get(Context, nullptr, LEother, UE, SE));
  EXPECT_NE(N, DISubrange::get(Context, nullptr, LIother, UE, SE));
  EXPECT_NE(N, DISubrange::get(Context, nullptr, LVother, UE, SE));
}

typedef MetadataTest DIGenericSubrangeTest;

TEST_F(DIGenericSubrangeTest, fortranAssumedRankInt) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  auto *LI = DIExpression::get(
      Context, {dwarf::DW_OP_consts, static_cast<uint64_t>(-10)});
  auto *UI = DIExpression::get(Context, {dwarf::DW_OP_consts, 10});
  auto *SI = DIExpression::get(Context, {dwarf::DW_OP_consts, 4});
  auto *UIother = DIExpression::get(Context, {dwarf::DW_OP_consts, 20});
  auto *UVother = DILocalVariable::get(Context, Scope, "ubother", File, 8, Type,
                                       2, Flags, 8, nullptr);
  auto *UEother = DIExpression::get(Context, {5, 6});
  auto *LIZero = DIExpression::get(Context, {dwarf::DW_OP_consts, 0});
  auto *UIZero = DIExpression::get(Context, {dwarf::DW_OP_consts, 0});

  auto *N = DIGenericSubrange::get(Context, nullptr, LI, UI, SI);

  auto Lower = N->getLowerBound();
  ASSERT_TRUE(Lower);
  ASSERT_TRUE(Lower.is<DIExpression *>());
  EXPECT_EQ(dyn_cast_or_null<DIExpression>(LI), Lower.get<DIExpression *>());

  auto Upper = N->getUpperBound();
  ASSERT_TRUE(Upper);
  ASSERT_TRUE(Upper.is<DIExpression *>());
  EXPECT_EQ(dyn_cast_or_null<DIExpression>(UI), Upper.get<DIExpression *>());

  auto Stride = N->getStride();
  ASSERT_TRUE(Stride);
  ASSERT_TRUE(Stride.is<DIExpression *>());
  EXPECT_EQ(dyn_cast_or_null<DIExpression>(SI), Stride.get<DIExpression *>());

  EXPECT_EQ(N, DIGenericSubrange::get(Context, nullptr, LI, UI, SI));

  EXPECT_NE(N, DIGenericSubrange::get(Context, nullptr, LI, UIother, SI));
  EXPECT_NE(N, DIGenericSubrange::get(Context, nullptr, LI, UEother, SI));
  EXPECT_NE(N, DIGenericSubrange::get(Context, nullptr, LI, UVother, SI));

  auto *NZeroLower = DIGenericSubrange::get(Context, nullptr, LIZero, UI, SI);
  EXPECT_NE(NZeroLower,
            DIGenericSubrange::get(Context, nullptr, nullptr, UI, SI));

  auto *NZeroUpper = DIGenericSubrange::get(Context, nullptr, LI, UIZero, SI);
  EXPECT_NE(NZeroUpper,
            DIGenericSubrange::get(Context, nullptr, LI, nullptr, SI));
}

TEST_F(DIGenericSubrangeTest, fortranAssumedRankVar) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  auto *LV =
      DILocalVariable::get(Context, Scope, "lb", File, 8, Type, 2, Flags, 8,
                           nullptr);
  auto *UV =
      DILocalVariable::get(Context, Scope, "ub", File, 8, Type, 2, Flags, 8,
                           nullptr);
  auto *SV =
      DILocalVariable::get(Context, Scope, "st", File, 8, Type, 2, Flags, 8,
                           nullptr);
  auto *SVother = DILocalVariable::get(Context, Scope, "stother", File, 8, Type,
                                       2, Flags, 8, nullptr);
  auto *SIother = DIExpression::get(
      Context, {dwarf::DW_OP_consts, static_cast<uint64_t>(-1)});
  auto *SEother = DIExpression::get(Context, {5, 6});

  auto *N = DIGenericSubrange::get(Context, nullptr, LV, UV, SV);

  auto Lower = N->getLowerBound();
  ASSERT_TRUE(Lower);
  ASSERT_TRUE(Lower.is<DIVariable *>());
  EXPECT_EQ(LV, Lower.get<DIVariable *>());

  auto Upper = N->getUpperBound();
  ASSERT_TRUE(Upper);
  ASSERT_TRUE(Upper.is<DIVariable *>());
  EXPECT_EQ(UV, Upper.get<DIVariable *>());

  auto Stride = N->getStride();
  ASSERT_TRUE(Stride);
  ASSERT_TRUE(Stride.is<DIVariable *>());
  EXPECT_EQ(SV, Stride.get<DIVariable *>());

  EXPECT_EQ(N, DIGenericSubrange::get(Context, nullptr, LV, UV, SV));

  EXPECT_NE(N, DIGenericSubrange::get(Context, nullptr, LV, UV, SVother));
  EXPECT_NE(N, DIGenericSubrange::get(Context, nullptr, LV, UV, SEother));
  EXPECT_NE(N, DIGenericSubrange::get(Context, nullptr, LV, UV, SIother));
}

TEST_F(DIGenericSubrangeTest, useDIBuilder) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  auto *LV =
      DILocalVariable::get(Context, Scope, "lb", File, 8, Type, 2, Flags, 8, nullptr);
  auto *UE = DIExpression::get(Context, {2, 3});
  auto *SE = DIExpression::get(Context, {3, 4});

  auto *LVother = DILocalVariable::get(Context, Scope, "lbother", File, 8, Type,
                                       2, Flags, 8, nullptr);
  auto *LIother = DIExpression::get(
      Context, {dwarf::DW_OP_consts, static_cast<uint64_t>(-1)});

  Module M("M", Context);
  DIBuilder DIB(M);

  auto *N = DIB.getOrCreateGenericSubrange(
      DIGenericSubrange::BoundType(nullptr), DIGenericSubrange::BoundType(LV),
      DIGenericSubrange::BoundType(UE), DIGenericSubrange::BoundType(SE));

  auto Lower = N->getLowerBound();
  ASSERT_TRUE(Lower);
  ASSERT_TRUE(Lower.is<DIVariable *>());
  EXPECT_EQ(LV, Lower.get<DIVariable *>());

  auto Upper = N->getUpperBound();
  ASSERT_TRUE(Upper);
  ASSERT_TRUE(Upper.is<DIExpression *>());
  EXPECT_EQ(UE, Upper.get<DIExpression *>());

  auto Stride = N->getStride();
  ASSERT_TRUE(Stride);
  ASSERT_TRUE(Stride.is<DIExpression *>());
  EXPECT_EQ(SE, Stride.get<DIExpression *>());

  EXPECT_EQ(
      N, DIB.getOrCreateGenericSubrange(DIGenericSubrange::BoundType(nullptr),
                                        DIGenericSubrange::BoundType(LV),
                                        DIGenericSubrange::BoundType(UE),
                                        DIGenericSubrange::BoundType(SE)));

  EXPECT_NE(
      N, DIB.getOrCreateGenericSubrange(DIGenericSubrange::BoundType(nullptr),
                                        DIGenericSubrange::BoundType(LVother),
                                        DIGenericSubrange::BoundType(UE),
                                        DIGenericSubrange::BoundType(SE)));
  EXPECT_NE(
      N, DIB.getOrCreateGenericSubrange(DIGenericSubrange::BoundType(nullptr),
                                        DIGenericSubrange::BoundType(LIother),
                                        DIGenericSubrange::BoundType(UE),
                                        DIGenericSubrange::BoundType(SE)));
}
typedef MetadataTest DIEnumeratorTest;

TEST_F(DIEnumeratorTest, get) {
  auto *N = DIEnumerator::get(Context, 7, false, "name");
  EXPECT_EQ(dwarf::DW_TAG_enumerator, N->getTag());
  EXPECT_EQ(7, N->getValue().getSExtValue());
  EXPECT_FALSE(N->isUnsigned());
  EXPECT_EQ("name", N->getName());
  EXPECT_EQ(N, DIEnumerator::get(Context, 7, false, "name"));

  EXPECT_NE(N, DIEnumerator::get(Context, 7, true, "name"));
  EXPECT_NE(N, DIEnumerator::get(Context, 8, false, "name"));
  EXPECT_NE(N, DIEnumerator::get(Context, 7, false, "nam"));

  TempDIEnumerator Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIEnumeratorTest, getWithLargeValues) {
  auto *N = DIEnumerator::get(Context, APInt::getMaxValue(128), false, "val");
  EXPECT_EQ(128U, N->getValue().countPopulation());
  EXPECT_EQ(N,
            DIEnumerator::get(Context, APInt::getMaxValue(128), false, "val"));
  EXPECT_NE(N,
            DIEnumerator::get(Context, APInt::getMinValue(128), false, "val"));
}

typedef MetadataTest DIBasicTypeTest;

TEST_F(DIBasicTypeTest, get) {
  auto *N =
      DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33, 26, 7,
                        DINode::FlagZero);
  EXPECT_EQ(dwarf::DW_TAG_base_type, N->getTag());
  EXPECT_EQ("special", N->getName());
  EXPECT_EQ(33u, N->getSizeInBits());
  EXPECT_EQ(26u, N->getAlignInBits());
  EXPECT_EQ(7u, N->getEncoding());
  EXPECT_EQ(0u, N->getLine());
  EXPECT_EQ(DINode::FlagZero, N->getFlags());
  EXPECT_EQ(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 7, DINode::FlagZero));

  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_unspecified_type,
                                "special", 33, 26, 7, DINode::FlagZero));
  EXPECT_NE(N,
            DIBasicType::get(Context, dwarf::DW_TAG_base_type, "s", 33, 26, 7,
                              DINode::FlagZero));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 32,
                                26, 7, DINode::FlagZero));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                25, 7, DINode::FlagZero));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 6, DINode::FlagZero));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 7, DINode::FlagBigEndian));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 7, DINode::FlagLittleEndian));

  TempDIBasicType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIBasicTypeTest, getWithLargeValues) {
  auto *N = DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special",
                             UINT64_MAX, UINT32_MAX - 1, 7, DINode::FlagZero);
  EXPECT_EQ(UINT64_MAX, N->getSizeInBits());
  EXPECT_EQ(UINT32_MAX - 1, N->getAlignInBits());
}

TEST_F(DIBasicTypeTest, getUnspecified) {
  auto *N =
      DIBasicType::get(Context, dwarf::DW_TAG_unspecified_type, "unspecified");
  EXPECT_EQ(dwarf::DW_TAG_unspecified_type, N->getTag());
  EXPECT_EQ("unspecified", N->getName());
  EXPECT_EQ(0u, N->getSizeInBits());
  EXPECT_EQ(0u, N->getAlignInBits());
  EXPECT_EQ(0u, N->getEncoding());
  EXPECT_EQ(0u, N->getLine());
  EXPECT_EQ(DINode::FlagZero, N->getFlags());
}

typedef MetadataTest DITypeTest;

TEST_F(DITypeTest, clone) {
  // Check that DIType has a specialized clone that returns TempDIType.
  DIType *N = DIBasicType::get(Context, dwarf::DW_TAG_base_type, "int", 32, 32,
                               dwarf::DW_ATE_signed, DINode::FlagZero);

  TempDIType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DITypeTest, cloneWithFlags) {
  // void (void)
  Metadata *TypesOps[] = {nullptr};
  Metadata *Types = MDTuple::get(Context, TypesOps);

  DIType *D =
      DISubroutineType::getDistinct(Context, DINode::FlagZero, 0, Types);
  EXPECT_EQ(DINode::FlagZero, D->getFlags());
  TempDIType D2 = D->cloneWithFlags(DINode::FlagRValueReference);
  EXPECT_EQ(DINode::FlagRValueReference, D2->getFlags());
  EXPECT_EQ(DINode::FlagZero, D->getFlags());

  TempDIType T =
      DISubroutineType::getTemporary(Context, DINode::FlagZero, 0, Types);
  EXPECT_EQ(DINode::FlagZero, T->getFlags());
  TempDIType T2 = T->cloneWithFlags(DINode::FlagRValueReference);
  EXPECT_EQ(DINode::FlagRValueReference, T2->getFlags());
  EXPECT_EQ(DINode::FlagZero, T->getFlags());
}

typedef MetadataTest DIDerivedTypeTest;

TEST_F(DIDerivedTypeTest, get) {
  DIFile *File = getFile();
  DIScope *Scope = getSubprogram();
  DIType *BaseType = getBasicType("basic");
  MDTuple *ExtraData = getTuple();
  unsigned DWARFAddressSpace = 8;
  DINode::DIFlags Flags5 = static_cast<DINode::DIFlags>(5);
  DINode::DIFlags Flags4 = static_cast<DINode::DIFlags>(4);

  auto *N =
      DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something", File,
                         1, Scope, BaseType, 2, 3, 4, DWARFAddressSpace, Flags5,
                         ExtraData);
  EXPECT_EQ(dwarf::DW_TAG_pointer_type, N->getTag());
  EXPECT_EQ("something", N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(1u, N->getLine());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(BaseType, N->getBaseType());
  EXPECT_EQ(2u, N->getSizeInBits());
  EXPECT_EQ(3u, N->getAlignInBits());
  EXPECT_EQ(4u, N->getOffsetInBits());
  EXPECT_EQ(DWARFAddressSpace, N->getDWARFAddressSpace().getValue());
  EXPECT_EQ(5u, N->getFlags());
  EXPECT_EQ(ExtraData, N->getExtraData());
  EXPECT_EQ(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, DWARFAddressSpace, Flags5, ExtraData));

  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_reference_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, DWARFAddressSpace, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "else",
                                  File, 1, Scope, BaseType, 2, 3,
                                  4, DWARFAddressSpace, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", getFile(), 1, Scope, BaseType, 2,
                                  3, 4, DWARFAddressSpace, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 2, Scope, BaseType, 2, 3,
                                  4, DWARFAddressSpace, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, getSubprogram(),
                                  BaseType, 2, 3, 4, DWARFAddressSpace, Flags5,
                                  ExtraData));
  EXPECT_NE(N, DIDerivedType::get(
                   Context, dwarf::DW_TAG_pointer_type, "something", File, 1,
                   Scope, getBasicType("basic2"), 2, 3, 4, DWARFAddressSpace,
                   Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 3, 3,
                                  4, DWARFAddressSpace, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 2,
                                  4, DWARFAddressSpace, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  5, DWARFAddressSpace, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, DWARFAddressSpace + 1, Flags5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, DWARFAddressSpace, Flags4, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, DWARFAddressSpace, Flags5, getTuple()));

  TempDIDerivedType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIDerivedTypeTest, getWithLargeValues) {
  DIFile *File = getFile();
  DIScope *Scope = getSubprogram();
  DIType *BaseType = getBasicType("basic");
  MDTuple *ExtraData = getTuple();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(5);

  auto *N = DIDerivedType::get(
      Context, dwarf::DW_TAG_pointer_type, "something", File, 1, Scope,
      BaseType, UINT64_MAX, UINT32_MAX - 1, UINT64_MAX - 2, UINT32_MAX - 3,
      Flags, ExtraData);
  EXPECT_EQ(UINT64_MAX, N->getSizeInBits());
  EXPECT_EQ(UINT32_MAX - 1, N->getAlignInBits());
  EXPECT_EQ(UINT64_MAX - 2, N->getOffsetInBits());
  EXPECT_EQ(UINT32_MAX - 3, N->getDWARFAddressSpace().getValue());
}

typedef MetadataTest DICompositeTypeTest;

TEST_F(DICompositeTypeTest, get) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DIScope *Scope = getSubprogram();
  DIType *BaseType = getCompositeType();
  uint64_t SizeInBits = 2;
  uint32_t AlignInBits = 3;
  uint64_t OffsetInBits = 4;
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(5);
  MDTuple *Elements = getTuple();
  unsigned RuntimeLang = 6;
  DIType *VTableHolder = getCompositeType();
  MDTuple *TemplateParams = getTuple();
  StringRef Identifier = "some id";

  auto *N = DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                 BaseType, SizeInBits, AlignInBits,
                                 OffsetInBits, Flags, Elements, RuntimeLang,
                                 VTableHolder, TemplateParams, Identifier);
  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(BaseType, N->getBaseType());
  EXPECT_EQ(SizeInBits, N->getSizeInBits());
  EXPECT_EQ(AlignInBits, N->getAlignInBits());
  EXPECT_EQ(OffsetInBits, N->getOffsetInBits());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(Elements, N->getElements().get());
  EXPECT_EQ(RuntimeLang, N->getRuntimeLang());
  EXPECT_EQ(VTableHolder, N->getVTableHolder());
  EXPECT_EQ(TemplateParams, N->getTemplateParams().get());
  EXPECT_EQ(Identifier, N->getIdentifier());

  EXPECT_EQ(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));

  EXPECT_NE(N, DICompositeType::get(Context, Tag + 1, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, "abc", File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, getFile(), Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line + 1, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, getSubprogram(), BaseType,
                   SizeInBits, AlignInBits, OffsetInBits, Flags, Elements,
                   RuntimeLang, VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, getBasicType("other"),
                   SizeInBits, AlignInBits, OffsetInBits, Flags, Elements,
                   RuntimeLang, VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits + 1, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits + 1,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits + 1, Flags, Elements, RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  DINode::DIFlags FlagsPOne = static_cast<DINode::DIFlags>(Flags + 1);
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, FlagsPOne, Elements, RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, getTuple(), RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang + 1,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                   getCompositeType(), TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, getTuple(), Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, "other"));

  // Be sure that missing identifiers get null pointers.
  EXPECT_FALSE(DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, "")
                   ->getRawIdentifier());
  EXPECT_FALSE(DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams)
                   ->getRawIdentifier());

  TempDICompositeType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DICompositeTypeTest, getWithLargeValues) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DIScope *Scope = getSubprogram();
  DIType *BaseType = getCompositeType();
  uint64_t SizeInBits = UINT64_MAX;
  uint32_t AlignInBits = UINT32_MAX - 1;
  uint64_t OffsetInBits = UINT64_MAX - 2;
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(5);
  MDTuple *Elements = getTuple();
  unsigned RuntimeLang = 6;
  DIType *VTableHolder = getCompositeType();
  MDTuple *TemplateParams = getTuple();
  StringRef Identifier = "some id";

  auto *N = DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                 BaseType, SizeInBits, AlignInBits,
                                 OffsetInBits, Flags, Elements, RuntimeLang,
                                 VTableHolder, TemplateParams, Identifier);
  EXPECT_EQ(SizeInBits, N->getSizeInBits());
  EXPECT_EQ(AlignInBits, N->getAlignInBits());
  EXPECT_EQ(OffsetInBits, N->getOffsetInBits());
}

TEST_F(DICompositeTypeTest, replaceOperands) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DIScope *Scope = getSubprogram();
  DIType *BaseType = getCompositeType();
  uint64_t SizeInBits = 2;
  uint32_t AlignInBits = 3;
  uint64_t OffsetInBits = 4;
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(5);
  unsigned RuntimeLang = 6;
  StringRef Identifier = "some id";

  auto *N = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier);

  auto *Elements = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getElements().get());
  N->replaceElements(Elements);
  EXPECT_EQ(Elements, N->getElements().get());
  N->replaceElements(nullptr);
  EXPECT_EQ(nullptr, N->getElements().get());

  DIType *VTableHolder = getCompositeType();
  EXPECT_EQ(nullptr, N->getVTableHolder());
  N->replaceVTableHolder(VTableHolder);
  EXPECT_EQ(VTableHolder, N->getVTableHolder());
  // As an extension, the containing type can be anything.  This is
  // used by Rust to associate vtables with their concrete type.
  DIType *BasicType = getBasicType("basic");
  N->replaceVTableHolder(BasicType);
  EXPECT_EQ(BasicType, N->getVTableHolder());
  N->replaceVTableHolder(nullptr);
  EXPECT_EQ(nullptr, N->getVTableHolder());

  auto *TemplateParams = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getTemplateParams().get());
  N->replaceTemplateParams(TemplateParams);
  EXPECT_EQ(TemplateParams, N->getTemplateParams().get());
  N->replaceTemplateParams(nullptr);
  EXPECT_EQ(nullptr, N->getTemplateParams().get());
}

TEST_F(DICompositeTypeTest, variant_part) {
  unsigned Tag = dwarf::DW_TAG_variant_part;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DIScope *Scope = getSubprogram();
  DIType *BaseType = getCompositeType();
  uint64_t SizeInBits = 2;
  uint32_t AlignInBits = 3;
  uint64_t OffsetInBits = 4;
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(5);
  unsigned RuntimeLang = 6;
  StringRef Identifier = "some id";
  DIDerivedType *Discriminator = cast<DIDerivedType>(getDerivedType());
  DIDerivedType *Discriminator2 = cast<DIDerivedType>(getDerivedType());

  EXPECT_NE(Discriminator, Discriminator2);

  auto *N = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      Discriminator);

  // Test the hashing.
  auto *Same = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      Discriminator);
  auto *Other = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      Discriminator2);
  auto *NoDisc = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr);

  EXPECT_EQ(N, Same);
  EXPECT_NE(Same, Other);
  EXPECT_NE(Same, NoDisc);
  EXPECT_NE(Other, NoDisc);

  EXPECT_EQ(N->getDiscriminator(), Discriminator);
}

TEST_F(DICompositeTypeTest, dynamicArray) {
  unsigned Tag = dwarf::DW_TAG_array_type;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DILocalScope *Scope = getSubprogram();
  DIType *BaseType = getCompositeType();
  uint64_t SizeInBits = 32;
  uint32_t AlignInBits = 32;
  uint64_t OffsetInBits = 4;
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(3);
  unsigned RuntimeLang = 6;
  StringRef Identifier = "some id";
  DIType *Type = getDerivedType();
  Metadata *DlVar1 = DILocalVariable::get(Context, Scope, "dl_var1", File, 8,
                                       Type, 2, Flags, 8, nullptr);
  Metadata *DlVar2 = DILocalVariable::get(Context, Scope, "dl_var2", File, 8,
                                       Type, 2, Flags, 8, nullptr);
  uint64_t Elements1[] = {dwarf::DW_OP_push_object_address, dwarf::DW_OP_deref};
  Metadata *DataLocation1 = DIExpression::get(Context, Elements1);

  uint64_t Elements2[] = {dwarf::DW_OP_constu, 0};
  Metadata *DataLocation2 = DIExpression::get(Context, Elements2);

  uint64_t Elements3[] = {dwarf::DW_OP_constu, 3};
  Metadata *Rank1 = DIExpression::get(Context, Elements3);

  uint64_t Elements4[] = {dwarf::DW_OP_constu, 4};
  Metadata *Rank2 = DIExpression::get(Context, Elements4);

  ConstantInt *RankInt1 = ConstantInt::get(Context, APInt(7, 0));
  ConstantAsMetadata *RankConst1 = ConstantAsMetadata::get(RankInt1);
  ConstantInt *RankInt2 = ConstantInt::get(Context, APInt(6, 0));
  ConstantAsMetadata *RankConst2 = ConstantAsMetadata::get(RankInt2);
  auto *N1 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DlVar1);

  auto *Same1 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DlVar1);

  auto *Other1 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DlVar2);

  EXPECT_EQ(N1, Same1);
  EXPECT_NE(Same1, Other1);
  EXPECT_EQ(N1->getDataLocation(), DlVar1);

  auto *N2 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1);

  auto *Same2 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1);

  auto *Other2 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation2);

  EXPECT_EQ(N2, Same2);
  EXPECT_NE(Same2, Other2);
  EXPECT_EQ(N2->getDataLocationExp(), DataLocation1);

  auto *N3 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1, nullptr, nullptr, Rank1);

  auto *Same3 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1, nullptr, nullptr, Rank1);

  auto *Other3 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1, nullptr, nullptr, Rank2);

  EXPECT_EQ(N3, Same3);
  EXPECT_NE(Same3, Other3);
  EXPECT_EQ(N3->getRankExp(), Rank1);

  auto *N4 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1, nullptr, nullptr, RankConst1);

  auto *Same4 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1, nullptr, nullptr, RankConst1);

  auto *Other4 = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier,
      nullptr, DataLocation1, nullptr, nullptr, RankConst2);

  EXPECT_EQ(N4, Same4);
  EXPECT_NE(Same4, Other4);
  EXPECT_EQ(N4->getRankConst(), RankInt1);
}

typedef MetadataTest DISubroutineTypeTest;

TEST_F(DISubroutineTypeTest, get) {
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(1);
  DINode::DIFlags FlagsPOne = static_cast<DINode::DIFlags>(Flags + 1);
  MDTuple *TypeArray = getTuple();

  auto *N = DISubroutineType::get(Context, Flags, 0, TypeArray);
  EXPECT_EQ(dwarf::DW_TAG_subroutine_type, N->getTag());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(TypeArray, N->getTypeArray().get());
  EXPECT_EQ(N, DISubroutineType::get(Context, Flags, 0, TypeArray));

  EXPECT_NE(N, DISubroutineType::get(Context, FlagsPOne, 0, TypeArray));
  EXPECT_NE(N, DISubroutineType::get(Context, Flags, 0, getTuple()));

  // Test the hashing of calling conventions.
  auto *Fast = DISubroutineType::get(
      Context, Flags, dwarf::DW_CC_BORLAND_msfastcall, TypeArray);
  auto *Std = DISubroutineType::get(Context, Flags,
                                    dwarf::DW_CC_BORLAND_stdcall, TypeArray);
  EXPECT_EQ(Fast,
            DISubroutineType::get(Context, Flags,
                                  dwarf::DW_CC_BORLAND_msfastcall, TypeArray));
  EXPECT_EQ(Std, DISubroutineType::get(
                     Context, Flags, dwarf::DW_CC_BORLAND_stdcall, TypeArray));

  EXPECT_NE(N, Fast);
  EXPECT_NE(N, Std);
  EXPECT_NE(Fast, Std);

  TempDISubroutineType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));

  // Test always-empty operands.
  EXPECT_EQ(nullptr, N->getScope());
  EXPECT_EQ(nullptr, N->getFile());
  EXPECT_EQ("", N->getName());
}

typedef MetadataTest DIFileTest;

TEST_F(DIFileTest, get) {
  StringRef Filename = "file";
  StringRef Directory = "dir";
  DIFile::ChecksumKind CSKind = DIFile::ChecksumKind::CSK_MD5;
  StringRef ChecksumString = "000102030405060708090a0b0c0d0e0f";
  DIFile::ChecksumInfo<StringRef> Checksum(CSKind, ChecksumString);
  StringRef Source = "source";
  auto *N = DIFile::get(Context, Filename, Directory, Checksum, Source);

  EXPECT_EQ(dwarf::DW_TAG_file_type, N->getTag());
  EXPECT_EQ(Filename, N->getFilename());
  EXPECT_EQ(Directory, N->getDirectory());
  EXPECT_EQ(Checksum, N->getChecksum());
  EXPECT_EQ(Source, N->getSource());
  EXPECT_EQ(N, DIFile::get(Context, Filename, Directory, Checksum, Source));

  EXPECT_NE(N, DIFile::get(Context, "other", Directory, Checksum, Source));
  EXPECT_NE(N, DIFile::get(Context, Filename, "other", Checksum, Source));
  DIFile::ChecksumInfo<StringRef> OtherChecksum(DIFile::ChecksumKind::CSK_SHA1, ChecksumString);
  EXPECT_NE(
      N, DIFile::get(Context, Filename, Directory, OtherChecksum));
  StringRef OtherSource = "other";
  EXPECT_NE(N, DIFile::get(Context, Filename, Directory, Checksum, OtherSource));
  EXPECT_NE(N, DIFile::get(Context, Filename, Directory, Checksum));
  EXPECT_NE(N, DIFile::get(Context, Filename, Directory));

  TempDIFile Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIFileTest, ScopeGetFile) {
  // Ensure that DIScope::getFile() returns itself.
  DIScope *N = DIFile::get(Context, "file", "dir");
  EXPECT_EQ(N, N->getFile());
}

typedef MetadataTest DICompileUnitTest;

TEST_F(DICompileUnitTest, get) {
  unsigned SourceLanguage = 1;
  DIFile *File = getFile();
  StringRef Producer = "some producer";
  bool IsOptimized = false;
  StringRef Flags = "flag after flag";
  unsigned RuntimeVersion = 2;
  StringRef SplitDebugFilename = "another/file";
  auto EmissionKind = DICompileUnit::FullDebug;
  MDTuple *EnumTypes = getTuple();
  MDTuple *RetainedTypes = getTuple();
  MDTuple *GlobalVariables = getTuple();
  MDTuple *ImportedEntities = getTuple();
  uint64_t DWOId = 0x10000000c0ffee;
  MDTuple *Macros = getTuple();
  StringRef SysRoot = "/";
  StringRef SDK = "MacOSX.sdk";
  auto *N = DICompileUnit::getDistinct(
      Context, SourceLanguage, File, Producer, IsOptimized, Flags,
      RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
      RetainedTypes, GlobalVariables, ImportedEntities, Macros, DWOId, true,
      false, DICompileUnit::DebugNameTableKind::Default, false, SysRoot, SDK);

  EXPECT_EQ(dwarf::DW_TAG_compile_unit, N->getTag());
  EXPECT_EQ(SourceLanguage, N->getSourceLanguage());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Producer, N->getProducer());
  EXPECT_EQ(IsOptimized, N->isOptimized());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(RuntimeVersion, N->getRuntimeVersion());
  EXPECT_EQ(SplitDebugFilename, N->getSplitDebugFilename());
  EXPECT_EQ(EmissionKind, N->getEmissionKind());
  EXPECT_EQ(EnumTypes, N->getEnumTypes().get());
  EXPECT_EQ(RetainedTypes, N->getRetainedTypes().get());
  EXPECT_EQ(GlobalVariables, N->getGlobalVariables().get());
  EXPECT_EQ(ImportedEntities, N->getImportedEntities().get());
  EXPECT_EQ(Macros, N->getMacros().get());
  EXPECT_EQ(DWOId, N->getDWOId());
  EXPECT_EQ(SysRoot, N->getSysRoot());
  EXPECT_EQ(SDK, N->getSDK());

  TempDICompileUnit Temp = N->clone();
  EXPECT_EQ(dwarf::DW_TAG_compile_unit, Temp->getTag());
  EXPECT_EQ(SourceLanguage, Temp->getSourceLanguage());
  EXPECT_EQ(File, Temp->getFile());
  EXPECT_EQ(Producer, Temp->getProducer());
  EXPECT_EQ(IsOptimized, Temp->isOptimized());
  EXPECT_EQ(Flags, Temp->getFlags());
  EXPECT_EQ(RuntimeVersion, Temp->getRuntimeVersion());
  EXPECT_EQ(SplitDebugFilename, Temp->getSplitDebugFilename());
  EXPECT_EQ(EmissionKind, Temp->getEmissionKind());
  EXPECT_EQ(EnumTypes, Temp->getEnumTypes().get());
  EXPECT_EQ(RetainedTypes, Temp->getRetainedTypes().get());
  EXPECT_EQ(GlobalVariables, Temp->getGlobalVariables().get());
  EXPECT_EQ(ImportedEntities, Temp->getImportedEntities().get());
  EXPECT_EQ(Macros, Temp->getMacros().get());
  EXPECT_EQ(SysRoot, Temp->getSysRoot());
  EXPECT_EQ(SDK, Temp->getSDK());

  auto *TempAddress = Temp.get();
  auto *Clone = MDNode::replaceWithPermanent(std::move(Temp));
  EXPECT_TRUE(Clone->isDistinct());
  EXPECT_EQ(TempAddress, Clone);
}

TEST_F(DICompileUnitTest, replaceArrays) {
  unsigned SourceLanguage = 1;
  DIFile *File = getFile();
  StringRef Producer = "some producer";
  bool IsOptimized = false;
  StringRef Flags = "flag after flag";
  unsigned RuntimeVersion = 2;
  StringRef SplitDebugFilename = "another/file";
  auto EmissionKind = DICompileUnit::FullDebug;
  MDTuple *EnumTypes = MDTuple::getDistinct(Context, None);
  MDTuple *RetainedTypes = MDTuple::getDistinct(Context, None);
  MDTuple *ImportedEntities = MDTuple::getDistinct(Context, None);
  uint64_t DWOId = 0xc0ffee;
  StringRef SysRoot = "/";
  StringRef SDK = "MacOSX.sdk";
  auto *N = DICompileUnit::getDistinct(
      Context, SourceLanguage, File, Producer, IsOptimized, Flags,
      RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
      RetainedTypes, nullptr, ImportedEntities, nullptr, DWOId, true, false,
      DICompileUnit::DebugNameTableKind::Default, false, SysRoot, SDK);

  auto *GlobalVariables = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getGlobalVariables().get());
  N->replaceGlobalVariables(GlobalVariables);
  EXPECT_EQ(GlobalVariables, N->getGlobalVariables().get());
  N->replaceGlobalVariables(nullptr);
  EXPECT_EQ(nullptr, N->getGlobalVariables().get());

  auto *Macros = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getMacros().get());
  N->replaceMacros(Macros);
  EXPECT_EQ(Macros, N->getMacros().get());
  N->replaceMacros(nullptr);
  EXPECT_EQ(nullptr, N->getMacros().get());
}

typedef MetadataTest DISubprogramTest;

TEST_F(DISubprogramTest, get) {
  DIScope *Scope = getCompositeType();
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  DIFile *File = getFile();
  unsigned Line = 2;
  DISubroutineType *Type = getSubroutineType();
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  unsigned ScopeLine = 3;
  DIType *ContainingType = getCompositeType();
  unsigned Virtuality = 2;
  unsigned VirtualIndex = 5;
  int ThisAdjustment = -3;
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(6);
  bool IsOptimized = false;
  MDTuple *TemplateParams = getTuple();
  DISubprogram *Declaration = getSubprogram();
  MDTuple *RetainedNodes = getTuple();
  MDTuple *ThrownTypes = getTuple();
  DICompileUnit *Unit = getUnit();
  DISubprogram::DISPFlags SPFlags =
      static_cast<DISubprogram::DISPFlags>(Virtuality);
  assert(!IsLocalToUnit && IsDefinition && !IsOptimized &&
         "bools and SPFlags have to match");
  SPFlags |= DISubprogram::SPFlagDefinition;

  auto *N = DISubprogram::get(
      Context, Scope, Name, LinkageName, File, Line, Type, ScopeLine,
      ContainingType, VirtualIndex, ThisAdjustment, Flags, SPFlags, Unit,
      TemplateParams, Declaration, RetainedNodes, ThrownTypes);

  EXPECT_EQ(dwarf::DW_TAG_subprogram, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(LinkageName, N->getLinkageName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(IsLocalToUnit, N->isLocalToUnit());
  EXPECT_EQ(IsDefinition, N->isDefinition());
  EXPECT_EQ(ScopeLine, N->getScopeLine());
  EXPECT_EQ(ContainingType, N->getContainingType());
  EXPECT_EQ(Virtuality, N->getVirtuality());
  EXPECT_EQ(VirtualIndex, N->getVirtualIndex());
  EXPECT_EQ(ThisAdjustment, N->getThisAdjustment());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(IsOptimized, N->isOptimized());
  EXPECT_EQ(Unit, N->getUnit());
  EXPECT_EQ(TemplateParams, N->getTemplateParams().get());
  EXPECT_EQ(Declaration, N->getDeclaration());
  EXPECT_EQ(RetainedNodes, N->getRetainedNodes().get());
  EXPECT_EQ(ThrownTypes, N->getThrownTypes().get());
  EXPECT_EQ(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine, ContainingType, VirtualIndex,
                                 ThisAdjustment, Flags, SPFlags, Unit,
                                 TemplateParams, Declaration, RetainedNodes,
                                 ThrownTypes));

  EXPECT_NE(N, DISubprogram::get(Context, getCompositeType(), Name, LinkageName,
                                 File, Line, Type, ScopeLine, ContainingType,
                                 VirtualIndex, ThisAdjustment, Flags, SPFlags,
                                 Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, "other", LinkageName, File,
                                 Line, Type, ScopeLine, ContainingType,
                                 VirtualIndex, ThisAdjustment, Flags, SPFlags,
                                 Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, "other", File, Line,
                                 Type, ScopeLine, ContainingType, VirtualIndex,
                                 ThisAdjustment, Flags, SPFlags, Unit,
                                 TemplateParams, Declaration, RetainedNodes,
                                 ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, getFile(),
                                 Line, Type, ScopeLine, ContainingType,
                                 VirtualIndex, ThisAdjustment, Flags, SPFlags,
                                 Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File,
                                 Line + 1, Type, ScopeLine, ContainingType,
                                 VirtualIndex, ThisAdjustment, Flags, SPFlags,
                                 Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 getSubroutineType(), ScopeLine, ContainingType,
                                 VirtualIndex, ThisAdjustment, Flags, SPFlags,
                                 Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(
                   Context, Scope, Name, LinkageName, File, Line, Type,
                   ScopeLine, ContainingType, VirtualIndex, ThisAdjustment,
                   Flags, SPFlags ^ DISubprogram::SPFlagLocalToUnit, Unit,
                   TemplateParams, Declaration, RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(
                   Context, Scope, Name, LinkageName, File, Line, Type,
                   ScopeLine, ContainingType, VirtualIndex, ThisAdjustment,
                   Flags, SPFlags ^ DISubprogram::SPFlagDefinition, Unit,
                   TemplateParams, Declaration, RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine + 1, ContainingType,
                                 VirtualIndex, ThisAdjustment, Flags, SPFlags,
                                 Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine, getCompositeType(),
                                 VirtualIndex, ThisAdjustment, Flags, SPFlags,
                                 Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(
                   Context, Scope, Name, LinkageName, File, Line, Type,
                   ScopeLine, ContainingType, VirtualIndex, ThisAdjustment,
                   Flags, SPFlags ^ DISubprogram::SPFlagVirtual, Unit,
                   TemplateParams, Declaration, RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine, ContainingType,
                                 VirtualIndex + 1, ThisAdjustment, Flags,
                                 SPFlags, Unit, TemplateParams, Declaration,
                                 RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(
                   Context, Scope, Name, LinkageName, File, Line, Type,
                   ScopeLine, ContainingType, VirtualIndex, ThisAdjustment,
                   Flags, SPFlags ^ DISubprogram::SPFlagOptimized, Unit,
                   TemplateParams, Declaration, RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine, ContainingType, VirtualIndex,
                                 ThisAdjustment, Flags, SPFlags, nullptr,
                                 TemplateParams, Declaration, RetainedNodes,
                                 ThrownTypes));
  EXPECT_NE(N,
            DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                              Type, ScopeLine, ContainingType, VirtualIndex,
                              ThisAdjustment, Flags, SPFlags, Unit, getTuple(),
                              Declaration, RetainedNodes, ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine, ContainingType, VirtualIndex,
                                 ThisAdjustment, Flags, SPFlags, Unit,
                                 TemplateParams, getSubprogram(), RetainedNodes,
                                 ThrownTypes));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine, ContainingType, VirtualIndex,
                                 ThisAdjustment, Flags, SPFlags, Unit,
                                 TemplateParams, Declaration, getTuple()));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, ScopeLine, ContainingType, VirtualIndex,
                                 ThisAdjustment, Flags, SPFlags, Unit,
                                 TemplateParams, Declaration, RetainedNodes,
                                 getTuple()));

  TempDISubprogram Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DILexicalBlockTest;

TEST_F(DILexicalBlockTest, get) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  unsigned Line = 5;
  unsigned Column = 8;

  auto *N = DILexicalBlock::get(Context, Scope, File, Line, Column);

  EXPECT_EQ(dwarf::DW_TAG_lexical_block, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Column, N->getColumn());
  EXPECT_EQ(N, DILexicalBlock::get(Context, Scope, File, Line, Column));

  EXPECT_NE(N,
            DILexicalBlock::get(Context, getSubprogram(), File, Line, Column));
  EXPECT_NE(N, DILexicalBlock::get(Context, Scope, getFile(), Line, Column));
  EXPECT_NE(N, DILexicalBlock::get(Context, Scope, File, Line + 1, Column));
  EXPECT_NE(N, DILexicalBlock::get(Context, Scope, File, Line, Column + 1));

  TempDILexicalBlock Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DILexicalBlockTest, Overflow) {
  DISubprogram *SP = getSubprogram();
  DIFile *F = getFile();
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, 2, 7);
    EXPECT_EQ(2u, LB->getLine());
    EXPECT_EQ(7u, LB->getColumn());
  }
  unsigned U16 = 1u << 16;
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, UINT32_MAX, U16 - 1);
    EXPECT_EQ(UINT32_MAX, LB->getLine());
    EXPECT_EQ(U16 - 1, LB->getColumn());
  }
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, UINT32_MAX, U16);
    EXPECT_EQ(UINT32_MAX, LB->getLine());
    EXPECT_EQ(0u, LB->getColumn());
  }
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, UINT32_MAX, U16 + 1);
    EXPECT_EQ(UINT32_MAX, LB->getLine());
    EXPECT_EQ(0u, LB->getColumn());
  }
}

typedef MetadataTest DILexicalBlockFileTest;

TEST_F(DILexicalBlockFileTest, get) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  unsigned Discriminator = 5;

  auto *N = DILexicalBlockFile::get(Context, Scope, File, Discriminator);

  EXPECT_EQ(dwarf::DW_TAG_lexical_block, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Discriminator, N->getDiscriminator());
  EXPECT_EQ(N, DILexicalBlockFile::get(Context, Scope, File, Discriminator));

  EXPECT_NE(N, DILexicalBlockFile::get(Context, getSubprogram(), File,
                                       Discriminator));
  EXPECT_NE(N,
            DILexicalBlockFile::get(Context, Scope, getFile(), Discriminator));
  EXPECT_NE(N,
            DILexicalBlockFile::get(Context, Scope, File, Discriminator + 1));

  TempDILexicalBlockFile Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DINamespaceTest;

TEST_F(DINamespaceTest, get) {
  DIScope *Scope = getFile();
  StringRef Name = "namespace";
  bool ExportSymbols = true;

  auto *N = DINamespace::get(Context, Scope, Name, ExportSymbols);

  EXPECT_EQ(dwarf::DW_TAG_namespace, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(N, DINamespace::get(Context, Scope, Name, ExportSymbols));
  EXPECT_NE(N, DINamespace::get(Context, getFile(), Name, ExportSymbols));
  EXPECT_NE(N, DINamespace::get(Context, Scope, "other", ExportSymbols));
  EXPECT_NE(N, DINamespace::get(Context, Scope, Name, !ExportSymbols));

  TempDINamespace Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIModuleTest;

TEST_F(DIModuleTest, get) {
  DIFile *File = getFile();
  DIScope *Scope = getFile();
  StringRef Name = "module";
  StringRef ConfigMacro = "-DNDEBUG";
  StringRef Includes = "-I.";
  StringRef APINotes = "/tmp/m.apinotes";
  unsigned LineNo = 4;
  bool IsDecl = true;

  auto *N = DIModule::get(Context, File, Scope, Name, ConfigMacro, Includes,
                          APINotes, LineNo, IsDecl);

  EXPECT_EQ(dwarf::DW_TAG_module, N->getTag());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(ConfigMacro, N->getConfigurationMacros());
  EXPECT_EQ(Includes, N->getIncludePath());
  EXPECT_EQ(APINotes, N->getAPINotesFile());
  EXPECT_EQ(LineNo, N->getLineNo());
  EXPECT_EQ(IsDecl, N->getIsDecl());
  EXPECT_EQ(N, DIModule::get(Context, File, Scope, Name, ConfigMacro, Includes,
                             APINotes, LineNo, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, getFile(), getFile(), Name, ConfigMacro,
                             Includes, APINotes, LineNo, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, File, Scope, "other", ConfigMacro,
                             Includes, APINotes, LineNo, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, File, Scope, Name, "other", Includes,
                             APINotes, LineNo, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, File, Scope, Name, ConfigMacro, "other",
                             APINotes, LineNo, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, File, Scope, Name, ConfigMacro, Includes,
                             "other", LineNo, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, getFile(), Scope, Name, ConfigMacro,
                             Includes, APINotes, LineNo, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, File, Scope, Name, ConfigMacro, Includes,
                             APINotes, 5, IsDecl));
  EXPECT_NE(N, DIModule::get(Context, File, Scope, Name, ConfigMacro, Includes,
                             APINotes, LineNo, false));

  TempDIModule Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DITemplateTypeParameterTest;

TEST_F(DITemplateTypeParameterTest, get) {
  StringRef Name = "template";
  DIType *Type = getBasicType("basic");
  bool defaulted = false;

  auto *N = DITemplateTypeParameter::get(Context, Name, Type, defaulted);

  EXPECT_EQ(dwarf::DW_TAG_template_type_parameter, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(N, DITemplateTypeParameter::get(Context, Name, Type, defaulted));

  EXPECT_NE(N, DITemplateTypeParameter::get(Context, "other", Type, defaulted));
  EXPECT_NE(N, DITemplateTypeParameter::get(Context, Name,
                                            getBasicType("other"), defaulted));
  EXPECT_NE(N, DITemplateTypeParameter::get(Context, Name, Type, true));

  TempDITemplateTypeParameter Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DITemplateValueParameterTest;

TEST_F(DITemplateValueParameterTest, get) {
  unsigned Tag = dwarf::DW_TAG_template_value_parameter;
  StringRef Name = "template";
  DIType *Type = getBasicType("basic");
  bool defaulted = false;
  Metadata *Value = getConstantAsMetadata();

  auto *N =
      DITemplateValueParameter::get(Context, Tag, Name, Type, defaulted, Value);
  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(Value, N->getValue());
  EXPECT_EQ(N, DITemplateValueParameter::get(Context, Tag, Name, Type,
                                             defaulted, Value));

  EXPECT_NE(N, DITemplateValueParameter::get(
                   Context, dwarf::DW_TAG_GNU_template_template_param, Name,
                   Type, defaulted, Value));
  EXPECT_NE(N, DITemplateValueParameter::get(Context, Tag, "other", Type,
                                             defaulted, Value));
  EXPECT_NE(N, DITemplateValueParameter::get(Context, Tag, Name,
                                             getBasicType("other"), defaulted,
                                             Value));
  EXPECT_NE(N,
            DITemplateValueParameter::get(Context, Tag, Name, Type, defaulted,
                                          getConstantAsMetadata()));
  EXPECT_NE(
      N, DITemplateValueParameter::get(Context, Tag, Name, Type, true, Value));

  TempDITemplateValueParameter Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIGlobalVariableTest;

TEST_F(DIGlobalVariableTest, get) {
  DIScope *Scope = getSubprogram();
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  DIFile *File = getFile();
  unsigned Line = 5;
  DIType *Type = getDerivedType();
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  MDTuple *templateParams = getTuple();
  DIDerivedType *StaticDataMemberDeclaration =
      cast<DIDerivedType>(getDerivedType());

  uint32_t AlignInBits = 8;

  auto *N = DIGlobalVariable::get(
      Context, Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
      IsDefinition, StaticDataMemberDeclaration, templateParams, AlignInBits,
      nullptr);

  EXPECT_EQ(dwarf::DW_TAG_variable, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(LinkageName, N->getLinkageName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(IsLocalToUnit, N->isLocalToUnit());
  EXPECT_EQ(IsDefinition, N->isDefinition());
  EXPECT_EQ(StaticDataMemberDeclaration, N->getStaticDataMemberDeclaration());
  EXPECT_EQ(templateParams, N->getTemplateParams());
  EXPECT_EQ(AlignInBits, N->getAlignInBits());
  EXPECT_EQ(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));

  EXPECT_NE(N, DIGlobalVariable::get(
                   Context, getSubprogram(), Name, LinkageName, File, Line,
                   Type, IsLocalToUnit, IsDefinition,
                   StaticDataMemberDeclaration, templateParams, AlignInBits,
                   nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, "other", LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, "other", File, Line,
                                     Type, IsLocalToUnit, IsDefinition,
                                     StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName,
                                     getFile(), Line, Type, IsLocalToUnit,
                                     IsDefinition, StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line + 1, Type, IsLocalToUnit,
                                     IsDefinition, StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, getDerivedType(), IsLocalToUnit,
                                     IsDefinition, StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, !IsLocalToUnit, IsDefinition,
                                     StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, !IsDefinition,
                                     StaticDataMemberDeclaration,
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     cast<DIDerivedType>(getDerivedType()),
                                     templateParams, AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     StaticDataMemberDeclaration, nullptr,
                                     AlignInBits, nullptr));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     StaticDataMemberDeclaration,
                                     templateParams, (AlignInBits << 1),
                                     nullptr));

  TempDIGlobalVariable Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIGlobalVariableExpressionTest;

TEST_F(DIGlobalVariableExpressionTest, get) {
  DIScope *Scope = getSubprogram();
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  DIFile *File = getFile();
  unsigned Line = 5;
  DIType *Type = getDerivedType();
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  MDTuple *templateParams = getTuple();
  auto *Expr = DIExpression::get(Context, {1, 2});
  auto *Expr2 = DIExpression::get(Context, {1, 2, 3});
  DIDerivedType *StaticDataMemberDeclaration =
      cast<DIDerivedType>(getDerivedType());
  uint32_t AlignInBits = 8;

  auto *Var = DIGlobalVariable::get(
      Context, Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
      IsDefinition, StaticDataMemberDeclaration, templateParams, AlignInBits,
      nullptr);
  auto *Var2 = DIGlobalVariable::get(
      Context, Scope, "other", LinkageName, File, Line, Type, IsLocalToUnit,
      IsDefinition, StaticDataMemberDeclaration, templateParams, AlignInBits,
      nullptr);
  auto *N = DIGlobalVariableExpression::get(Context, Var, Expr);

  EXPECT_EQ(Var, N->getVariable());
  EXPECT_EQ(Expr, N->getExpression());
  EXPECT_EQ(N, DIGlobalVariableExpression::get(Context, Var, Expr));
  EXPECT_NE(N, DIGlobalVariableExpression::get(Context, Var2, Expr));
  EXPECT_NE(N, DIGlobalVariableExpression::get(Context, Var, Expr2));

  TempDIGlobalVariableExpression Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DILocalVariableTest;

TEST_F(DILocalVariableTest, get) {
  DILocalScope *Scope = getSubprogram();
  StringRef Name = "name";
  DIFile *File = getFile();
  unsigned Line = 5;
  DIType *Type = getDerivedType();
  unsigned Arg = 6;
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);
  uint32_t AlignInBits = 8;

  auto *N =
      DILocalVariable::get(Context, Scope, Name, File, Line, Type, Arg, Flags,
                           AlignInBits, nullptr);
  EXPECT_TRUE(N->isParameter());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(Arg, N->getArg());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(AlignInBits, N->getAlignInBits());
  EXPECT_EQ(N, DILocalVariable::get(Context, Scope, Name, File, Line, Type, Arg,
                                    Flags, AlignInBits, nullptr));

  EXPECT_FALSE(
      DILocalVariable::get(Context, Scope, Name, File, Line, Type, 0, Flags,
                           AlignInBits, nullptr)->isParameter());
  EXPECT_NE(N, DILocalVariable::get(Context, getSubprogram(), Name, File, Line,
                                    Type, Arg, Flags, AlignInBits, nullptr));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, "other", File, Line, Type,
                                    Arg, Flags, AlignInBits, nullptr));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, getFile(), Line, Type,
                                    Arg, Flags, AlignInBits, nullptr));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line + 1, Type,
                                    Arg, Flags, AlignInBits, nullptr));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line,
                                    getDerivedType(), Arg, Flags, AlignInBits,
                                    nullptr));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line, Type,
                                    Arg + 1, Flags, AlignInBits, nullptr));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line, Type,
                                    Arg, Flags, (AlignInBits << 1), nullptr));

  TempDILocalVariable Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DILocalVariableTest, getArg256) {
  EXPECT_EQ(255u, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                       0, nullptr, 255, DINode::FlagZero, 0,
                                       nullptr)
                      ->getArg());
  EXPECT_EQ(256u, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                       0, nullptr, 256, DINode::FlagZero, 0,
                                       nullptr)
                      ->getArg());
  EXPECT_EQ(257u, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                       0, nullptr, 257, DINode::FlagZero, 0,
                                       nullptr)
                      ->getArg());
  unsigned Max = UINT16_MAX;
  EXPECT_EQ(Max, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                      0, nullptr, Max, DINode::FlagZero, 0,
                                      nullptr)
                     ->getArg());
}

typedef MetadataTest DIExpressionTest;

TEST_F(DIExpressionTest, get) {
  uint64_t Elements[] = {2, 6, 9, 78, 0};
  auto *N = DIExpression::get(Context, Elements);
  EXPECT_EQ(makeArrayRef(Elements), N->getElements());
  EXPECT_EQ(N, DIExpression::get(Context, Elements));

  EXPECT_EQ(5u, N->getNumElements());
  EXPECT_EQ(2u, N->getElement(0));
  EXPECT_EQ(6u, N->getElement(1));
  EXPECT_EQ(9u, N->getElement(2));
  EXPECT_EQ(78u, N->getElement(3));
  EXPECT_EQ(0u, N->getElement(4));

  TempDIExpression Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));

  // Test DIExpression::prepend().
  uint64_t Elts0[] = {dwarf::DW_OP_LLVM_fragment, 0, 32};
  auto *N0 = DIExpression::get(Context, Elts0);
  uint8_t DIExprFlags = DIExpression::ApplyOffset;
  DIExprFlags |= DIExpression::DerefBefore;
  DIExprFlags |= DIExpression::DerefAfter;
  DIExprFlags |= DIExpression::StackValue;
  auto *N0WithPrependedOps = DIExpression::prepend(N0, DIExprFlags, 64);
  uint64_t Elts1[] = {dwarf::DW_OP_deref,
                      dwarf::DW_OP_plus_uconst, 64,
                      dwarf::DW_OP_deref,
                      dwarf::DW_OP_stack_value,
                      dwarf::DW_OP_LLVM_fragment, 0, 32};
  auto *N1 = DIExpression::get(Context, Elts1);
  EXPECT_EQ(N0WithPrependedOps, N1);

  // Test DIExpression::append().
  uint64_t Elts2[] = {dwarf::DW_OP_deref, dwarf::DW_OP_plus_uconst, 64,
                      dwarf::DW_OP_deref, dwarf::DW_OP_stack_value};
  auto *N2 = DIExpression::append(N0, Elts2);
  EXPECT_EQ(N0WithPrependedOps, N2);
}

TEST_F(DIExpressionTest, isValid) {
#define EXPECT_VALID(...)                                                      \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    EXPECT_TRUE(DIExpression::get(Context, Elements)->isValid());              \
  } while (false)
#define EXPECT_INVALID(...)                                                    \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    EXPECT_FALSE(DIExpression::get(Context, Elements)->isValid());             \
  } while (false)

  // Empty expression should be valid.
  EXPECT_TRUE(DIExpression::get(Context, None));

  // Valid constructions.
  EXPECT_VALID(dwarf::DW_OP_plus_uconst, 6);
  EXPECT_VALID(dwarf::DW_OP_constu, 6, dwarf::DW_OP_plus);
  EXPECT_VALID(dwarf::DW_OP_deref);
  EXPECT_VALID(dwarf::DW_OP_LLVM_fragment, 3, 7);
  EXPECT_VALID(dwarf::DW_OP_plus_uconst, 6, dwarf::DW_OP_deref);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_plus_uconst, 6);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_LLVM_fragment, 3, 7);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_plus_uconst, 6,
               dwarf::DW_OP_LLVM_fragment, 3, 7);

  // Invalid constructions.
  EXPECT_INVALID(~0u);
  EXPECT_INVALID(dwarf::DW_OP_plus, 0);
  EXPECT_INVALID(dwarf::DW_OP_plus_uconst);
  EXPECT_INVALID(dwarf::DW_OP_LLVM_fragment);
  EXPECT_INVALID(dwarf::DW_OP_LLVM_fragment, 3);
  EXPECT_INVALID(dwarf::DW_OP_LLVM_fragment, 3, 7, dwarf::DW_OP_plus_uconst, 3);
  EXPECT_INVALID(dwarf::DW_OP_LLVM_fragment, 3, 7, dwarf::DW_OP_deref);

#undef EXPECT_VALID
#undef EXPECT_INVALID
}

TEST_F(DIExpressionTest, createFragmentExpression) {
#define EXPECT_VALID_FRAGMENT(Offset, Size, ...)                               \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    DIExpression* Expression = DIExpression::get(Context, Elements);           \
    EXPECT_TRUE(DIExpression::createFragmentExpression(                        \
      Expression, Offset, Size).hasValue());                                   \
  } while (false)
#define EXPECT_INVALID_FRAGMENT(Offset, Size, ...)                             \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    DIExpression* Expression = DIExpression::get(Context, Elements);           \
    EXPECT_FALSE(DIExpression::createFragmentExpression(                       \
      Expression, Offset, Size).hasValue());                                   \
  } while (false)

  // createFragmentExpression adds correct ops.
  Optional<DIExpression*> R = DIExpression::createFragmentExpression(
    DIExpression::get(Context, {}), 0, 32);
  EXPECT_EQ(R.hasValue(), true);
  EXPECT_EQ(3u, (*R)->getNumElements());
  EXPECT_EQ(dwarf::DW_OP_LLVM_fragment, (*R)->getElement(0));
  EXPECT_EQ(0u, (*R)->getElement(1));
  EXPECT_EQ(32u, (*R)->getElement(2));

  // Valid fragment expressions.
  EXPECT_VALID_FRAGMENT(0, 32, {});
  EXPECT_VALID_FRAGMENT(0, 32, dwarf::DW_OP_deref);
  EXPECT_VALID_FRAGMENT(0, 32, dwarf::DW_OP_LLVM_fragment, 0, 32);
  EXPECT_VALID_FRAGMENT(16, 16, dwarf::DW_OP_LLVM_fragment, 0, 32);

  // Invalid fragment expressions (incompatible ops).
  EXPECT_INVALID_FRAGMENT(0, 32, dwarf::DW_OP_constu, 6, dwarf::DW_OP_plus);
  EXPECT_INVALID_FRAGMENT(0, 32, dwarf::DW_OP_constu, 14, dwarf::DW_OP_minus);
  EXPECT_INVALID_FRAGMENT(0, 32, dwarf::DW_OP_constu, 16, dwarf::DW_OP_shr);
  EXPECT_INVALID_FRAGMENT(0, 32, dwarf::DW_OP_constu, 16, dwarf::DW_OP_shl);
  EXPECT_INVALID_FRAGMENT(0, 32, dwarf::DW_OP_constu, 16, dwarf::DW_OP_shra);
  EXPECT_INVALID_FRAGMENT(0, 32, dwarf::DW_OP_plus_uconst, 6);

#undef EXPECT_VALID_FRAGMENT
#undef EXPECT_INVALID_FRAGMENT
}

TEST_F(DIExpressionTest, replaceArg) {
#define EXPECT_REPLACE_ARG_EQ(Expr, OldArg, NewArg, ...)                       \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    ArrayRef<uint64_t> Expected = Elements;                                    \
    DIExpression *Expression = DIExpression::replaceArg(Expr, OldArg, NewArg); \
    EXPECT_EQ(Expression->getElements(), Expected);                            \
  } while (false)

  auto N = DIExpression::get(
      Context, {dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_LLVM_arg, 1,
                dwarf::DW_OP_plus, dwarf::DW_OP_LLVM_arg, 2, dwarf::DW_OP_mul});
  EXPECT_REPLACE_ARG_EQ(N, 0, 1, dwarf::DW_OP_LLVM_arg, 0,
                        dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_plus,
                        dwarf::DW_OP_LLVM_arg, 1, dwarf::DW_OP_mul);
  EXPECT_REPLACE_ARG_EQ(N, 0, 2, dwarf::DW_OP_LLVM_arg, 1,
                        dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_plus,
                        dwarf::DW_OP_LLVM_arg, 1, dwarf::DW_OP_mul);
  EXPECT_REPLACE_ARG_EQ(N, 2, 0, dwarf::DW_OP_LLVM_arg, 0,
                        dwarf::DW_OP_LLVM_arg, 1, dwarf::DW_OP_plus,
                        dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_mul);
  EXPECT_REPLACE_ARG_EQ(N, 2, 1, dwarf::DW_OP_LLVM_arg, 0,
                        dwarf::DW_OP_LLVM_arg, 1, dwarf::DW_OP_plus,
                        dwarf::DW_OP_LLVM_arg, 1, dwarf::DW_OP_mul);

#undef EXPECT_REPLACE_ARG_EQ
}

TEST_F(DIExpressionTest, foldConstant) {
  const ConstantInt *Int;
  const ConstantInt *NewInt;
  DIExpression *Expr;
  DIExpression *NewExpr;

#define EXPECT_FOLD_CONST(StartWidth, StartValue, EndWidth, EndValue, NumElts)  \
  Int = ConstantInt::get(Context, APInt(StartWidth, StartValue));               \
  std::tie(NewExpr, NewInt) = Expr->constantFold(Int);                          \
  ASSERT_EQ(NewInt->getBitWidth(), EndWidth##u);                                \
  EXPECT_EQ(NewInt->getValue(), APInt(EndWidth, EndValue));                     \
  EXPECT_EQ(NewExpr->getNumElements(), NumElts##u)

  // Unfoldable expression should return the original unmodified Int/Expr.
  Expr = DIExpression::get(Context, {dwarf::DW_OP_deref});
  EXPECT_FOLD_CONST(32, 117, 32, 117, 1);
  EXPECT_EQ(NewExpr, Expr);
  EXPECT_EQ(NewInt, Int);
  EXPECT_TRUE(NewExpr->startsWithDeref());

  // One unsigned bit-width conversion.
  Expr = DIExpression::get(
      Context, {dwarf::DW_OP_LLVM_convert, 72, dwarf::DW_ATE_unsigned});
  EXPECT_FOLD_CONST(8, 12, 72, 12, 0);

  // Two unsigned bit-width conversions (mask truncation).
  Expr = DIExpression::get(
      Context, {dwarf::DW_OP_LLVM_convert, 8, dwarf::DW_ATE_unsigned,
                dwarf::DW_OP_LLVM_convert, 16, dwarf::DW_ATE_unsigned});
  EXPECT_FOLD_CONST(32, -1, 16, 0xff, 0);

  // Sign extension.
  Expr = DIExpression::get(
      Context, {dwarf::DW_OP_LLVM_convert, 32, dwarf::DW_ATE_signed});
  EXPECT_FOLD_CONST(16, -1, 32, -1, 0);

  // Get non-foldable operations back in the new Expr.
  uint64_t Elements[] = {dwarf::DW_OP_deref, dwarf::DW_OP_stack_value};
  ArrayRef<uint64_t> Expected = Elements;
  Expr = DIExpression::get(
      Context, {dwarf::DW_OP_LLVM_convert, 32, dwarf::DW_ATE_signed});
  Expr = DIExpression::append(Expr, Expected);
  ASSERT_EQ(Expr->getNumElements(), 5u);
  EXPECT_FOLD_CONST(16, -1, 32, -1, 2);
  EXPECT_EQ(NewExpr->getElements(), Expected);

#undef EXPECT_FOLD_CONST
}

typedef MetadataTest DIObjCPropertyTest;

TEST_F(DIObjCPropertyTest, get) {
  StringRef Name = "name";
  DIFile *File = getFile();
  unsigned Line = 5;
  StringRef GetterName = "getter";
  StringRef SetterName = "setter";
  unsigned Attributes = 7;
  DIType *Type = getBasicType("basic");

  auto *N = DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                SetterName, Attributes, Type);

  EXPECT_EQ(dwarf::DW_TAG_APPLE_property, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(GetterName, N->getGetterName());
  EXPECT_EQ(SetterName, N->getSetterName());
  EXPECT_EQ(Attributes, N->getAttributes());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes, Type));

  EXPECT_NE(N, DIObjCProperty::get(Context, "other", File, Line, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, getFile(), Line, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line + 1, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, "other",
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   "other", Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes + 1, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes,
                                   getBasicType("other")));

  TempDIObjCProperty Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIImportedEntityTest;

TEST_F(DIImportedEntityTest, get) {
  unsigned Tag = dwarf::DW_TAG_imported_module;
  DIScope *Scope = getSubprogram();
  DINode *Entity = getCompositeType();
  DIFile *File = getFile();
  unsigned Line = 5;
  StringRef Name = "name";

  auto *N =
      DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line, Name);

  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Entity, N->getEntity());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(
      N, DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line, Name));

  EXPECT_NE(N,
            DIImportedEntity::get(Context, dwarf::DW_TAG_imported_declaration,
                                  Scope, Entity, File, Line, Name));
  EXPECT_NE(N, DIImportedEntity::get(Context, Tag, getSubprogram(), Entity,
                                     File, Line, Name));
  EXPECT_NE(N, DIImportedEntity::get(Context, Tag, Scope, getCompositeType(),
                                     File, Line, Name));
  EXPECT_NE(N, DIImportedEntity::get(Context, Tag, Scope, Entity, nullptr, Line,
                                     Name));
  EXPECT_NE(N, DIImportedEntity::get(Context, Tag, Scope, Entity, File,
                                     Line + 1, Name));
  EXPECT_NE(N, DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line,
                                     "other"));

  TempDIImportedEntity Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));

  MDTuple *Elements1 = getTuple();
  MDTuple *Elements2 = getTuple();
  auto *Ne = DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line,
                                   Name, Elements1);

  EXPECT_EQ(Elements1, Ne->getElements().get());

  EXPECT_EQ(Ne, DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line,
                                      Name, Elements1));
  EXPECT_NE(Ne, DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line,
                                      "ModOther", Elements1));
  EXPECT_NE(Ne, DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line,
                                      Name, Elements2));
  EXPECT_NE(
      Ne, DIImportedEntity::get(Context, Tag, Scope, Entity, File, Line, Name));

  TempDIImportedEntity Tempe = Ne->clone();
  EXPECT_EQ(Ne, MDNode::replaceWithUniqued(std::move(Tempe)));
}

typedef MetadataTest MetadataAsValueTest;

TEST_F(MetadataAsValueTest, MDNode) {
  MDNode *N = MDNode::get(Context, None);
  auto *V = MetadataAsValue::get(Context, N);
  EXPECT_TRUE(V->getType()->isMetadataTy());
  EXPECT_EQ(N, V->getMetadata());

  auto *V2 = MetadataAsValue::get(Context, N);
  EXPECT_EQ(V, V2);
}

TEST_F(MetadataAsValueTest, MDNodeMDNode) {
  MDNode *N = MDNode::get(Context, None);
  Metadata *Ops[] = {N};
  MDNode *N2 = MDNode::get(Context, Ops);
  auto *V = MetadataAsValue::get(Context, N2);
  EXPECT_TRUE(V->getType()->isMetadataTy());
  EXPECT_EQ(N2, V->getMetadata());

  auto *V2 = MetadataAsValue::get(Context, N2);
  EXPECT_EQ(V, V2);

  auto *V3 = MetadataAsValue::get(Context, N);
  EXPECT_TRUE(V3->getType()->isMetadataTy());
  EXPECT_NE(V, V3);
  EXPECT_EQ(N, V3->getMetadata());
}

TEST_F(MetadataAsValueTest, MDNodeConstant) {
  auto *C = ConstantInt::getTrue(Context);
  auto *MD = ConstantAsMetadata::get(C);
  Metadata *Ops[] = {MD};
  auto *N = MDNode::get(Context, Ops);

  auto *V = MetadataAsValue::get(Context, MD);
  EXPECT_TRUE(V->getType()->isMetadataTy());
  EXPECT_EQ(MD, V->getMetadata());

  auto *V2 = MetadataAsValue::get(Context, N);
  EXPECT_EQ(MD, V2->getMetadata());
  EXPECT_EQ(V, V2);
}

typedef MetadataTest ValueAsMetadataTest;

TEST_F(ValueAsMetadataTest, UpdatesOnRAUW) {
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV0(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  auto *MD = ValueAsMetadata::get(GV0.get());
  EXPECT_TRUE(MD->getValue() == GV0.get());
  ASSERT_TRUE(GV0->use_empty());

  std::unique_ptr<GlobalVariable> GV1(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  GV0->replaceAllUsesWith(GV1.get());
  EXPECT_TRUE(MD->getValue() == GV1.get());
}

TEST_F(ValueAsMetadataTest, TempTempReplacement) {
  // Create a constant.
  ConstantAsMetadata *CI =
      ConstantAsMetadata::get(ConstantInt::get(Context, APInt(8, 0)));

  auto Temp1 = MDTuple::getTemporary(Context, None);
  auto Temp2 = MDTuple::getTemporary(Context, {CI});
  auto *N = MDTuple::get(Context, {Temp1.get()});

  // Test replacing a temporary node with another temporary node.
  Temp1->replaceAllUsesWith(Temp2.get());
  EXPECT_EQ(N->getOperand(0), Temp2.get());

  // Clean up Temp2 for teardown.
  Temp2->replaceAllUsesWith(nullptr);
}

TEST_F(ValueAsMetadataTest, CollidingDoubleUpdates) {
  // Create a constant.
  ConstantAsMetadata *CI =
      ConstantAsMetadata::get(ConstantInt::get(Context, APInt(8, 0)));

  // Create a temporary to prevent nodes from resolving.
  auto Temp = MDTuple::getTemporary(Context, None);

  // When the first operand of N1 gets reset to nullptr, it'll collide with N2.
  Metadata *Ops1[] = {CI, CI, Temp.get()};
  Metadata *Ops2[] = {nullptr, CI, Temp.get()};

  auto *N1 = MDTuple::get(Context, Ops1);
  auto *N2 = MDTuple::get(Context, Ops2);
  ASSERT_NE(N1, N2);

  // Tell metadata that the constant is getting deleted.
  //
  // After this, N1 will be invalid, so don't touch it.
  ValueAsMetadata::handleDeletion(CI->getValue());
  EXPECT_EQ(nullptr, N2->getOperand(0));
  EXPECT_EQ(nullptr, N2->getOperand(1));
  EXPECT_EQ(Temp.get(), N2->getOperand(2));

  // Clean up Temp for teardown.
  Temp->replaceAllUsesWith(nullptr);
}

typedef MetadataTest DIArgListTest;

TEST_F(DIArgListTest, get) {
  SmallVector<ValueAsMetadata *, 2> VMs;
  VMs.push_back(
      ConstantAsMetadata::get(ConstantInt::get(Context, APInt(8, 0))));
  VMs.push_back(
      ConstantAsMetadata::get(ConstantInt::get(Context, APInt(2, 0))));
  DIArgList *DV0 = DIArgList::get(Context, VMs);
  DIArgList *DV1 = DIArgList::get(Context, VMs);
  EXPECT_EQ(DV0, DV1);
}

TEST_F(DIArgListTest, UpdatesOnRAUW) {
  Type *Ty = Type::getInt1PtrTy(Context);
  ConstantAsMetadata *CI =
      ConstantAsMetadata::get(ConstantInt::get(Context, APInt(8, 0)));
  std::unique_ptr<GlobalVariable> GV0(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  auto *MD0 = ValueAsMetadata::get(GV0.get());

  SmallVector<ValueAsMetadata *, 2> VMs;
  VMs.push_back(CI);
  VMs.push_back(MD0);
  auto *AL = DIArgList::get(Context, VMs);
  EXPECT_EQ(AL->getArgs()[0], CI);
  EXPECT_EQ(AL->getArgs()[1], MD0);

  std::unique_ptr<GlobalVariable> GV1(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  auto *MD1 = ValueAsMetadata::get(GV1.get());
  GV0->replaceAllUsesWith(GV1.get());
  EXPECT_EQ(AL->getArgs()[0], CI);
  EXPECT_EQ(AL->getArgs()[1], MD1);
}

typedef MetadataTest TrackingMDRefTest;

TEST_F(TrackingMDRefTest, UpdatesOnRAUW) {
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV0(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  TypedTrackingMDRef<ValueAsMetadata> MD(ValueAsMetadata::get(GV0.get()));
  EXPECT_TRUE(MD->getValue() == GV0.get());
  ASSERT_TRUE(GV0->use_empty());

  std::unique_ptr<GlobalVariable> GV1(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  GV0->replaceAllUsesWith(GV1.get());
  EXPECT_TRUE(MD->getValue() == GV1.get());

  // Reset it, so we don't inadvertently test deletion.
  MD.reset();
}

TEST_F(TrackingMDRefTest, UpdatesOnDeletion) {
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  TypedTrackingMDRef<ValueAsMetadata> MD(ValueAsMetadata::get(GV.get()));
  EXPECT_TRUE(MD->getValue() == GV.get());
  ASSERT_TRUE(GV->use_empty());

  GV.reset();
  EXPECT_TRUE(!MD);
}

TEST(NamedMDNodeTest, Search) {
  LLVMContext Context;
  ConstantAsMetadata *C =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), 1));
  ConstantAsMetadata *C2 =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), 2));

  Metadata *const V = C;
  Metadata *const V2 = C2;
  MDNode *n = MDNode::get(Context, V);
  MDNode *n2 = MDNode::get(Context, V2);

  Module M("MyModule", Context);
  const char *Name = "llvm.NMD1";
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(Name);
  NMD->addOperand(n);
  NMD->addOperand(n2);

  std::string Str;
  raw_string_ostream oss(Str);
  NMD->print(oss);
  EXPECT_STREQ("!llvm.NMD1 = !{!0, !1}\n",
               oss.str().c_str());
}

typedef MetadataTest FunctionAttachmentTest;
TEST_F(FunctionAttachmentTest, setMetadata) {
  Function *F = getFunction("foo");
  ASSERT_FALSE(F->hasMetadata());
  EXPECT_EQ(nullptr, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(nullptr, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  DISubprogram *SP1 = getSubprogram();
  DISubprogram *SP2 = getSubprogram();
  ASSERT_NE(SP1, SP2);

  F->setMetadata("dbg", SP1);
  EXPECT_TRUE(F->hasMetadata());
  EXPECT_EQ(SP1, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(SP1, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  F->setMetadata(LLVMContext::MD_dbg, SP2);
  EXPECT_TRUE(F->hasMetadata());
  EXPECT_EQ(SP2, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(SP2, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  F->setMetadata("dbg", nullptr);
  EXPECT_FALSE(F->hasMetadata());
  EXPECT_EQ(nullptr, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(nullptr, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  MDTuple *T1 = getTuple();
  MDTuple *T2 = getTuple();
  ASSERT_NE(T1, T2);

  F->setMetadata("other1", T1);
  F->setMetadata("other2", T2);
  EXPECT_TRUE(F->hasMetadata());
  EXPECT_EQ(T1, F->getMetadata("other1"));
  EXPECT_EQ(T2, F->getMetadata("other2"));
  EXPECT_EQ(nullptr, F->getMetadata("dbg"));

  F->setMetadata("other1", T2);
  F->setMetadata("other2", T1);
  EXPECT_EQ(T2, F->getMetadata("other1"));
  EXPECT_EQ(T1, F->getMetadata("other2"));

  F->setMetadata("other1", nullptr);
  F->setMetadata("other2", nullptr);
  EXPECT_FALSE(F->hasMetadata());
  EXPECT_EQ(nullptr, F->getMetadata("other1"));
  EXPECT_EQ(nullptr, F->getMetadata("other2"));
}

TEST_F(FunctionAttachmentTest, getAll) {
  Function *F = getFunction("foo");

  MDTuple *T1 = getTuple();
  MDTuple *T2 = getTuple();
  MDTuple *P = getTuple();
  DISubprogram *SP = getSubprogram();

  F->setMetadata("other1", T2);
  F->setMetadata(LLVMContext::MD_dbg, SP);
  F->setMetadata("other2", T1);
  F->setMetadata(LLVMContext::MD_prof, P);
  F->setMetadata("other2", T2);
  F->setMetadata("other1", T1);

  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  F->getAllMetadata(MDs);
  ASSERT_EQ(4u, MDs.size());
  EXPECT_EQ(LLVMContext::MD_dbg, MDs[0].first);
  EXPECT_EQ(LLVMContext::MD_prof, MDs[1].first);
  EXPECT_EQ(Context.getMDKindID("other1"), MDs[2].first);
  EXPECT_EQ(Context.getMDKindID("other2"), MDs[3].first);
  EXPECT_EQ(SP, MDs[0].second);
  EXPECT_EQ(P, MDs[1].second);
  EXPECT_EQ(T1, MDs[2].second);
  EXPECT_EQ(T2, MDs[3].second);
}

TEST_F(FunctionAttachmentTest, Verifier) {
  Function *F = getFunction("foo");
  F->setMetadata("attach", getTuple());
  F->setIsMaterializable(true);

  // Confirm this is materializable.
  ASSERT_TRUE(F->isMaterializable());

  // Materializable functions cannot have metadata attachments.
  EXPECT_TRUE(verifyFunction(*F));

  // Function declarations can.
  F->setIsMaterializable(false);
  EXPECT_FALSE(verifyModule(*F->getParent()));
  EXPECT_FALSE(verifyFunction(*F));

  // So can definitions.
  (void)new UnreachableInst(Context, BasicBlock::Create(Context, "bb", F));
  EXPECT_FALSE(verifyModule(*F->getParent()));
  EXPECT_FALSE(verifyFunction(*F));
}

TEST_F(FunctionAttachmentTest, RealEntryCount) {
  Function *F = getFunction("foo");
  EXPECT_FALSE(F->getEntryCount().hasValue());
  F->setEntryCount(12304, Function::PCT_Real);
  auto Count = F->getEntryCount();
  EXPECT_TRUE(Count.hasValue());
  EXPECT_EQ(12304u, Count->getCount());
  EXPECT_EQ(Function::PCT_Real, Count->getType());
}

TEST_F(FunctionAttachmentTest, SyntheticEntryCount) {
  Function *F = getFunction("bar");
  EXPECT_FALSE(F->getEntryCount().hasValue());
  F->setEntryCount(123, Function::PCT_Synthetic);
  auto Count = F->getEntryCount(true /*allow synthetic*/);
  EXPECT_TRUE(Count.hasValue());
  EXPECT_EQ(123u, Count->getCount());
  EXPECT_EQ(Function::PCT_Synthetic, Count->getType());
}

TEST_F(FunctionAttachmentTest, SubprogramAttachment) {
  Function *F = getFunction("foo");
  DISubprogram *SP = getSubprogram();
  F->setSubprogram(SP);

  // Note that the static_cast confirms that F->getSubprogram() actually
  // returns an DISubprogram.
  EXPECT_EQ(SP, static_cast<DISubprogram *>(F->getSubprogram()));
  EXPECT_EQ(SP, F->getMetadata("dbg"));
  EXPECT_EQ(SP, F->getMetadata(LLVMContext::MD_dbg));
}

typedef MetadataTest DistinctMDOperandPlaceholderTest;
TEST_F(DistinctMDOperandPlaceholderTest, getID) {
  EXPECT_EQ(7u, DistinctMDOperandPlaceholder(7).getID());
}

TEST_F(DistinctMDOperandPlaceholderTest, replaceUseWith) {
  // Set up some placeholders.
  DistinctMDOperandPlaceholder PH0(7);
  DistinctMDOperandPlaceholder PH1(3);
  DistinctMDOperandPlaceholder PH2(0);
  Metadata *Ops[] = {&PH0, &PH1, &PH2};
  auto *D = MDTuple::getDistinct(Context, Ops);
  ASSERT_EQ(&PH0, D->getOperand(0));
  ASSERT_EQ(&PH1, D->getOperand(1));
  ASSERT_EQ(&PH2, D->getOperand(2));

  // Replace them.
  auto *N0 = MDTuple::get(Context, None);
  auto *N1 = MDTuple::get(Context, N0);
  PH0.replaceUseWith(N0);
  PH1.replaceUseWith(N1);
  PH2.replaceUseWith(nullptr);
  EXPECT_EQ(N0, D->getOperand(0));
  EXPECT_EQ(N1, D->getOperand(1));
  EXPECT_EQ(nullptr, D->getOperand(2));
}

TEST_F(DistinctMDOperandPlaceholderTest, replaceUseWithNoUser) {
  // There is no user, but we can still call replace.
  DistinctMDOperandPlaceholder(7).replaceUseWith(MDTuple::get(Context, None));
}

// Test various assertions in metadata tracking. Don't run these tests if gtest
// will use SEH to recover from them. Two of these tests get halfway through
// inserting metadata into DenseMaps for tracking purposes, and then they
// assert, and we attempt to destroy an LLVMContext with broken invariants,
// leading to infinite loops.
#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG) && !defined(GTEST_HAS_SEH)
TEST_F(DistinctMDOperandPlaceholderTest, MetadataAsValue) {
  // This shouldn't crash.
  DistinctMDOperandPlaceholder PH(7);
  EXPECT_DEATH(MetadataAsValue::get(Context, &PH),
               "Unexpected callback to owner");
}

TEST_F(DistinctMDOperandPlaceholderTest, UniquedMDNode) {
  // This shouldn't crash.
  DistinctMDOperandPlaceholder PH(7);
  EXPECT_DEATH(MDTuple::get(Context, &PH), "Unexpected callback to owner");
}

TEST_F(DistinctMDOperandPlaceholderTest, SecondDistinctMDNode) {
  // This shouldn't crash.
  DistinctMDOperandPlaceholder PH(7);
  MDTuple::getDistinct(Context, &PH);
  EXPECT_DEATH(MDTuple::getDistinct(Context, &PH),
               "Placeholders can only be used once");
}

TEST_F(DistinctMDOperandPlaceholderTest, TrackingMDRefAndDistinctMDNode) {
  // TrackingMDRef doesn't install an owner callback, so it can't be detected
  // as an invalid use.  However, using a placeholder in a TrackingMDRef *and*
  // a distinct node isn't possible and we should assert.
  //
  // (There's no positive test for using TrackingMDRef because it's not a
  // useful thing to do.)
  {
    DistinctMDOperandPlaceholder PH(7);
    MDTuple::getDistinct(Context, &PH);
    EXPECT_DEATH(TrackingMDRef Ref(&PH), "Placeholders can only be used once");
  }
  {
    DistinctMDOperandPlaceholder PH(7);
    TrackingMDRef Ref(&PH);
    EXPECT_DEATH(MDTuple::getDistinct(Context, &PH),
                 "Placeholders can only be used once");
  }
}
#endif

typedef MetadataTest DebugVariableTest;
TEST_F(DebugVariableTest, DenseMap) {
  DenseMap<DebugVariable, uint64_t> DebugVariableMap;

  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  DIType *Type = getDerivedType();
  DINode::DIFlags Flags = static_cast<DINode::DIFlags>(7);

  DILocation *InlinedLoc = DILocation::get(Context, 2, 7, Scope);

  DILocalVariable *VarA =
      DILocalVariable::get(Context, Scope, "A", File, 5, Type, 2, Flags, 8, nullptr);
  DILocalVariable *VarB =
      DILocalVariable::get(Context, Scope, "B", File, 7, Type, 3, Flags, 8, nullptr);

  DebugVariable DebugVariableA(VarA, NoneType(), nullptr);
  DebugVariable DebugVariableInlineA(VarA, NoneType(), InlinedLoc);
  DebugVariable DebugVariableB(VarB, NoneType(), nullptr);
  DebugVariable DebugVariableFragB(VarB, {{16, 16}}, nullptr);

  DebugVariableMap.insert({DebugVariableA, 2});
  DebugVariableMap.insert({DebugVariableInlineA, 3});
  DebugVariableMap.insert({DebugVariableB, 6});
  DebugVariableMap.insert({DebugVariableFragB, 12});

  EXPECT_EQ(DebugVariableMap.count(DebugVariableA), 1u);
  EXPECT_EQ(DebugVariableMap.count(DebugVariableInlineA), 1u);
  EXPECT_EQ(DebugVariableMap.count(DebugVariableB), 1u);
  EXPECT_EQ(DebugVariableMap.count(DebugVariableFragB), 1u);

  EXPECT_EQ(DebugVariableMap.find(DebugVariableA)->second, 2u);
  EXPECT_EQ(DebugVariableMap.find(DebugVariableInlineA)->second, 3u);
  EXPECT_EQ(DebugVariableMap.find(DebugVariableB)->second, 6u);
  EXPECT_EQ(DebugVariableMap.find(DebugVariableFragB)->second, 12u);
}

} // end namespace
